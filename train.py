import torch
import numpy as np
import h5py
import math
import matplotlib.pyplot as plt

from transformer_model import Transformer
#simple proof of concept model

class dataset(torch.utils.data.Dataset):
  def __init__(self, file_path='datasets/testdataset.h5py'):
    super().__init__()
    self.file = h5py.File(file_path, 'r+')

  def __len__(self):
    return len(self.file['reactant_elem'])
  
  def __getitem__(self, idx):
    return (self.file['reactant_elem'][idx], self.file['reactant_charge'][idx], self.file['reactant_pos'][idx],
    self.file['reactant_bond_idx'][idx], self.file['reactant_bond_type'][idx], 
    self.file['product_elem'][idx], self.file['product_charge'][idx], self.file['product_pos'][idx],
    self.file['product_bond_idx'][idx], self.file['product_bond_type'][idx], 
    self.file['theozyme_elem'][idx], self.file['theozyme_pos'][idx], self.file['theozyme_bond_idx'][idx], self.file['theozyme_bond_type'][idx]
    )

def get_target(TE, TP, TBI, max_length=1000):
  TE = TE.roll(-1, 1)
  TP = TP.roll(-1, 1)
  #so the last element is not the first target
  TBI += 1 #as i have rolled each element down one

  return TE, TP, TBI

THEOZYME_SIZE = 1000
COMPOUND_SIZE = 500
ELEMENT_SIZE = 20
BOND_SIZE = 8

NHEADS = 2
NBLOCKS = 2
DMODEL  = 20 #embedding size
DFF = 255 #high dim upscale in fead forward layer

BATCH_SIZE = 4

data = dataset()
loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
model = Transformer(nheads=NHEADS, nblocks=NBLOCKS, dff=DFF, dmodel=DMODEL, elem_size=ELEMENT_SIZE, nbonds=BOND_SIZE, theozyme_size=THEOZYME_SIZE, compound_size=COMPOUND_SIZE)
# model.load_state_dict(torch.load('weights/transformer.pt'))

loss_func_elemnt = torch.nn.CrossEntropyLoss()
loss_func_bonding = torch.nn.CrossEntropyLoss()
loss_func_pos = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

epochs = 100

outputs = []
losses = []

for epoch in range(epochs):
  for RE, RC, RP, RBI, RBT, PE, PC, PP, PBI, PBT, TE, TP, TBI, TBT in loader:
    tgtE, tgtP, tgtBI = get_target(TE, TP, TBI, THEOZYME_SIZE)
    pred_elem, pred_pos, pred_bonding = model(RE, RC, RP, RBI, RBT, PE, PC, PP, PBI, PBT, tgtE, tgtP, tgtBI, TBT)

    bonding = torch.sparse_coo_tensor(TBI, TBT, pred_bonding.shape)
    bonding = bonding.to_dense()

    zeros = torch.zeros((TE.shape[0], TE.shape[1], pred_elem.shape[-1]))
    mask = torch.arange(pred_elem.shape[-1])[None, None, :] == TE[:, :, None]
    zeros[mask] = 1
    TE = zeros

    loss_elem = loss_func_elemnt(pred_elem, TE)
    loss_pos = loss_func_pos(pred_pos, TP)
    loss_bonding = loss_func_pos(pred_bonding, bonding)
    loss = loss_elem + loss_bonding + loss_pos


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append([loss_elem.detach().numpy(), loss_pos.detach().numpy(), loss_bonding.detach().numpy()])
  print(f"epoch {epoch} : loss {loss_elem, loss_pos, loss_bonding}")
  torch.save(model.state_dict(), 'weights/transformer.pt')

n = 10
losses = np.array(losses).T
plt.plot(np.arange(n), losses[0, -n:, ])
plt.plot(np.arange(n), losses[1, -n:, ])
plt.plot(np.arange(n), losses[2, -n:, ])
plt.show()