import torch
import numpy as np
import h5py
import math
import matplotlib.pyplot as plt
import time
import wandb

from transformer_model import Transformer
#simple proof of concept model

class dataset(torch.utils.data.Dataset):
  def __init__(self, file_path='datasets/dense_bonding10page.h5py'):
    super().__init__()
    self.file = h5py.File(file_path, 'r+')

  def __len__(self):
    return len(self.file['reactant_elem'])
  
  def __getitem__(self, idx):
    return (self.file['reactant_elem'][idx], self.file['reactant_charge'][idx], self.file['reactant_pos'][idx],
    self.file['reactant_adj'][idx], 
    self.file['product_elem'][idx], self.file['product_charge'][idx], self.file['product_pos'][idx],
    self.file['product_adj'][idx], 
    self.file['theozyme_elem'][idx], self.file['theozyme_pos'][idx], self.file['theozyme_adj'][idx])

def get_target(TE, TP, TBI, max_length=1000):
  TE = TE.roll(-1, 1)
  TP = TP.roll(-1, 1)
  TBI = TBI.roll(-1, 1)
  #so the last element is not the first target

  return TE, TP, TBI

THEOZYME_SIZE = 1000
COMPOUND_SIZE = 1000
ELEMENT_SIZE = 30
BOND_SIZE = 8

NHEADS = 1
NBLOCKS = 1
DMODEL  = 5 #embedding size
DFF = 10 #high dim upscale in fead forward layer

BATCH_SIZE = 1
LEARNING_RATE = 0.02
EPOCHS = 5

wandb.init(
    # set the wandb project where this run will be logged
    project="Theozmye-Transformer",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": LEARNING_RATE,
    "n_heads": NHEADS,
    "n_blocks": NBLOCKS,
    "DMODEL": DMODEL,
    "DFF": DFF,
    "batch_size": BATCH_SIZE,
    "dataset": "10page_dense",
    "epochs": EPOCHS,
    }
)

data = dataset()
loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
model = Transformer(nheads=NHEADS, nblocks=NBLOCKS, dff=DFF, dmodel=DMODEL, elem_size=ELEMENT_SIZE, nbonds=BOND_SIZE, theozyme_size=THEOZYME_SIZE, compound_size=COMPOUND_SIZE)
# model.load_state_dict(torch.load('weights/transformer.pt'))

loss_func_elemnt = torch.nn.CrossEntropyLoss()
loss_func_bonding = torch.nn.CrossEntropyLoss()
loss_func_pos = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

epochs = EPOCHS
batch_length = len(loader)


outputs = []
losses = []
avg_loss = []


for epoch in range(epochs):
  count = 0
  for RE, RC, RP, RADJ, PE, PC, PP, PADJ, TE, TP, TADJ in loader:
    start = time.time()
    tgtE, tgtP, tgtADJ = get_target(TE, TP, TADJ, THEOZYME_SIZE)

    pred_elem, pred_pos, pred_bonding = model(RE, RC, RP, RADJ, PE, PC, PP, PADJ, tgtE, tgtP, tgtADJ)

    loss_elem = loss_func_elemnt(pred_elem, torch.nn.functional.one_hot(TE.long(), ELEMENT_SIZE).float())
    loss_pos = loss_func_pos(pred_pos, TP)
    loss_bonding = loss_func_pos(pred_bonding, torch.nn.functional.one_hot(TADJ.long(), BOND_SIZE).float())
    loss = loss_elem + loss_bonding + loss_pos

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append([loss_elem.detach().numpy(), loss_pos.detach().numpy(), loss_bonding.detach().numpy()])
    print(f"{count}/{batch_length} : loss {loss.detach().numpy()} : {round(time.time()-start, 2)}s")
    count += 1

  print(f"epoch {epoch} : loss {loss_elem, loss_pos, loss_bonding}")
  torch.save(model.state_dict(), 'weights/transformer.pt')
  start = batch_length * epoch
  end = batch_length * (epoch+1)
  temp_losses = np.array(losses[start:end])
  avg_loss.append([
      temp_losses[0].mean(),
      temp_losses[1].mean(),
      temp_losses[2].mean(),
      temp_losses[3].mean(),
  ])

  wandb.log({
      "epoch" : epoch,
      "loss" : avg_loss[epoch][0],
      "loss-elem" : avg_loss[epoch][1],
      "loss-pos" : avg_loss[epoch][2],
      "loss-pos" : avg_loss[epoch][3],
      "weights" : model.state_dict()
  })
n = 10
losses = np.array(losses).T
plt.plot(np.arange(n), losses[0, -n:, ])
plt.plot(np.arange(n), losses[1, -n:, ])
plt.plot(np.arange(n), losses[2, -n:, ])
plt.show()