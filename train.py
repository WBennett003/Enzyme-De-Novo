import torch
import numpy as np
import h5py
import math
import matplotlib.pyplot as plt
import time
import wandb

from transformer_model import Transformer
from visualisation import plot_prediction, plot_elem_confusion, plot_bonding_confusion, plot_bonding_matrix

if torch.cuda.is_available():
  dev = "CUDA:0"
else:
  dev = "CPU"

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

#Hyperparameters
THEOZYME_SIZE = 250
COMPOUND_SIZE = 250
ELEMENT_SIZE = 16
BOND_SIZE = 4

NHEADS = 6
NBLOCKS = 6
DMODEL  = 255 #embedding size
DFF = 2056 #high dim upscale in fead forward layer

BATCH_SIZE = 10
LEARNING_RATE = 0.02
EPOCHS = 20

wandb.init(
    # set the wandb project where this run will be logged
    project="Theozmye-Transformer",
    
    # track hyperparameters and run metadata
    config={
    "elems" : ELEMENT_SIZE,
    "bonds" : BOND_SIZE,
    "theozyme size" :  THEOZYME_SIZE,
    "compound size" : COMPOUND_SIZE,
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
    print(f"{count}/{batch_length} : loss {loss} : {round(time.time()-start, 2)}s")
    count += 1

  start = batch_length * epoch
  end = batch_length * (epoch+1)
  temp_losses = np.array(losses[start:end])
  avg_loss.append([
      temp_losses[:, 0].mean(),
      temp_losses[:, 1].mean(),
      temp_losses[:, 2].mean(),
  ])
  print(f"epoch {epoch} : loss {avg_loss[epoch]}")
  elem_fig = plot_elem_confusion(TE, pred_elem.argmax(-1), get_figure=True, n_elems=ELEMENT_SIZE)
  bond_fig = plot_bonding_confusion(TADJ, pred_bonding.argmax(-1), get_figure=True, n_bonds=BOND_SIZE)
  matrix_fig = plot_bonding_matrix(TADJ[0], pred_bonding[0].argmax(-1), get_figure=True)

  wandb.log({
      "epoch" : epoch,
      "loss" : sum(avg_loss[epoch]),
      "loss-elem" : avg_loss[epoch][0],
      "loss-pos" : avg_loss[epoch][1],
      "loss-bonding" : avg_loss[epoch][2],
      "elem_figure" : wandb.Image(elem_fig),
      "bond_figure" : wandb.Image(bond_fig),
      "maxtrix_figure" : wandb.Image(matrix_fig),
  })
  torch.save(model.state_dict(), 'weights/transformer.pt')

