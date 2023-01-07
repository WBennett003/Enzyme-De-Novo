import torch
import wandb

def look_ahead_mask(shape):
  mask = torch.arange(shape)[None, :] > torch.arange(shape)[:, None]
  return mask


def look_ahead_attention(q, k, v, look_ahead_mask=None):
  x = torch.matmul(q, torch.einsum('ijk->ikj', k))
  x = x/k.shape[1]**0.5

  x[:, look_ahead_mask] = float('-inf')      

  x = torch.softmax(x, dim=2)
  x = torch.matmul(x, v)
  return x

def scaled_product_attention(q, k, v, weights=None, mask=None, padding_mask=None, look_ahead_mask=None):
  x = torch.matmul(q, torch.einsum('ijk->ikj', k))
  x = x/k.shape[1]**0.5

  if weights is not None:
    x = (x[:, :, :, None] * weights).sum(-1)

  if look_ahead_mask is not None:
    x[:, look_ahead_mask] = float('-inf')      

  
  if padding_mask is not None:
    x[padding_mask] = float('-inf')

  x = torch.softmax(x, dim=2)
  x = torch.matmul(x, v)
  return x

class Graph_Attention(torch.nn.Module):
  def __init__(self, insize, outsize, nheads, attn=scaled_product_attention):
    super().__init__()
    self.insize = insize
    self.outsize = outsize
    self.nheads = nheads

    self.attn = attn

    self.wq = [torch.nn.Linear(self.insize, self.insize) for i in range(nheads)]
    self.wk = [torch.nn.Linear(self.insize, self.insize) for i in range(nheads)]
    self.wv = [torch.nn.Linear(self.insize, self.insize) for i in range(nheads)]

    self.out = torch.nn.Linear(self.insize*nheads, self.outsize)

  def forward(self, nodefeatures, nodeadj, padding_mask=None, look_ahead_mask=None):
    qkv = []
    for head in range(self.nheads):
      wq = self.wq[head](nodefeatures)#.relu()
      wk = self.wk[head](nodefeatures)#.relu()
      wv = self.wv[head](nodefeatures)#.relu() 
      attn = self.attn(wq, wk, wv, weights=nodeadj, look_ahead_mask=look_ahead_mask)
  
      qkv.append(attn)
    qkv = torch.concat(qkv, axis=-1)
    qkv = self.out(qkv)
    return qkv

class MHA(torch.nn.Module):
  def __init__(self, dff, nheads, attn=scaled_product_attention):
    super().__init__()
    self.dff = dff
    self.nheads = nheads

    self.attn = attn

    
    self.wq = [torch.nn.Linear(self.dff, self.dff) for i in range(nheads)]
    self.wk = [torch.nn.Linear(self.dff, self.dff) for i in range(nheads)]
    self.wv = [torch.nn.Linear(self.dff, self.dff) for i in range(nheads)]

    self.out = torch.nn.Linear(self.dff*nheads, self.dff)


  def forward(self, Q, K, V, mask=None):
    qkv = []
    for head in range(self.nheads):
      wq = self.wq[head](Q)#.relu()
      wk = self.wk[head](K)#.relu()
      wv = self.wv[head](V)#.relu() 
      attn = self.attn(wq, wk, wv, mask)
  
      qkv.append(attn)
    qkv = torch.concat(qkv, axis=-1)
    qkv = self.out(qkv)
    return qkv

class compound_layer(torch.nn.Module):
  def __init__(self, dmodel, nheads, elem_size, n_bond_types, charge_size, dff):
    super().__init__()
    self.elem_embedding = torch.nn.Embedding(elem_size, dmodel)
    
    self.charge_layer = torch.nn.Embedding(charge_size, dmodel)

    self.bond_embedding = torch.nn.Embedding(n_bond_types, dmodel)

    # self.charge_layer = torch.nn.Sequential(
    #   torch.nn.Linear(1, dff),
    #   torch.nn.ReLU()
    # )
    self.pos_layer = torch.nn.Sequential(
      torch.nn.Linear(3, dmodel),
      torch.nn.ReLU()
    )
    self.ff = torch.nn.Sequential(
      torch.nn.Linear(3*dmodel, dmodel),
      torch.nn.ReLU()      
    )
    self.bonding_layer = Graph_Attention(dmodel, dmodel, nheads=nheads)


  def forward(self, CE, CC, CP, CADJ):
    padding_mask = get_padding(CE)
    CE = self.elem_embedding(CE)
    CC = self.charge_layer(CC)
    CP = self.pos_layer(CP)

    CB = self.bond_embedding(CADJ)

    C = torch.concat([CE, CC, CP], axis=-1)
    C = self.ff(C)

    C = self.bonding_layer(C, CB, padding_mask=padding_mask)

    return C

class theozyme_layer(torch.nn.Module):
  def __init__(self, dmodel, nheads, elem_size, n_bond_types, dff):
    super().__init__()
    self.elem_embedding = torch.nn.Embedding(elem_size, dmodel)
    self.bonding_embedding = torch.nn.Embedding(n_bond_types, dmodel)
    self.pos_layer = torch.nn.Sequential(
      torch.nn.Linear(3, dmodel),
      torch.nn.ReLU()
    )
    self.ff = torch.nn.Sequential(
      torch.nn.Linear(2*dmodel, dmodel),
      torch.nn.ReLU(),
    )
    self.bonding_layer = Graph_Attention(dmodel, dmodel, nheads=nheads)

  def forward(self, TE, TP, TADJ):
    padding_mask = get_padding(TE)
    mask = look_ahead_mask(TE.shape[1])
    TE = self.elem_embedding(TE)
    TP = self.pos_layer(TP)
    T = torch.concat([TE, TP], axis=-1)
    T = self.ff(T)
    TB = self.bonding_embedding(TADJ)
    T = self.bonding_layer(T, TB, padding_mask=padding_mask, look_ahead_mask=mask)
    return T

class encoder_layer(torch.nn.Module):
  def __init__(self, dmodel, nheads, dff):
    super().__init__()

    self.attn = MHA(dmodel, nheads=nheads)
    self.feadforward = torch.nn.Sequential(
        torch.nn.Linear(dmodel, dff),
        torch.nn.ReLU(),
        torch.nn.Linear(dff, dmodel)
    )

    self.norm = torch.nn.LayerNorm(dmodel)

  def forward(self, x, padding_mask=None):
    attn = self.attn(x, x, x)
    x = self.norm(attn+x)
    ff = self.feadforward(x)
    x = self.norm(ff+x)
    return x


def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class decoder_layer(torch.nn.Module):   
  def __init__(self, dmodel, nheads, dff):
    super().__init__()

    self.attn = MHA(dmodel, nheads=nheads, attn=look_ahead_attention)
    self.feadforward = torch.nn.Sequential(
        torch.nn.Linear(dmodel, dff),
        torch.nn.ReLU(),
        torch.nn.Linear(dff, dmodel)
    )
    self.norm = torch.nn.LayerNorm(dmodel)

    self.cross_attention = MHA(dmodel, nheads)


  def forward(self, latent_x, x, mask, src_padding_mask=None, tgt_padding_mask=None):

    attn = self.attn(x, x, x, mask)
    x = self.norm(attn+x)

    attn = self.cross_attention(x, latent_x, latent_x, mask=True)
    x = self.norm(attn+x)

    ff = self.feadforward(x)
    x = self.norm(ff+x)

    return x

class Transducer(torch.nn.Module): # Latent into Elem, Charge, Pos, Bond idx, and bond types
  def __init__(self, theozyme_size, nbonds, elem_size, dmodel, dff):
    super().__init__()
    self.dmodel = dmodel


    self.elem_layer = torch.nn.Sequential(
      torch.nn.Linear(dmodel, dmodel),
      torch.nn.ReLU(),
      torch.nn.Linear(dmodel, elem_size),
      torch.nn.Softmax()
    )

    self.pos_layer = torch.nn.Sequential(
      torch.nn.Linear(dmodel, dmodel),
      torch.nn.ReLU(),
      torch.nn.Linear(dmodel, 3),
    )

    self.bonding_layer1 = torch.nn.Sequential(
      torch.nn.Linear(dmodel, dmodel*theozyme_size),
      torch.nn.ReLU()
      )
    self.bonding_layer2 = torch.nn.Sequential(
      torch.nn.Linear(dmodel, dmodel),
      torch.nn.ReLU(),
      torch.nn.Linear(dmodel, nbonds),
      torch.nn.Softmax()
    )

  def forward(self, latent_x):
    elem = self.elem_layer(latent_x)
    pos = self.pos_layer(latent_x)

    upscale_x = self.bonding_layer1(latent_x)
    upscale_x = upscale_x.reshape((-1, upscale_x.shape[1], upscale_x.shape[1], self.dmodel))
    latent_x = self.bonding_layer2(upscale_x)
    return elem, pos, latent_x # returns the elem, pos, latent_X
    

class Transformer(torch.nn.Module):
    def __init__(self, compound_size=500, compound_channel_size=5, theozyme_size=1000, theozyme_channel_size=7,elem_size=50, nbonds=8, dropout=0.1, charge_size=8, dmodel=20, dff=200, nheads=4, nblocks=4):   
        super().__init__()
        self.compound_size = compound_size
        self.compound_channel_size = compound_channel_size
        self.theozyme_size = theozyme_size
        self.theozyme_channel_size = theozyme_channel_size
        self.nbonds = nbonds
        self.charge_size = charge_size
        self.dff = dff
        self.dmodel = dmodel
        self.nheads = nheads
        self.nblocks = nblocks

        self.reactant_layer = compound_layer(dmodel, nheads, elem_size, nbonds, charge_size, dff)
        self.product_layer = compound_layer(dmodel, nheads, elem_size, nbonds, charge_size, dff)

        self.compound_layer = torch.nn.Sequential(
          torch.nn.Linear(2*dmodel, dmodel),
          torch.nn.ReLU()
        )
        
        self.theozyme_layer = theozyme_layer(dmodel, nheads, elem_size, nbonds, dff)

        self.encoder = [encoder_layer(dmodel, nheads, dff) for i in range(nblocks)]
        self.decoder = [decoder_layer(dmodel, nheads, dff) for i in range(nblocks)]

        self.output = Transducer(theozyme_size, nbonds, elem_size, dmodel=dmodel, dff=dff)

    def forward(self, RE, RC, RP, RADJ, PE, PC, PP, PADJ, TE, TP, TADJ):
        mask = look_ahead_mask(self.theozyme_size)

        rx = self.reactant_layer(RE, RC, RP, RADJ)

        px = self.product_layer(PE, PC, PP, PADJ)
        
        x = torch.concat([rx, px], axis=-1)
        x = self.compound_layer(x)

        target_x = self.theozyme_layer(TE, TP, TADJ)
        
        for encoder in self.encoder:
          x = encoder(x)

        for decoder in self.decoder:
          target_x = decoder(x, target_x, mask)

        xe, xp, xbonding = self.output(target_x)

        return xe, xp, xbonding

    def train(self, dataset, epochs=10, batch_size=10, learning_rate=0.02, weights_dir=None):
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_szie, shuffle=True)
        
        if weights_dir is not None:
            self.load_state_dict(torch.load(weights_dir))

        loss_func_elemnt = torch.nn.CrossEntropyLoss()
        loss_func_bonding = torch.nn.CrossEntropyLoss()
        loss_func_pos = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning)

        batch_length = len(loader)

        outputs = []
        losses = []
        avg_loss = []

        for epoch in range(epochs):
            count = 0
            for RE, RC, RP, RADJ, PE, PC, PP, PADJ, TE, TP, TADJ in loader:
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
                print(f"{count}/{batch_length} : loss {loss.detach().numpy()}")
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
                "weights" : self.state_dict()
            })

def get_padding(element): #arg element : (Batch size, sequence length, int)
  lengths = element.argmin(1)#gets length of nonpadded array, as it returns the index of the first zero
  mask = torch.arange(element.shape[1])[None, :] > lengths[:, None]
  mask = mask[:, :, None].repeat(1, 1, mask.shape[1])
  
  return mask
