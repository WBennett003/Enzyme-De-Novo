import torch

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

  if mask is not None:
    mask = mask.long()
    zeros = torch.zeros((x.shape[0], x.shape[1], x.shape[2]))
    # zeros = torch.ones(x.shape) * float('-inf')
    for b in range(x.shape[0]):
      wa = (x[b, mask[b, 0, :], mask[b, 1, :]][:, None] * weights[b])
      wa = wa.sum(axis=1)
      zeros[b, mask[b, 0, :], mask[b, 1, :]] = wa
    x = zeros

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

  def forward(self, nodefeatures, node_indices, nodeattributes, padding_mask=None, look_ahead_mask=None):
    qkv = []
    for head in range(self.nheads):
      wq = self.wq[head](nodefeatures)#.relu()
      wk = self.wk[head](nodefeatures)#.relu()
      wv = self.wv[head](nodefeatures)#.relu() 
      attn = self.attn(wq, wk, wv, mask=node_indices, weights=nodeattributes, look_ahead_mask=look_ahead_mask)
  
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
  def __init__(self, dmodel, nheads, elem_size, n_bond_types, charge_size):
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


  def forward(self, CE, CC, CP, CBI, CBT):
    padding_mask = get_padding(CE)
    CE = self.elem_embedding(CE)
    CC = self.charge_layer(CC)
    CP = self.pos_layer(CP)

    CB = self.bond_embedding(CBT)

    C = torch.concat([CE, CC, CP], axis=-1)
    C = self.ff(C)

    C = self.bonding_layer(C, CBI, CB, padding_mask=padding_mask)

    return C

class theozyme_layer(torch.nn.Module):
  def __init__(self, dff, nheads, elem_size, n_bond_types):
    super().__init__()
    self.elem_embedding = torch.nn.Embedding(elem_size, dff)
    self.pos_layer = torch.nn.Sequential(
      torch.nn.Linear(3, dff),
      torch.nn.ReLU()
    )
    self.ff = torch.nn.Sequential(
      torch.nn.Linear(2*dff, dff),
      torch.nn.ReLU()      
    )
    self.bonding_layer = Graph_Attention(dff, dff, nheads=nheads)

  def forward(self, TE, TP, TBI, TBT):
    padding_mask = get_padding(TE)
    mask = look_ahead_mask(TE.shape[1])
    TE = self.elem_embedding(TE)
    TP = self.pos_layer(TP)
    T = torch.concat([TE, TP], axis=-1)
    T = self.ff(T)
    T = self.bonding_layer(T, TBI, TBT, padding_mask=padding_mask, look_ahead_mask=mask)
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

        self.reactant_layer = compound_layer(dmodel, nheads, elem_size, nbonds, charge_size)
        self.product_layer = compound_layer(dmodel, nheads, elem_size, nbonds, charge_size)

        self.compound_layer = torch.nn.Sequential(
          torch.nn.Linear(2*dmodel, dmodel),
          torch.nn.ReLU()
        )
        
        self.theozyme_layer = theozyme_layer(dmodel, nheads, elem_size, nbonds)

        self.encoder = [encoder_layer(dmodel, nheads, dff) for i in range(nblocks)]
        self.decoder = [decoder_layer(dmodel, nheads, dff) for i in range(nblocks)]

        self.output = Transducer(theozyme_size, nbonds, elem_size, dmodel=dmodel, dff=dff)

    def forward(self, RE, RC, RP, RBI, RBT, PE, PC, PP, PBI, PBT, TE, TP, TBI, TBT):
        mask = look_ahead_mask(self.theozyme_size)

        rx = self.reactant_layer(RE, RC, RP, RBI, RBT)

        px = self.product_layer(PE, PC, PP, PBI, PBT)
        
        x = torch.concat([rx, px], axis=-1)
        x = self.compound_layer(x)

        target_x = self.theozyme_layer(TE, TP, TBI, TBT)
        
        for encoder in self.encoder:
          x = encoder(x)

        for decoder in self.decoder:
          target_x = decoder(x, target_x, mask)

        xe, xp, xbonding = self.output(target_x)

        return xe, xp, xbonding

def get_padding(element): #arg element : (Batch size, sequence length, int)
  lengths = element.argmin(1)#gets length of nonpadded array, as it returns the index of the first zero
  mask = torch.arange(element.shape[1])[None, :] > lengths[:, None]
  mask = mask[:, :, None].repeat(1, 1, mask.shape[1])
  
  return mask
