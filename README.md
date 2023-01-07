# Enzyme-De-Novo

This is a project to try and collect enzyme data which can then be used to train generative models to create enzymes for a given chemical reaction.

## TODO list
- normalising position values
- make ajacency matrix and COO tensors easier to convert between
- add hydrogens to pdb files
- add non-interacting residues to the theozyme not in describtion
- create transition states from reactants and products
- order systems so Ri and Pi is the same atom
- add Sterioisomerism to bonding matrix, so nbonds could include Trans-double Cis-double ect
- create better visualisation tools and model analysis tools
- improve the model architecture maybe have a latent feature and latent edge and have each attention be graph attention



### Dataset
The data is collected from the M-CSA dataset which gives the ChemID of the reactants and products which allows for a 2D .mol files to be downloaded, but using RDKit it attempts to generate a 3D conformer, it also provides a list of PDB id of the enzymes which can be downloaded and the catalytic residues which then are selected from the pdb. 

the .mol files provide both the features and ajacency (Atom properties and bonds) but .pdb only provides the features, but since its made of Amino acids, assuming they are not modified the ajacency will be constant and so i created a look up table for the non-hydrogen bonds. 

TODO: need to add hydrogens to .pdb files and also find non-catalytic residues in the Theozyme

Overview:

Element : Long 1D Arr (index for Sparse)
Charge : Long 1D arr (index for Sparse)
Pos : float 3D arr

edge_indexes : [node1, node2]  n x 2 Long arr
edge_types : int 1D arr (index for Sparse) need to add steroisomerism to bonding types so Cis-double bond, Trans-tripple bond ect

The Ajacency ndarray can be split into two for N-body compound, an Bonding matrix and a Steroisomerism matrix, where they share the same shape of n \times n except for the add channel size d which represents the types of bond or sterioisomerism. But they could be concatenated together if you do not wish to embed them.

Torch-geometric uses a COO tensor which is constructured with a more memory efficent list of the edges, [node1, node2, edge_type]. associated with an array of edge tokens.

### Model

This is a Transformer model from Attention is all you need with graph embeddings for the Compound and target theozyme into a latent space which is feed into the transformer encoder and decoder respectively, there is also a transducer module to convert the latent space into the element, position and bonding predictions. This is a very "janky" method for this and will try and change it into a more robust method.

The Graph Attention which embeds the inputed graphs is a weighted masked attention where non-bonded elements in the attention product is set to zero and the bonded weights are dotted with bonded elements. this creates the inital latent-graph tensor.


