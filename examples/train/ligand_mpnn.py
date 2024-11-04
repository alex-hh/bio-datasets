"""End-to-end example of training with bio datasets.

Reproduce training of ProteinMPNN / LigandMPNN.

`LigandMPNN was trained on protein assemblies in the PDB (as of Dec 16, 2022) determined
by X-ray crystallography or cryoEM to better than 3.5 Å resolution and with a total length of
less than 6,000 residues. The train/test split was based on protein sequences clustered at a
30% sequence identity cutoff. We evaluated LigandMPNN sequence design performance on
a test set of 317 protein structures containing a small molecule, 74 with nucleic acids, and 83
with a transition metal (Figure 2A). For comparison, we retrained ProteinMPNN on the same
training dataset of PDB biounits as LigandMPNN, except none of the context atoms were
provided during training.`

`The median sequence recoveries (ten designed sequences
per protein) near small molecules were 50.4% for Rosetta using the genpot energy function
(18), 50.4% for ProteinMPNN, and 63.3% for LigandMPNN. For residues near nucleotides,
median sequence recoveries were 35.2% for Rosetta (11) (using Rosetta energy optimized
for protein-DNA interfaces), 34.0% for ProteinMPNN, and 50.5% for LigandMPNN, and for
residues near metals, 36.0% for Rosetta (18), 40.6% for ProteinMPNN, and 77.5% for
LigandMPNN (Figure 2A). Sequence recoveries were consistently higher for LigandMPNN
over most proteins in the validation data set`

Splits available at: https://github.com/dauparas/LigandMPNN/tree/main/training

N.B. if we use an order-agnostic autoregressive model, then chain order doesn't matter.

reusing data across splits:
https://huggingface.co/docs/datasets/en/repository_structure
"""
