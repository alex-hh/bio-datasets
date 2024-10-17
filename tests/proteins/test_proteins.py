import numpy as np
from biotite.structure.filter import filter_amino_acids
from biotite.structure.residues import residue_iter

from bio_datasets.protein import Protein
from bio_datasets.protein import constants as protein_constants


def test_residue_atom_order(pdb_atom_array):
    total_residues = 0
    correct_residues = 0
    amino_acid_filter = filter_amino_acids(pdb_atom_array)
    pdb_atom_array = pdb_atom_array[amino_acid_filter]
    for residue_atoms in residue_iter(pdb_atom_array):
        atom_names = residue_atoms.atom_name
        expected_atom_names = np.array(
            protein_constants.residue_atoms_ordered[residue_atoms.res_name[0]]
        )
        total_residues += 1
        if len(atom_names) != len(expected_atom_names):
            # missing atoms are ok
            continue
        assert (
            np.all(
                atom_names
                == np.array(
                    protein_constants.residue_atoms_ordered[residue_atoms.res_name[0]]
                )
            ),
            f"Observed: {atom_names} != Expected: {np.array(protein_constants.residue_atoms_ordered[residue_atoms.res_name[0]])}",
        )
        correct_residues += 1
    assert correct_residues / total_residues > 0.5


# def test_reorder_atoms(afdb_atom_array):
#     # afdb has CB before O
#     protein = Protein(afdb_atom_array)
#     test_residue_atom_order(protein.atoms)


def test_fill_missing_atoms(pdb_atom_array):
    """
    REMARK 465 MISSING RESIDUES
    REMARK 465 THE FOLLOWING RESIDUES WERE NOT LOCATED IN THE
    REMARK 465 EXPERIMENT. (M=MODEL NUMBER; RES=RESIDUE NAME; C=CHAIN
    REMARK 465 IDENTIFIER; SSSEQ=SEQUENCE NUMBER; I=INSERTION CODE.)
    REMARK 465
    REMARK 465   M RES C SSSEQI
    REMARK 465     MSE A     1
    REMARK 465     GLY A     2
    REMARK 465     GLU A    95
    REMARK 465     GLY A    96
    REMARK 465     GLY A    97
    REMARK 465     SER A    98
    REMARK 465     LEU A    99
    REMARK 465     GLU A   100
    REMARK 465     HIS A   101
    REMARK 465     HIS A   102
    REMARK 465     HIS A   103
    REMARK 465     HIS A   104
    REMARK 465     HIS A   105
    REMARK 465     HIS A   106
    REMARK 470
    REMARK 470 MISSING ATOM
    REMARK 470 THE FOLLOWING RESIDUES HAVE MISSING ATOMS(M=MODEL NUMBER;
    REMARK 470 RES=RESIDUE NAME; C=CHAIN IDENTIFIER; SSEQ=SEQUENCE NUMBER;
    REMARK 470 I=INSERTION CODE):
    REMARK 470   M RES CSSEQI  ATOMS
    REMARK 470     LYS A  15    CG   CD   CE   NZ
    REMARK 470     PHE A  17    CG   CD1  CD2  CE1  CE2  CZ
    REMARK 470     SER A  27    OG
    REMARK 470     GLN A  30    CG   CD   OE1  NE2
    REMARK 470     LYS A  31    CG   CD   CE   NZ
    REMARK 470     ASN A  34    CG   OD1  ND2
    REMARK 470     LEU A  36    CG   CD1  CD2
    REMARK 470     LYS A  46    CG   CD   CE   NZ
    REMARK 470     ARG A  47    CG   CD   NE   CZ   NH1  NH2
    REMARK 470     ARG A  55    CG   CD   NE   CZ   NH1  NH2
    REMARK 470     LYS A  62    CG   CD   CE   NZ
    REMARK 470     GLU A  73    CG   CD   OE1  OE2

    """
    pdb_atom_array = pdb_atom_array[filter_amino_acids(pdb_atom_array)]
    # 1qys has missing atoms
    protein = Protein(pdb_atom_array)
    # todo check for nans
    for raw_residue, filled_residue in zip(
        residue_iter(pdb_atom_array[filter_amino_acids(pdb_atom_array)]),
        residue_iter(protein.atoms),
    ):
        expected_atom_names = np.array(
            protein_constants.residue_atoms_ordered[raw_residue.res_name[0]]
        )
        if len(raw_residue) != len(expected_atom_names):
            missing_atoms = np.setdiff1d(expected_atom_names, raw_residue.atom_name)
            assert np.all(np.isin(missing_atoms, filled_residue.atom_name))
            filled_atom_mask = np.isin(filled_residue.atom_name, missing_atoms)
            assert np.all(np.isnan(filled_residue.coord[filled_atom_mask]))


# n.b. filling missing residues is not yet implemented - would require
# some decision on handling non-consecutive residue indices
# def test_fill_missing_residues(pdb_atom_array_1aq1):
#     """
#     REMARK 465 MISSING RESIDUES
#     REMARK 465 THE FOLLOWING RESIDUES WERE NOT LOCATED IN THE
#     REMARK 465 EXPERIMENT. (M=MODEL NUMBER; RES=RESIDUE NAME; C=CHAIN
#     REMARK 465 IDENTIFIER; SSSEQ=SEQUENCE NUMBER; I=INSERTION CODE.)
#     REMARK 465
#     REMARK 465   M RES C SSSEQI
#     REMARK 465     ARG A    36
#     REMARK 465     LEU A    37
#     REMARK 465     ASP A    38
#     REMARK 465     THR A    39
#     REMARK 465     GLU A    40
#     REMARK 465     THR A    41
#     REMARK 465     GLU A    42
#     REMARK 465     GLY A    43
#     REMARK 465     ALA A   149
#     REMARK 465     ARG A   150
#     REMARK 465     ALA A   151
#     REMARK 465     PHE A   152
#     REMARK 465     GLY A   153
#     REMARK 465     VAL A   154
#     REMARK 465     PRO A   155
#     REMARK 465     VAL A   156
#     REMARK 465     ARG A   157
#     REMARK 465     THR A   158
#     REMARK 465     TYR A   159
#     REMARK 465     THR A   160
#     REMARK 465     HIS A   161
#     REMARK 500
#     """
#     # 1aq1 has missing residues
#     pass
