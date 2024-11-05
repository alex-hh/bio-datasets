from typing import Optional

from biotite import structure as bs

from bio_datasets.structure.residue import ResidueDictionary, register_preset_res_dict

from .nucleic import NucleotideChain, residue_atoms, residue_elements, rna_nucleotides

rna_residue_atoms = {res: residue_atoms[res] for res in rna_nucleotides}
rna_residue_elements = {res: residue_elements[res] for res in rna_nucleotides}
# biotite excludes some less central atoms
backbone_atoms = [
    "OP3",
    "P",
    "OP1",
    "OP2",
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "O2'",
    "C1'",
]

register_preset_res_dict(
    "rna", residue_names=rna_nucleotides, backbone_atoms=backbone_atoms
)


# TODO: add RNAMixin if we want to add more dna specific functionality
class RNAChain(NucleotideChain):
    def __init__(
        self,
        atoms: bs.AtomArray,
        residue_dictionary: Optional[ResidueDictionary] = None,
        verbose: bool = False,
        backbone_only: bool = False,
        drop_hydrogens: bool = True,
        map_nonstandard_nucleotides: bool = False,
        nonstandard_as_lowercase: bool = False,
        raise_error_on_unexpected: bool = False,
        replace_unexpected_with_unknown: bool = False,
    ):
        if residue_dictionary is None:
            residue_dictionary = ResidueDictionary.from_preset("rna")
        super().__init__(
            atoms,
            residue_dictionary=residue_dictionary,
            verbose=verbose,
            backbone_only=backbone_only,
            drop_hydrogens=drop_hydrogens,
            map_nonstandard_nucleotides=map_nonstandard_nucleotides,
            nonstandard_as_lowercase=nonstandard_as_lowercase,
            raise_error_on_unexpected=raise_error_on_unexpected,
            replace_unexpected_with_unknown=replace_unexpected_with_unknown,
        )
