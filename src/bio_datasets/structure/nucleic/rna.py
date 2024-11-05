import copy
from dataclasses import field
from typing import Optional, Dict, List

import numpy as np
from biotite import structure as bs

from bio_datasets.structure.residue import ResidueDictionary
from .nucleic import rna_nucleotides, residue_atoms, residue_elements, NucleotideChain


rna_residue_atoms = {res: residue_atoms[res] for res in rna_nucleotides}
rna_residue_elements = {res: residue_elements[res] for res in rna_nucleotides}
# biotite excludes some less central atoms
backbone_atoms = [
  'OP3',
  'P',
  'OP1',
  'OP2',
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


class RNADictionary(ResidueDictionary):
    
    """Defaults configure a dictionary with just the 4 standard bases"""

    # TODO: these are actually all constants
    residue_names: np.ndarray = field(
        default_factory=lambda: copy.deepcopy(rna_nucleotides)
    )
    residue_types: np.ndarray = field(
        default_factory=lambda: copy.deepcopy(rna_nucleotides)
    )
    residue_atoms: Dict[str, List[str]] = field(
        default_factory=lambda: copy.deepcopy(rna_residue_atoms)
    )
    residue_elements: Dict[str, List[str]] = field(
        default_factory=lambda: copy.deepcopy(rna_residue_elements)
    )
    backbone_atoms: List[str] = field(default_factory=lambda: backbone_atoms)
    unknown_residue_name: str = field(default_factory=lambda: "UNK")


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
    ):
        if residue_dictionary is None:
            residue_dictionary = RNADictionary()
        super().__init__(
            atoms,
            residue_dictionary=residue_dictionary,
            verbose=verbose,
            backbone_only=backbone_only,
            drop_hydrogens=drop_hydrogens,
            map_nonstandard_nucleotides=map_nonstandard_nucleotides,
            nonstandard_as_lowercase=nonstandard_as_lowercase,
        )
