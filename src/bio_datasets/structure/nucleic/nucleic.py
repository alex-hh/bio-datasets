import copy
from dataclasses import field
from typing import Optional, Dict, List

import numpy as np
from biotite import structure as bs
from biotite.structure.info.ccd import get_ccd
from biotite.structure.io.pdbx import get_component
from biotite.structure.filter import _canonical_nucleotide_list, _phosphate_backbone

from bio_datasets.structure.biomolecule import BiomoleculeChain
from bio_datasets.structure.residue import ResidueDictionary


dna_nucleotides = ["DA", "DC", "DG", "DT"]
rna_nucleotides = ["A", "C", "G", "U"]


def get_residue_atoms_and_elements(residue_names):
    residue_atoms = {}
    residue_elements = {}
    ccd_data = get_ccd()
    for resname in residue_names:
        comp = get_component(ccd_data, res_name=resname)
        atoms = [
            at
            for at in comp.atom_name
            if not at.startswith("H") and not at.startswith("D")
        ]
        elements = [
            elem
            for at, elem in zip(comp.atom_name, comp.element)
            if elem != "H" and elem != "D" and at != "OXT"
        ]
        assert len(atoms) == len(elements)
        residue_atoms[resname] = atoms
        residue_elements[resname] = elements

    return residue_atoms, residue_elements


residue_atoms, residue_elements = get_residue_atoms_and_elements(_canonical_nucleotide_list)


class NucleotideDictionary(ResidueDictionary):
    
    """Defaults configure a dictionary with just the 20 standard amino acids"""

    # TODO: these are actually all constants
    residue_names: np.ndarray = field(
        default_factory=lambda: copy.deepcopy(_canonical_nucleotide_list)
    )
    residue_atoms: Dict[str, List[str]] = field(
        default_factory=lambda: copy.deepcopy(residue_atoms)
    )
    residue_elements: Dict[str, List[str]] = field(
        default_factory=lambda: copy.deepcopy(residue_elements)
    )
    backbone_atoms: List[str] = field(default_factory=lambda: _phosphate_backbone)  # just core bond-forming? atoms
    unknown_residue_name: str = field(default_factory=lambda: "UNK")  # TODO: check if this is correct


class NucleotideChain(BiomoleculeChain):
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
            residue_dictionary = NucleotideDictionary()
        if map_nonstandard_nucleotides:
            # https://www.biotite-python.org/latest/apidoc/biotite.structure.map_nucleotide.html#biotite.structure.map_nucleotide
            raise NotImplementedError("Matching non-standard nucleotides not yet implemented")
        super().__init__(
            atoms,
            residue_dictionary=residue_dictionary,
            verbose=verbose,
            backbone_only=backbone_only,
            drop_hydrogens=drop_hydrogens,
        )
