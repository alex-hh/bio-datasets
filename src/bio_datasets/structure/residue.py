import functools
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from biotite import structure as bs
from biotite.structure.info.ccd import get_ccd
from biotite.structure.io.pdbx import get_component
from biotite.structure.residues import get_residue_starts

from bio_datasets.np_utils import map_categories_to_indices


def get_ccd_dict():
    with open(
        Path(__file__).parent.parent
        / "structure"
        / "library"
        / "ccd_residue_dictionary.json",
        "r",
    ) as f:
        return json.load(f)


def get_atom_elements():
    ccd_data = get_ccd()
    return list(np.unique(ccd_data["chem_comp_atom"]["type_symbol"].as_array()))


def get_residue_frequencies():
    freq_path = Path(__file__).parent.parent / "structure" / "library" / "cc-counts.tdd"
    with open(freq_path, "r") as f:
        _ = next(f)  # skip header
        return {line.split()[0]: int(line.split()[1]) for line in f}


ALL_ELEMENT_TYPES = get_atom_elements()
_PRESET_RESIDUE_DICTIONARY_KWARGS = {}  # kwargs to pass to ResidueDictionary.from_ccd


def register_preset_res_dict(preset_name: str, **kwargs):
    _PRESET_RESIDUE_DICTIONARY_KWARGS[preset_name] = kwargs


# c.f. docstring of biotite.structure.filter.filter_amino_acids
PROTEIN_TYPES = [
    "D-PEPTIDE LINKING",
    "D-PEPTIDE NH3 AMINO TERMINUS",
    "D-BETA-PEPTIDE, C-GAMMA LINKING",
    "D-GAMMA-PEPTIDE, C-DELTA LINKING",
    "L-PEPTIDE COOH CARBOXY TERMINUS",
    "L-PEPTIDE LINKING",
    "L-BETA-PEPTIDE, C-GAMMA LINKING",
    "L-GAMMA-PEPTIDE, C-DELTA LINKING",
    "L-PEPTIDE COOH CARBOXY TERMINUS",
    "L-PEPTIDE NH3 AMINO TERMINUS",
    "L-PEPTIDE LINKING",
    "PEPTIDE LINKING",
]


# c.f. docstring of biotite.structure.filter.filter_nucleotides
DNA_TYPES = [
    "DNA LINKING",
    "DNA OH 3 PRIME TERMINUS",
    "DNA OH 5 PRIME TERMINUS",
    "L-DNA LINKING",
]


# c.f. docstring of biotite.structure.filter.filter_nucleotides
RNA_TYPES = [
    "RNA LINKING",
    "RNA OH 3 PRIME TERMINUS",
    "RNA OH 5 PRIME TERMINUS",
    "L-RNA LINKING",
]


# c.f. docstring of biotite.structure.filter.filter_carbohydrates
CARBOHYDRATE_TYPES = [
    "D-SACCHARIDE",
    "D-SACCHARIDE, ALPHA LINKING",
    "D-SACCHARIDE, BETA LINKING",
    "L-SACCHARIDE",
    "L-SACCHARIDE, ALPHA LINKING",
    "L-SACCHARIDE, BETA LINKING",
    "SACCHARIDE",
]


CHEMICAL_TYPES = ["NON-POLYMER", "OTHER", "PEPTIDE-LIKE"]


# we dont store these in memory bc too large
def get_component_types():
    ccd_data = get_ccd()
    res_names = ccd_data["chem_comp"]["id"].as_array()
    res_types = ccd_data["chem_comp"]["type"].as_array()
    return {name: type for name, type in zip(res_names, res_types)}


def get_component_categories(chem_component_types: Dict[str, str]):
    categories = {}
    for name, chem_type in chem_component_types.items():
        chem_type = chem_type.strip().upper()
        if chem_type in PROTEIN_TYPES:
            categories[name] = "protein"
        elif chem_type in DNA_TYPES:
            categories[name] = "dna"
        elif chem_type in RNA_TYPES:
            categories[name] = "rna"
        elif chem_type in CARBOHYDRATE_TYPES:
            categories[name] = "carbohydrate"
        elif chem_type in CHEMICAL_TYPES:
            categories[name] = "small_molecule"
        else:
            raise ValueError(f"Unknown chemical component type: {chem_type}")
    return categories


def get_component_3to1():
    ccd_data = get_ccd()
    res_names = ccd_data["chem_comp"]["id"].as_array()
    res_types = ccd_data["chem_comp"]["one_letter_code"].as_array()
    return {name: code for name, code in zip(res_names, res_types) if code}


# TODO: auto convert unknown residues?
@dataclass
class ResidueDictionary:
    residue_names: List[str]
    residue_letters: List[
        str
    ]  # one letter codes. n.b. care needs to be taken about ambiguity for different molecule types
    residue_atoms: Dict[str, List]  # defines composition and atom order
    residue_elements: Dict[str, List[str]]
    unknown_residue_name: str
    # types define one-hot representations, and help with vectorised standardisation
    atom_types: List[str]
    element_types: List[str]
    residue_categories: Optional[Dict[str, str]] = None
    backbone_atoms: Optional[List[str]] = None
    conversions: Optional[List[Dict]] = None
    _expected_relative_atom_indices_mapping: Optional[Dict[str, List[int]]] = None

    def __post_init__(self):
        if self.residue_letters is not None:
            assert len(np.unique(self.residue_letters)) == len(
                self.residue_letters
            ), "Duplicate residue types"
            assert len(self.residue_letters) == len(self.residue_names)
        if self.residue_categories is not None:
            assert len(self.residue_categories) == len(
                self.residue_names
            ), "Mismatch between number of residue names and categories"
        assert len(np.unique(self.element_types)) == len(
            self.element_types
        ), "Duplicate element types"
        assert len(np.unique(self.atom_types)) == len(
            self.atom_types
        ), "Duplicate atom types"
        if self.backbone_atoms is not None:
            assert len(np.unique(self.backbone_atoms)) == len(
                self.backbone_atoms
            ), "Duplicate backbone atoms"
        assert len(np.unique(self.residue_names)) == len(
            self.residue_names
        ), "Duplicate residue names"
        # TODO: assert backbone_atoms are in correct order
        assert all([res in self.residue_atoms for res in self.residue_names])
        assert all([res in self.residue_elements for res in self.residue_names])
        for res, ats in self.residue_atoms.items():
            if len(ats) == 0:
                print("Warning: zero atoms for residue", res)
        if self.conversions is not None:
            for conversion in self.conversions:
                assert conversion["to_residue"] in self.residue_names
                # tuples get converted to lists during serialization so we need to convert them back for eq checks
                conversion["atom_swaps"] = [
                    tuple(swaps) for swaps in conversion["atom_swaps"]
                ]

    @classmethod
    @functools.lru_cache(maxsize=10)
    def from_ccd_dict(
        cls,
        residue_names: Optional[List[str]] = None,
        category: Optional[str] = None,
        backbone_atoms: Optional[List[str]] = None,
        unknown_residue_name: str = "UNK",
        conversions: Optional[List[Dict]] = None,
        minimum_pdb_entries: int = 1,  # ligands might often be unique - arguably res dict not that useful for these cases?
    ):
        """Hydrogens and OXT are not included in the pre-built dictionary"""
        ccd_dict = get_ccd_dict()
        frequencies = get_residue_frequencies()

        res_names = ccd_dict["residue_names"]
        res_categories = ccd_dict["residue_categories"]
        if ccd_dict.get("residue_letters") is None:
            res_letter_mapping = get_component_3to1()
            res_letters = [res_letter_mapping[res] for res in res_names]
            assert all(
                [res_letter for res_letter in res_letters]
            )  # unknown types are ?
        else:
            res_letters = ccd_dict["residue_letters"]

        if category is not None:
            assert category in [
                "protein",
                "dna",
                "rna",
                "saccharide",
                "small_molecule",
            ], f"Unknown category: {category}"

        def keep_res(res_name):
            res_filter = frequencies.get(res_name, 0) >= minimum_pdb_entries
            if residue_names is not None:
                res_filter = res_filter and res_name in residue_names
            if category is not None:
                res_filter = res_filter and res_categories[res_name] == category
            return res_filter

        selected_res_names = []
        selected_res_letters = []
        for res_name, res_letter in zip(res_names, res_letters):
            if keep_res(res_name):
                selected_res_names.append(res_name)
                selected_res_letters.append(res_letter)
        res_names = selected_res_names
        res_letters = selected_res_letters

        categories = set([res_categories[res] for res in res_names])
        if backbone_atoms is not None:
            assert (
                len(categories) == 1
            ), "Backbone atoms only supported for single category dictionaries"

        residue_atoms = {res: ccd_dict["residue_atoms"][res] for res in res_names}
        residue_elements = {res: ccd_dict["residue_elements"][res] for res in res_names}
        residue_categories = {res: res_categories[res] for res in res_names}
        if ccd_dict["_expected_relative_atom_indices_mapping"] is not None:
            _expected_relative_atom_indices_mapping = {
                res: ccd_dict["_expected_relative_atom_indices_mapping"][res]
                for res in res_names
            }
        else:
            _expected_relative_atom_indices_mapping = None
        element_types = list(
            sorted(list(set(itertools.chain(*residue_elements.values()))))
        )
        atom_types = list(sorted(list(set(itertools.chain(*residue_atoms.values())))))

        return cls(
            residue_names=res_names,
            residue_letters=res_letters,
            residue_atoms=residue_atoms,
            residue_elements=residue_elements,
            residue_categories=residue_categories,
            backbone_atoms=backbone_atoms,
            unknown_residue_name=unknown_residue_name,
            element_types=element_types,
            atom_types=atom_types,
            conversions=conversions,
            _expected_relative_atom_indices_mapping=_expected_relative_atom_indices_mapping,
        )

    @classmethod
    def from_preset(cls, preset_name: str, **extra_kwargs):
        return cls.from_ccd_dict(
            **_PRESET_RESIDUE_DICTIONARY_KWARGS[preset_name], **extra_kwargs
        )

    def __str__(self):
        return f"ResidueDictionary ({len(self.residue_names)}) residue types"

    @property
    def residue_sizes(self):
        return np.array(
            [len(self.residue_atoms[resname]) for resname in self.residue_names]
        )

    def get_res_name_relative_atom_indices_mapping(self, res_name: str) -> np.ndarray:
        if res_name == self.unknown_residue_name:
            # n.b. in some structures, UNK also contains CB, CG, ...
            residue_atom_list = self.backbone_atoms or []
        else:
            residue_atom_list = self.residue_atoms[res_name]
        # TODO: can we vectorise this?
        atom_indices_mapping = []
        # relative_indices = np.argwhere(
        atom_indices_mapping = np.full(len(self.atom_types), -100, dtype=int)
        atom_in_res_mask = np.isin(self.atom_types, residue_atom_list)
        relative_indices = np.array(
            [
                residue_atom_list.index(atom_type)
                for atom_type in np.array(self.atom_types)[atom_in_res_mask]
            ]
        )
        atom_indices_mapping[atom_in_res_mask] = relative_indices
        return atom_indices_mapping

    @property
    def relative_atom_indices_mapping(
        self, resnames: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Get a mapping from atom type index to expected index relative to the start of a given residue.
        """
        if self._expected_relative_atom_indices_mapping is not None:
            return np.array(
                [
                    self._expected_relative_atom_indices_mapping[resname]
                    for resname in self.residue_names
                ]
            )
        assert self.atom_types is not None
        all_atom_indices_mapping = []
        expected_relative_atom_indices_mapping = {}
        for resname in self.residue_names if resnames is None else resnames:
            res_mapping = self.get_res_name_relative_atom_indices_mapping(resname)
            expected_relative_atom_indices_mapping[resname] = list(res_mapping)
            all_atom_indices_mapping.append(res_mapping)
        self._expected_relative_atom_indices_mapping = (
            expected_relative_atom_indices_mapping
        )
        return np.stack(all_atom_indices_mapping, axis=0)

    @property
    def max_residue_size(self):
        return self.residue_sizes.max()

    @property
    def total_element_types(self):
        """How many element types across all proteins"""
        assert self.element_types is not None
        return len(self.element_types)

    @property
    def standard_atoms_by_residue(self):
        """Return a fixed size array of atom names for each residue type.

        Shape (num_residue_types x max_atoms_per_residue)
        e.g. for proteins we use atom14 (21 x 14)
        """
        arr = np.full((len(self.residue_names), self.max_residue_size), "", dtype="U6")
        for ix, residue_name in enumerate(self.residue_names):
            residue_atoms = self.residue_atoms[residue_name]
            arr[ix, : len(residue_atoms)] = residue_atoms
        return arr

    def get_residue_sizes(
        self, restype_index: np.ndarray, chain_id: np.ndarray
    ) -> np.ndarray:
        return self.residue_sizes[restype_index]

    def get_residue_category(self, restype_index: np.ndarray) -> np.ndarray:
        return self.residue_categories[restype_index]

    def get_expected_relative_atom_indices(self, restype_index, atomtype_index):
        return self.relative_atom_indices_mapping[restype_index, atomtype_index]

    def get_atom_names(
        self,
        restype_index: np.ndarray,
        relative_atom_index: np.ndarray,
        chain_id: np.ndarray,
    ):
        return self.standard_atoms_by_residue[
            restype_index,
            relative_atom_index,
        ]

    def res_name_to_index(self, res_name: np.ndarray) -> np.ndarray:
        # n.b. protein resnames are sorted in alphabetical order, apart from UNK
        if not np.all(np.isin(res_name, np.array(self.residue_names))):
            raise ValueError(
                f"res_name contains elements not in the allowed list: "
                f"{np.unique(res_name[~np.isin(res_name, np.array(self.residue_names))])}"
            )
        return map_categories_to_indices(res_name, self.residue_names)

    def res_letter_to_index(self, res_letter: np.ndarray) -> np.ndarray:
        if not np.all(np.isin(res_letter, np.array(self.residue_letters))):
            raise ValueError(
                f"res_letter contains elements not in the allowed list: "
                f"{np.unique(res_letter[~np.isin(res_letter, np.array(self.residue_letters))])}"
            )
        return map_categories_to_indices(res_letter, self.residue_letters)

    def atomtype_index_full_to_short(self):
        # return a num_residues, num_full, num_short mapping array (e.g. atom37 -> atom14 for each residue)
        raise NotImplementedError()

    def res_name_to_onehot(self, res_name: np.ndarray) -> np.ndarray:
        masks = [res_name == r for r in self.residue_names]
        return np.stack(masks, axis=-1)

    def res_letter_to_onehot(self, res_letter: np.ndarray) -> np.ndarray:
        masks = [res_letter == r for r in self.residue_letters]
        return np.stack(masks, axis=-1)

    def res_letter_to_name(self, res_letter: np.ndarray) -> np.ndarray:
        return np.array(self.residue_names)[self.res_letter_to_index(res_letter)]

    def decode_res_letter_index(self, res_letter_index: np.ndarray) -> np.ndarray:
        return "".join(np.array(self.residue_letters)[res_letter_index])

    def atom_full_to_atom_short(self):
        # eg atom37->atom14
        raise NotImplementedError()


def tile_residue_annotation_to_atoms(
    atoms: bs.AtomArray, residue_annotation: np.ndarray, residue_starts: np.ndarray
) -> np.ndarray:
    # use residue index as cumsum of residue starts
    assert len(residue_annotation) == len(residue_starts)
    residue_index = np.cumsum(get_residue_starts_mask(atoms, residue_starts)) - 1
    return residue_annotation[residue_index]


def get_residue_starts_mask(
    atoms: bs.AtomArray, residue_starts: Optional[np.ndarray] = None
) -> np.ndarray:
    if residue_starts is None:
        residue_starts = get_residue_starts(atoms)
    mask = np.zeros(len(atoms), dtype=bool)
    mask[residue_starts] = True
    return mask
