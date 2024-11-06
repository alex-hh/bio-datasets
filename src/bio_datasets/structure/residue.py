import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from biotite import structure as bs
from biotite.structure.info.ccd import get_ccd
from biotite.structure.io.pdbx import get_component
from biotite.structure.residues import get_residue_starts

from bio_datasets.np_utils import map_categories_to_indices


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


def get_component_types():
    ccd_data = get_ccd()
    res_names = ccd_data["chem_comp"]["id"].as_array()
    res_types = ccd_data["chem_comp"]["type"].as_array()
    return {name: type for name, type in zip(res_names, res_types)}


CHEM_COMPONENT_TYPES = get_component_types()


# TODO: speed these up
def get_component_categories():
    categories = {}
    for name, chem_type in CHEM_COMPONENT_TYPES.items():
        chem_type = chem_type.strip().upper()
        if chem_type in PROTEIN_TYPES:
            categories[name] = "protein"
        elif chem_type in DNA_TYPES:
            categories[name] = "dna"
        elif chem_type in RNA_TYPES:
            categories[name] = "rna"
        elif chem_type in CARBOHYDRATE_TYPES:
            categories[name] = "saccharide"
        elif chem_type in CHEMICAL_TYPES:
            categories[name] = "chemical"
        else:
            raise ValueError(f"Unknown chemical component type: {chem_type}")
    return categories


def get_component_3to1():
    ccd_data = get_ccd()
    res_names = ccd_data["chem_comp"]["id"].as_array()
    res_types = ccd_data["chem_comp"]["one_letter_code"].as_array()
    return {name: code for name, code in zip(res_names, res_types) if code}


CHEM_COMPONENT_CATEGORIES = get_component_categories()
CHEM_COMPONENT_3TO1 = get_component_3to1()


# TODO: auto convert unknown residues?
@dataclass
class ResidueDictionary:
    residue_names: List[str]
    residue_atoms: Dict[str, List]  # defines composition and atom order
    residue_elements: Dict[str, List[str]]
    unknown_residue_name: str
    # types define one-hot representations, and help with vectorised standardisation
    atom_types: List[str]
    element_types: List[str]
    residue_types: Optional[List[str]] = None  # one letter codes
    backbone_atoms: Optional[List[str]] = None
    conversions: Optional[List[Dict]] = None

    def __post_init__(self):
        if self.residue_types is not None:
            assert len(np.unique(self.residue_types)) == len(
                self.residue_types
            ), "Duplicate residue types"
            assert len(self.residue_types) == len(self.residue_names)
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
    def from_ccd(
        cls,
        residue_names: Optional[List[str]] = None,
        category: Optional[str] = None,
        keep_hydrogens: bool = False,
        keep_oxt: bool = False,  # keeping it will add an extra atom to each residue during standardisation
        backbone_atoms: Optional[List[str]] = None,
        unknown_residue_name: str = "UNK",
        conversions: Optional[List[Dict]] = None,
        minimum_pdb_entries: int = 100,  # ligands might often be unique - but then what's benefit of residue dictionary for unique ligands? SmallMolecule doens't even use residue dictionary
    ):
        ccd_data = get_ccd()
        frequencies = get_residue_frequencies()
        res_names = np.unique(ccd_data["chem_comp_atom"]["comp_id"].as_array(str))
        res_names = [
            res for res in res_names if frequencies.get(res, 0) >= minimum_pdb_entries
        ]
        if not keep_hydrogens:
            res_names = [
                res for res in res_names if res != "H" and res != "D" and res != "D8U"
            ]
        if residue_names is not None:
            res_names = [res for res in res_names if res in residue_names]
        if category is not None:
            assert category in [
                "protein",
                "dna",
                "rna",
                "saccharide",
                "chemical",
            ], f"Unknown category: {category}"
            mask = np.array(
                [CHEM_COMPONENT_CATEGORIES[name] == category for name in res_names]
            )
            res_names = list(res_names[mask])
        else:
            categories = list(
                set([CHEM_COMPONENT_CATEGORIES[name] for name in res_names])
            )
        # one-letter codes only used for unambiguous and complete residue dictionaries
        # TODO: check this covers all unknown cases
        if len(categories) == 1 and categories[0] != "chemical":
            res_types = [CHEM_COMPONENT_3TO1[name] for name in res_names]
            assert all([res_type and not res_type == "?" for res_type in res_types])
        else:
            res_types = None

        if backbone_atoms is not None:
            assert (
                len(categories) == 1
            ), "Backbone atoms only supported for single category dictionaries"

        res_atom_names = {}
        res_element_types = {}
        chem_comp_atom = ccd_data["chem_comp_atom"]
        for (
            name
        ) in (
            res_names
        ):  # TODO: is this an inefficient loop? We can just load the full table and filter as required...
            res_mask = chem_comp_atom["comp_id"].as_array(str) == name
            assert np.any(res_mask), f"No atoms found for residue {name}"
            res_elements_arr = chem_comp_atom["type_symbol"].as_array(str)[res_mask]
            res_atom_names_arr = chem_comp_atom["atom_id"].as_array(str)[res_mask]
            assert len(res_elements_arr) == len(res_atom_names_arr)

            mask = np.ones(len(res_atom_names_arr), dtype=bool)
            if not keep_hydrogens:
                mask &= (res_elements_arr != "H") & (res_elements_arr != "D")
            if not keep_oxt:
                mask &= res_atom_names_arr != "OXT"

            res_atom_names[name] = list(res_atom_names_arr[mask])
            res_element_types[name] = list(res_elements_arr[mask])

        all_res_mask = np.isin(chem_comp_atom["comp_id"].as_array(str), res_names)
        all_res_elements = np.unique(
            chem_comp_atom["type_symbol"].as_array(str)[all_res_mask]
        )
        all_res_atom_names = np.unique(
            chem_comp_atom["atom_id"].as_array(str)[all_res_mask]
        )

        return cls(
            residue_names=list(res_names),
            residue_types=res_types,
            residue_atoms=res_atom_names,
            residue_elements=res_element_types,
            backbone_atoms=backbone_atoms,
            unknown_residue_name=unknown_residue_name,
            element_types=list(all_res_elements),
            atom_types=list(all_res_atom_names),
            conversions=conversions,
        )

    @classmethod
    def from_preset(cls, preset_name: str, **extra_kwargs):
        return cls.from_ccd(
            **_PRESET_RESIDUE_DICTIONARY_KWARGS[preset_name], **extra_kwargs
        )

    def __str__(self):
        return f"ResidueDictionary ({len(self.residue_names)}) residue types"

    @property
    def residue_sizes(self):
        return np.array(
            [len(self.residue_atoms[resname]) for resname in self.residue_names]
        )

    @property
    def residue_categories(self):
        return np.array(
            [CHEM_COMPONENT_CATEGORIES[resname] for resname in self.residue_names]
        )

    def get_resname_relative_atom_indices_mapping(self, resname: str) -> np.ndarray:
        if resname == self.unknown_residue_name:
            # n.b. in some structures, UNK also contains CB, CG, ...
            residue_atom_list = self.backbone_atoms or []
        else:
            residue_atom_list = self.residue_atoms[resname]
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
        assert self.atom_types is not None
        all_atom_indices_mapping = []
        for resname in self.residue_names if resnames is None else resnames:
            all_atom_indices_mapping.append(
                self.get_resname_relative_atom_indices_mapping(resname)
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

    def resname_to_index(self, resname: np.ndarray) -> np.ndarray:
        # n.b. protein resnames are sorted in alphabetical order, apart from UNK
        if not np.all(np.isin(resname, np.array(self.residue_names))):
            raise ValueError(
                f"resname contains elements not in the allowed list: "
                f"{np.unique(resname[~np.isin(resname, np.array(self.residue_names))])}"
            )
        return map_categories_to_indices(resname, self.residue_names)

    def restype_to_index(self, restype: np.ndarray) -> np.ndarray:
        if not np.all(np.isin(restype, np.array(self.residue_types))):
            raise ValueError(
                f"restype contains elements not in the allowed list: "
                f"{np.unique(restype[~np.isin(restype, np.array(self.residue_names))])}"
            )
        return map_categories_to_indices(restype, self.residue_types)

    def atomtype_index_full_to_short(self):
        # return a num_residues, num_full, num_short mapping array (e.g. atom37 -> atom14 for each residue)
        raise NotImplementedError()

    def resname_to_onehot(self, resname: np.ndarray) -> np.ndarray:
        masks = [resname == r for r in self.residue_names]
        return np.stack(masks, axis=-1)

    def restype_to_onehot(self, restype: np.ndarray) -> np.ndarray:
        masks = [restype == r for r in self.residue_types]
        return np.stack(masks, axis=-1)

    def restype_to_resname(self, restype: np.ndarray) -> np.ndarray:
        return np.array(self.residue_names)[self.restype_to_index(restype)]

    def decode_restype_index(self, restype_index: np.ndarray) -> np.ndarray:
        return "".join(np.array(self.residue_types)[restype_index])

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
