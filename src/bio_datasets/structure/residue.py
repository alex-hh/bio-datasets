import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from biotite import structure as bs
from biotite.structure.info.ccd import get_ccd
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
_PRESET_RESIDUE_DICTIONARY_KWARGS = (
    {}
)  # kwargs to pass to ResidueDictionary.from_ccd_dict


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


RES_NAMES = get_ccd()["chem_comp"]["id"].as_array()

# we dont store these in memory bc too large
def get_component_types():
    ccd_data = get_ccd()
    res_types = ccd_data["chem_comp"]["type"].as_array()
    return {name: type for name, type in zip(RES_NAMES, res_types)}


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


# useful to store this globally because it helps us split chains in complex.py
CHEM_COMPONENT_CATEGORIES = get_component_categories(get_component_types())


def get_component_3to1():
    ccd_data = get_ccd()
    res_names = ccd_data["chem_comp"]["id"].as_array()
    res_types = ccd_data["chem_comp"]["one_letter_code"].as_array()
    return {name: code for name, code in zip(res_names, res_types) if code}


def get_res_categories(res_name: np.ndarray):
    unique_resnames = np.unique(res_name)
    unique_restype_indices = map_categories_to_indices(res_name, list(unique_resnames))
    unique_categories = np.array(
        [CHEM_COMPONENT_CATEGORIES[resname] for resname in unique_resnames]
    )
    return unique_categories[unique_restype_indices]


# TODO: support inferring chirality from residue name
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

    def __post_init__(self):
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
                conversion["element_swaps"] = [
                    tuple(swaps) for swaps in conversion["element_swaps"]
                ]
        self._expected_relative_atom_indices_mapping = None

    @classmethod
    def from_ccd_dict(
        cls,
        residue_names: Optional[List[str]] = None,
        category: Optional[str] = None,
        atom_types: Optional[List[str]] = None,
        backbone_atoms: Optional[List[str]] = None,
        unknown_residue_name: str = "UNK",
        conversions: Optional[List[Dict]] = None,
        minimum_pdb_entries: int = 1,  # ligands might often be unique - arguably res dict not that useful for these cases?
        **kwargs,
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

        element_types = list(
            sorted(list(set(itertools.chain(*residue_elements.values()))))
        )
        if atom_types is None:
            atom_types = list(
                sorted(list(set(itertools.chain(*residue_atoms.values()))))
            )

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
            **kwargs,
        )

    @classmethod
    def from_preset(cls, preset_name: str, **extra_kwargs):
        return cls.from_ccd_dict(
            **_PRESET_RESIDUE_DICTIONARY_KWARGS[preset_name], **extra_kwargs
        )

    @classmethod
    def from_ccd(
        cls,
        residue_names: Optional[List[str]] = None,
        category: Optional[str] = None,
        keep_hydrogens: bool = False,
        keep_oxt: bool = False,  # keeping it will add an extra atom to each residue during standardisation
        backbone_atoms: Optional[List[str]] = None,
        unknown_residue_name: str = "UNK",
        conversions: Optional[List[Dict]] = None,
        minimum_pdb_entries: int = 1,  # ligands might often be unique - but then what's benefit of residue dictionary for unique ligands? SmallMolecule doens't even use residue dictionary
    ):
        ccd_data = get_ccd()
        chem_component_3to1 = get_component_3to1()
        chem_component_categories = get_component_categories(get_component_types())
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
                "carbohydrate",
                "small_molecule",
            ], f"Unknown category: {category}"
            mask = np.array(
                [chem_component_categories[name] == category for name in res_names]
            )
            res_names = list(res_names[mask])
        else:
            categories = list(
                set([chem_component_categories[name] for name in res_names])
            )

        res_letters = [chem_component_3to1[name] for name in res_names]
        res_categories = {name: chem_component_categories[name] for name in res_names}
        assert all([res_letter for res_letter in res_letters])

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

        element_types = list(
            sorted(list(set(itertools.chain(*res_element_types.values()))))
        )
        atom_types = list(sorted(list(set(itertools.chain(*res_atom_names.values())))))
        return cls(
            residue_names=list(res_names),
            residue_letters=res_letters,
            residue_atoms=res_atom_names,
            residue_elements=res_element_types,
            residue_categories=res_categories,
            backbone_atoms=backbone_atoms,
            unknown_residue_name=unknown_residue_name,
            element_types=element_types,
            atom_types=atom_types,
            conversions=conversions,
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

    def relative_atom_indices_mapping(
        self, resnames: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Get a mapping from atom type index to expected index relative to the start of a given residue.
        """
        assert self.atom_types is not None
        all_atom_indices_mapping = []
        expected_relative_atom_indices_mapping = {}

        for resname in self.residue_names if resnames is None else resnames:
            res_mapping = self.get_res_name_relative_atom_indices_mapping(resname)
            expected_relative_atom_indices_mapping[resname] = list(res_mapping)
            all_atom_indices_mapping.append(res_mapping)
        return np.stack(all_atom_indices_mapping, axis=0)

    @property
    def max_residue_size(self):
        """Max number of atoms in any residue.

        (Across CCD, max is 240)
        """
        return self.residue_sizes.max()

    @property
    def total_element_types(self):
        """How many element types across all proteins"""
        assert self.element_types is not None
        return len(self.element_types)

    # TODO: would be good to cache but cant cache list arg.
    def standard_atoms_by_residue(self, resnames: Optional[List[str]] = None):
        """Return a fixed size array of atom names for each residue type.

        Shape (num_residue_types x max_atoms_per_residue)
        e.g. for proteins we use atom14 (21 x 14)

        N.B.This function remains a bit of a bottleneck.
        """
        resnames = resnames or self.residue_names
        arr = np.full((len(resnames), self.max_residue_size), "", dtype="U6")
        for ix, residue_name in enumerate(resnames):
            residue_atoms = self.residue_atoms[residue_name]
            arr[ix, : len(residue_atoms)] = residue_atoms
        return arr

    def standard_elements_by_residue(self, resnames: Optional[List[str]] = None):
        resnames = resnames or self.residue_names
        arr = np.full((len(resnames), self.max_residue_size), "", dtype="U6")
        for ix, residue_name in enumerate(resnames):
            residue_elements = self.residue_elements[residue_name]
            arr[ix, : len(residue_elements)] = residue_elements
        return arr

    def get_residue_sizes(
        self, restype_index: np.ndarray, chain_id: np.ndarray
    ) -> np.ndarray:
        return self.residue_sizes[restype_index]

    def get_residue_categories(self, restype_index: np.ndarray) -> np.ndarray:
        restype_indices = np.unique(restype_index)
        resnames = np.array(self.residue_names)[restype_indices]
        assert self.residue_categories is not None
        cat_arr = np.array([self.residue_categories[resname] for resname in resnames])
        subset_restype_indices = np.searchsorted(restype_indices, restype_index)
        return cat_arr[subset_restype_indices]

    def get_expected_relative_atom_indices(self, restype_index, atomtype_index):
        if len(self.residue_names) > 100:
            # for large dictionaries, only compute atom indices for subset of residues
            restype_indices = np.unique(restype_index)
            resnames = np.array(self.residue_names)[restype_indices]
            # index relative to restype_indices of restype_index
            subset_restype_indices = np.searchsorted(restype_indices, restype_index)
            mapping = self.relative_atom_indices_mapping(resnames)[
                subset_restype_indices, atomtype_index
            ]
        else:
            # for small dictionaries, just compute and cache the full mapping
            if self._expected_relative_atom_indices_mapping is None:
                self._expected_relative_atom_indices_mapping = (
                    self.relative_atom_indices_mapping()
                )
            mapping = self._expected_relative_atom_indices_mapping[
                restype_index, atomtype_index
            ]
        return mapping

    def get_atom_names(
        self,
        restype_index: np.ndarray,
        relative_atom_index: np.ndarray,
        chain_id: np.ndarray,
    ):
        # chain_id is used by ProteinDictionary -- TODO: maybe just accept atoms directly
        restype_indices = np.unique(restype_index)
        resnames = list(np.array(self.residue_names)[restype_indices])
        # index relative to restype_indices of restype_index
        subset_restype_indices = np.searchsorted(restype_indices, restype_index)
        return self.standard_atoms_by_residue(resnames)[
            subset_restype_indices,
            relative_atom_index,
        ]

    def get_elements(
        self,
        restype_index: np.ndarray,
        relative_atom_index: np.ndarray,
        chain_id: np.ndarray,
    ):
        restype_indices = np.unique(restype_index)
        resnames = list(np.array(self.residue_names)[restype_indices])
        # index relative to restype_indices of restype_index
        subset_restype_indices = np.searchsorted(restype_indices, restype_index)
        return self.standard_elements_by_residue(resnames)[
            subset_restype_indices,
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

    def decode_restype_index(self, restype_index: np.ndarray) -> np.ndarray:
        return "".join(np.array(self.residue_letters)[restype_index])

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


def create_complete_atom_array_from_restype_index(
    restype_index: np.ndarray,
    residue_dictionary: ResidueDictionary,
    chain_id: Tuple[str, np.ndarray],
    extra_fields: Optional[List[str]] = None,
    res_id: Optional[np.ndarray] = None,
    backbone_only: bool = False,
    residue_index_offset: int = 0,
):
    """
    Populate annotations from restype_index, assuming all atoms are present.
    """
    if isinstance(chain_id, np.ndarray):
        assert len(chain_id) == len(restype_index)
        unique_chain_ids = np.unique(chain_id)
        chain_atom_arrays = []
        chain_residue_starts = []
        residue_starts_offset = 0
        res_index_offset = 0
        for single_chain_id in unique_chain_ids:
            chain_mask = chain_id == single_chain_id
            if res_id is not None:
                chain_res_id = res_id[chain_mask]
            else:
                chain_res_id = None
            (
                atom_array,
                residue_starts,
                full_annot_names,
            ) = create_complete_atom_array_from_restype_index(
                restype_index=restype_index[chain_mask],
                residue_dictionary=residue_dictionary,
                chain_id=single_chain_id,
                extra_fields=extra_fields,
                backbone_only=backbone_only,
                residue_index_offset=res_index_offset,
                res_id=chain_res_id,
            )
            chain_atom_arrays.append(atom_array)
            chain_residue_starts.append(residue_starts + residue_starts_offset)
            residue_starts_offset += len(atom_array)
            res_index_offset += atom_array.res_index.max() + 1

        concatenated_array = sum(chain_atom_arrays, bs.AtomArray(length=0))
        for key in atom_array._annot.keys():
            if key not in concatenated_array._annot:
                concatenated_array.set_annotation(
                    key,
                    np.concatenate([atoms._annot[key] for atoms in chain_atom_arrays]),
                )
        return (
            concatenated_array,
            np.concatenate(chain_residue_starts),
            full_annot_names,
        )
    else:

        if backbone_only:
            residue_sizes = len(residue_dictionary.backbone_atoms) * len(restype_index)
        else:
            residue_sizes = residue_dictionary.get_residue_sizes(
                restype_index, chain_id
            )
            # (n_residues,) NOT (n_atoms,)

        residue_starts = np.concatenate(
            [[0], np.cumsum(residue_sizes)[:-1]]
        )  # (n_residues,)
        new_atom_array = bs.AtomArray(length=np.sum(residue_sizes))
        chain_id = np.full(len(new_atom_array), chain_id, dtype="U4")
        new_atom_array.set_annotation(
            "chain_id",
            chain_id,
        )
        full_annot_names = [
            "chain_id",
        ]
        residue_index = (
            np.cumsum(get_residue_starts_mask(new_atom_array, residue_starts)) - 1
        )
        relative_atom_index = (
            np.arange(len(new_atom_array)) - residue_starts[residue_index]
        )
        atom_names = new_atom_array.atom_name
        new_atom_array.set_annotation("restype_index", restype_index[residue_index])
        atom_names = residue_dictionary.get_atom_names(
            new_atom_array.restype_index, relative_atom_index, chain_id
        )
        new_atom_array.set_annotation("atom_name", atom_names)
        new_atom_array.set_annotation(
            "res_name",
            np.array(residue_dictionary.residue_names)[new_atom_array.restype_index],
        )

        new_atom_array.set_annotation("chain_res_index", residue_index)
        new_atom_array.set_annotation("res_index", residue_index + residue_index_offset)
        if res_id is not None:
            assert res_id.shape == restype_index.shape
            new_atom_array.set_annotation("res_id", res_id[residue_index])
        else:
            new_atom_array.set_annotation("res_id", residue_index + 1)
        elements = residue_dictionary.get_elements(
            new_atom_array.restype_index, relative_atom_index, chain_id
        )
        new_atom_array.set_annotation("element", elements)
        new_atom_array.set_annotation(
            "elemtype_index",
            map_categories_to_indices(
                new_atom_array.element, residue_dictionary.element_types
            ),
        )
        full_annot_names += [
            "atom_name",
            "restype_index",
            "elemtype_index",
            "res_name",
            "chain_res_index",
            "res_index",
            "res_id",
        ]
        if extra_fields is not None:
            for f in extra_fields:
                new_atom_array.add_annotation(
                    f, dtype=float if f in ["occupancy", "b_factor"] else int
                )
        return new_atom_array, residue_starts, full_annot_names
