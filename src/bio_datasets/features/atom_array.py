"""Biotite atom array feature type for compatibility with HF datasets.

We write an Arrow ExtensionType for the AtomArray.
A couple of options for this:
1. Atom14
2. Backbone
3. Atom37
The issue with other formats is that the list of constituents could vary.
"""
import gzip
import os
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from io import BytesIO, StringIO
from os import PathLike
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa
from biotite import structure as bs
from biotite.structure import get_chains
from biotite.structure.filter import filter_amino_acids
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile
from biotite.structure.residues import get_residue_starts
from datasets import Array1D, Array2D, config
from datasets.download import DownloadConfig
from datasets.features.features import Feature, get_nested_type
from datasets.table import array_cast, cast_array_to_feature
from datasets.utils.file_utils import is_local_path, xopen, xsplitext
from datasets.utils.py_utils import no_op_if_value_is_null, string_to_dict

from bio_datasets import config as bio_config
from bio_datasets.protein import constants as protein_constants
from bio_datasets.protein.protein import (
    BACKBONE_ATOMS,
    Protein,
    ProteinComplex,
    create_complete_atom_array_from_aa_index,
    filter_backbone,
    get_residue_starts_mask,
)

if bio_config.FOLDCOMP_AVAILABLE:
    import foldcomp

if bio_config.FASTPDB_AVAILABLE:
    import fastpdb


FILE_TYPE_TO_EXT = {
    "pdb": "pdb",
    "PDB": "pdb",
    "CIF": "cif",
    "cif": "cif",
    "FCZ": "fcz",
    "fcz": "fcz",
    "foldcomp": "fcz",
}


extra_annots = [
    "b_factor",
    "occupancy",
    "charge",
    "element",  # seems redundant
    "atom_id",
]


def filter_chains(structure, chain_ids):
    # TODO: double-check numeric chain id is ok...
    all_chains = get_chains(structure)
    if len(all_chains) == 0:
        raise ValueError("No chains found in the input file.")
    if chain_ids is None:
        return structure
    if isinstance(chain_ids, str):
        chain_ids = [chain_ids]
    for chain in chain_ids:
        if chain not in all_chains:
            raise ValueError(f"Chain {chain} not found in input file")
    chain_filter = [a.chain_id in chain_ids for a in structure]
    structure = structure[chain_filter]
    return structure


def infer_bytes_format(b: bytes) -> str:
    """
    Infer the file format of a bytes object from its contents.
    """
    if b.startswith(b"FCMP"):
        return "fcz"
    else:
        # otherwise, assume pdb for now
        return "pdb"


def load_structure(
    fpath_or_handler,
    format="pdb",
    model: int = 1,
    extra_fields=None,
):
    """
    TODO: support foldcomp format, binary cif format
    TODO: support model choice / multiple models (multiple conformations)
    Args:
        fpath: filepath to either pdb or cif file
        chain: the chain id or list of chain ids to load
    Returns:
        biotite.structure.AtomArray
    """
    format = FILE_TYPE_TO_EXT[format]
    if isinstance(fpath_or_handler, str) and fpath_or_handler.endswith(".gz"):
        with gzip.open(fpath_or_handler, "rt") as f:
            fpath_or_handler = StringIO(f.read())
    if format == "cif":
        pdbxf = CIFFile.read(fpath_or_handler)
        structure = pdbxf.get_structure(
            model=model,
            extra_fields=extra_fields,
        )
    elif format == "pdb":
        if bio_config.FASTPDB_AVAILABLE:
            pdbf = fastpdb.PDBFile.read(fpath_or_handler)
        else:
            pdbf = PDBFile.read(fpath_or_handler)
        structure = pdbf.get_structure(
            model=model,
            extra_fields=extra_fields,
        )
    elif format == "fcz":
        if not bio_config.FOLDCOMP_AVAILABLE:
            raise ImportError(
                "Foldcomp is not installed. Please install it with `pip install foldcomp`"
            )
        import foldcomp

        if is_open_compatible(fpath_or_handler):
            with open(fpath_or_handler, "rb") as fcz:
                fcz_binary = fcz.read()
        else:
            raise ValueError(f"Unsupported file type: expected path or bytes handler")
        (_, pdb_str) = foldcomp.decompress(fcz_binary)
        lines = pdb_str.splitlines()
        pdbf = PDBFile()
        pdbf.lines = lines
        structure = pdbf.get_structure(
            model=model,
            extra_fields=extra_fields,
        )
    else:
        raise ValueError(f"Unsupported file format: {format}")

    return structure


def atom_array_from_dict(d: dict) -> bs.AtomArray:
    sequence = d["sequence"]
    annots_keys = [k for k in d.keys() if k in extra_annots]
    if "backbone_coords" in d:
        backbone_coords = d["backbone_coords"]
        assert len(sequence) == len(d["backbone_coords"]["N"])
        atoms = []
        for res_ix, aa in enumerate(sequence):
            # TODO: better support for non-standard amino acids
            if aa == "U":
                aa = "C"
            if aa == "O":
                aa = "K"
            res_name = protein_constants.restype_1to3[aa]
            for atom_name in BACKBONE_ATOMS:
                annots = {}
                for k in annots_keys:
                    annots[k] = d[k][res_ix]
                atom = bs.Atom(
                    coord=backbone_coords[atom_name][res_ix],
                    chain_id="A",
                    res_id=res_ix + 1,
                    res_name=res_name,
                    hetero=False,
                    atom_name=atom_name,
                    element=atom_name[0],
                    **annots,
                )
                atoms.append(atom)
        arr = bs.array(atoms)
        return arr
    elif "atom37_coords" in d:
        raise NotImplementedError("Atom37 not supported yet")
    else:
        raise ValueError("No coordinates found")


def encode_biotite_atom_array(
    array: bs.AtomArray, encode_with_foldcomp: bool = False, name: Optional[str] = None
) -> str:
    """
    Encode a biotite AtomArray to pdb string bytes.

    TODO: support foldcomp encoding
    """
    pdb = pdb.PDBFile()
    pdb.set_structure(array)
    contents = "\n".join(pdb.lines) + "\n"
    if encode_with_foldcomp:
        import foldcomp

        if name is None:
            name = getattr(array, "name", str(uuid.uuid4()))
        return foldcomp.compress(name, contents)
    else:
        return contents.encode()


def is_open_compatible(file):
    return isinstance(file, (str, PathLike))


def infer_type_from_structure_file_dict(d: dict) -> Tuple[Optional[str], Optional[str]]:
    if "type" in d and d["type"] is not None:
        return d["type"]
    elif "path" in d:
        path = d["path"]
        if path.endswith(".gz"):
            path = path[:-3]
        ext = xsplitext(path)[1][1:]
        return ext
    elif "bytes" in d:
        return infer_bytes_format(d["bytes"])
    else:
        return None


def load_structure_from_file_dict(
    d: dict,
    token_per_repo_id: Optional[Dict[str, int]] = None,
    extra_fields: Optional[List[str]] = None,
) -> bs.AtomArray:
    if token_per_repo_id is None:
        token_per_repo_id = {}

    path, bytes_ = d.get("path"), d.get("bytes")
    file_type = infer_type_from_structure_file_dict(d)
    if bytes_ is None:
        if path is None:
            raise ValueError(
                f"A structure should have one of 'path' or 'bytes' but both are None in {d}."
            )
        else:
            if is_local_path(path):
                atom_array = load_structure(
                    path,
                    format=file_type,
                    extra_fields=extra_fields,
                )
            else:
                source_url = path.split("::")[-1]
                pattern = (
                    config.HUB_DATASETS_URL
                    if source_url.startswith(config.HF_ENDPOINT)
                    else config.HUB_DATASETS_HFFS_URL
                )
                try:
                    repo_id = string_to_dict(source_url, pattern)["repo_id"]
                    token = token_per_repo_id.get(repo_id)
                except ValueError:
                    token = None
                download_config = DownloadConfig(token=token)
                with xopen(path, "r", download_config=download_config) as f:
                    atom_array = load_structure(
                        f,
                        format=file_type or "pdb",
                        extra_fields=extra_fields,
                    )

    else:
        if path is not None:
            if file_type == "fcz":
                fhandler = BytesIO(bytes_)
            elif file_type == "pdb":
                fhandler = StringIO(bytes_.decode())
            elif file_type == "cif":
                fhandler = StringIO(bytes_.decode())
            else:
                raise ValueError(
                    f"Unsupported file format: {file_type} for bytes input"
                )
            atom_array = load_structure(
                fhandler,
                format=file_type,
                extra_fields=extra_fields,
            )
        else:
            if file_type == "fcz":
                _, pdb = foldcomp.decompress(bytes_)
            else:
                pdb = bytes_.decode()
            contents = StringIO(pdb)
            # assume pdb format in bytes for now - is bytes only possible internally?
            atom_array = load_structure(
                contents,
                format="pdb",
                extra_fields=extra_fields,
            )
    return atom_array


# n.b. metadata like chain_id, pdb_id, etc. should be stored separately
@dataclass
class _AtomArrayFeatureMixin:
    with_box: bool = False
    with_bonds: bool = False
    with_occupancy: bool = False
    with_b_factor: bool = False
    with_res_id: bool = False  # can be inferred...
    with_atom_id: bool = False
    with_charge: bool = False
    with_element: bool = False
    with_ins_code: bool = False
    with_hetero: bool = False

    @property
    def extra_fields(self):
        # values that can be passed to biotite load_structure
        extra_fields = []
        if self.with_occupancy:
            extra_fields.append("occupancy")
        if self.with_b_factor:
            extra_fields.append("b_factor")
        if self.with_atom_id:
            extra_fields.append("atom_id")
        if self.with_charge:
            extra_fields.append("charge")
        return extra_fields


@dataclass
class AtomArrayFeature(_AtomArrayFeatureMixin, Feature):
    """
    AtomArray [`Feature`] to read macromolecular atomic structure data from a PDB or CIF file.

    This feature stores the array directly as a pa struct (basically a dictionary of arrays),
    as defined in the AtomArrayExtensionType.

    Input: The BiomolecularStructureFile feature accepts as (encodeable) input (Q. where would 'input' typically occur):
    - A `biotite.structure.AtomArray` object.
    - TODO: a Biopython structure object
    - TODO: a file handler or file contents string?
    - A dictionary with the required keys:
        - sequence
        - backbone_coords: a dictionary with keys:
            - N
            - CA
            - C
            - O
        - atom37_coords: an array of shape (N, 37, 3)
        - atom37_mask: a boolean array of shape (N, 37)
        Only backbone_coords or atom37_coords + mask need to be provided, not both.
        All other keys are optional, but must correspond to fields in the AtomArrayExtensionType:
        - chain_id
        - res_id
        - ins_code
        - res_name
        - hetero
        - atom_name
        - box
        - bonds
        - occupancy
        - b_factor
        - atom_id
        - charge

    From biotite docs:
    Atom attributes:

    The following annotation categories are mandatory:

    =========  ===========  =================  =======================================
    Category   Type         Examples           Description
    =========  ===========  =================  =======================================
    chain_id   string (U4)  'A','S','AB', ...  Polypeptide chain
    res_id     int          1,2,3, ...         Sequence position of residue
    ins_code   string (U1)  '', 'A','B',..     PDB insertion code (iCode)
    res_name   string (U5)  'GLY','ALA', ...   Residue name
    hetero     bool         True, False        False for ``ATOM``, true for ``HETATM``
    atom_name  string (U6)  'CA','N', ...      Atom name
    element    string (U2)  'C','O','SE', ...  Chemical Element
    =========  ===========  =================  =======================================

    For all :class:`Atom`, :class:`AtomArray` and :class:`AtomArrayStack`
    objects these annotations are initially set with default values.
    Additionally to these annotations, an arbitrary amount of annotation
    categories can be added via :func:`add_annotation()` or
    :func:`set_annotation()`.
    The annotation arrays can be accessed either via the method
    :func:`get_annotation()` or directly (e.g. ``array.res_id``).

    The following annotation categories are optionally used by some
    functions:

    =========  ===========  =================   ============================
    Category   Type         Examples            Description
    =========  ===========  =================   ============================
    atom_id    int          1,2,3, ...          Atom serial number
    b_factor   float        0.9, 12.3, ...      Temperature factor
    occupancy  float        .1, .3, .9, ...     Occupancy
    charge     int          -2,-1,0,1,2, ...    Electric charge of the atom
    =========  ===========  =================   ============================

    Bond information can be associated to an :class:`AtomArray` or
    :class:`AtomArrayStack` by setting the ``bonds`` attribute with a
    :class:`BondList`.
    A :class:`BondList` specifies the indices of atoms that form chemical
    bonds.
    """

    all_atoms_present: ClassVar[
        bool
    ] = False  # to use this, need to be decoding to Protein (ProteinAtomArrayFeature)
    drop_sidechains: ClassVar[bool] = False
    requires_encoding: bool = True
    requires_decoding: bool = True
    decode: bool = True
    coords_dtype: str = "float32"
    b_factor_is_plddt: bool = False
    b_factor_dtype: str = "float32"
    chain_id: Optional[
        str
    ] = None  # single chain id - means we will intepret structure as a single chain
    id: Optional[str] = None
    # Automatically constructed
    _type: str = field(
        default="AtomArray", init=False, repr=False
    )  # registered feature name

    def _make_features_dict(self):
        features = [
            ("coords", Array2D((None, 3), self.coords_dtype)),
            ("aa_index", Array1D((None,), "uint8")),
            ("chain_id", Array1D((None,), "string")),
        ]
        if not self.all_atoms_present:
            features.append(("atom_name", Array1D((None,), "string")))
            features.append(("residue_starts", Array1D((None,), "uint16")))
        if self.with_res_id:
            features.append(("res_id", Array1D((None,), "uint16")))
        if self.with_hetero:
            features.append(("hetero", Array1D((None,), "bool")))
        if self.with_ins_code:
            features.append(("ins_code", Array1D((None,), "string")))
        if self.with_box:
            features.append(("box", Array2D((3, 3), "float32")))
        if self.with_bonds:
            features.append(("bond_edges", Array2D((None, 2), "uint16")))
            features.append(("bond_types", Array1D((None,), "uint8")))
        if self.with_occupancy:
            features.append(("occupancy", Array1D((None,), "float16")))
        if self.with_b_factor:
            # TODO: maybe have specific storage format for plddt bfactor (fixed range)
            features.append(("b_factor", Array1D((None,), self.b_factor_dtype)))
        if self.with_charge:
            features.append(("charge", Array1D((None,), "int8")))
        if self.with_element:
            features.append(("element", Array1D((None,), "string")))
        return OrderedDict(
            features
        )  # order may not be important due to Features.recursive_reorder

    def __post_init__(self):
        # init the StructFeature - since it inherits from dict, pa type inference is automatic (via get_nested_type)
        self._features = self._make_features_dict()

    def __call__(self):
        return get_nested_type(self._features)

    @property
    def required_keys(self):
        required_keys = ["coords", "atom_name", "res_name", "chain_id"]
        if self.with_box:
            required_keys.append("box")
        if self.with_bonds:
            required_keys.append("bonds")
        return required_keys

    def cast_storage(self, array: pa.StructArray) -> pa.StructArray:
        array_fields = {field.name for field in array.type}
        # c.f. cast_array_to_feature: since we don't inherit from dict, we reproduce the logic here
        null_array = pa.array([None] * len(array))
        arrays = [
            cast_array_to_feature(
                array.field(name) if name in array_fields else null_array, subfeature
            )
            for name, subfeature in self._features.items()
        ]
        return pa.StructArray.from_arrays(
            arrays, names=list(self._features), mask=array.is_null()
        )

    def encode_example(self, value: Union[bs.AtomArray, dict]) -> dict:
        if isinstance(value, Protein):
            value = value.atoms
        if isinstance(value, dict):
            if "bytes" in value or "path" in value or "type" in value:
                # if it's already encoded, we don't need to encode it again
                struct = load_structure_from_file_dict(
                    value, extra_fields=self.extra_fields
                )
                return self.encode_example(struct)
            if all([attr in value for attr in self.required_keys]):
                return value
            if "sequence" in value:
                value = atom_array_from_dict(value)
            else:
                raise ValueError(
                    "Cannot encode dict without sequence or (bytes/path/type)"
                )
            return self.encode_example(value)
        elif isinstance(value, bs.AtomArray):
            if self.all_atoms_present:
                orig_length = len(value)
                value, residue_starts = Protein.standardise_atoms(
                    value, backbone_only=self.drop_sidechains
                )
                if len(value) != orig_length:
                    raise ValueError(
                        "Standardisation changed the length of the atom array"
                    )
            else:
                residue_starts = get_residue_starts(value)
            if len(value) > 65535:
                raise ValueError(
                    "AtomArray too large to fit in uint16 (residue starts)"
                )
            atom_array_struct = {
                "coords": value.coord,
                "aa_index": np.array(
                    [
                        protein_constants.restype_order_with_x[
                            protein_constants.restype_3to1[res_name]
                        ]
                        for res_name in value.res_name[residue_starts]
                    ]
                ),
            }
            if not self.all_atoms_present:
                atom_array_struct["residue_starts"] = residue_starts
                atom_array_struct["atom_name"] = value.atom_name
            if self.chain_id is None:
                atom_array_struct["chain_id"] = value.chain_id[residue_starts]
            else:
                atom_array_struct["chain_id"] = None
            for attr in [
                "box",
                "occupancy",
                "b_factor",
                "atom_id",
                "charge",
                "element",
                "res_id",
                "ins_code",
                "hetero",
            ]:
                if getattr(self, f"with_{attr}"):
                    if (
                        attr == "b_factor" and self.b_factor_is_plddt
                    ) or attr == "res_id":
                        # residue-level annotation
                        atom_array_struct[attr] = getattr(value, attr)[residue_starts]
                    else:
                        atom_array_struct[attr] = getattr(value, attr)
            if self.with_bonds:
                bonds_array = value.bond_list.as_array()
                assert bonds_array.ndim == 2 and bonds_array.shape[1] == 3
                atom_array_struct["bond_edges"] = bonds_array[:, :2]
                atom_array_struct["bond_types"] = bonds_array[:, 2]
            return atom_array_struct
        elif isinstance(value, (str, os.PathLike)):
            if os.path.exists(value):
                file_type = xsplitext(value)[1][1:].lower()
                return self.encode_example(
                    load_structure(
                        value, format=file_type, extra_fields=self.extra_fields
                    )
                )
        elif isinstance(value, bytes):
            # assume it encodes file contents.
            # TODO: automatically check for foldcomp format
            file_type = infer_bytes_format(value)
            fhandler = BytesIO(value)
            return self.encode_example(
                load_structure(
                    fhandler, format=file_type, extra_fields=self.extra_fields
                )
            )
        else:
            raise ValueError(f"Unsupported value type: {type(value)}")

    def decode_example(self, value: dict, token_per_repo_id=None) -> "bs.AtomArray":
        """
        def add_annotation(self, category, dtype):
        Add an annotation category, if not already existing.

        Initially the new annotation is filled with the *zero*
        representation of the given type.

        Parameters
        ----------
        category : str
            The annotation category to be added.
        dtype : type or str
            A type instance or a valid *NumPy* *dtype* string.
            Defines the type of the annotation

        See Also
        --------
        set_annotation

        Notes
        -----
        If the annotation category already exists, a compatible dtype is chosen,
        that is also able to represent the old values.
        if category not in self._annot:
            self._annot[str(category)] = np.zeros(self._array_length, dtype=dtype)
        elif np.can_cast(self._annot[str(category)].dtype, dtype):
            self._annot[str(category)] = self._annot[str(category)].astype(dtype)
        elif np.can_cast(dtype, self._annot[str(category)].dtype):
            # The existing dtype is more general
            pass
        else:
            raise ValueError(
                f"Cannot cast '{str(category)}' "
                f"with dtype '{self._annot[str(category)].dtype}' into '{dtype}'"
        """
        # TODO: null check
        # TODO: optimise this...if we set format to numpy, everything is a numpy array which should be ideal
        num_atoms = len(value["coords"])
        if self.all_atoms_present:
            aa_index = value.pop("aa_index")
            if self.chain_id is None:
                chain_id = value.pop("chain_id")  # residue-level annotation
            else:
                chain_id = np.full(len(aa_index), self.chain_id)
                del value["chain_id"]
            atoms, residue_starts, _ = create_complete_atom_array_from_aa_index(
                aa_index, chain_id, backbone_only=self.drop_sidechains
            )
            residue_index = (
                np.cumsum(get_residue_starts_mask(atoms, residue_starts)) - 1
            )
        else:
            atoms = bs.AtomArray(num_atoms)
            residue_starts = value.pop("residue_starts")
            residue_index = (
                np.cumsum(get_residue_starts_mask(atoms, residue_starts)) - 1
            )
            # residue-level annotations -> atom-level annotations
            if "res_id" in value:
                atoms.set_annotation("res_id", value.pop("res_id")[residue_index])
            else:
                atoms.set_annotation("res_id", residue_index + 1)  # 1-based residue ids
            atoms.set_annotation("aa_index", value.pop("aa_index")[residue_index])
            if "chain_id" in value and self.chain_id is None:
                atoms.set_annotation("chain_id", value.pop("chain_id")[residue_index])
            elif self.chain_id is not None:
                atoms.set_annotation("chain_id", np.full(num_atoms, self.chain_id))
            atoms.set_annotation(
                "res_name", np.array(protein_constants.resnames)[atoms.aa_index]
            )

        if self.b_factor_is_plddt and "b_factor" in value:
            atoms.set_annotation("b_factor", value.pop("b_factor")[residue_index])

        atoms.coord = value.pop("coords")
        if not self.with_element:
            atoms.set_annotation("element", np.char.array(atoms.atom_name).astype("U1"))
        if "bond_edges" in value:
            bonds_array = value.pop("bond_edges")
            bond_types = value.pop("bond_types")
            bonds_array = np.concatenate([bonds_array, bond_types[:, None]], axis=1)
            bonds = bs.BondList(num_atoms, bonds_array)
            atoms.bond_list = bonds

        # anything left in value is an atom-level annotation
        for key, value in value.items():
            atoms.set_annotation(key, value)

        return atoms


@dataclass
class StructureFeature(_AtomArrayFeatureMixin, Feature):
    """Structure [`Feature`] to read (bio)molecular atomic structure data from supported file types.
    The file contents are serialized as bytes, file path and file type within an Arrow table.
    The file contents are automatically decoded to a biotite AtomArray (if mode=="array") or a
    Biopython structure (if mode=="structure") when loading data from the dataset.

    This is similar to the Image/Audio features in the HF datasets library.

    - AtomArray documentation: https://www.biotite-python.org/latest/apidoc/biotite.structure.AtomArray.html#biotite.structure.AtomArray
    - Structure documentation: https://biopython.org/wiki/The_Biopython_Structural_Bioinformatics_FAQ#the-structure-object

    Input: The StructureFeature accepts as (encodeable) input (e.g. as structure values in the outputs of dataset_builder.generate_examples()):
    - A `str`: Absolute path to the structure file (i.e. random access is allowed).
    - A `dict` with the keys:

        - `path`: String with relative path of the structure file to the archive file.
        - `bytes`: Bytes of the structure file.

      This is useful for archived files with sequential access.
    - A `biotite.structure.AtomArray` object.
    - TODO: a Biopython structure object
    - TODO: a file handler or file contents string?

    Args:
        decode (`bool`, defaults to `True`):
            Whether to decode the structure data. If `False`,
            returns the underlying dictionary in the format `{"path": structure_path, "bytes": structure_bytes, "type": structure_type}`.
    """

    requires_encoding: bool = True
    requires_decoding: bool = True
    decode: bool = True
    id: Optional[str] = None
    encode_with_foldcomp: bool = False
    pa_type: ClassVar[Any] = pa.struct(
        {"bytes": pa.binary(), "path": pa.string(), "type": pa.string()}
    )
    _type: str = field(default="Structure", init=False, repr=False)

    def __call__(self):
        return self.pa_type

    def encode_example(self, value: Union[str, bytes, bs.AtomArray]) -> dict:
        """Encode example into a format for Arrow.

        This determines what gets written to the Arrow file.
        TODO: accept Protein as input?
        """
        file_type = infer_type_from_structure_file_dict(value)
        if isinstance(value, str):
            return {"path": value, "bytes": None, "type": file_type}
        elif isinstance(value, bytes):
            # just assume pdb format for now
            return {"path": None, "bytes": value, "type": file_type}
        elif isinstance(value, bs.AtomArray):
            return {
                "path": None,
                "bytes": encode_biotite_atom_array(
                    value,
                    encode_with_foldcomp=self.encode_with_foldcomp,
                ),
                "type": "pdb" if not self.encode_with_foldcomp else "fcz",
            }
        elif value.get("path") is not None and os.path.isfile(value["path"]):
            path = value["path"]
            # we set "bytes": None to not duplicate the data if they're already available locally
            # (this assumes invocation in what context?)
            return {"bytes": None, "path": path, "type": file_type or "pdb"}
        elif value.get("bytes") is not None or value.get("path") is not None:
            # store the Structure bytes, and path is optionally used to infer the Structure format using the file extension
            path = value.get("path")
            return {"bytes": value.get("bytes"), "path": path, "type": file_type}
        else:
            raise ValueError(
                f"A structure sample should have one of 'path' or 'bytes' but they are missing or None in {value}."
            )

    def decode_example(self, value: dict, token_per_repo_id=None) -> "bs.AtomArray":
        """Decode example structure file into AtomArray data.

        Args:
            value (`str` or `dict`):
                A string with the absolute structure file path, a dictionary with
                keys:

                - `path`: String with absolute or relative structure file path.
                - `bytes`: The bytes of the structure file.
                - `type`: The type of the structure file (e.g. "pdb", "cif", "fcz")
                Must be not None.
            token_per_repo_id (`dict`, *optional*):
                To access and decode
                structure files from private repositories on the Hub, you can pass
                a dictionary repo_id (`str`) -> token (`bool` or `str`).

        Returns:
            `biotite.AtomArray`
        """
        if not self.decode:
            raise RuntimeError(
                "Decoding is disabled for this feature. Please use Structure(decode=True) instead."
            )

        array = load_structure_from_file_dict(
            value, token_per_repo_id=token_per_repo_id, extra_fields=self.extra_fields
        )
        return array

    def cast_storage(self, storage: pa.StructArray) -> pa.StructArray:
        if pa.types.is_struct(storage.type):
            if storage.type.get_field_index("bytes") >= 0:
                bytes_array = storage.field("bytes")
            else:
                bytes_array = pa.array([None] * len(storage), type=pa.binary())
            if storage.type.get_field_index("path") >= 0:
                path_array = storage.field("path")
            else:
                path_array = pa.array([None] * len(storage), type=pa.string())
            storage = pa.StructArray.from_arrays(
                [bytes_array, path_array, storage.field("type")],
                names=["bytes", "path", "type"],
                mask=storage.is_null(),
            )
        else:
            raise ValueError(f"Unsupported storage type: {storage.type}")
        return array_cast(storage, self.pa_type)

    def embed_storage(self, storage: pa.StructArray) -> pa.StructArray:
        """Embed the file contents into the Arrow table.

        Configured by the embed_external_files flag in Dataset.push_to_hub / DatasetsDict
        TODO: check this is working as expected
        """

        @no_op_if_value_is_null
        def path_to_bytes(path):
            with xopen(path, "rb") as f:
                bytes_ = f.read()
            if path.endswith(".gz"):
                assert is_local_path(path), "Gzipped files must have local file paths."
                with gzip.open(path, "rb") as f:
                    bytes_ = f.read()
            return bytes_

        bytes_array = pa.array(
            [
                (path_to_bytes(x["path"]) if x["bytes"] is None else x["bytes"])
                if x is not None
                else None
                for x in storage.to_pylist()
            ],
            type=pa.binary(),
        )
        path_array = pa.array(
            [
                os.path.basename(path) if path is not None else None
                for path in storage.field("path").to_pylist()
            ],
            type=pa.string(),
        )
        type_array = storage.field("type")
        storage = pa.StructArray.from_arrays(
            [bytes_array, path_array, type_array],
            ["bytes", "path", "type"],
            mask=bytes_array.is_null(),
        )
        return array_cast(storage, self.pa_type)


@dataclass
class ProteinStructureFeature(StructureFeature):
    _type: str = field(default="ProteinStructure", init=False, repr=False)

    def encode_example(self, value: Union[Protein, dict, bs.AtomArray]) -> dict:
        if isinstance(value, bs.AtomArray):
            value = value[filter_amino_acids(value)]
            if "element" not in value._annot:
                value.set_annotation(
                    "element", np.char.array(value.atom_name).astype("U1")
                )
            value = value[~np.isin(value.element, ["H", "D"])]
        return super().encode_example(value)

    def decode_example(
        self, encoded: dict, token_per_repo_id=None
    ) -> Union["Protein", "ProteinComplex"]:
        atoms = super().decode_example(encoded, token_per_repo_id=token_per_repo_id)
        # TODO: check this always excludes hetatms
        # TODO: filter amino acids in encode_example also where possible
        atoms = atoms[filter_amino_acids(atoms)]
        chain_ids = np.unique(atoms.chain_id)
        if len(chain_ids) > 1:
            assert (
                not self.drop_sidechains
            ), "Cannot drop sidechains for multi-chain proteins."
            return ProteinComplex.from_atoms(atoms)
        return Protein(atoms, backbone_only=self.drop_sidechains)


@dataclass
class ProteinAtomArrayFeature(AtomArrayFeature):

    """Decodes to a `bio_datasets.protein.Protein` or `bio_datasets.protein.ProteinComplex` object.

    These objects have standardised atoms (with nans for any missing atoms),
    and are guaranteed to contain no HETATMs or hydrogens.
    """

    all_atoms_present: bool = False
    drop_sidechains: bool = False
    internal_coords_type: str = None  # foldcomp, idealised, or pnerf
    _type: str = field(
        default="Protein", init=False, repr=False
    )  # registered feature name

    @classmethod
    def from_preset(cls, preset: str, **kwargs):
        if preset == "afdb":
            return cls(
                with_b_factor=True,
                b_factor_is_plddt=True,
                # b_factor_dtype="uint8"
                b_factor_dtype="float16",
                coords_dtype="float16",
                all_atoms_present=True,
                chain_id="A",
                **kwargs,
            )
        elif preset == "pdb":
            return cls(with_b_factor=False, coords_dtype="float16", **kwargs)
        else:
            raise ValueError(f"Unknown preset: {preset}")

    def encode_example(self, value: Union[Protein, dict, bs.AtomArray]) -> dict:
        # TODO: share this code
        if isinstance(value, bs.AtomArray):
            if self.drop_sidechains:
                value = value[filter_backbone(value)]
            if "element" not in value._annot:
                value.set_annotation(
                    "element", np.char.array(value.atom_name).astype("U1")
                )
            value = value[~np.isin(value.element, ["H", "D"])]
            # TODO: check this always excludes hetatms
            return super().encode_example(value[filter_amino_acids(value)])
        if isinstance(value, (Protein, ProteinComplex)):
            if self.drop_sidechains:
                value = value.backbone()
            return super().encode_example(value.atoms)
        return super().encode_example(value)

    def decode_example(
        self, encoded: dict, token_per_repo_id=None
    ) -> Union["Protein", "ProteinComplex"]:
        atoms = super().decode_example(encoded, token_per_repo_id=token_per_repo_id)
        atoms = atoms[filter_amino_acids(atoms)]
        chain_ids = np.unique(atoms.chain_id)
        if len(chain_ids) > 1:
            assert (
                not self.drop_sidechains
            ), "Cannot drop sidechains for multi-chain proteins."
            return ProteinComplex.from_atoms(atoms)
        return Protein(atoms, backbone_only=self.drop_sidechains)
