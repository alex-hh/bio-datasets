import gzip
import os
from os import PathLike
from typing import Optional

import numpy as np

from bio_datasets import config as bio_config

if bio_config.FASTPDB_AVAILABLE:
    import fastpdb

if bio_config.FOLDCOMP_AVAILABLE:
    import foldcomp

from biotite import structure as bs
from biotite.file import InvalidFileError
from biotite.structure.atoms import repeat
from biotite.structure.io import pdbx
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx.convert import (
    _get_block,
    _get_transformations,
    _parse_operation_expression,
)
from biotite.structure.residues import get_residue_starts
from biotite.structure.util import matrix_rotate

from .residue import (
    ResidueDictionary,
    create_complete_atom_array_from_restype_index,
    get_residue_starts_mask,
)

FILE_TYPE_TO_EXT = {
    "pdb": "pdb",
    "PDB": "pdb",
    "CIF": "cif",
    "cif": "cif",
    "bcif": "bcif",
    "FCZ": "fcz",
    "fcz": "fcz",
    "foldcomp": "fcz",
}


def is_open_compatible(file):
    return isinstance(file, (str, PathLike))


def fill_missing_polymer_chain_residues(
    chain_atoms, complete_res_ids, restype_index, residue_dict, chain_id
):
    """Fill in missing residues for a single polymer chain."""
    missing_res_mask = ~np.isin(complete_res_ids, chain_atoms.res_id)
    missing_atoms, _, _ = create_complete_atom_array_from_restype_index(
        restype_index[missing_res_mask],
        residue_dict,
        chain_id,
        res_id=complete_res_ids[missing_res_mask],
    )

    missing_atoms.set_annotation(
        "altloc_id", np.full(len(missing_atoms), ".").astype("str")
    )
    missing_atoms.set_annotation(
        "auth_asym_id",
        np.full(len(missing_atoms), chain_id).astype("str"),
    )
    for chain_level_annot in ["label_asym_id", "label_entity_id"]:
        if chain_level_annot in chain_atoms._annot:
            annot_array = chain_atoms._annot[chain_level_annot]
            first_val = annot_array[0]
            assert np.all(annot_array == first_val)
            missing_atoms.set_annotation(
                chain_level_annot,
                np.full(len(missing_atoms), first_val).astype(annot_array.dtype),
            )

    missing_atoms.set_annotation(
        "auth_seq_id", np.full(len(missing_atoms), -1).astype(int)
    )
    if "occupancy" in chain_atoms._annot:
        raise NotImplementedError("occupancy not supported yet")
    complete_atoms = chain_atoms + missing_atoms

    for annot, chain_annot in chain_atoms._annot.items():
        if annot not in missing_atoms._annot:
            # hopefully this is ok...
            complete_atoms.set_annotation(
                annot,
                np.concatenate(
                    [chain_annot, np.zeros(len(missing_atoms), dtype=chain_annot.dtype)]
                ),
            )
        else:
            complete_atoms.set_annotation(
                annot,
                np.concatenate([chain_annot, missing_atoms._annot[annot]]),
            )

    residue_starts = get_residue_starts(complete_atoms)

    res_perm = np.argsort(complete_atoms.res_id[residue_starts])
    residue_sizes = np.diff(np.append(residue_starts, len(complete_atoms)))

    permuted_residue_starts = residue_starts[res_perm]
    permuted_residue_sizes = residue_sizes[res_perm]

    permuted_residue_starts_atom = np.repeat(
        permuted_residue_starts, permuted_residue_sizes
    )
    post_perm_res_changes = (
        permuted_residue_starts_atom[1:] != permuted_residue_starts_atom[:-1]
    )
    post_perm_residue_starts = np.concatenate(
        [[0], np.where(post_perm_res_changes)[0] + 1]
    )
    _post_perm_res_index = (
        np.cumsum(get_residue_starts_mask(complete_atoms, post_perm_residue_starts)) - 1
    )

    permuted_relative_atom_index = (
        np.arange(len(complete_atoms)) - post_perm_residue_starts[_post_perm_res_index]
    )

    atom_perm = permuted_residue_starts_atom + permuted_relative_atom_index

    complete_atoms = complete_atoms[atom_perm]
    return complete_atoms


def fill_missing_polymer_residues(
    structure, entity_poly_seq, poly_entity_ids, poly_chain_ids
):
    """Fill in missing residues for polymer entities."""
    processed_chain_atoms = []
    residue_dict = ResidueDictionary.from_ccd_dict()
    for entity_chain_ids, entity_id in zip(poly_chain_ids, poly_entity_ids):
        poly_seq_entity_mask = (
            entity_poly_seq["entity_id"].as_array(int, -1) == entity_id
        )
        if not poly_seq_entity_mask.any():
            for chain_id in entity_chain_ids.split(","):
                processed_chain_atoms.append(
                    structure[
                        (structure.label_entity_id == str(entity_id))
                        & (structure.auth_asym_id == chain_id)
                    ]
                )
        else:
            complete_res_ids = entity_poly_seq["num"].as_array(int, -1)[
                poly_seq_entity_mask
            ]
            entity_res_name = entity_poly_seq["mon_id"].as_array(str)[
                poly_seq_entity_mask
            ]
            entity_restype_index = residue_dict.res_name_to_index(entity_res_name)

            for chain_id in entity_chain_ids.split(","):
                chain_atoms = structure[
                    (structure.auth_asym_id == chain_id)
                    & (structure.label_entity_id == str(entity_id))
                ]
                complete_atoms = fill_missing_polymer_chain_residues(
                    chain_atoms,
                    complete_res_ids,
                    entity_restype_index,
                    residue_dict,
                    chain_id,
                )
                complete_atoms.set_annotation(
                    "label_entity_id",
                    np.full(len(complete_atoms), entity_id).astype(int),
                )

                processed_chain_atoms.append(complete_atoms)
    return processed_chain_atoms


def _fill_missing_residues(structure: bs.AtomArray, block):
    processed_chain_atoms = []
    entity_poly = block["entity_poly"]
    entity_poly_seq = block["entity_poly_seq"]
    poly_chain_ids = entity_poly["pdbx_strand_id"].as_array(str)
    poly_entity_ids = entity_poly["entity_id"].as_array(int, -1)
    entity_ids = block["entity"]["id"].as_array(int, -1)

    nonpoly_entity_mask = ~np.isin(entity_ids, poly_entity_ids)
    nonpoly_entity_ids = entity_ids[nonpoly_entity_mask]
    for entity_id in nonpoly_entity_ids:
        processed_chain_atoms.append(
            structure[structure.label_entity_id == str(entity_id)]
        )

    processed_chain_atoms += fill_missing_polymer_residues(
        structure, entity_poly_seq, poly_entity_ids, poly_chain_ids
    )

    filled_structure = sum(processed_chain_atoms, bs.AtomArray(length=0))
    for key in structure._annot.keys():
        if key not in filled_structure._annot:
            filled_structure.set_annotation(
                key,
                np.concatenate(
                    [chain_atoms._annot[key] for chain_atoms in processed_chain_atoms]
                ),
            )
    return filled_structure


def get_pdbx_structure(
    pdbx_file,
    data_block=None,
    model=1,
    altloc="first",
    extra_fields=None,
    include_bonds=False,
    fill_missing_residues: bool = False,
):
    """Modified from biotite.structure.io.pdbx.get_structure to return canonical chain_id and res_id
    and also add auth_chain_id and auth_res_id annotations.

    TODO: support use_author_fields. But n.b. fill_missing_polymer_chain_residues relies
    on atoms.res_id matching the canonical `label_seq_id` res_id.
    """
    # there are also auth_comp_id, auth_atom_id for res_name, atom_name, but these seem a bit unnecessary.
    extra_fields = extra_fields or []
    extra_fields += [
        f
        for f in ["auth_asym_id", "auth_seq_id", "label_entity_id"]
        if f not in extra_fields
    ]
    structure = pdbx.get_structure(
        pdbx_file,
        data_block=data_block,
        model=model,
        extra_fields=extra_fields,
        use_author_fields=False,
        include_bonds=include_bonds,
        altloc=altloc,
    )
    block = _get_block(pdbx_file, data_block)

    if not fill_missing_residues:
        filled_structure = structure
    else:
        filled_structure = _fill_missing_residues(structure, block)

    return filled_structure


def _load_cif_structure(
    fpath_or_handler,
    file_type,
    model=1,
    extra_fields=None,
    fill_missing_residues=False,
    altloc="first",
    include_bonds=False,
):
    """Load a structure from cif or binary cif format.

    We wrap biotite.structure.io.pdbx.get_structure to return canonical chain_id and res_id
    and also add auth_chain_id and auth_res_id annotations.

    Cif files contain canonical labelling of res id chain id etc.
    as well as 'auth' labelling, which is what is shown in the pdb file.

    Optionally fill in missing residues with nan coordinates and standard atom names,
    by cross-referencing the entity_poly_seq header with the atom_site information and
    the CCD dictionary.

    TODO: an alternative to standardising here would be standardising within standardise_atoms
    if an additional kwarg (some map from res id to label res id) is provided.

    This would then generalise to e.g. aligning to uniprot as well, which would be extremely nice.
    Would be good to write some generic residue mapping utilites to allow this.
    """
    # we use filter_altloc all to make it easier to get the chain id mapping
    if file_type == "cif":
        pdbxf = pdbx.CIFFile.read(fpath_or_handler)
    else:
        pdbxf = pdbx.BinaryCIFFile.read(fpath_or_handler)
    return get_pdbx_structure(
        pdbxf,
        data_block=None,
        model=model,
        extra_fields=extra_fields,
        fill_missing_residues=fill_missing_residues,
        altloc=altloc,
        include_bonds=include_bonds,
    )


def _load_pdb_structure(
    fpath_or_handler,
    model=1,
    extra_fields=None,
    include_bonds=False,
):
    if bio_config.FASTPDB_AVAILABLE:
        pdbf = fastpdb.PDBFile.read(fpath_or_handler)
    else:
        pdbf = PDBFile.read(fpath_or_handler)
    structure = pdbf.get_structure(
        model=model,
        extra_fields=extra_fields,
        include_bonds=include_bonds,
    )
    return structure


def _load_foldcomp_structure(
    fpath_or_handler,
    model=1,
    extra_fields=None,
    include_bonds=False,
):
    if not bio_config.FOLDCOMP_AVAILABLE:
        raise ImportError(
            "Foldcomp is not installed. Please install it with `pip install foldcomp`"
        )

    if is_open_compatible(fpath_or_handler):
        with open(fpath_or_handler, "rb") as fcz:
            fcz_binary = fcz.read()
    else:
        raise ValueError("Unsupported file type: expected path or bytes handler")
    (_, pdb_str) = foldcomp.decompress(fcz_binary)
    lines = pdb_str.splitlines()
    pdbf = PDBFile()
    pdbf.lines = lines
    structure = pdbf.get_structure(
        model=model,
        extra_fields=extra_fields,
        include_bonds=include_bonds,
    )
    return structure


def load_structure(
    fpath_or_handler,
    file_type: Optional[str] = None,
    model: int = 1,
    extra_fields=None,
    fill_missing_residues=False,
    include_bonds=False,
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
    if isinstance(fpath_or_handler, (str, PathLike)) and fpath_or_handler.endswith(
        ".gz"
    ):
        file_type = (
            file_type or os.path.splitext(os.path.splitext(fpath_or_handler)[0])[1][1:]
        )
        # https://github.com/biotite-dev/biotite/issues/193
        with gzip.open(fpath_or_handler, "rt") as f:
            return load_structure(
                f,
                file_type=file_type,
                model=model,
                extra_fields=extra_fields,
                fill_missing_residues=fill_missing_residues,
                include_bonds=include_bonds,
            )

    if file_type is None and isinstance(fpath_or_handler, (str, PathLike)):
        file_type = os.path.splitext(fpath_or_handler)[1][1:]
    assert (
        file_type is not None
    ), "Format must be specified if fpath_or_handler is not a path"

    file_type = FILE_TYPE_TO_EXT[file_type]
    if fill_missing_residues:
        assert file_type in [
            "cif",
            "bcif",
        ], "Fill missing residues only supported for cif files"

    if file_type in ["cif", "bcif"]:
        return _load_cif_structure(
            fpath_or_handler,
            file_type=file_type,
            model=model,
            extra_fields=extra_fields,
            fill_missing_residues=fill_missing_residues,
            include_bonds=include_bonds,
        )

    elif file_type == "pdb":
        return _load_pdb_structure(
            fpath_or_handler,
            model=model,
            extra_fields=extra_fields,
            include_bonds=include_bonds,
        )
    elif file_type == "fcz":
        return _load_foldcomp_structure(
            fpath_or_handler,
            model=model,
            extra_fields=extra_fields,
            include_bonds=include_bonds,
        )
    else:
        raise ValueError(f"Unsupported file format: {file_type}")


def _apply_transformations(structure, transformation_dict, operations):
    """
    Get subassembly by applying the given operations to the input
    structure containing affected asym IDs.
    """
    # Additional first dimesion for 'structure.repeat()'
    assembly_coord = np.zeros((len(operations),) + structure.coord.shape)
    assembly_chain_ids = []
    # Apply corresponding transformation for each copy in the assembly
    for i, operation in enumerate(operations):
        coord = structure.coord
        # Execute for each transformation step
        # in the operation expression
        for op_step in operation:
            rotation_matrix, translation_vector = transformation_dict[op_step]
            # Rotate
            coord = matrix_rotate(coord, rotation_matrix)
            # Translate
            coord += translation_vector

        chain_id = structure.chain_id
        if (coord == structure.coord).all():
            # null operation should not change the chain ID
            assembly_chain_ids.append(chain_id.copy())
        else:
            operation_str = "-".join(list(operation))
            assembly_chain_ids.append(
                np.char.add(
                    chain_id, np.array(["-" + operation_str]).astype(chain_id.dtype)
                )
            )
        assembly_coord[i] = coord

    assembly = repeat(structure, assembly_coord)
    assembly.chain_id = np.concatenate(assembly_chain_ids)
    return assembly


def get_assembly_with_missing_residues(  # noqa: CCR001
    pdbx_file,
    data_block=None,
    assembly_id=None,
    model=None,
    altloc="first",
    extra_fields=None,
    include_bonds=False,
    fill_missing_residues=False,
):
    """Modified from biotite.structure.io.pdbx.get_assembly to fill in missing residues.

    We also return `label` fields rather than `auth` fields, but add `auth` fields as annotations.
    """
    block = _get_block(pdbx_file, data_block)

    try:
        assembly_gen_category = block["pdbx_struct_assembly_gen"]
    except KeyError:
        raise InvalidFileError("File has no 'pdbx_struct_assembly_gen' category")

    try:
        struct_oper_category = block["pdbx_struct_oper_list"]
    except KeyError:
        raise InvalidFileError("File has no 'pdbx_struct_oper_list' category")

    assembly_ids = assembly_gen_category["assembly_id"].as_array(str)
    if assembly_id is None:
        assembly_id = assembly_ids[0]
    elif assembly_id not in assembly_ids:
        raise KeyError(f"File has no Assembly ID '{assembly_id}'")

    ### Calculate all possible transformations
    transformations = _get_transformations(struct_oper_category)

    ### Get structure according to additional parameters
    # Include 'label_asym_id' as annotation array
    # for correct asym ID filtering
    extra_fields = [] if extra_fields is None else extra_fields
    if "label_asym_id" in extra_fields:
        extra_fields_and_asym = extra_fields
    else:
        # The operations apply on asym IDs
        # -> they need to be included to select the correct atoms
        extra_fields_and_asym = extra_fields + ["label_asym_id"]
    structure = get_pdbx_structure(
        pdbx_file,
        data_block=data_block,
        model=model,
        altloc=altloc,
        extra_fields=extra_fields_and_asym,
        include_bonds=include_bonds,
        fill_missing_residues=fill_missing_residues,
    )

    ### Get transformations and apply them to the affected asym IDs
    assembly = None
    for _id, op_expr, asym_id_expr in zip(
        assembly_gen_category["assembly_id"].as_array(str),
        assembly_gen_category["oper_expression"].as_array(str),
        assembly_gen_category["asym_id_list"].as_array(str),
    ):
        # Find the operation expressions for given assembly ID
        # We already asserted that the ID is actually present
        if _id == assembly_id:
            operations = _parse_operation_expression(op_expr)
            asym_ids = asym_id_expr.split(",")
            # Filter affected asym IDs
            sub_structure = structure[..., np.isin(structure.label_asym_id, asym_ids)]
            sub_assembly = _apply_transformations(
                sub_structure, transformations, operations
            )
            # Merge the chains with asym IDs for this operation
            # with chains from other operations
            if assembly is None:
                assembly = sub_assembly
            else:
                assembly += sub_assembly

    # Remove 'label_asym_id', if it was not included in the original
    # user-supplied 'extra_fields'
    if "label_asym_id" not in extra_fields:
        assembly.del_annotation("label_asym_id")

    return assembly


def load_assembly(
    fpath_or_handler,
    file_type="cif",
    model=1,
    assembly_id=None,
    extra_fields=None,
    fill_missing_residues=False,
    include_bonds=False,
):
    """Load biological assembly from cif/bcif file.

    TODO: add support for pdb files.
    """
    if isinstance(fpath_or_handler, (str, PathLike)) and fpath_or_handler.endswith(
        ".gz"
    ):
        file_type = (
            file_type or os.path.splitext(os.path.splitext(fpath_or_handler)[0])[1][1:]
        )
        # https://github.com/biotite-dev/biotite/issues/193
        with gzip.open(fpath_or_handler, "rt") as f:
            return load_assembly(
                f,
                file_type=file_type,
                model=model,
                extra_fields=extra_fields,
                fill_missing_residues=fill_missing_residues,
                include_bonds=include_bonds,
            )

    if file_type in ["cif", "bcif"]:
        if file_type == "cif":
            cf = pdbx.CIFFile.read(fpath_or_handler)
        else:
            cf = pdbx.BinaryCIFFile.read(fpath_or_handler)
        return get_assembly_with_missing_residues(
            cf,
            data_block=None,
            assembly_id=assembly_id,
            model=model,
            altloc="first",
            extra_fields=extra_fields,
            include_bonds=include_bonds,
            fill_missing_residues=fill_missing_residues,
        )
    else:
        raise ValueError(f"Unsupported file format: {file_type}")
