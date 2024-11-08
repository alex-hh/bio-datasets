"""Python script for converting cif to bcif.

molstar has a js script: https://molstar.org/docs/data-access-tools/convert-to-bcif/
but python probably nicer.
"""
import argparse
import gzip
import json
import os

from biotite.structure.io.pdbx import BinaryCIFFile, CIFFile, compress
from biotite.structure.io.pdbx.convert import _get_block, _get_or_create_block

# inferred from 1aq1 binary cif file provided by pdb.
with open(os.path.join(os.path.dirname(__file__), "bcif_dtypes.json"), "r") as f:
    BCIF_FILE_DTYPES = json.load(f)


# just anything required for reconstructing the assembly.
LITE_COLUMNS_TO_KEEP = [
    # asym coords
    "atom_site",
    # required for fill missing residues
    "entity",
    "entity_poly",
    "entity_poly_seq",
    # required for assembly reconstruction
    "cell",
    "struct_asym",
    "pdbx_struct_assembly",  # needed?
    "pdbx_struct_assembly_gen",
    "pdbx_struct_oper_list",
    "symmetry",
]


def single_cif_to_bcif(
    input_file: str,
    output_file: str,
    lite: bool = False,
):
    if input_file.endswith(".gz"):
        with gzip.open(input_file, "rt") as f:
            inf = CIFFile.read(f)
    else:
        inf = CIFFile.read(input_file)
    outf = BinaryCIFFile()
    out_block = _get_or_create_block(outf, "1aq1")
    Category = out_block.subcomponent_class()
    in_block = _get_block(inf, None)
    for key, in_category in in_block.items():
        if lite and key not in LITE_COLUMNS_TO_KEEP:
            continue
        out_category = Category()
        for in_column, in_data in in_category.items():
            try:
                # exptl_crystal is causing type coercion issues. TODO: fix
                arr = in_data.as_array(BCIF_FILE_DTYPES[key][in_column])
            except Exception:
                arr = in_data.as_array()

            out_category[in_column] = arr
        out_block[key] = out_category
    outf = compress(outf)
    outf.write(output_file)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--lite", action="store_true")
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    single_cif_to_bcif(
        args.input_path,
        args.output_path,
        lite=args.lite,
    )


def dir_main():
    parser = create_parser()
    args = parser.parse_args()
    for file in os.listdir(args.input_path):
        output_file = os.path.join(args.output_path, file + ".bcif")
        single_cif_to_bcif(
            os.path.join(args.input_path, file),
            output_file,
            lite=args.lite,
        )


if __name__ == "__main__":
    main()
