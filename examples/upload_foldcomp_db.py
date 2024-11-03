"""
Upload a foldcomp database to the hub.

# TODO: rewrite with foldcompdb?

storage requirements for backbone coordinates:
coords are 4 x 3 x 2 bytes (float16) = 24 bytes per residue
bfactor is 2 bytes per residue
residue starts are 2 bytes per residue
sequence is 1 byte per residue
atom name is ~ 4 bytes per residue

so we can save a load of bytes by not storing the atom names
or residue starts for AF - since they can be inferred from the sequence

so total storage we really need is

12/24 bytes for backbone coords (discretised/float16)
1? byte for side chain coords
1? byte for bfactors (discretised)
1 byte for sequence

i.e. 15/27 bytes per residue in total; foldcomp appears to store ~16!

For non-AF models, saving atom names is fine; we could infer residue starts
from atom names but we might as well just save them too.

c.f.https://github.com/huggingface/datasets/tree/main/templates
"""
import argparse
import itertools
import os
from typing import Optional

import foldcomp
from datasets import Dataset, NamedSplit, Value

from bio_datasets.features import (
    Features,
    ProteinAtomArrayFeature,
    ProteinStructureFeature,
)


def examples_generator(db_file, max_examples: Optional[int] = None):
    assert os.path.exists(db_file)
    with foldcomp.open(db_file, decompress=True) as db:
        for (name, pdb_str) in itertools.islice(db, max_examples):
            # if we opened with decompress False, we wouldn't get name
            pdb_bytes = foldcomp.compress(name, pdb_str)
            # pdb_bytes = bytes(pdb_str)
            example = {
                "name": name,
                "structure": {"bytes": pdb_bytes, "path": None, "type": "fcz"},
            }
            yield example


def main(
    repo_id: str,
    db_file: str,
    as_array: bool,
    config_name: Optional[str] = None,
    max_examples: Optional[int] = None,
    coords_dtype: str = "float32",
    backbone_only: bool = False,
):
    # from_generator calls GeneratorBasedBuilder.download_and_prepare and as_dataset
    features = Features(
        name=Value("string"),
        structure=ProteinAtomArrayFeature.from_preset(
            "afdb", drop_sidechains=backbone_only
        )
        if as_array
        else ProteinStructureFeature(with_b_factor=True),
    )
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        ds = Dataset.from_generator(
            examples_generator,
            gen_kwargs={"db_file": db_file, "max_examples": max_examples},
            features=features,
            cache_dir=temp_dir,
        )
        ds.push_to_hub(repo_id, config_name=config_name or "default")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id", type=str)
    parser.add_argument("--foldcomp_db_name", type=str)
    parser.add_argument("--foldcomp_db_path", type=str)
    parser.add_argument("--as_array", action="store_true")
    parser.add_argument("--config_name", type=str, default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--coords_dtype", type=str, default="float32")
    parser.add_argument("--backbone_only", action="store_true")
    args = parser.parse_args()
    if args.foldcomp_db_name is None and args.foldcomp_db_path is None:
        raise ValueError("Either foldcomp_db_name or foldcomp_db_path must be provided")
    if args.foldcomp_db_name is not None:
        try:
            os.chdir("data")
            foldcomp.setup(args.foldcomp_db_name)
        except KeyError as e:
            # https://github.com/steineggerlab/foldcomp/issues/60
            print("Ignoring foldcomp setup error: ", e)
        os.chdir("..")
        args.foldcomp_db_path = os.path.join("data", args.foldcomp_db_name)
    main(
        args.repo_id,
        args.foldcomp_db_path,
        args.as_array,
        config_name=args.config_name,
        max_examples=args.max_examples,
        coords_dtype=args.coords_dtype,
        backbone_only=args.backbone_only,
    )
