"""
Upload the CATH dataset to the hub.

c.f.https://github.com/huggingface/datasets/tree/main/templates
"""
import argparse
import json
import tempfile
from typing import Dict, List, Optional

import tqdm

from bio_datasets import Dataset, Features, NamedSplit, Value
from bio_datasets.features import ProteinAtomArrayFeature
from bio_datasets.structure.protein import ProteinDictionary


def load_coords(
    jsonl_file: str,
    disable_tqdm: bool = False,
    split_ids: Optional[Dict[str, List[str]]] = None,
):
    """Split-specific jsonl files should be created by running data_creation_scripts/create_cath_splits.py"""
    entries_by_split = {split: [] for split in split_ids.keys()}
    name_to_split = {
        name: split for split, names in split_ids.items() for name in names
    }
    with open(jsonl_file) as f:
        lines = (
            f.readlines()
        )  # get a list rather than iterator to allow tqdm to know progress
        for line in tqdm.tqdm(lines, disable=disable_tqdm):
            coords_dict = json.loads(line)
            if split_ids:
                split = name_to_split.get(coords_dict["name"], None)
                if split is None:
                    # print(f"Skipping {coords_dict['name']} because it's not in the split_ids")
                    continue
            else:
                split = "train"
            entries_by_split[split].append(coords_dict)
    return entries_by_split


def examples_generator(coords: List[Dict], features: Features):
    for coords_dict in coords:
        example = coords_dict
        if "N" in coords_dict:
            coords = {
                "N": example.pop("N"),
                "CA": example.pop("CA"),
                "C": example.pop("C"),
                "O": example.pop("O"),
            }
            coords_dict["coords"] = coords
        chain_id = coords_dict["name"].split(".")[1]
        # a dictionary with this format is one of the formats accepted by AtomArray.encode_example
        example["backbone"] = {
            "sequence": coords_dict["seq"],
            "backbone_coords": coords_dict["coords"],
            "chain_id": chain_id * len(coords_dict["seq"]),
        }
        example["num_chains"] = coords_dict["num_chains"]
        example["name"] = coords_dict["name"]
        example["CATH"] = coords_dict["CATH"]
        yield example


def main(
    repo_id: str,
    cath_jsonl_path: str,
    cath_splits_path: str,
    config_name: str = "default",
):
    with open(cath_splits_path) as f:
        splits = json.load(f)

    splits = {k: v for k, v in splits.items() if k in ["train", "validation", "test"]}
    coords_by_split = load_coords(cath_jsonl_path, split_ids=splits)
    for split_name, split_coords in coords_by_split.items():
        print(f"Processing {split_name} split", len(split_coords), "examples")
        assert len(split_coords) > 0, f"No examples found for split {split_name}"
        # from_generator calls GeneratorBasedBuilder.download_and_prepare and as_dataset
        features = Features(
            backbone=ProteinAtomArrayFeature(
                residue_dictionary=ProteinDictionary.from_preset("protein"),
                backbone_only=True,
            ),
            num_chains=Value("int32"),
            name=Value("string"),
            CATH=Value("string"),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            ds = Dataset.from_generator(
                examples_generator,
                gen_kwargs={"coords": split_coords, "features": features},
                features=features,
                split=NamedSplit(split_name),
                cache_dir=temp_dir,
            )
            ds.push_to_hub(repo_id, split=split_name, config_name=config_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id", type=str)
    parser.add_argument("--config_name", type=str, default="default")
    parser.add_argument("--cath_jsonl_path", type=str, required=True)
    parser.add_argument("--cath_splits_path", type=str, required=True)
    args = parser.parse_args()
    main(
        args.repo_id,
        args.cath_jsonl_path,
        args.cath_splits_path,
        config_name=args.config_name,
    )
