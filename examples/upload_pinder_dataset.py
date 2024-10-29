"""Script demonstrating how to build the Pinder dataset.

Each bound complex in the Pinder dataset is represented by a single ProteinAtomArray feature.
Where available, predicted and experimentally determined ('apo') unbound states for both receptor
and ligand are also represented by ProteinAtomArray features.
We only upload the 'canonical' apo conformation for each protein.

The sequences of the unbound states are not aligned to the bound state, so that each unbound state
covers an identical set of residues to the bound state. Any missing coordinates in the unbound states
are set to NaN.
"""
import argparse
import contextlib
import io
import logging
import math
import os
import pathlib
import tempfile
from typing import List, Optional, Tuple

import biotite.sequence.align as align
import numpy as np
import pandas as pd
import tqdm
from biotite import structure as bs
from datasets import Dataset, Features, NamedSplit, Sequence, Value
from google.cloud.storage import Blob
from pinder.core.utils import cloud as pinder_cloud_utils
from bio_datasets.structure.protein import ProteinMixin, constants as protein_constants


UPLOAD = "upload_from_filename"
DOWNLOAD = "download_to_filename"


def process_many(
    path_pairs: List[Tuple[str, Blob]],
    method: str,
    global_timeout: int = 600,
    timeout: Tuple[int, int] = (10, 320),
    retries: int = 5,
    threads: int = 4,
) -> None:
    f"""Serial version of process_many.

    Parameters
    ----------
    path_pairs : list
        tuples of (path_str, Blob)
    method : str
        "{UPLOAD}" | "{DOWNLOAD}"
    global_timeout : int=600
        timeout to wait for all operations to complete
    timeout : tuple, default=(5, 120)
        timeout forwarded to Blob method
    retries : int, default=5
        how many times to attempt the method
    threads : int, default=4
        how many threads to use in the thread pool
    """
    if not len(path_pairs):
        return
    if method not in pinder_cloud_utils.BLOB_ACTIONS:
        raise Exception(f"{method=} not in {pinder_cloud_utils.BLOB_ACTIONS=}")
    for path, blob in path_pairs:
        pinder_cloud_utils.retry_blob_method(
            blob,
            method,
            path,
            retries=retries,
            timeout=timeout,
        )


pinder_cloud_utils.process_many = process_many
from pinder.core import PinderSystem, get_index, get_metadata
from pinder.core.index.utils import IndexEntry
from pinder.core.loader.structure import Structure
from pinder.core.structure.atoms import (
    _align_and_map_sequences,
    _get_seq_aligned_structures,
    _get_structure_and_res_info,
    apply_mask,
    get_seq_alignments,
    mask_from_res_list,
)

from bio_datasets import ProteinAtomArrayFeature


def mask_structure(structure: Structure, mask: np.ndarray) -> Structure:
    """Apply mask to structure."""
    return Structure(
        filepath=structure.filepath,
        uniprot_map=structure.uniprot_map,
        pinder_id=structure.pinder_id,
        atom_array=apply_mask(structure.atom_array.copy(), mask),
    )


def align_sequence_to_ref(
    ref_seq: str,
    subject_seq: str,
    ref_numbering: list[int] | None = None,
    subject_numbering: list[int] | None = None,
) -> tuple[str, str, list[int], list[int]]:
    """Modified from pinder.core.structure.atoms.align_sequences, to retain all ref residues,
    so that only subject residues are dropped.

    Pinder might have chosen not to do this to avoid having to represent insertions.

    Aligns two sequences and returns a tuple containing the aligned sequences
    and their respective numbering.

    Parameters
    ----------
    ref_seq : (str)
        The reference sequence to align to.
    subject_seq : (str)
        The subject sequence to be aligned.
    ref_numbering : (list[int], optional)
        List of residue numbers for the reference sequence.
    subject_numbering : (list[int], optional)
        List of residue numbers for the subject sequence.

    Returns:
        tuple: A tuple containing:
            - Aligned reference sequence (str)
            - Aligned subject sequence (str)
            - Numbering of the aligned reference sequence (list[int])
            - Numbering of the aligned subject sequence (list[int])

    Raises
    ------
        ValueError if the sequences cannot be aligned.
    """
    alignments = get_seq_alignments(ref_seq, subject_seq)
    aln = alignments[0]
    s = align.alignment.get_symbols(aln)

    if ref_numbering is None:
        ref_numbering = list(range(1, len(ref_seq) + 1))

    if subject_numbering is None:
        subject_numbering = list(range(1, len(subject_seq) + 1))

    # assigning reference numbering to all residues in a given sequence
    # (e.g. PDB chain)
    # p = subject protein sequence, u = ref sequence
    ref_numbering_mapped = []
    subject_numbering_mapped = []
    subject_sequence_mapped = ""
    ref_sequence_mapped = ""
    ui = -1
    pj = -1
    for p, u in zip(s[0], s[1]):
        if u:
            ui += 1
        if p:
            pj += 1
        if u:
            # ref_numbering_mapped.append(ui + 1)  # 1-based
            ref_numbering_mapped.append(ref_numbering[ui])
            subject_numbering_mapped.append(subject_numbering[pj])
            ref_sequence_mapped += u
            subject_sequence_mapped += p
    return (
        ref_sequence_mapped,
        subject_sequence_mapped,
        ref_numbering_mapped,
        subject_numbering_mapped,
    )


def get_subject_positions_in_ref_masks(
    ref_at,
    target_at,
    pdb_engine: str = "fastpdb",
):
    """Whereas _get_seq_aligned_structures computes masks for the parts
    of the two sequences that are mutually alignable, this returns the positions
    of the subject sequence in the reference sequence, allowing us to map subject
    coords onto the reference by doing ref_coords[subject_mask_in_ref], subj_coords[subj_mask].
    """
    ref_info = _get_structure_and_res_info(ref_at, pdb_engine)
    subj_info = _get_structure_and_res_info(target_at, pdb_engine)

    # Aligns two sequences and maps the numbering
    # align_sequences aligns a subject to a reference
    # then returns the alignable parts of the sequences together with their numbering
    subj_resid_map, alns = _align_and_map_sequences(ref_info, subj_info)
    # Need to remove ref residues in order to match subject
    ref_structure, _, _ = ref_info
    subj_structure, _, _ = subj_info

    assert isinstance(ref_structure, (bs.AtomArray, bs.AtomArrayStack))
    assert isinstance(subj_structure, (bs.AtomArray, bs.AtomArrayStack))

    ref_ids, ref_residues = bs.get_residues(ref_structure)
    subj_ids, subj_residues = bs.get_residues(subj_structure)
    # identify residues with sequence match
    subj_resid_seq_match_map = {
        subj_id: ref_id
        for subj_id, ref_id in subj_resid_map.items()
        if subj_residues[list(subj_ids).index(subj_id)]
        == ref_residues[list(ref_ids).index(ref_id)]
    }

    subj_resid_mutation_map = {
        subj_id: ref_id
        for subj_id, ref_id in subj_resid_map.items()
        if subj_id not in subj_resid_seq_match_map
    }

    # uncomment to debug
    # for subj_id, ref_id in subj_resid_mutation_map.items():
    #     atoms = subj_structure[subj_structure.res_id == subj_id]
    #     ref_atoms = ref_structure[ref_structure.res_id == ref_id]
    #     assert len(atoms) == len(ref_atoms), f"{atoms} != {ref_atoms} {ref_atoms.ins_code}"

    # values are positions in ref that subject aligns to and has matching sequence
    subj_mask_in_ref = mask_from_res_list(
        ref_structure, list(subj_resid_seq_match_map.values())
    )
    subj_mask = mask_from_res_list(
        subj_structure, list(subj_resid_seq_match_map.keys())
    )
    # TODO: we could include backbone at aligned but non-identical positions
    return subj_mask_in_ref, subj_mask


class PinderDataset:

    """Class to handle aligning of apo sequences to complex and standardisation of structures.

    We use sequence alignment because then the atom types will be the same.
    """

    def __init__(
        self,
        index,
        metadata,
        cleanup: bool = False,
        dataset_path: Optional[str] = None,
    ):
        self.index = index
        self.metadata = metadata
        self.cleanup = cleanup
        self.dataset_path = (
            pathlib.Path(dataset_path) if dataset_path is not None else None
        )

    def __len__(self):
        return len(self.index)

    def get_aligned_structures(
        self,
        ref_struct,
        target_struct,
        mode: str = "ref",  # "ref" or "intersection"
    ):
        """TODO: test that we get same result as applying get_seq_aligned_structure

        Source: https://github.com/pinder-org/pinder/blob/8ad1ead7a174736635c13fa7266d9ca54cf9f44e/src/pinder-core/pinder/core/loader/structure.py#L146
        """
        # N.B. pinder utils have stuff for handling multi-chain cases, so we need to assert that these are single-chain structures.
        ref_chains = bs.get_chains(ref_struct.atom_array)
        target_chains = bs.get_chains(target_struct.atom_array)
        assert len(set(target_chains)) == 1
        assert len(set(ref_chains)) == 1

        ref_at = ref_struct.atom_array.copy()
        ref_at, _ = ProteinMixin.standardise_atoms(ref_at, drop_oxt=True)
        target_at = target_struct.atom_array.copy()
        target_at, _ = ProteinMixin.standardise_atoms(target_at, drop_oxt=True)

        if mode == "ref":
            # We drop any target residues that aren't present in the reference.
            subj_mask_in_ref, subj_mask = get_subject_positions_in_ref_masks(
                ref_at, target_at
            )
            # TODO: add assert that mapped positions agree

            # the below also automatically handles renumbering.
            aligned_target_at = ref_at.copy()
            # We'll assume that the sequence is the same at positions that don't align, so only coords need to be masked
            aligned_target_at.coord[~subj_mask_in_ref] = np.nan
            if "b_factor" in ref_at._annot:
                aligned_target_at.b_factor[~subj_mask_in_ref] = np.nan
            aligned_target_at.coord[subj_mask_in_ref] = target_at[subj_mask].coord
            target_at = aligned_target_at

        elif mode == "intersection":
            # handles masking and renumbering
            ref_at, target_at = _get_seq_aligned_structures(ref_at, target_at)
            raise NotImplementedError("TODO: standardise atoms")

        else:
            raise ValueError(f"Invalid mode: {mode}")

        # make copies so we don't modify the original
        ref_struct = Structure(
            filepath=ref_struct.filepath,
            uniprot_map=ref_struct.uniprot_map,
            pinder_id=ref_struct.pinder_id,
            atom_array=ref_at,
            pdb_engine=ref_struct.pdb_engine,
        )
        target_struct = Structure(
            filepath=target_struct.filepath,
            uniprot_map=target_struct.uniprot_map,
            pinder_id=target_struct.pinder_id,
            atom_array=target_at,
            pdb_engine=target_struct.pdb_engine,
        )

        return ref_struct, target_struct

    def _filter_non_standard(self, struct):
        struct = struct.filter("hetero", [False])
        at = struct.atom_array
        at = at[np.isin(at.res_name, protein_constants.resnames)]
        struct.atom_array = at
        return struct

    def make_structures(self, system: PinderSystem):
        """We have to choose which reference to align to: choices are complex, apo (unbound) or predicted (unbound).

        Bound makes most sense I think.
        """
        # TODO: we could save apo even if only one chain is present
        has_apo = system.entry.apo_R and system.entry.apo_L
        apo_count = int(system.entry.apo_R) + int(system.entry.apo_L)
        has_pred = system.entry.predicted_R and system.entry.predicted_L
        pred_count = int(system.entry.predicted_R) + int(system.entry.predicted_L)

        native_R = self._filter_non_standard(system.native_R)
        native_L = self._filter_non_standard(system.native_L)
        if len(native_R.atom_array) == 0 or len(native_L.atom_array) == 0:
            print(
                f"Skipping {system.entry.id} because it has no atoms after excluding "
                f"hetero atoms and non-standard aas, R {len(native_R.atom_array)} L {len(native_L.atom_array)}"
            )
            return None

        native_R_at, _ = ProteinMixin.standardise_atoms(native_R.atom_array, drop_oxt=True)
        native_L_at, _ = ProteinMixin.standardise_atoms(native_L.atom_array, drop_oxt=True)
        native_R.atom_array = native_R_at
        native_L.atom_array = native_L_at

        holo_receptor = self._filter_non_standard(system.holo_receptor)
        holo_ligand = self._filter_non_standard(system.holo_ligand)

        # apo_complex = system.create_apo_complex() - this superimposes structures, which gives info away about interaction
        # https://github.com/pinder-org/pinder/blob/8ad1ead7a174736635c13fa7266d9ca54cf9f44e/examples/pinder-system.ipynb
        if has_apo:
            apo_R, apo_L = system.apo_receptor, system.apo_ligand
            apo_R = self._filter_non_standard(apo_R)
            apo_L = self._filter_non_standard(apo_L)
        else:
            apo_R, apo_L = None, None
        if has_pred:
            pred_R, pred_L = system.pred_receptor, system.pred_ligand
            pred_R = self._filter_non_standard(pred_R)
            pred_L = self._filter_non_standard(pred_L)

        else:
            pred_R, pred_L = None, None

        if has_apo:
            # only change to native R is standardise atoms - already called above so should do nothing
            native_R_v1, apo_R = self.get_aligned_structures(
                native_R,
                apo_R,
                mode="ref",
            )
            native_L_v1, apo_L = self.get_aligned_structures(
                native_L,
                apo_L,
                mode="ref",
            )
            assert len(native_R_v1.atom_array) == len(
                native_R.atom_array
            ), f"{len(native_R_v1.atom_array)} != {len(native_R.atom_array)}"
            assert len(native_L_v1.atom_array) == len(
                native_L.atom_array
            ), f"{len(native_L_v1.atom_array)} != {len(native_L.atom_array)}"

        if has_pred:
            _, pred_R = self.get_aligned_structures(
                native_R,
                pred_R,
                mode="ref",
            )
            _, pred_L = self.get_aligned_structures(
                native_L,
                pred_L,
                mode="ref",
            )

        holo_receptor_at = holo_receptor.atom_array.copy()
        holo_ligand_at = holo_ligand.atom_array.copy()
        holo_receptor_at, _ = ProteinMixin.standardise_atoms(holo_receptor_at, drop_oxt=True)
        holo_ligand_at, _ = ProteinMixin.standardise_atoms(holo_ligand_at, drop_oxt=True)
        holo_receptor = Structure(
            filepath=holo_receptor.filepath,
            uniprot_map=holo_receptor.uniprot_map,
            pinder_id=holo_receptor.pinder_id,
            atom_array=holo_receptor_at,
            pdb_engine=holo_receptor.pdb_engine,
        )
        holo_ligand = Structure(
            filepath=holo_ligand.filepath,
            uniprot_map=holo_ligand.uniprot_map,
            pinder_id=holo_ligand.pinder_id,
            atom_array=holo_ligand_at,
            pdb_engine=holo_ligand.pdb_engine,
        )
        if self.cleanup:
            os.remove(native_L.filepath)
            os.remove(holo_receptor.filepath)
            os.remove(holo_ligand.filepath)
            if apo_count > 0:
                if apo_R is not None:
                    os.remove(apo_R.filepath)
                if apo_L is not None and apo_L.filepath != apo_R.filepath:
                    os.remove(apo_L.filepath)
            if pred_count > 0:
                if pred_R is not None:
                    os.remove(pred_R.filepath)
                if pred_L is not None and pred_L.filepath != pred_R.filepath:
                    os.remove(pred_L.filepath)
        native = native_R + native_L
        # TODO: add uniprot seq and mapping to native
        assert len(holo_receptor_at) == len(
            native_R.atom_array
        ), f"{len(holo_receptor_at)} != {len(native_R.atom_array)}"
        assert len(holo_ligand_at) == len(
            native_L.atom_array
        ), f"{len(holo_ligand_at)} != {len(native_L.atom_array)}"
        return {
            "complex": native,
            "apo_receptor": apo_R,
            "apo_ligand": apo_L,
            "pred_receptor": pred_R,
            "pred_ligand": pred_L,
            "holo_receptor": holo_receptor,
            "holo_ligand": holo_ligand,
        }

    def _get_item_from_id(self, id):
        row = self.index[self.index["id"] == id].iloc[0]
        metadata = self.metadata[self.metadata["id"] == id].iloc[0]
        # n.b. PinderSystem will automatically download if entry can't be found locally
        # TODO: if necessary, renumber reference ids to always be contiguous (before alignment)
        # TODO: check whether paths exist and sleep if not (prevent parsing error due to truncated file download...)
        system = PinderSystem(
            entry=IndexEntry(**row.to_dict()), dataset_path=self.dataset_path
        )
        if system.entry.predicted_R:
            uniprot_seq_R = system.pred_receptor.sequence
        else:
            uniprot_seq_R = None
        if system.entry.predicted_L:
            uniprot_seq_L = system.pred_ligand.sequence
        else:
            uniprot_seq_L = None
        structures = self.make_structures(system)
        if structures is None:
            return None
        holo_receptor = structures.pop("holo_receptor")
        holo_ligand = structures.pop("holo_ligand")

        structures["receptor_resids_with_uniprot_mapping"] = list(
            sorted(list(holo_receptor.resolved_pdb2uniprot.keys()))
        )
        structures["receptor_mapped_uniprot_resids"] = [
            holo_receptor.resolved_pdb2uniprot[res_id]
            for res_id in structures["receptor_resids_with_uniprot_mapping"]
        ]
        structures["ligand_resids_with_uniprot_mapping"] = list(
            sorted(list(holo_ligand.resolved_pdb2uniprot.keys()))
        )
        structures["ligand_mapped_uniprot_resids"] = np.array(
            [
                holo_ligand.resolved_pdb2uniprot[res_id]
                for res_id in structures["ligand_resids_with_uniprot_mapping"]
            ],
            dtype="uint16",
        )
        structures["receptor_uniprot_accession"] = system.entry.uniprot_R
        structures["ligand_uniprot_accession"] = system.entry.uniprot_L
        # TODO: get metadata
        structures["receptor_uniprot_seq"] = uniprot_seq_R
        structures["ligand_uniprot_seq"] = uniprot_seq_L
        for key in [
            "cluster_id",
            "cluster_id_R",
            "cluster_id_L",
        ]:
            structures[key] = row[key]
        for metadata_key in [
            "method",
            "resolution",
            "probability",  # probability that the protein complex is a true biological complex
            "oligomeric_count",
            "ECOD_names_R",
            "ECOD_names_L",
        ]:
            structures[metadata_key] = metadata[metadata_key]
        structures["id"] = row["id"]
        structures["pdb_id"] = row["pdb_id"]
        return structures

    def __getitem__(self, idx):
        id = self.index.iloc[idx]["id"]
        return self._get_item_from_id(id)


@contextlib.contextmanager
def suppress_output():
    # Suppress stdout and stderr
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        # Disable logging temporarily
        logging.disable(logging.CRITICAL)
        try:
            yield
        finally:
            # Re-enable logging
            logging.disable(logging.NOTSET)


def examples_generator(
    index: List[pd.DataFrame],
    metadata: pd.DataFrame,
    dataset_path: str,
    max_examples: Optional[int] = None,
    cleanup: bool = False,
):
    for index_df in index:
        ds = PinderDataset(
            index_df, metadata, dataset_path=dataset_path, cleanup=cleanup
        )
        print(f"Dataset length: {len(ds)}")
        for i in tqdm.tqdm(range(len(ds)), disable=len(index) > 1):
            if max_examples is not None and i >= max_examples:
                break
            try:
                with suppress_output():
                    ex = ds[i]
            except Exception as e:
                print(f"Error getting example {index_df.iloc[i]['id']}", e)
                # raise e
                continue
            if ex is None:
                print(f"Skipping None example {index_df.iloc[i]['id']}")
                continue
            ex["complex"] = ex["complex"].atom_array
            ex["apo_receptor"] = (
                ex["apo_receptor"].atom_array
                if ex["apo_receptor"] is not None
                else None
            )
            ex["apo_ligand"] = (
                ex["apo_ligand"].atom_array if ex["apo_ligand"] is not None else None
            )
            ex["pred_receptor"] = (
                ex["pred_receptor"].atom_array
                if ex["pred_receptor"] is not None
                else None
            )
            ex["pred_ligand"] = (
                ex["pred_ligand"].atom_array if ex["pred_ligand"] is not None else None
            )
            yield ex


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--keep_representatives_only", action="store_true")
    parser.add_argument("--system_ids", type=List[str], default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--num_proc", type=int, default=None)
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()

    index = get_index()
    metadata = get_metadata()

    if args.system_ids is not None:
        index = index[index["id"].isin(args.system_ids)]
    else:
        if args.subset is not None:
            if args.subset == "cluster_representatives":
                assert args.split == "train"
                index = (
                    index[index["split"] == args.split].groupby("cluster_id").head(1)
                )
                split = "cluster_representatives_train"
            else:
                assert args.split == "test"
                index = index[
                    (index[f"{args.subset}"]) & (index["split"] == args.split)
                ]
                split = args.subset
        else:
            index = index[index["split"] == args.split]
            split = args.split
        if args.keep_representatives_only:
            index = index.groupby("cluster_id").head(1)

    cluster_ids = list(index.cluster_id.unique())
    print(f"Pinder dataset: {len(cluster_ids)} clusters; {len(index)} systems")

    # TODO: decide on appropriate metadata (probably most of stuff from metadata...)
    features = Features(
        **{
            "id": Value("string"),
            "cluster_id": Value("string"),
            "pdb_id": Value("string"),
            "complex": ProteinAtomArrayFeature(with_res_id=True),
            "apo_receptor": ProteinAtomArrayFeature(with_res_id=True),
            "apo_ligand": ProteinAtomArrayFeature(with_res_id=True),
            "pred_receptor": ProteinAtomArrayFeature(with_res_id=True),
            "pred_ligand": ProteinAtomArrayFeature(with_res_id=True),
            "receptor_uniprot_accession": Value("string"),
            "ligand_uniprot_accession": Value("string"),
            "receptor_uniprot_seq": Value("string"),
            "ligand_uniprot_seq": Value("string"),
            # TODO: switch to array1d when following issue fixed:
            # https://github.com/huggingface/datasets/issues/7243
            # the two sequences basically define the keys and values of a dictionary mapping resids to uniprot ids
            "receptor_resids_with_uniprot_mapping": Sequence(Value("uint16")),
            "receptor_mapped_uniprot_resids": Sequence(Value("uint16")),
            "ligand_resids_with_uniprot_mapping": Sequence(Value("uint16")),
            "ligand_mapped_uniprot_resids": Sequence(Value("uint16")),
            # selected metadata: anything else can be obtained from the pinder metadata file.
            "oligomeric_count": Value("uint16"),
            "resolution": Value("float16"),
            "probability": Value("float16"),
            "method": Value("string"),
        }
    )
    # TODO: does this memmap? do I need to use GeneratorBasedBuilder explicitly?
    # to do full set, i can just do this in a loop with memory control.

    # TODO: implement sharding and num_proc
    print(f"Index length: {len(index)} metadata length: {len(metadata)}")
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_proc is None:
            index_list = [index]
        else:
            shard_size = math.ceil(len(index) / args.num_proc)
            index_list = [
                index.iloc[i : i + shard_size] for i in range(0, len(index), shard_size)
            ]
        print(f"Index list length: {len(index_list)} {[len(df) for df in index_list]}")
        dataset = Dataset.from_generator(
            examples_generator,
            features=features,
            gen_kwargs={
                "index": index_list,
                "metadata": metadata,
                "dataset_path": args.dataset_path,
                "max_examples": args.max_examples,
                "cleanup": args.cleanup,
            },
            split=NamedSplit(split),
            cache_dir=temp_dir,
            num_proc=args.num_proc,
        )
        dataset.push_to_hub(
            "graph-transformers/pinder",
            split=split,
        )
