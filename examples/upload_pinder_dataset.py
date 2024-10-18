"""Script demonstrating how to build the Pinder dataset.

What we really want to upload for each protein is a single ProteinComplex feature,
and two sets of Protein features for each ligand and each receptor
(representing the unbound and - where available - predicted unbound states).
The chains should all be pre-aligned to either the native complex or the predicted unbound entry using
the pinder utils so that they are all the same size, with missing coordinates masked out.

If we choose to pre-align to the native complex, we need to also return full sequences
and mappings to native complex from full sequences.

To store the structures - we could even drop the missing coordinates and rey on the numbering
being the same.

Not sure if we also want to superimpose? Might be a bit dangerous to superimpose
the unbound complex jointly onto the bound complex: this leaks information.

We should also upload a raw version where we don't do any alignment or masking.
"""
import argparse
from typing import List

import biotite.sequence.align as align
import numpy as np
from biotite import structure as bs
from datasets import Dataset, Features, Value
from pinder.core import PinderSystem, get_index, get_metadata
from pinder.core.loader.structure import Structure
from pinder.core.structure.atoms import (
    _align_and_map_sequences,
    _get_seq_aligned_structures,
    _get_structure_and_res_info,
    apply_mask,
    get_seq_alignments,
    mask_from_res_list,
)

from bio_datasets import ProteinStructureFeature


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

    # values are positions in ref that subject aligns to
    subj_mask_in_ref = mask_from_res_list(ref_structure, list(subj_resid_map.values()))
    subj_mask = mask_from_res_list(subj_structure, list(subj_resid_map.keys()))
    return subj_mask_in_ref, subj_mask


class PinderDataset:
    def __init__(self, index, metadata):
        self.index = index
        self.metadata = metadata

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
        target_at = target_struct.atom_array.copy()

        if mode == "ref":
            # We drop any ref residues that aren't present in the target.
            subj_mask_in_ref, subj_mask = get_subject_positions_in_ref_masks(
                ref_at, target_at
            )

            # the below also automatically handles renumbering.
            aligned_target_at = ref_at.copy()
            # We'll assume that the sequence is the same at positions that don't align, so only coords need to be masked
            aligned_target_at[~subj_mask_in_ref].coord = np.nan
            if "b_factor" in ref_at._annot:
                aligned_target_at[~subj_mask_in_ref].b_factor = np.nan
            aligned_target_at[subj_mask_in_ref].coord = target_at[subj_mask].coord
            target_at = aligned_target_at

        elif mode == "intersection":
            # handles masking and renumbering
            ref_mask, target_mask = _get_seq_aligned_structures(ref_at, target_at)
            # handle differing atoms later to be consistent across all cases
            # equivalent to align_common_sequence with remove_differing_atoms True and remove_differing_annotations True
            # ref_at_mod = ref_at.copy()
            # target_at_mod = target_at.copy()
            # # remove differing atoms:
            # ref_target_mask = bs.filter_intersection(ref_at_mod, target_at_mod)
            # target_ref_mask = bs.filter_intersection(target_at_mod, ref_at_mod)
            # # remove differing annotations:
            # ref_at_mod, target_at_mod = bs.surgery.fix_annotation_mismatch(
            #     ref_at_mod, target_at_mod, ["element", "ins_code", "b_factor"]
            # )
            # ref_at = ref_at_mod[ref_target_mask].copy()
            # target_at = target_at_mod[target_ref_mask].copy()

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

    def make_structures(
        self, system: PinderSystem, remove_differing_atoms: bool = True
    ):
        """We have to choose which reference to align to: choices are complex, apo (unbound) or predicted (unbound).

        Bound makes most sense I think.
        """
        has_apo = system.entry.apo_R and system.entry.apo_L
        has_pred = system.entry.predicted_R and system.entry.predicted_L

        # apo_complex = system.create_apo_complex() - his superimposes structures, which gives info away about interaction
        # https://github.com/pinder-org/pinder/blob/8ad1ead7a174736635c13fa7266d9ca54cf9f44e/examples/pinder-system.ipynb
        if has_apo:
            apo_R, apo_L = system.apo_receptor, system.apo_ligand
        else:
            apo_R, apo_L = None, None
        if has_pred:
            pred_R, pred_L = system.pred_receptor, system.pred_ligand
        else:
            pred_R, pred_L = None, None

        if apo_R is not None:
            _, apo_R = self.get_aligned_structures(
                system.native_R,
                apo_R,
                mode="ref",
            )
            _, apo_L = self.get_aligned_structures(
                system.native_L,
                apo_L,
                mode="ref",
            )

        if pred_R is not None:
            _, pred_R = self.get_aligned_structures(
                system.native_R,
                pred_R,
                mode="ref",
            )
            _, pred_L = self.get_aligned_structures(
                system.native_L,
                pred_L,
                mode="ref",
            )

        native = system.native_R + system.native_L
        # TODO: add uniprot seq and mapping to native
        return {
            "complex": native,
            "apo_receptor": apo_R,
            "apo_ligand": apo_L,
            "pred_receptor": pred_R,
            "pred_ligand": pred_L,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--max_length_monomer", type=int, default=None)
    parser.add_argument("--keep_representatives_only", action="store_true")
    parser.add_argument("--system_ids", type=List[str], default=None)
    args = parser.parse_args()

    index = get_index()
    metadata = get_metadata()

    if args.system_ids is not None:
        index = index[index["id"].isin(args.system_ids)]
    else:
        if args.subset is not None:
            index = index[
                (index["subset"] == args.subset) & (index["split"] == args.split)
            ]
        else:
            index = index[index["split"] == args.split]
        index = index[
            (index["length1"] <= args.max_length_monomer)
            & (index["length2"] <= args.max_length_monomer)
        ]
        if args.keep_representatives_only:
            index = index.groupby("cluster_id").head(1)

    cluster_ids = list(index.cluster_id.unique())
    print(f"Pinder dataset: {len(cluster_ids)} clusters; {len(index)} systems")

    # TODO: decide on appropriate metadata (probably most of stuff from metadata...)
    features = Features(
        {
            "id": Value("string"),
            "cluster_id": Value("string"),
            "complex": ProteinStructureFeature(),
            "unbound_receptor": ProteinStructureFeature(),
            "unbound_ligand": ProteinStructureFeature(),
            "unbound_type": Value("string"),
        }
    )
    dataset = Dataset.from_generator(
        iter(PinderDataset(index, metadata)), features=features
    )
