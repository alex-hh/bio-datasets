"""Use utils from nerfax parser to convert protein into internal coordinates."""
import nerfax
import numpy as np
from biotite import structure as bs

from .protein import filter_atom_names


def load_backbone_coord_array(structure: bs.AtomArray):
    xyz = np.stack(
        [structure[filter_atom_names(structure, at)].coord for at in ["N", "CA", "C"]],
        axis=1,
    )  # L, 3, 3 -> Lx3, 3
    return xyz


def get_backbone_internals(structure: bs.AtomArray):
    # https://github.com/PeptoneLtd/nerfax/blob/2dd1ea019197cd0e273a8d5b920cc850c6b03460/nerfax/mpnerf_constants.py#L590
    xyz = load_backbone_coord_array(structure)
    lengths, angles, torsions = nerfax.parser.xyz_to_internal_coords(
        nerfax.parser.insert_zero(xyz).reshape((-1, 3))
    )  # Dummy reference residue used - does that have to happen at this stage?
    angles = angles.at[0, 0].set(1.0)  # (Dummy non-zero angle required)
    return lengths, angles, torsions


def get_sidechain_internals(structure: bs.AtomArray):
    pass
