# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Constants used in AlphaFold."""

import collections
from typing import List, Mapping, Tuple

import numpy as np

# Distance from one CA to next CA [trans configuration: omega = 180].
ca_ca = 3.80209737096

# Format: The list for each AA type contains chi1, chi2, chi3, chi4 in
# this order (or a relevant subset from chi1 onwards). ALA and GLY don't have
# chi angles so their chi angle lists are empty.
chi_angles_atoms = {
    "ALA": [],
    # Chi5 in arginine is always 0 +- 5 degrees, so ignore it.
    "ARG": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "NE"],
        ["CG", "CD", "NE", "CZ"],
    ],
    "ASN": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "ASP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "CYS": [["N", "CA", "CB", "SG"]],
    "GLN": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ],
    "GLU": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ],
    "GLY": [],
    "HIS": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
    "ILE": [["N", "CA", "CB", "CG1"], ["CA", "CB", "CG1", "CD1"]],
    "LEU": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "LYS": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "CE"],
        ["CG", "CD", "CE", "NZ"],
    ],
    "MET": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "SD"],
        ["CB", "CG", "SD", "CE"],
    ],
    "PHE": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "PRO": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"]],
    "SER": [["N", "CA", "CB", "OG"]],
    "THR": [["N", "CA", "CB", "OG1"]],
    "TRP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "TYR": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "VAL": [["N", "CA", "CB", "CG1"]],
}

# If chi angles given in fixed-length array, this matrix determines how to mask
# them for each AA type. The order is as per restype_order (see below).
chi_angles_mask = [
    [0.0, 0.0, 0.0, 0.0],  # ALA
    [1.0, 1.0, 1.0, 1.0],  # ARG
    [1.0, 1.0, 0.0, 0.0],  # ASN
    [1.0, 1.0, 0.0, 0.0],  # ASP
    [1.0, 0.0, 0.0, 0.0],  # CYS
    [1.0, 1.0, 1.0, 0.0],  # GLN
    [1.0, 1.0, 1.0, 0.0],  # GLU
    [0.0, 0.0, 0.0, 0.0],  # GLY
    [1.0, 1.0, 0.0, 0.0],  # HIS
    [1.0, 1.0, 0.0, 0.0],  # ILE
    [1.0, 1.0, 0.0, 0.0],  # LEU
    [1.0, 1.0, 1.0, 1.0],  # LYS
    [1.0, 1.0, 1.0, 0.0],  # MET
    [1.0, 1.0, 0.0, 0.0],  # PHE
    [1.0, 1.0, 0.0, 0.0],  # PRO
    [1.0, 0.0, 0.0, 0.0],  # SER
    [1.0, 0.0, 0.0, 0.0],  # THR
    [1.0, 1.0, 0.0, 0.0],  # TRP
    [1.0, 1.0, 0.0, 0.0],  # TYR
    [1.0, 0.0, 0.0, 0.0],  # VAL
]

# The following chi angles are pi periodic: they can be rotated by a multiple
# of pi without affecting the structure.
chi_pi_periodic = [
    [0.0, 0.0, 0.0, 0.0],  # ALA
    [0.0, 0.0, 0.0, 0.0],  # ARG
    [0.0, 0.0, 0.0, 0.0],  # ASN
    [0.0, 1.0, 0.0, 0.0],  # ASP
    [0.0, 0.0, 0.0, 0.0],  # CYS
    [0.0, 0.0, 0.0, 0.0],  # GLN
    [0.0, 0.0, 1.0, 0.0],  # GLU
    [0.0, 0.0, 0.0, 0.0],  # GLY
    [0.0, 0.0, 0.0, 0.0],  # HIS
    [0.0, 0.0, 0.0, 0.0],  # ILE
    [0.0, 0.0, 0.0, 0.0],  # LEU
    [0.0, 0.0, 0.0, 0.0],  # LYS
    [0.0, 0.0, 0.0, 0.0],  # MET
    [0.0, 1.0, 0.0, 0.0],  # PHE
    [0.0, 0.0, 0.0, 0.0],  # PRO
    [0.0, 0.0, 0.0, 0.0],  # SER
    [0.0, 0.0, 0.0, 0.0],  # THR
    [0.0, 0.0, 0.0, 0.0],  # TRP
    [0.0, 1.0, 0.0, 0.0],  # TYR
    [0.0, 0.0, 0.0, 0.0],  # VAL
    [0.0, 0.0, 0.0, 0.0],  # UNK
]

# Atoms positions relative to the 8 rigid groups, defined by the pre-omega, phi,
# psi and chi angles:
# 0: 'backbone group',
# 1: 'pre-omega-group', (empty)
# 2: 'phi-group', (currently empty, because it defines only hydrogens)
# 3: 'psi-group',
# 4,5,6,7: 'chi1,2,3,4-group'
# The atom positions are relative to the axis-end-atom of the corresponding
# rotation axis. The x-axis is in direction of the rotation axis, and the y-axis
# is defined such that the dihedral-angle-definiting atom (the last entry in
# chi_angles_atoms above) is in the xy-plane (with a positive y-coordinate).
# format: [atomname, group_idx, rel_position]
rigid_group_atom_positions = {
    "ALA": [
        ["N", 0, (-0.525, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.529, -0.774, -1.205)],
        ["O", 3, (0.627, 1.062, 0.000)],
    ],
    "ARG": [
        ["N", 0, (-0.524, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.524, -0.778, -1.209)],
        ["O", 3, (0.626, 1.062, 0.000)],
        ["CG", 4, (0.616, 1.390, -0.000)],
        ["CD", 5, (0.564, 1.414, 0.000)],
        ["NE", 6, (0.539, 1.357, -0.000)],
        ["NH1", 7, (0.206, 2.301, 0.000)],
        ["NH2", 7, (2.078, 0.978, -0.000)],
        ["CZ", 7, (0.758, 1.093, -0.000)],
    ],
    "ASN": [
        ["N", 0, (-0.536, 1.357, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.531, -0.787, -1.200)],
        ["O", 3, (0.625, 1.062, 0.000)],
        ["CG", 4, (0.584, 1.399, 0.000)],
        ["ND2", 5, (0.593, -1.188, 0.001)],
        ["OD1", 5, (0.633, 1.059, 0.000)],
    ],
    "ASP": [
        ["N", 0, (-0.525, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, 0.000, -0.000)],
        ["CB", 0, (-0.526, -0.778, -1.208)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.593, 1.398, -0.000)],
        ["OD1", 5, (0.610, 1.091, 0.000)],
        ["OD2", 5, (0.592, -1.101, -0.003)],
    ],
    "CYS": [
        ["N", 0, (-0.522, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, 0.000, 0.000)],
        ["CB", 0, (-0.519, -0.773, -1.212)],
        ["O", 3, (0.625, 1.062, -0.000)],
        ["SG", 4, (0.728, 1.653, 0.000)],
    ],
    "GLN": [
        ["N", 0, (-0.526, 1.361, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, 0.000)],
        ["CB", 0, (-0.525, -0.779, -1.207)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.615, 1.393, 0.000)],
        ["CD", 5, (0.587, 1.399, -0.000)],
        ["NE2", 6, (0.593, -1.189, -0.001)],
        ["OE1", 6, (0.634, 1.060, 0.000)],
    ],
    "GLU": [
        ["N", 0, (-0.528, 1.361, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.526, -0.781, -1.207)],
        ["O", 3, (0.626, 1.062, 0.000)],
        ["CG", 4, (0.615, 1.392, 0.000)],
        ["CD", 5, (0.600, 1.397, 0.000)],
        ["OE1", 6, (0.607, 1.095, -0.000)],
        ["OE2", 6, (0.589, -1.104, -0.001)],
    ],
    "GLY": [
        ["N", 0, (-0.572, 1.337, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.517, -0.000, -0.000)],
        ["O", 3, (0.626, 1.062, -0.000)],
    ],
    "HIS": [
        ["N", 0, (-0.527, 1.360, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, 0.000, 0.000)],
        ["CB", 0, (-0.525, -0.778, -1.208)],
        ["O", 3, (0.625, 1.063, 0.000)],
        ["CG", 4, (0.600, 1.370, -0.000)],
        ["CD2", 5, (0.889, -1.021, 0.003)],
        ["ND1", 5, (0.744, 1.160, -0.000)],
        ["CE1", 5, (2.030, 0.851, 0.002)],
        ["NE2", 5, (2.145, -0.466, 0.004)],
    ],
    "ILE": [
        ["N", 0, (-0.493, 1.373, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, -0.000)],
        ["CB", 0, (-0.536, -0.793, -1.213)],
        ["O", 3, (0.627, 1.062, -0.000)],
        ["CG1", 4, (0.534, 1.437, -0.000)],
        ["CG2", 4, (0.540, -0.785, -1.199)],
        ["CD1", 5, (0.619, 1.391, 0.000)],
    ],
    "LEU": [
        ["N", 0, (-0.520, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.522, -0.773, -1.214)],
        ["O", 3, (0.625, 1.063, -0.000)],
        ["CG", 4, (0.678, 1.371, 0.000)],
        ["CD1", 5, (0.530, 1.430, -0.000)],
        ["CD2", 5, (0.535, -0.774, 1.200)],
    ],
    "LYS": [
        ["N", 0, (-0.526, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, 0.000)],
        ["CB", 0, (-0.524, -0.778, -1.208)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.619, 1.390, 0.000)],
        ["CD", 5, (0.559, 1.417, 0.000)],
        ["CE", 6, (0.560, 1.416, 0.000)],
        ["NZ", 7, (0.554, 1.387, 0.000)],
    ],
    "MET": [
        ["N", 0, (-0.521, 1.364, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, 0.000, 0.000)],
        ["CB", 0, (-0.523, -0.776, -1.210)],
        ["O", 3, (0.625, 1.062, -0.000)],
        ["CG", 4, (0.613, 1.391, -0.000)],
        ["SD", 5, (0.703, 1.695, 0.000)],
        ["CE", 6, (0.320, 1.786, -0.000)],
    ],
    "PHE": [
        ["N", 0, (-0.518, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, 0.000, -0.000)],
        ["CB", 0, (-0.525, -0.776, -1.212)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.607, 1.377, 0.000)],
        ["CD1", 5, (0.709, 1.195, -0.000)],
        ["CD2", 5, (0.706, -1.196, 0.000)],
        ["CE1", 5, (2.102, 1.198, -0.000)],
        ["CE2", 5, (2.098, -1.201, -0.000)],
        ["CZ", 5, (2.794, -0.003, -0.001)],
    ],
    "PRO": [
        ["N", 0, (-0.566, 1.351, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, 0.000)],
        ["CB", 0, (-0.546, -0.611, -1.293)],
        ["O", 3, (0.621, 1.066, 0.000)],
        ["CG", 4, (0.382, 1.445, 0.0)],
        # ['CD', 5, (0.427, 1.440, 0.0)],
        ["CD", 5, (0.477, 1.424, 0.0)],  # manually made angle 2 degrees larger
    ],
    "SER": [
        ["N", 0, (-0.529, 1.360, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.518, -0.777, -1.211)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["OG", 4, (0.503, 1.325, 0.000)],
    ],
    "THR": [
        ["N", 0, (-0.517, 1.364, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, -0.000)],
        ["CB", 0, (-0.516, -0.793, -1.215)],
        ["O", 3, (0.626, 1.062, 0.000)],
        ["CG2", 4, (0.550, -0.718, -1.228)],
        ["OG1", 4, (0.472, 1.353, 0.000)],
    ],
    "TRP": [
        ["N", 0, (-0.521, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, 0.000)],
        ["CB", 0, (-0.523, -0.776, -1.212)],
        ["O", 3, (0.627, 1.062, 0.000)],
        ["CG", 4, (0.609, 1.370, -0.000)],
        ["CD1", 5, (0.824, 1.091, 0.000)],
        ["CD2", 5, (0.854, -1.148, -0.005)],
        ["CE2", 5, (2.186, -0.678, -0.007)],
        ["CE3", 5, (0.622, -2.530, -0.007)],
        ["NE1", 5, (2.140, 0.690, -0.004)],
        ["CH2", 5, (3.028, -2.890, -0.013)],
        ["CZ2", 5, (3.283, -1.543, -0.011)],
        ["CZ3", 5, (1.715, -3.389, -0.011)],
    ],
    "TYR": [
        ["N", 0, (-0.522, 1.362, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, -0.000, -0.000)],
        ["CB", 0, (-0.522, -0.776, -1.213)],
        ["O", 3, (0.627, 1.062, -0.000)],
        ["CG", 4, (0.607, 1.382, -0.000)],
        ["CD1", 5, (0.716, 1.195, -0.000)],
        ["CD2", 5, (0.713, -1.194, -0.001)],
        ["CE1", 5, (2.107, 1.200, -0.002)],
        ["CE2", 5, (2.104, -1.201, -0.003)],
        ["OH", 5, (4.168, -0.002, -0.005)],
        ["CZ", 5, (2.791, -0.001, -0.003)],
    ],
    "VAL": [
        ["N", 0, (-0.494, 1.373, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, -0.000)],
        ["CB", 0, (-0.533, -0.795, -1.213)],
        ["O", 3, (0.627, 1.062, -0.000)],
        ["CG1", 4, (0.540, 1.429, -0.000)],
        ["CG2", 4, (0.533, -0.776, 1.203)],
    ],
}

# Reordered atoms in standard order
# order should be consistent with atom_types and CCD (TODO: check this)
residue_atoms = {
    "ALA": ["N", "CA", "C", "O", "CB"],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
    "CYS": ["N", "CA", "C", "O", "CB", "SG"],
    "SEC": ["N", "CA", "C", "O", "CB", "SE"],  # Selenocysteine (added)
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
    "GLY": ["N", "CA", "C", "O"],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CE1", "NE2", "CD2"],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
    "MSE": ["N", "CA", "C", "O", "CB", "CG", "SE", "CE"],  # Selenomethionine (added)
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD"],
    # "PYL": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "CG2", "CD2", "CE2", "N2"],  # Pyrrolysine (added)
    "SER": ["N", "CA", "C", "O", "CB", "OG"],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2"],
    "TRP": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "NE1",
        "CE2",
        "CD2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
    ],
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
    "UNK": [
        "N",
        "CA",
        "C",
        "O",
    ],  # n.b. other atoms can be present in unk residues - we typically discard
}

restype_name_to_atom14_names = {
    restype: [atom_name for atom_name in residue_atoms[restype]]
    + [""] * (14 - len(residue_atoms[restype]))
    for restype in residue_atoms
}

# Naming swaps for ambiguous atom names.
# Due to symmetries in the amino acids the naming of atoms is ambiguous in
# 4 of the 20 amino acids.
# (The LDDT paper lists 7 amino acids as ambiguous, but the naming ambiguities
# in LEU, VAL and ARG can be resolved by using the 3d constellations of
# the 'ambiguous' atoms and their neighbours)
# Because for LEU, VAL and ARG, no ambiguous exist when the prediction output is chi angle instead of the location of individual atoms.
# For the rest, ASP and others, when you rotate the bond 180 degree, you get the same configuraiton due to symmetry.

residue_atom_renaming_swaps = {
    "ASP": {"OD1": "OD2"},
    "GLU": {"OE1": "OE2"},
    "PHE": {"CD1": "CD2", "CE1": "CE2"},
    "TYR": {"CD1": "CD2", "CE1": "CE2"},
}

# Van der Waals radii [Angstroem] of the atoms (from Wikipedia)
van_der_waals_radius = {
    "C": 1.7,
    "N": 1.55,
    "O": 1.52,
    "S": 1.8,
}

Bond = collections.namedtuple("Bond", ["atom1_name", "atom2_name", "length", "stddev"])
BondAngle = collections.namedtuple(
    "BondAngle",
    ["atom1_name", "atom2_name", "atom3name", "angle_rad", "stddev"],
)

# Between-residue bond lengths for general bonds (first element) and for Proline
# (second element).
between_res_bond_length_c_n = [1.329, 1.341]
between_res_bond_length_stddev_c_n = [0.014, 0.016]

# Between-residue cos_angles.
between_res_cos_angles_c_n_ca = [-0.5203, 0.0353]  # degrees: 121.352 +- 2.315
between_res_cos_angles_ca_c_n = [-0.4473, 0.0311]  # degrees: 116.568 +- 1.995

# This mapping is used when we need to store atom data in a format that requires
# fixed atom data size for every residue (e.g. a numpy array).
# atom types for the 20 standard amino acids
atom_types = [
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
]
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
atom_type_num = len(atom_types)  # := 37.

# A compact atom encoding with 14 columns
# pylint: disable=line-too-long
# pylint: disable=bad-whitespace

# This is the standard residue order when coding AA type as a number.
# Reproduce it by taking 3-letter AA codes and sorting them alphabetically.
restypes = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]
restype_order = {restype: i for i, restype in enumerate(restypes)}
restype_num = len(restypes)  # := 20.
unk_restype_index = restype_num  # Catch-all index for unknown restypes.

restypes_with_x = restypes + ["X"]
restype_order_with_x = {restype: i for i, restype in enumerate(restypes_with_x)}

restype_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "X": "UNK",
}


# NB: restype_3to1 differs from Bio.PDB.protein_letters_3to1 by being a simple
# 1-to-1 mapping of 3 letter names to one letter names. The latter contains
# many more, and less common, three letter names as keys and maps many of these
# to the same one letter name (including 'X' and 'U' which we don't use here).
restype_3to1 = {v: k for k, v in restype_1to3.items()}

# Define a restype name for all unknown residues.
unk_restype = "UNK"

resnames = [restype_1to3[r] for r in restypes] + [unk_restype]
resname_to_idx = {resname: i for i, resname in enumerate(resnames)}


def sequence_to_onehot(
    sequence: str, mapping: Mapping[str, int], map_unknown_to_x: bool = False
) -> np.ndarray:
    """Maps the given sequence into a one-hot encoded matrix.

    Args:
      sequence: An amino acid sequence.
      mapping: A dictionary mapping amino acids to integers.
      map_unknown_to_x: If True, any amino acid that is not in the mapping will be
        mapped to the unknown amino acid 'X'. If the mapping doesn't contain
        amino acid 'X', an error will be thrown. If False, any amino acid not in
        the mapping will throw an error.

    Returns:
      A numpy array of shape (seq_len, num_unique_aas) with one-hot encoding of
      the sequence.

    Raises:
      ValueError: If the mapping doesn't contain values from 0 to
        num_unique_aas - 1 without any gaps.
    """
    num_entries = max(mapping.values()) + 1

    if sorted(set(mapping.values())) != list(range(num_entries)):
        raise ValueError(
            "The mapping must have values from 0 to num_unique_aas-1 "
            "without any gaps. Got: %s" % sorted(mapping.values())
        )

    one_hot_arr = np.zeros((len(sequence), num_entries), dtype=np.int32)

    for aa_index, aa_type in enumerate(sequence):
        if map_unknown_to_x:
            if aa_type.isalpha() and aa_type.isupper():
                aa_id = mapping.get(aa_type, mapping["X"])
            else:
                raise ValueError(f"Invalid character in the sequence: {aa_type}")
        else:
            aa_id = mapping[aa_type]
        one_hot_arr[aa_index, aa_id] = 1

    return one_hot_arr


def _make_standard_atom_mask() -> np.ndarray:
    """Returns [num_res_types, num_atom_types] mask array."""
    # +1 to account for unknown (all 0s).
    mask = np.zeros([restype_num + 1, atom_type_num], dtype=np.int32)
    for restype, restype_letter in enumerate(restypes):
        restype_name = restype_1to3[restype_letter]
        atom_names = residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = atom_order[atom_name]
            mask[restype, atom_type] = 1
    return mask


STANDARD_ATOM_MASK = _make_standard_atom_mask()


# A one hot representation for the first and second atoms defining the axis
# of rotation for each chi-angle in each residue.
def chi_angle_atom(atom_index: int) -> np.ndarray:
    """Define chi-angle rigid groups via one-hot representations."""
    chi_angles_index = {}
    one_hots = []

    for k, v in chi_angles_atoms.items():
        indices = [atom_types.index(s[atom_index]) for s in v]
        indices.extend([-1] * (4 - len(indices)))
        chi_angles_index[k] = indices

    for r in restypes:
        res3 = restype_1to3[r]
        one_hot = np.eye(atom_type_num)[chi_angles_index[res3]]
        one_hots.append(one_hot)

    one_hots.append(np.zeros([4, atom_type_num]))  # Add zeros for residue `X`.
    one_hot = np.stack(one_hots, axis=0)
    one_hot = np.transpose(one_hot, [0, 2, 1])

    return one_hot


chi_atom_1_one_hot = chi_angle_atom(1)
chi_atom_2_one_hot = chi_angle_atom(2)

# # An array like chi_angles_atoms but using indices rather than names.
# chi_angles_atom_indices = [chi_angles_atoms[restype_1to3[r]] for r in restypes]
# chi_angles_atom_indices = tree.map_structure(
#     lambda atom_name: atom_order[atom_name], chi_angles_atom_indices
# )
# chi_angles_atom_indices = np.array(
#     [
#         chi_atoms + ([[0, 0, 0, 0]] * (4 - len(chi_atoms)))
#         for chi_atoms in chi_angles_atom_indices
#     ]
# )

# Mapping from (res_name, atom_name) pairs to the atom's chi group index
# and atom index within that group.
chi_groups_for_atom = collections.defaultdict(list)
for res_name, chi_angle_atoms_for_res in chi_angles_atoms.items():
    for chi_group_i, chi_group in enumerate(chi_angle_atoms_for_res):
        for atom_i, atom in enumerate(chi_group):
            chi_groups_for_atom[(res_name, atom)].append((chi_group_i, atom_i))
chi_groups_for_atom = dict(chi_groups_for_atom)


restype_atom14_ambiguous_atoms = np.zeros((21, 14), dtype=np.float32)
restype_atom14_ambiguous_atoms_swap_idx = np.tile(np.arange(14, dtype=int), (21, 1))


def _make_atom14_ambiguity_feats():
    for res, pairs in residue_atom_renaming_swaps.items():
        res_idx = restype_order[restype_3to1[res]]
        for atom1, atom2 in pairs.items():
            atom1_idx = restype_name_to_atom14_names[res].index(atom1)
            atom2_idx = restype_name_to_atom14_names[res].index(atom2)
            restype_atom14_ambiguous_atoms[res_idx, atom1_idx] = 1
            restype_atom14_ambiguous_atoms[res_idx, atom2_idx] = 1
            restype_atom14_ambiguous_atoms_swap_idx[res_idx, atom1_idx] = atom2_idx
            restype_atom14_ambiguous_atoms_swap_idx[res_idx, atom2_idx] = atom1_idx


_make_atom14_ambiguity_feats()


def aatype_to_str_sequence(aatype):
    return "".join([restypes_with_x[aatype[i]] for i in range(len(aatype))])


### ALPHAFOLD MULTIMER STUFF ###
def _make_chi_atom_indices():
    """Returns atom indices needed to compute chi angles for all residue types.

    Returns:
      A tensor of shape [residue_types=21, chis=4, atoms=4]. The residue types are
      in the order specified in residue_constants.restypes + unknown residue type
      at the end. For chi angles which are not defined on the residue, the
      positions indices are by default set to 0.
    """
    chi_atom_indices = []
    for residue_name in restypes:
        residue_name = restype_1to3[residue_name]
        residue_chi_angles = chi_angles_atoms[residue_name]
        atom_indices = []
        for chi_angle in residue_chi_angles:
            atom_indices.append([atom_order[atom] for atom in chi_angle])
        for _ in range(4 - len(atom_indices)):
            atom_indices.append([0, 0, 0, 0])  # For chi angles not defined on the AA.
        chi_atom_indices.append(atom_indices)

    chi_atom_indices.append([[0, 0, 0, 0]] * 4)  # For UNKNOWN residue.

    return np.array(chi_atom_indices)


def _make_renaming_matrices():
    """Matrices to map atoms to symmetry partners in ambiguous case."""
    # As the atom naming is ambiguous for 7 of the 20 amino acids, provide
    # alternative groundtruth coordinates where the naming is swapped
    restype_3 = [restype_1to3[res] for res in restypes]
    restype_3 += ["UNK"]
    # Matrices for renaming ambiguous atoms.
    all_matrices = {res: np.eye(14, dtype=np.float32) for res in restype_3}
    for resname, swap in residue_atom_renaming_swaps.items():
        correspondences = np.arange(14)
        for source_atom_swap, target_atom_swap in swap.items():
            source_index = restype_name_to_atom14_names[resname].index(source_atom_swap)
            target_index = restype_name_to_atom14_names[resname].index(target_atom_swap)
            correspondences[source_index] = target_index
            correspondences[target_index] = source_index
            renaming_matrix = np.zeros((14, 14), dtype=np.float32)
            for index, correspondence in enumerate(correspondences):
                renaming_matrix[index, correspondence] = 1.0
        all_matrices[resname] = renaming_matrix.astype(np.float32)
    renaming_matrices = np.stack([all_matrices[restype] for restype in restype_3])
    return renaming_matrices


def _make_restype_atom37_mask():
    """Mask of which atoms are present for which residue type in atom37."""
    # create the corresponding mask
    restype_atom37_mask = np.zeros([21, 37], dtype=np.float32)
    for restype, restype_letter in enumerate(restypes):
        restype_name = restype_1to3[restype_letter]
        atom_names = residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = atom_order[atom_name]
            restype_atom37_mask[restype, atom_type] = 1
    return restype_atom37_mask


def _make_restype_atom14_mask():
    """Mask of which atoms are present for which residue type in atom14."""
    restype_atom14_mask = []

    for rt in restypes:
        atom_names = restype_name_to_atom14_names[restype_1to3[rt]]
        restype_atom14_mask.append([(1.0 if name else 0.0) for name in atom_names])

    restype_atom14_mask.append([0.0] * 14)
    restype_atom14_mask = np.array(restype_atom14_mask, dtype=np.float32)
    return restype_atom14_mask


def _make_restype_atom37_to_atom14():
    """Map from atom37 to atom14 per residue type."""
    restype_atom37_to_atom14 = []  # mapping (restype, atom37) --> atom14
    for rt in restypes:
        atom_names = restype_name_to_atom14_names[restype_1to3[rt]]
        atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
        restype_atom37_to_atom14.append(
            [
                (atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0)
                for name in atom_types
            ]
        )

    restype_atom37_to_atom14.append([0] * 37)
    restype_atom37_to_atom14 = np.array(restype_atom37_to_atom14, dtype=np.int32)
    return restype_atom37_to_atom14


def _make_restype_atom14_to_atom37():
    """Map from atom14 to atom37 per residue type."""
    restype_atom14_to_atom37 = []  # mapping (restype, atom14) --> atom37
    for rt in restypes:
        atom_names = restype_name_to_atom14_names[restype_1to3[rt]]
        restype_atom14_to_atom37.append(
            [(atom_order[name] if name else 0) for name in atom_names]
        )
    # Add dummy mapping for restype 'UNK'
    restype_atom14_to_atom37.append([0] * 14)
    restype_atom14_to_atom37 = np.array(restype_atom14_to_atom37, dtype=np.int32)
    return restype_atom14_to_atom37


def _make_restype_atom14_is_ambiguous():
    """Mask which atoms are ambiguous in atom14."""
    # create an ambiguous atoms mask.  shape: (21, 14)
    restype_atom14_is_ambiguous = np.zeros((21, 14), dtype=np.float32)
    for resname, swap in residue_atom_renaming_swaps.items():
        for atom_name1, atom_name2 in swap.items():
            restype = restype_order[restype_3to1[resname]]
            atom_idx1 = restype_name_to_atom14_names[resname].index(atom_name1)
            atom_idx2 = restype_name_to_atom14_names[resname].index(atom_name2)
            restype_atom14_is_ambiguous[restype, atom_idx1] = 1
            restype_atom14_is_ambiguous[restype, atom_idx2] = 1

    return restype_atom14_is_ambiguous


def _make_restype_rigidgroup_base_atom37_idx():
    """Create Map from rigidgroups to atom37 indices."""
    # Create an array with the atom names.
    # shape (num_restypes, num_rigidgroups, 3_atoms): (21, 8, 3)
    base_atom_names = np.full([21, 8, 3], "", dtype=object)

    # 0: backbone frame
    base_atom_names[:, 0, :] = ["C", "CA", "N"]

    # 3: 'psi-group'
    base_atom_names[:, 3, :] = ["CA", "C", "O"]

    # 4,5,6,7: 'chi1,2,3,4-group'
    for restype, restype_letter in enumerate(restypes):
        resname = restype_1to3[restype_letter]
        for chi_idx in range(4):
            if chi_angles_mask[restype][chi_idx]:
                atom_names = chi_angles_atoms[resname][chi_idx]
                base_atom_names[restype, chi_idx + 4, :] = atom_names[1:]

    # Translate atom names into atom37 indices.
    lookuptable = atom_order.copy()
    lookuptable[""] = 0
    restype_rigidgroup_base_atom37_idx = np.vectorize(lambda x: lookuptable[x])(
        base_atom_names
    )
    return restype_rigidgroup_base_atom37_idx


CHI_ATOM_INDICES = _make_chi_atom_indices()
RENAMING_MATRICES = _make_renaming_matrices()
RESTYPE_ATOM14_TO_ATOM37 = _make_restype_atom14_to_atom37()
RESTYPE_ATOM37_TO_ATOM14 = _make_restype_atom37_to_atom14()
RESTYPE_ATOM37_MASK = _make_restype_atom37_mask()
RESTYPE_ATOM14_MASK = _make_restype_atom14_mask()
RESTYPE_ATOM14_IS_AMBIGUOUS = _make_restype_atom14_is_ambiguous()
RESTYPE_RIGIDGROUP_BASE_ATOM37_IDX = _make_restype_rigidgroup_base_atom37_idx()

# Create mask for existing rigid groups.
RESTYPE_RIGIDGROUP_MASK = np.zeros([21, 8], dtype=np.float32)
RESTYPE_RIGIDGROUP_MASK[:, 0] = 1
RESTYPE_RIGIDGROUP_MASK[:, 3] = 1
RESTYPE_RIGIDGROUP_MASK[:20, 4:] = chi_angles_mask
