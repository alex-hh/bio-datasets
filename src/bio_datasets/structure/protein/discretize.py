import numpy as np


def discretize_to_bits(
    arr: np.ndarray, bits: int, max_val: float, min_val: float = 0, signed: bool = False
) -> np.ndarray:
    """
    Discretize a NumPy array of floats/integers to the specified number of bits with automatic scaling.

    Args:
        arr (np.ndarray): Array of floats or integers to be discretized.
        bits (int): Number of bits for discretization.
        min_val (float): Minimum possible value in the input range.
        max_val (float): Maximum possible value in the input range.
        signed (bool): Whether to use signed or unsigned integer representation.

    Returns:
        np.ndarray: Array of discretized integers.
    """
    if signed:
        # Signed range: [-2^(bits-1), 2^(bits-1) - 1]
        max_int_val = (2 ** (bits - 1)) - 1
        min_int_val = -(2 ** (bits - 1))
    else:
        # Unsigned range: [0, 2^bits - 1]
        max_int_val = (2**bits) - 1
        min_int_val = 0

    # Normalize to the range [0, 1]
    normalized = (arr - min_val) / (max_val - min_val)

    # Scale to the target integer range
    scaled = normalized * (max_int_val - min_int_val) + min_int_val

    # Discretize to integer values
    discretized = np.round(scaled).astype(np.int32)

    return discretized


def decode_from_bits(
    arr: np.ndarray, bits: int, max_val: float, min_val: float = 0, signed: bool = False
) -> np.ndarray:
    """
    Decode a NumPy array of discretized integers back to the original floating-point values.

    Args:
        arr (np.ndarray): Array of discretized integers.
        bits (int): Number of bits used for discretization.
        min_val (float): Minimum possible value in the input range.
        max_val (float): Maximum possible value in the input range.
        signed (bool): Whether the original discretization used signed or unsigned integers.

    Returns:
        np.ndarray: Array of decoded floating-point values.
    """
    if signed:
        # Signed range: [-2^(bits-1), 2^(bits-1) - 1]
        max_int_val = (2 ** (bits - 1)) - 1
        min_int_val = -(2 ** (bits - 1))
    else:
        # Unsigned range: [0, 2^bits - 1]
        max_int_val = (2**bits) - 1
        min_int_val = 0

    # Scale back to the normalized range [0, 1]
    normalized = (arr - min_int_val) / (max_int_val - min_int_val)

    # Scale to the original value range [min_val, max_val]
    decoded = normalized * (max_val - min_val) + min_val

    return decoded


class Compressor:
    def __init__(self, config: dict):
        self.config = config

    def compress(self, internal_coords):
        """Compress the internal coordinates. Return internal_coords tuple and compression config."""
        bond_lengths, bond_angles, dihedrals = internal_coords
        pass

    def decompress(self, compressed_internal_coords):
        """Decompress the internal coordinates. Return a tuple of the form (bond_lengths, bond_angles, dihedrals)"""
        pass
