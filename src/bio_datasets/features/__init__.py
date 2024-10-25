__all__ = [
    "AtomArrayFeature",
    "StructureFeature",
    "ProteinAtomArrayFeature",
    "ProteinStructureFeature",
]

from datasets.features.features import register_feature

from .atom_array import (
    AtomArrayFeature,
    ProteinAtomArrayFeature,
    ProteinStructureFeature,
    StructureFeature,
)

register_feature(StructureFeature, "StructureFeature")
register_feature(AtomArrayFeature, "AtomArrayFeature")
register_feature(ProteinAtomArrayFeature, "ProteinAtomArrayFeature")
register_feature(ProteinStructureFeature, "ProteinStructureFeature")
