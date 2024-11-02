__all__ = [
    "AtomArrayFeature",
    "StructureFeature",
    "ProteinAtomArrayFeature",
    "ProteinStructureFeature",
    "Features",
]

from typing import Dict

from datasets.features.features import FeatureType, register_feature

from .atom_array import (
    AtomArrayFeature,
    ProteinAtomArrayFeature,
    ProteinStructureFeature,
    StructureFeature,
)
from .features import Features

_BIO_FEATURE_TYPES: Dict[str, FeatureType] = {}


def register_bio_feature(feature_cls):
    _BIO_FEATURE_TYPES[feature_cls.__name__] = feature_cls
    register_feature(feature_cls, feature_cls.__name__)


register_bio_feature(StructureFeature)
register_bio_feature(AtomArrayFeature)
register_bio_feature(ProteinAtomArrayFeature)
register_bio_feature(ProteinStructureFeature)
