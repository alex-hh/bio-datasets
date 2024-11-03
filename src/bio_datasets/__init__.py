import inspect

import datasets
from datasets.packaged_modules import _PACKAGED_DATASETS_MODULES, _hash_python_lines

from .features import *
from .packaged_modules.structurefolder import structurefolder
from .protein import Protein, ProteinChain, ProteinComplex

datasets.features.features.Features = Features

_PACKAGED_BIO_MODULES = {
    "structurefolder": (
        structurefolder.__name__,
        _hash_python_lines(inspect.getsource(structurefolder).splitlines()),
    )
}

_PACKAGED_DATASETS_MODULES.update(_PACKAGED_BIO_MODULES)
