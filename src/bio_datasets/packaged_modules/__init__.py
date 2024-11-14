"""Builders for bio datasets.

We register builders as 'packaged modules' in the datasets library.
This allows us to load data using load_dataset.

To achieve this, each builder must be defined in a separate file in the builders/ subdirectory.
That file should define a single class inheriting from `datasets.DatasetBuilder`.
"""
import inspect

from datasets.packaged_modules import _PACKAGED_DATASETS_MODULES, _hash_python_lines

from .structurefolder import structurefolder

_PACKAGED_BIO_MODULES = {
    "structurefolder": (
        structurefolder.__name__,
        _hash_python_lines(inspect.getsource(structurefolder).splitlines()),
    )
}

_PACKAGED_DATASETS_MODULES.update(_PACKAGED_BIO_MODULES)
