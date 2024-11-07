# Bio Datasets

Bringing bio (molecules and more) to the HuggingFace Datasets library.

This (unofficial!) extension to Datasets is designed to make the following things as easy as possible:

1. efficient storage of biological data for ML
2. low-overhead loading of data into standardised python objects from bio data libraries, convenient for downstream processing
3. sharing of datasets large and small

We aim to do these three things and *no more*, leaving you to get on with the science!

## Efficient conversion between storage and usage formats

The best format for storing data is typically not the most convenient format for data to be loaded into for downstream applications. The Datasets library abstracts the details of these choices and the logic for converting between formats into Feature classes, dictating how data of particular types should be stored and loaded. We extend the Datasets library by creating Feature types for optimised storage and loading of biological data, starting with biomolecular structures.

The main formats we support for storing and loading biomolecular structure data are:


| Feature name |   Storage format    |  Loaded as  |
| ------------ | --------------------| ------------|
|  AtomArrayFeature / ProteinAtomArrayFeature  | arrays of cartesian or (*experimental*) discretised internal coordinates and annotations | `biotite.structure.AtomArray` / `bio_datasets.ProteinChain` / `bio_datasets.ProteinComplex` (`bio_datasets` classes are lightweight wrappers around `AtomArray`)|
|  StructureFeature / ProteinStructureFeature   | byte string encoded file format embedded into parquet columns: PDB / compressed PDB (gzip / foldcomp fcz) | `biotite.structure.AtomArray` / `bio_datasets.ProteinChain` / `bio_datasets.ProteinComplex` |


## Installation

```bash
git clone https://github.com/alex-hh/bio-datasets.git && cd bio-datasets
pip install .
```

## Usage

### Loading data from the Hub

In the Datasets library, datasets are associated with metadata annotations describing the feature types present in the dataset, and how those feature types should be stored and loaded.

This makes it easy to share datasets in efficient storage formats, while allowing people who download the dataset to directly access the data in a convenient format for downstream use.

To illustrate, we provide examples of datasets pre-configured with Bio Datasets Feature types that can be downloaded from the hub.

```python
import foldcomp
import bio_datasets  # necessary to register the custom feature types with the datasets library
from datasets import load_dataset

dataset = load_dataset(
    "biodatasets/afdb_e_coli",
    split="train",
)
ex = dataset[0]  # a dict with keys `name` and `structure` (a `biotite.structure.AtomArray` wrapped in a `bio_datasets.Protein` object for standardisation.)
print(type(ex["structure"]))
```
```
<class 'bio_datasets.structure.protein.protein.ProteinChain'>
```

That's it: when you access data from a dataset with preset Bio Datasets feature types, the datapoints that it returns will be Python dictionaries containing your Protein data formatted as a `bio_datasets.protein.ProteinChain` object (basically a biotite AtomArray with some added convenience methods for Protein ML.)

The trick is that the data was stored together with the required Feature type information, which we can inspect directly:

```python
print(dataset.info.features)

```
```
{'name': Value(dtype='string', id=None),
 'structure': ProteinStructureFeature(requires_encoding=True, requires_decoding=True, decode=True, id=None, with_occupancy=False, with_b_factor=True, with_atom_id=False, with_charge=False, encode_with_foldcomp=False)}
```

To summarise: this dataset contains two features: 'name', which is a string, and 'structure' which is a `bio_datasets.ProteinStructureFeature`. Features of this type will automatically be loaded as `bio_datasets.Protein` instances when bio_datasets is installed; and as dictionaries containing the fields `path`, `bytes` (the file contents) and `type` (the file format) fields when not.

We can also inspect the raw data format used for storage by discarding the feature information:

```python
dataset.info.features = None
dataset[0]["structure"]  # a dictionary with key bytes whose value is the foldcomp byte string
```

### Creating a dataset with bio feature types


#### Building and sharing datasets from local files

To streamline the processes of building your own datasets from local files, we provide some utility classes for building datasets from standard formats. For example suppose you have a local directory containing PDB files. You want to write an efficient data loader, and ideally to reduce the size of the directory to make it easier to share with your collaborators. bio-datasets allows you to achieve these two things with almost no effort.

#### Fully-flexible dataset construction

If your local data doesn't fit the format assumed by ProteinStructureFolder, but you still want a way to create a Dataset instance (for example to convert your local data into a more efficient storage format for sharing, or to exploit Dataset's fast memory-mapped retrieval), you simply need to configure a Dataset constructor with information on your Feature types:

This feature configuration is performed by creating a `datasets.Features` object mapping column names to feature types.

Each Feature type supports various configuration options (see details in \__init__ methods) controlling the formats in which data is stored and loaded.

<!-- https://huggingface.co/docs/datasets/v3.0.1/en/audio_dataset#create-a-dataset-builder-class -->
<!-- https://huggingface.co/docs/datasets/en/about_dataset_load#datasets-datasetbuilder -->
For example, using a generator-based Dataset constructor:

```python
from datasets import Dataset, Features
from bio_datasets import ProteinStructureFeature


def examples_generator(pdb_file_list):
    for file_path in pdb_file_list:
        yield {"structure": {"path": file_path}}  # generate examples in 'raw' format


# create a dataset which will save data to disk as a foldcomp-encoded byte string, but which will automatically
# decode that data to biotite atom arrays during loading / iteration
features = Features(structure=ProteinStructureFeature(encode_with_foldcomp=True))
ds = Dataset.from_generator(examples_generator, gen_kwargs={"pdb_file_list": pdb_file_list}, features=features)
ds[0]

# share your bio dataset to the HuggingFace hub!
ds.push_to_hub(HUB_REPO_ID)
```

The `examples_generator` function yields single datapoints in unprocessed formats compatible with the corresponding feature:

In this case the unprocessed data for the 'structure' column is passed as a python dictionary containing a single key 'path' whose value is the path to a PDB file. Refer to the documentation of individual Feature types for more information on supported input formats. (Under the hood these will be encoded via feature.encode_example).

### Performance tips: choose Feature types to trade off between efficiency of storage and loading

`bio_datasets.StructureFeature` feature data is stored internally
as PDB format byte-strings (optionally compressed with foldcomp or gzip). bio_datasets automatically handles conversion from this format to the
biotite AtomArray format for downstream processing.
Of course, parsing the PDB format to biotite format involves some overhead (though it's
still possible to iterate over ~100 pdb files a second; and we'll automatically load files
using [fastpdb](https://github.com/biotite-dev/fastpdb) if you have it installed)

If you want even faster processing, we also support storing data in a native array format
that supports blazingly fast iteration over fully featurised samples.
Let's convert the `bio_datasets.StructureFeature` data to the `bio_datasets.AtomArrayFeature` type, and compare iteration speed:


```python
from datasets import Features, Value, load_dataset
from bio_datasets import AtomArrayFeature

dataset = load_dataset(
    "biodatasets/afdb_e_coli",
    split="train",
)
pdb_time = timeit.timeit(stmt="""[ex for ex in dataset]""", number=1, globals=globals())
def convert_structure_to_array(ex, features):
    return features.encode_example(ex)

new_features = Features(name=Value("string"), structure=AtomArrayFeature())
array_dataset = dataset.map(convert_structure_to_array, features=new_features, fn_kwargs={"features": new_features})
array_time = timeit.timeit(stmt="""[ex for ex in array_dataset]""", number=1, globals=globals())

print(
  f"Iterated over {len(dataset)} examples in "
  f"{pdb_time:.1f}s with PDB storage vs {array_time:.1f}s with array storage\n"
  f" i.e. {len(dataset)/pdb_time:.1f} samples/s vs {len(dataset)/array_time:.1f} samples/s"
)
```
```
Iterated over 8726 examples in 47.5s with PDB storage vs 7.0s with array storage
 i.e. 183.6 samples/s vs 1237.8 samples/s
```

All of the Datasets library's methods for faster loading, including batching and
multiprocessing can also be applied to further optimise performance!

To combine the fast iteration offered by array-based storage with foldcomp-style compression,
we offer an **experimental** option to store structure data in a foldcomp-style discretised internal
coordinate-based representation.

### Sharing data to the hub

`ds.push_to_hub` will automatically save information about the Feature types stored
in the dataset. If a user with bio-datasets installed downloads the dataset, their bio
data will then automatically be decoded in the way specified by the Features.

### Creating your own feature types

TODO: add docs.


## Roadmap

* Support for other biological data types: protein-ligand complexes, MD, single cell / omics, ...


## Contributions

We would love to receive contributions of code (e.g. new feature types!),
suggestions for new data formats/feature types, and sharing of compatible bio datasets
e.g. to the HuggingFace Hub
