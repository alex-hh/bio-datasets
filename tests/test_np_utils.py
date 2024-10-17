import numpy as np

from bio_datasets.np_utils import map_categories_to_indices


def test_map_categories_to_indices():
    categories = ["A", "B", "C"]
    arr = np.array(["A", "B", "C", "B", "C", "A"])
    assert np.all(
        map_categories_to_indices(arr, categories) == np.array([0, 1, 2, 1, 2, 0])
    )
