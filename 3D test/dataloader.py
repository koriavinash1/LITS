import tensorflow as tf
import os
import numpy as np
import nibabel as nib
from config import FLAGS


data_path = FLAGS.train_dir

def get_all_volumes_paths(base_path):
    return [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith('.nii')]

def get_input_paths(paths):
    return [path for path in paths if path.split("/").pop().split(".")[0].split("_")[0].split("-")[0] == "volume"]

def get_segmentation_paths(paths):
    return [path for path in paths if path.split("/").pop().split(".")[0].split("_")[0].split("-")[0] == "segmentation"]

def get_volumes(paths):
    return [np.clip(nib.load(path).get_data(), -100, 400) for path in paths]

def data_normalization(volumes):
    # horizontal shift
    volumes = np.add(volumes, 100)
    volumes = volumes/500
    return volumes
    
class DataSet(object):
    def __init__(self, volumes, segmentations):
        self._num_examples = len(volumes)
        self._volumes = volumes
        self._segmentations = segmentations
        self._epochs_completed = 0
        self._index_in_epoch = 0
        # print self._num_examples
        # print len(self._segmentations)
        assert self._num_examples == len(self._segmentations)

    @property
    def volumes(self):
        return self._volumes

    @property
    def segmentations(self):
        return self._segmentations

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def shuffle_in_unison(self, a, b):
        a = np.array(a)
        b = np.array(b)
        assert len(a) == len(b)
        shuffled_a = np.empty(a.shape, dtype=a.dtype)
        shuffled_b = np.empty(b.shape, dtype=b.dtype)
        permutation = np.random.permutation(len(a))
        for old_index, new_index in enumerate(permutation):
            shuffled_a[new_index] = a[old_index]
            shuffled_b[new_index] = b[old_index]
        return shuffled_a, shuffled_b

    def next_batch(self, batch_size):
        if self._index_in_epoch > len(self._volumes)-batch_size:
            self._index_in_epoch = 0
            self._epochs_completed +=1
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        end = self._index_in_epoch
        return self.shuffle_in_unison(np.reshape(self._volumes[start:end], (batch_size, 128,128,128,1)), self._segmentations[start:end])


def load_dataSets(test_volume_number = 1):
    class DataSets(object):
        pass
    data_sets = DataSets()
    paths = get_all_volumes_paths(data_path)
    volume_paths = get_input_paths(paths)
    segmentation_paths = get_segmentation_paths(paths)

    # train_volumes = np.random.randn(10,128,128,128)
    # train_segmentations = np.random.randn(10,128,128,128)
    print "extracting volumes...."
    train_volumes = get_volumes(volume_paths)
    train_volumes = data_normalization(train_volumes)
    print "extracting segmentations...."
    train_segmentations = get_volumes(segmentation_paths).astype("float32")
    # train_segmentations = data_normalization(train_segmentations)

    test_volumes = train_volumes[:test_volume_number]
    test_segmentations = train_segmentations[:test_volume_number]
    np.where(test_segmentations == 0, test_segmentations, 1) 
    
    train_volumes = train_volumes[test_volume_number:]
    train_segmentations = train_segmentations[test_volume_number:]
    np.where(train_segmentations == 0, train_segmentations, 1) 

    data_sets.train = DataSet(train_volumes, train_segmentations)
    data_sets.test = DataSet(test_volumes, test_segmentations)
    return data_sets    