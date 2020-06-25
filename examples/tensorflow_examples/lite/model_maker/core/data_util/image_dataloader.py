from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import random

import tensorflow as tf
import tensorflow_datasets as tfds


def load_image(path):
  #Loads image
  image_raw = tf.io.read_file(path)
  image_tensor = tf.cond(
      tf.image.is_jpeg(image_raw),
      lambda: tf.image.decode_jpeg(image_raw, channels=3),
      lambda: tf.image.decode_png(image_raw, channels=3))
  return image_tensor

class ImageClassifierDataLoader(DataLoader):
  #DataLoader for image classifier.

  def __init__(self, dataset, size, num_classes, index_to_label):
    super(ImageClassifierDataLoader, self).__init__(dataset, size)
    self.num_classes = num_classes
    self.index_to_label = index_to_label

  def split(self, fraction):
    """Splits dataset into two sub-datasets with the given fraction.

    Primarily used for splitting the data set into training and testing sets.

    Args:
      fraction: float, demonstrates the fraction of the first returned
        subdataset in the original data.

    Returns:
      The splitted two sub dataset.
    """
    ds = self.dataset

    train_size = int(self.size * fraction)
    trainset = ImageClassifierDataLoader(
        ds.take(train_size), train_size, self.num_classes, self.index_to_label)

    test_size = self.size - train_size
    testset = ImageClassifierDataLoader(
        ds.skip(train_size), test_size, self.num_classes, self.index_to_label)

    return trainset, testset

  @classmethod
  def from_folder(cls, filename, shuffle=True):
    """Image analysis for image classification load images with labels.

    Assume the image data of the same label are in the same subdirectory.

    Args:
      filename: Name of the file.
      shuffle: boolean, if shuffle, random shuffle data.

    Returns:
      ImageDataset containing images and labels and other related info.
    """
    data_root = os.path.abspath(filename)

    # Assumes the image data of the same label are in the same subdirectory,
    # gets image path and label names.
    all_image_paths = list(tf.io.gfile.glob(data_root + r'/*/*'))
    all_image_size = len(all_image_paths)
    if all_image_size == 0:
      raise ValueError('Image size is zero')

    if shuffle:
      # Random shuffle data.
      random.shuffle(all_image_paths)

    label_names = sorted(
        name for name in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, name)))
    all_label_size = len(label_names)
    label_to_index = dict(
        (name, index) for index, name in enumerate(label_names))
    all_image_labels = [
        label_to_index[os.path.basename(os.path.dirname(path))]
        for path in all_image_paths
    ]

    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)

    autotune = tf.data.experimental.AUTOTUNE
    image_ds = path_ds.map(load_image, num_parallel_calls=autotune)

    # Loads label.
    label_ds = tf.data.Dataset.from_tensor_slices(
        tf.cast(all_image_labels, tf.int64))

    # Creates  a dataset if (image, label) pairs.
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

    tf.compat.v1.logging.info(
        'Load image with size: %d, num_label: %d, labels: %s.', all_image_size,
        all_label_size, ', '.join(label_names))
    return ImageClassifierDataLoader(image_label_ds, all_image_size,
                                     all_label_size, label_names)


class DataLoader(object):
  """This class provides generic utilities for loading customized domain data that will be used later in model retraining.

  For different ML problems or tasks, such as image classification, text
  classification etc., a subclass is provided to handle task-specific data
  loading requirements.
  """

  def __init__(self, dataset, size):
    """Init function for class `DataLoader`.

    In most cases, one should use helper functions like `from_folder` to create
    an instance of this class.

    Args:
      dataset: A tf.data.Dataset object that contains a potentially large set of
        elements, where each element is a pair of (input_data, target). The
        `input_data` means the raw input data, like an image, a text etc., while
        the `target` means some ground truth of the raw input data, such as the
        classification label of the image etc.
      size: The size of the dataset. tf.data.Dataset donesn't support a function
        to get the length directly since it's lazy-loaded and may be infinite.
    """
    self.dataset = dataset
    self.size = size

  def split(self, fraction):
    """Splits dataset into two sub-datasets with the given fraction.

    Primarily used for splitting the data set into training and testing sets.

    Args:
      fraction: float, demonstrates the fraction of the first returned
        subdataset in the original data.

    Returns:
      The splitted two sub dataset.
    """
    ds = self.dataset

    train_size = int(self.size * fraction)
    trainset = DataLoader(ds.take(train_size), train_size)

    test_size = self.size - train_size
    testset = DataLoader(ds.skip(test_size), test_size)

    return trainset, testset