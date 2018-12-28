import numpy as np

# Dataset names.
_CLOUDS = 'clouds'

# Max number of entries in the colormap for each dataset.
_DATASET_MAX_ENTRIES = {
    _CLOUDS: 4,
}


def create_clouds_label_colormap():
  """Creates a label colormap used in ADE20K segmentation benchmark.
  Returns:
    A colormap for visualizing segmentation results.
  """
  return np.asarray([
      [0, 0, 0],
      [0, 191, 255],
      [103, 103, 103],
      [231, 231, 231],
  ])


def get_clouds_name():
  return _CLOUDS

def bit_get(val, idx):
  """Gets the bit value.
  Args:
    val: Input value, int or numpy int array.
    idx: Which bit of the input val.
  Returns:
    The "idx"-th bit of input val.
  """
  return (val >> idx) & 1


def create_label_colormap(dataset=_CLOUDS):
  """Creates a label colormap for the specified dataset.
  Args:
    dataset: The colormap used in the dataset.
  Returns:
    A numpy array of the dataset colormap.
  Raises:
    ValueError: If the dataset is not supported.
  """
  if dataset == _CLOUDS:
    return create_clouds_label_colormap()
  else:
    raise ValueError('Unsupported dataset.')


def label_to_color_image(label, dataset=_CLOUDS):
  """Adds color defined by the dataset colormap to the label.
  Args:
    label: A 2D array with integer type, storing the segmentation label.
    dataset: The colormap used in the dataset.
  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the dataset color map.
  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  if np.max(label) >= _DATASET_MAX_ENTRIES[dataset]:
    raise ValueError('label value too large.')

  colormap = create_label_colormap(dataset)
  return colormap[label]
