import numpy as np
from skimage.measure import label

volume_data = np.random.randint(0, 2, size=(100, 100, 100)) 

labeled_volume = label(volume_data, connectivity=3)  
unique_labels, label_counts = np.unique(labeled_volume, return_counts=True)

min_volume_threshold = 100
for label, count in zip(unique_labels, label_counts):
    if count < min_volume_threshold:
        volume_data[labeled_volume == label] = 0

import numpy as np
import scipy.ndimage as ndimage

volume_data = np.random.randint(0, 2, size=(100, 100, 100))  
binary_volume = volume_data > 0  

labeled_array, num_features = ndimage.label(binary_volume)  
min_volume_threshold = 100  
filtered_volume = np.zeros_like(labeled_array)
for label in range(1, num_features + 1):
    volume = np.sum(labeled_array == label)
    if volume >= min_volume_threshold:
        filtered_volume[labeled_array == label] = 1

