import numpy as np

def sliding_window_3d(data, patch_size=(96, 96, 96), step=(1, 1, 1), threshold=0.05):
    """
    Perform sliding window scanning on a 3D numpy array.

    Parameters:
    - data: 3D numpy array with shape (depth, height, width)
    - patch_size: Tuple specifying the size of the patch (depth, height, width)
    - step: Tuple specifying the step for sliding (depth, height, width)

    Returns:
    - List of patches extracted from the input data
    """

    depth, height, width = data.shape
    d_step, h_step, w_step = step
    d_patch, h_patch, w_patch = patch_size

    patches = []

    for d in range(0, depth - d_patch + 1, d_step):
        for h in range(0, height - h_patch + 1, h_step):
            for w in range(0, width - w_patch + 1, w_step):
                patch = data[d:d + d_patch, h:h + h_patch, w:w + w_patch]
                
                if np.sum(patch >0.5) > np.prod(patch.shape)* threshold:
                    patches.append(patch)
                

    return patches

# Example usage:
# Assuming you have a 3D numpy array 'data' with shape (512, 512, 200)
# and you want to extract patches of size (96, 96, 96) with a step of (32, 32, 32)
data = np.random.random((512, 512, 200))  # Replace this with your actual data
patches = sliding_window_3d(data, patch_size=(96, 96, 96), step=(32, 32, 32), threshold=0.9)
print(len(patches))
# 'patches' now contains a list of patches extracted from the input data






