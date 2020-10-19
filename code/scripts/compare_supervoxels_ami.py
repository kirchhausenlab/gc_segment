import os

import zarr
from sklearn.metrics.cluster import adjusted_mutual_info_score
import numpy as np

root = '/scratch/fiborganelles/2019_09_19_TR_632019_Mito_Perox_A77C2/masks/Golgis/Golgi3/Golgi3.zarr/volumes/'

sk = zarr.open(os.path.join(root, 'superpixels_sklearn'), 'r')
itk = zarr.open(os.path.join(root, 'supervoxels_sitk_pw_25'), 'r')

# baseline, must be a cube
block_size = 20
size = 50
blocks_per_dim = int(np.ceil(size / block_size))
print(f'blocks per dim {blocks_per_dim}')
total_blocks = blocks_per_dim**3
print(f'total blocks {total_blocks}')

a = np.arange(total_blocks).reshape(
    blocks_per_dim, blocks_per_dim, blocks_per_dim)
a = np.repeat(a, block_size, 0)
a = np.repeat(a, block_size, 1)
a = np.repeat(a, block_size, 2)


print(f'baseline shape before prune {a.shape}')

baseline = a[:size, :size, :size]
print(f'baseline shape after prune {baseline.shape}')

score = adjusted_mutual_info_score(
    sk[50:100, 150:200, 200:250].flatten(),
    # itk[50:100, 150:200, 200:250].flatten()
    baseline.flatten()
)

print(score)
