import fvdb
import torch

print("fvdb file:", fvdb.__file__)
print("fvdb version:", getattr(fvdb, "__version__", "unknown"))

pts = fvdb.JaggedTensor([torch.randn(100, 3, device="cuda")])
grid = fvdb.GridBatch.from_points(pts, voxel_sizes=0.1)

print("grid_count:", grid.grid_count)
print("total_voxels:", grid.total_voxels)
print("workflow test done")