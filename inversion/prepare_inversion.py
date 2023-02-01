import meshio
import numpy as np
from rsqsim_api.fault.segment import RsqSimSegment
import os.path
from rsqsim_api.fault.utilities import merge_multiple_nearly_adjacent_segments

cutde_dir = "../greens_functions"
dip = 30
#Parameters
# Lidar offsets and weights
lidar_v_offset = 0.0


# OIC offsets and weights
oic_e_offset = 0.0
oic_n_offset = 0.0


# Bathymetry offsets and weights
bathy_v_offset = 0.0


# Read fault mesh
fault = RsqSimSegment.from_stl(os.path.join(cutde_dir, f"hundalee{dip}_remeshed2km.stl"))
fault.build_laplacian_matrix()

trace = merge_multiple_nearly_adjacent_segments(list(fault.trace.geoms), 200.)
np.save(f"hundalee{dip}_trace.npy", np.array(trace.coords))
edge_numbers = fault.find_edge_patch_numbers(top=False)
edge_binary = np.zeros((len(fault.patch_numbers),))
edge_binary[edge_numbers] = 1.0
edge_binary = np.hstack([edge_binary, edge_binary])
np.save("inputs/laplacian.npy", fault.laplacian)
np.save("inputs/edge_binary.npy", edge_binary)

# Read greens functions
lidar_gfs = np.load(os.path.join(cutde_dir, f"hundalee_lidar_greens_functions_{dip}dip.npy"))
np.save("inputs/lidar_gfs.npy", lidar_gfs)
oic_gfs = np.load(os.path.join(cutde_dir, f"hundalee_oic_greens_functions_{dip}dip.npy"))
np.save("inputs/oic_gfs.npy", oic_gfs)
bathy_gfs = np.load(os.path.join(cutde_dir, f"hundalee_offshore_greens_functions_{dip}dip.npy"))
np.save("inputs/bathy_gfs.npy", bathy_gfs)

# Read data to apply offsets
lidar_data = np.loadtxt("curated_data/filtered_lidar.gmt")
oic_data = np.loadtxt("curated_data/selected_oic.gmt")
bathy_data = np.loadtxt("curated_data/vertical_offshore_median_flipped.gmt")

lidar_data[:, -1] += lidar_v_offset
lidar_obs = lidar_data[:, -1]
np.save("inputs/lidar_obs.npy", lidar_obs)
oic_data[:, 2] += oic_e_offset
oic_data[:, -1] += oic_n_offset
oic_obs = np.hstack([oic_data[:, 2], oic_data[:, -1]])
np.save("inputs/oic_obs.npy", oic_obs)

bathy_data[:, -1] += bathy_v_offset
bathy_obs = bathy_data[:, -1]
np.save("inputs/bathy_obs.npy", bathy_obs)




