import numpy as np
import pandas as pd
import cutde.halfspace as HS
import os.path
import meshio
from glob import glob
from shapely.geometry import LineString, Point
import geopandas as gpd

# # Change obj to stl
# for mesh in glob("hundalee*.obj"):
#     meshio.read(mesh).write(mesh.replace("obj", "stl"))

selected_lidar = pd.read_csv("../inversion/curated_data/selected_lidar.gmt", delim_whitespace=True)
selected_oic = pd.read_csv("../inversion/curated_data/selected_oic.gmt", delim_whitespace=True, header=None, names=["x", "y", "de", "x1", "y1", "dn"])
selected_offshore = pd.read_csv("../inversion/curated_data/vertical_offshore_median_flipped.gmt", delim_whitespace=True, header=None, names=["x", "y", "disp"])

lidar_obs = np.column_stack((selected_lidar.NZTM_E.values, selected_lidar.NZTM_N.values, 0. * selected_lidar.NZTM_E.values))
oic_obs = np.column_stack((selected_oic.x.values, selected_oic.y.values, 0. * selected_oic.x.values))
offshore_obs = np.column_stack((selected_offshore.x.values, selected_offshore.y.values,
                                0. * selected_offshore.x.values))

# Turn lidar obs into linestring for filtering
lidar_linestring = LineString(lidar_obs[:, :2])
gpd.GeoSeries(lidar_linestring).to_file("lidar_linestring.gpkg", driver="GPKG")
lidar_distances = np.array([lidar_linestring.project(Point(pt)) for pt in lidar_obs[:, :2]])
filter_spacing = 100.
filter_width = 100.
filtered_distances = np.arange(0, lidar_distances.max(), filter_spacing)
filtered_lidar = []
for distance in filtered_distances:
    dist_point = lidar_linestring.interpolate(distance)
    mask = np.logical_and(lidar_distances > distance - filter_width / 2., lidar_distances < distance + filter_width / 2.)
    filtered_lidar.append([dist_point.x, dist_point.y, 0., np.median(selected_lidar.Uplift[mask])])

filtered_lidar = np.array(filtered_lidar)
np.savetxt("filtered_lidar.gmt", filtered_lidar, fmt="%.3f")

bounds = [1600000.,5280000., 1665000., 5330000.]
grid_x = np.arange(bounds[0], bounds[2], 2000)
grid_y = np.arange(bounds[1], bounds[3], 2000)
grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
grid = np.column_stack((grid_xx.flatten(), grid_yy.flatten(), 0. * grid_xx.flatten()))

small_bounds = [1636000., 5289000., 1649000., 5304000.]
small_grid_x = np.arange(small_bounds[0], small_bounds[2], 500)
small_grid_y = np.arange(small_bounds[1], small_bounds[3], 500)
small_grid_xx, small_grid_yy = np.meshgrid(small_grid_x, small_grid_y)
small_grid = np.column_stack((small_grid_xx.flatten(), small_grid_yy.flatten(), 0. * small_grid_xx.flatten()))

tiny_grid_x = np.arange(small_bounds[0], small_bounds[2], 50)
tiny_grid_y = np.arange(small_bounds[1], small_bounds[3], 50)
tiny_grid_xx, tiny_grid_yy = np.meshgrid(tiny_grid_x, tiny_grid_y)
tiny_grid = np.column_stack((tiny_grid_xx.flatten(), tiny_grid_yy.flatten(), 0. * tiny_grid_xx.flatten()))




# fault_meshes = {}
# for mesh in glob("hundalee*_remeshed2km.stl"):
#     mesh_dip = int(mesh.split("hundalee")[1].split("_remeshed2km.stl")[0])
#     fault_meshes[mesh_dip] = meshio.read(mesh)

for dip in [30, 40, 50]:
    # mesh = fault_meshes[dip]
    # vertices = mesh.points
    # triangle_nums = mesh.cells_dict["triangle"]
    # triangles = vertices[triangle_nums]
    triangles = np.load(f"hundalee{dip}_remeshed2km.npy")

    gfs_for_plotting = HS.disp_matrix(obs_pts=grid, tris=triangles,
                                      nu=0.25)
    no_tension = gfs_for_plotting[:, :, :, :2]
    np.save("../inversion/plotting/hundalee_greens_functions_grid_{}.npy".format(dip), no_tension)

    gfs_for_plotting = HS.disp_matrix(obs_pts=small_grid, tris=triangles,
                                        nu=0.25)
    no_tension = gfs_for_plotting[:, :, :, :2]
    np.save("../inversion/plotting/hundalee_greens_functions_small_grid_{}.npy".format(dip), no_tension)

    gfs_for_plotting = HS.disp_matrix(obs_pts=tiny_grid, tris=triangles,
                                        nu=0.25)
    no_tension = gfs_for_plotting[:, :, :, :2]
    np.save("../inversion/plotting/hundalee_greens_functions_tiny_grid_{}.npy".format(dip), no_tension)

    ones_array = np.ones(triangles.shape[0], dtype=np.float64)

    for label, obs in zip(["lidar", "oic", "offshore"], [filtered_lidar[:, :-1], oic_obs, offshore_obs]):
        print("Calculating Greens functions for {} at {} degrees dip".format(label, dip))

        disps = HS.disp_matrix(obs_pts=obs, tris=triangles,
                                  nu=0.25)
        ss_disps = disps[:, :, :, 0]
        ds_disps = disps[:, :, :, 1]

        if label == "oic":
            e_ss_disps = ss_disps[:, 0, :]
            n_ss_disps = ss_disps[:, 1, :]
            e_ds_disps = ds_disps[:, 0, :]
            n_ds_disps = ds_disps[:, 1, :]
            combined_gfs = np.vstack([np.hstack([e_ss_disps, n_ss_disps]),
                                      np.hstack([e_ds_disps, n_ds_disps])])
            print(label)
            print(combined_gfs.shape)
        else:
            print(label)
            print(ss_disps.shape)
            combined_gfs = np.hstack([ss_disps[:, 2, :], ds_disps[:, 2, :]])
            print(combined_gfs.shape)



        np.save("hundalee_{}_greens_functions_{}dip.npy".format(label, dip), combined_gfs)






