from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import meshio
from rsqsim_api.visualisation.utilities import plot_coast

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

mesh = meshio.read("../greens_functions/hundalee30_remeshed2km.stl")
mesh_tris = mesh.cells_dict["triangle"]
mesh_points = mesh.points

champion_slip = np.load("outputs/champion_slip.npy")
initial_slip = np.load("outputs/initial_slip.npy")
# champion_slip = np.load("outputs/initial_slip.npy")
champion_ss = champion_slip[:champion_slip.shape[0]//2]
champion_ds = champion_slip[champion_slip.shape[0]//2:]
initial_ss = initial_slip[:initial_slip.shape[0]//2]
initial_ds = initial_slip[initial_slip.shape[0]//2:]

edge_triangles = np.load("inputs/edge_binary.npy")
edge_triangles = edge_triangles[:edge_triangles.shape[0]//2]

champion_combined = np.linalg.norm(np.column_stack([champion_ss, champion_ds]), axis=1)

gf_grid = np.load("plotting/hundalee_greens_functions_grid_30.npy")
vert_gfs = gf_grid[:, 2, :, :]
vert_disps = np.matmul(vert_gfs[:, :, 0], champion_ss) + np.matmul(vert_gfs[:, :, 1], champion_ds)
vert_disps = vert_disps.reshape(grid_xx.shape)
east_gfs = gf_grid[:, 0, :, :]
east_disps = np.matmul(east_gfs[:, :, 0], champion_ss) + np.matmul(east_gfs[:, :, 1], champion_ds)
north_gfs = gf_grid[:, 1, :, :]
north_disps = np.matmul(north_gfs[:, :, 0], champion_ss) + np.matmul(north_gfs[:, :, 1], champion_ds)

small_gf_grid = np.load("plotting/hundalee_greens_functions_small_grid_30.npy")
small_vert_gfs = small_gf_grid[:, 2, :, :]
small_vert_disps = np.matmul(small_vert_gfs[:, :, 0], champion_ss) + np.matmul(small_vert_gfs[:, :, 1], champion_ds)
small_vert_disps = small_vert_disps.reshape(small_grid_xx.shape)
small_east_gfs = small_gf_grid[:, 0, :, :]
small_east_disps = np.matmul(small_east_gfs[:, :, 0], champion_ss) + np.matmul(small_east_gfs[:, :, 1], champion_ds)
small_north_gfs = small_gf_grid[:, 1, :, :]
small_north_disps = np.matmul(small_north_gfs[:, :, 0], champion_ss) + np.matmul(small_north_gfs[:, :, 1], champion_ds)

tiny_gf_grid = np.load("plotting/hundalee_greens_functions_tiny_grid_30.npy")
tiny_vert_gfs = tiny_gf_grid[:, 2, :, :]
tiny_vert_disps = np.matmul(tiny_vert_gfs[:, :, 0], champion_ss) + np.matmul(tiny_vert_gfs[:, :, 1], champion_ds)
tiny_vert_disps = tiny_vert_disps.reshape(tiny_grid_xx.shape)

# Read in observations
lidar_obs = np.loadtxt("curated_data/filtered_lidar.gmt")
oic_obs = np.loadtxt("curated_data/selected_oic.gmt")
offshore_obs = np.loadtxt("curated_data/vertical_offshore_median_flipped.gmt")

# Read in greens functions
lidar_gfs = np.load("inputs/lidar_gfs.npy")
oic_gfs = np.load("inputs/oic_gfs.npy")
offshore_gfs = np.load("inputs/bathy_gfs.npy")

# Model deformation from greens functions
lidar_deform = np.matmul(lidar_gfs, champion_slip)
oic_deform = np.matmul(oic_gfs, champion_slip)
offshore_deform = np.matmul(offshore_gfs, champion_slip)

# Read in trace
trace = np.load("hundalee30_trace.npy")


fig, ax = plt.subplots(2, 3, figsize=(8, 5), sharex="row", sharey="row")
ax[0, 0].set_title("Wider region disps with fault mesh", fontsize=7)
ax[0, 0].imshow(vert_disps[-1::-1,:], extent=[bounds[0], bounds[2], bounds[1], bounds[3]], cmap=cm.coolwarm, vmin=-3, vmax=3.)
ax[0, 0].triplot(mesh_points[:, 0], mesh_points[:, 1], mesh_tris, color="black", linewidth=0.5)
ax[0, 0].quiver(grid_xx, grid_yy, east_disps, north_disps, scale=20, color="black")
im2 = ax[0, 1].tripcolor(mesh_points[:, 0], mesh_points[:, 1], mesh_tris, facecolors=champion_combined, cmap=cm.magma, vmin=0., vmax=5.)
ax[0, 1].set_title("Slip distribution", fontsize=7)
ax[0, 1].set_aspect("equal")
ax[1, 0].imshow(tiny_vert_disps[-1::-1,:], extent=[small_bounds[0], small_bounds[2], small_bounds[1], small_bounds[3]], cmap=cm.coolwarm, vmin=-3, vmax=3.)
ax[1, 0].set_title("Region of interest disps", fontsize=7)
ax[1, 0].quiver(small_grid_xx, small_grid_yy, small_east_disps, small_north_disps, scale=10, color="black")
ax[1, 0].set_aspect("equal")
ax[0, 2].set_aspect("equal")
ax[0, 2].set_axis_off()
ax[1, 1].scatter(lidar_obs[:, 0], lidar_obs[:, 1], c=lidar_obs[:, -1], s=20, cmap=cm.coolwarm, vmin=-3, vmax=3.)
ax[1, 1].set_title("Observations for coast lidar, OIC, bathy", fontsize=7)
ax[1, 1].scatter(offshore_obs[:, 0], offshore_obs[:, 1], c=offshore_obs[:, -1], s=20, cmap=cm.coolwarm, vmin=-3, vmax=3.)
ax[1, 1].quiver(oic_obs[:, 0], oic_obs[:, 1], oic_obs[:, 2], oic_obs[:, -1], scale=20, color="black")
ax[1, 2].scatter(lidar_obs[:, 0], lidar_obs[:, 1], c=lidar_deform, s=20, cmap=cm.coolwarm, vmin=-3, vmax=3.)
im = ax[1, 2].scatter(offshore_obs[:, 0], offshore_obs[:, 1], c=offshore_deform, s=20, cmap=cm.coolwarm, vmin=-3, vmax=3.)
ax[1, 2].quiver(oic_obs[:, 0], oic_obs[:, 1], oic_deform[:oic_deform.shape[0]//2], oic_deform[oic_deform.shape[0]//2:], scale=20, color="black")
ax[1, 2].set_title("Model for coast lidar, OIC, bathy", fontsize=7)
ax[1, 1].set_aspect("equal")
ax[1, 2].set_aspect("equal")

for ax_i in ax[0, :-1]:
    plot_coast(ax_i, clip_boundary=bounds)
    ax_i.plot(trace[:, 0], trace[:, 1], color="red", linewidth=0.5)
    ax_i.set_xlim(bounds[0], bounds[2])
    ax_i.set_ylim(bounds[1], bounds[3])

for ax_i in ax[1, :]:
    plot_coast(ax_i, clip_boundary=small_bounds)
    ax_i.plot(trace[:, 0], trace[:, 1], color="red", linewidth=0.5)
    ax_i.set_xlim(small_bounds[0], small_bounds[2])
    ax_i.set_ylim(small_bounds[1], small_bounds[3])

for ax_i in ax.flatten():
    ax_i.set_xticks([])
    ax_i.set_yticks([])




fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.1, 0.03, 0.35])
cbar_ax2 = fig.add_axes([0.85, 0.6, 0.03, 0.2])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar2 = fig.colorbar(im2, cax=cbar_ax2)
cbar.set_label("Vertical displacement (m)")
cbar2.set_label("Slip (m)")
plt.savefig("crap_inversion_results.png", dpi=300, bbox_inches="tight")





