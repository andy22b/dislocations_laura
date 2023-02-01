import numpy as np
from dislocations.inversions import HundaleeInversion
import pygmo as pg

# Load the data
laplacian = np.load("inputs/laplacian.npy")
edge_binary = np.load("inputs/edge_binary.npy")
lidar_gfs = np.load("inputs/lidar_gfs.npy")
oic_gfs = np.load("inputs/oic_gfs.npy")
bathy_gfs = np.load("inputs/bathy_gfs.npy")
lidar_obs = np.load("inputs/lidar_obs.npy")
oic_obs = np.load("inputs/oic_obs.npy")
bathy_obs = np.load("inputs/bathy_obs.npy")

num_patches = laplacian.shape[0]

# Set the weights
lidar_weight = 1.e4
oic_weight = 1.e3
bathy_weight = 1.0

# Set the regularization parameters
laplacian_weight = 1.e4
edge_weight = 1.0e5

# Set the initial guess
initial_ss = np.zeros((num_patches,))
initial_ds = 4. * np.ones((num_patches,))
initial_slip = np.hstack([initial_ss, initial_ds])

# Run inversion
inversion = HundaleeInversion(laplacian, edge_binary, lidar_gfs, lidar_obs,
                                oic_gfs, oic_obs, bathy_gfs, bathy_obs,
                                lidar_weight, oic_weight, bathy_weight,
                                laplacian_weight, edge_weight)


# Choose optimisation algorithm
nl = pg.nlopt('slsqp')
# Tolerance to make algorithm stop trying to improve result
nl.ftol_abs = 2E1

# Set up basin-hopping metaalgorithm
algo = pg.algorithm(uda=pg.mbh(nl, stop=1, perturb=.4))
# Lots of output to check on progress
algo.set_verbosity(1)

# set up inversion class to run algorithm on
pop = pg.population(prob=inversion)

# Tell population object what starting values will be
pop.push_back(initial_slip)

# Run algorithm
pop = algo.evolve(pop)

# Best slip distribution
preferred_slip = pop.champion_x
print(preferred_slip)
np.save("outputs/champion_slip.npy", preferred_slip)
