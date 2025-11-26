#%%
from dataclasses import (
    dataclass,
    field,
)
from jax import (
    config,
    hessian,
)
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import (
    Array,
    Float,
    install_import_hook,
)
from gpjax.kernels.computations import DenseKernelComputation
import jax
from jax.scipy.linalg import expm
from functools import partial 
from jax import lax 
import sys 
import numpy as np 
import scipy
import scipy.linalg as la
from jax.scipy.linalg import solve_triangular
config.update("jax_enable_x64", True)

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx

def prepare_spatiotemporal_data(df):
    """
    Example placeholder for reading (lon, lat, time) from a DataFrame
    along with velocity (u, v).
    """
    pos = jnp.array([df["lon"], df["lat"], df["time"]])  # shape = (3, N)
    vel = jnp.array([df["ubar"], df["vbar"]])            # shape = (2, N)
    return pos, vel

def label_position_spatiotemporal(pos_3d: jnp.ndarray) -> jnp.ndarray:
    """
    Convert (3, N) -> (2N, 4). The last dimension is the z-label in {0,1}.
    The first 3 columns are (x, y, t).
    """
    n_points = pos_3d.shape[1]
    # Repeated label [0,1,0,1,...] of length 2*N
    label = jnp.tile(jnp.array([0.0, 1.0]), n_points)
    # Repeat (x,y,t) across the second axis, then stack the label
    return jnp.vstack((jnp.repeat(pos_3d, repeats=2, axis=1), label)).T

def unlabel_position_spatiotemporal(X_labelled: jnp.ndarray) -> jnp.ndarray:
    """
    Revert from (2N, 4) back to (3, N) by ignoring every other row
    and discarding the last z-label dimension.
    """
    # X_labelled has shape (2N, 4) -> the columns are (x, y, t, label)
    data = X_labelled[:, :3]   # (2N, 3)
    labels = X_labelled[:, 3]  # (2N, )
    
    # Keep only rows where label == 0.0, i.e. the first derivative index
    original_data = data[labels == 0.0]
    return original_data.T     # shape (3, N)

def stack_velocity(vel_2d: jnp.ndarray) -> jnp.ndarray:
    """
    Convert 2D velocity (2, N) into a single column vector (2N, 1).
    """
    return vel_2d.T.flatten().reshape(-1, 1)

def dataset_4d(pos_3d: jnp.ndarray, vel_2d: jnp.ndarray):
    """
    Create a GPJax Dataset with spatio-temporal inputs (x,y,t)
    and 2D velocity outputs (u,v).
    """
    X_labelled = label_position_spatiotemporal(pos_3d)   # (2N, 4)
    Y_stacked = stack_velocity(vel_2d)                  # (2N, 1)
    return gpx.Dataset(X_labelled, Y_stacked)

def dataset_5d(pos_3d: jnp.ndarray, vel_2d: jnp.ndarray):
    """
    Create a GPJax Dataset with spatio-temporal inputs (x,y,t)
    and 2D velocity outputs (u,v).
    """
    X_labelled = label_position_spatiotemporal(pos_3d)   # (2N, 4)

    X_labelled_full = jnp.hstack([X_labelled, jnp.repeat(0, X_labelled.shape[0]).reshape(-1,1)])

    Y_stacked = stack_velocity(vel_2d)                  # (2N, 1)
    return gpx.Dataset(X_labelled_full, Y_stacked)

def cov_ellipse(cov):
    eigenvalues, eigenvectors = jnp.linalg.eigh(cov)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    axis_lengths = jnp.sqrt(eigenvalues)
    theta = jnp.linspace(0, 2 * jnp.pi, 500)
    circle = jnp.array([jnp.cos(theta), jnp.sin(theta)])  # Shape (2, N)
    ellipse = eigenvectors @ (axis_lengths[:, None] * circle)  # Shape (2, N)
    return ellipse

def optimise_mll(posterior, dataset, key, NIters=1000):
    objective = lambda p, d: -gpx.objectives.conjugate_mll(p, d)
    opt_posterior, history = gpx.fit_scipy(
        model=posterior,
        objective=objective,
        train_data=dataset,
    )
    return opt_posterior

def latent_distribution(opt_posterior, X_test, dataset_train):
    """
    Predict at spatio-temporal test inputs, shape (2N_test, 4).
    """
    latent = opt_posterior.predict(X_test, train_data=dataset_train)
    return latent.mean, latent.covariance()


#%%
from kernel import Matern32new

@dataclass
class HelmholtzKernel(gpx.kernels.stationary.StationaryKernel):
    """
    Spatio-temporal Helmholtz kernel: Spatial RBF for (x, y) and temporal Matern 3/2 for (t).
    Partial derivatives are only computed for spatial dimensions (x, y).
    """
    potential_kernel: gpx.kernels.stationary.StationaryKernel = field(
        default_factory=lambda: gpx.kernels.RBF(active_dims=[0, 1])  # Spatial only
    )
    stream_kernel: gpx.kernels.stationary.StationaryKernel = field(
        default_factory=lambda: gpx.kernels.RBF(active_dims=[0, 1])  # Spatial only
    )
    compute_engine = DenseKernelComputation()

    def __call__(self, X: jnp.ndarray, Xp: jnp.ndarray) -> jnp.ndarray:
        """
        X, Xp each have shape (4, ) = (x, y, t, z-label).
        The spatial kernel acts on (x, y) dimensions, while a separate Matern 3/2 kernel
        acts on the time dimension (t). Partial derivatives are computed for (x, y) only.
        """
        z = jnp.array(X[3], dtype=int)   # z in {0,1}
        zp = jnp.array(Xp[3], dtype=int) # zp in {0,1}
        sign = (-1) ** (z + zp)

        # Extract spatial (x, y) and temporal (t) components
        X_spatial = X[:2]    # (x, y)
        Xp_spatial = Xp[:2]  # (x', y')

        # Compute the Hessians for the spatial kernel (2 x 2 matrix)
        pot_hess = -jnp.array(hessian(self.potential_kernel)(X_spatial, Xp_spatial), dtype=jnp.float64)
        str_hess = -jnp.array(hessian(self.stream_kernel)(X_spatial, Xp_spatial), dtype=jnp.float64)

        # Select the appropriate spatial derivatives
        potential_dvtve = pot_hess[z, zp] 
        stream_dvtve = str_hess[1 - z, 1 - zp] 

        # Combine spatial and temporal components
        return potential_dvtve + sign * stream_dvtve
    
@dataclass
class HelmholtzKernelSpatioTemporal(gpx.kernels.stationary.StationaryKernel):
    """
    Spatio-temporal Helmholtz kernel: Spatial RBF for (x, y) and temporal Matern 3/2 for (t).
    Partial derivatives are only computed for spatial dimensions (x, y).
    """
    potential_kernel: gpx.kernels.stationary.StationaryKernel = field(
        default_factory=lambda: gpx.kernels.RBF(active_dims=[0, 1])  # Spatial only
    )
    stream_kernel: gpx.kernels.stationary.StationaryKernel = field(
        default_factory=lambda: gpx.kernels.RBF(active_dims=[0, 1])  # Spatial only
    )
    time_kernel: gpx.kernels.stationary.StationaryKernel = field(
        default_factory=lambda: gpx.kernels.Matern32(active_dims=[2])  # Time only
    )
    compute_engine = DenseKernelComputation()

    def __call__(self, X: jnp.ndarray, Xp: jnp.ndarray) -> jnp.ndarray:
        """
        X, Xp each have shape (4, ) = (x, y, t, z-label).
        The spatial kernel acts on (x, y) dimensions, while a separate Matern 3/2 kernel
        acts on the time dimension (t). Partial derivatives are computed for (x, y) only.
        """
        z = jnp.array(X[3], dtype=int)   # z in {0,1}
        zp = jnp.array(Xp[3], dtype=int) # zp in {0,1}
        sign = (-1) ** (z + zp)

        # Extract spatial (x, y) and temporal (t) components
        X_spatial = X[:2]    # (x, y)
        Xp_spatial = Xp[:2]  # (x', y')

        # Compute the Hessians for the spatial kernel (2 x 2 matrix)
        pot_hess = -jnp.array(hessian(self.potential_kernel)(X_spatial, Xp_spatial), dtype=jnp.float64)
        str_hess = -jnp.array(hessian(self.stream_kernel)(X_spatial, Xp_spatial), dtype=jnp.float64)

        # Compute the time kernel
        X_spacetime = X[:3]    # shape (3,) = (x, y, t)
        Xp_spacetime = Xp[:3]
        time_cov = self.time_kernel(X_spacetime, Xp_spacetime)

        # Select the appropriate spatial derivatives
        potential_dvtve = pot_hess[z, zp] * time_cov
        stream_dvtve = str_hess[1 - z, 1 - zp] * time_cov

        # Combine spatial and temporal components
        return potential_dvtve + sign * stream_dvtve
    
@dataclass
class HelmholtzKernelSpatioTemporal_derivative(gpx.kernels.stationary.StationaryKernel):
    """
    Spatio-temporal Helmholtz kernel: Spatial RBF for (x, y) and temporal Matern 3/2 for (t).
    Partial derivatives are only computed for spatial dimensions (x, y).
    """
    potential_kernel: gpx.kernels.stationary.StationaryKernel = field(
        default_factory=lambda: gpx.kernels.RBF(active_dims=[0, 1])  # Spatial only
    )
    stream_kernel: gpx.kernels.stationary.StationaryKernel = field(
        default_factory=lambda: gpx.kernels.RBF(active_dims=[0, 1])  # Spatial only
    )
    time_kernel: gpx.kernels.stationary.StationaryKernel = field(
        default_factory=lambda: gpx.kernels.Matern32(active_dims=[2])  # Time only
    )
    compute_engine = DenseKernelComputation()

    def __call__(self, X: jnp.ndarray, Xp: jnp.ndarray) -> jnp.ndarray:
        """
        X, Xp each have shape (4, ) = (x, y, t, z-label).
        The spatial kernel acts on (x, y) dimensions, while a separate Matern 3/2 kernel
        acts on the time dimension (t). Partial derivatives are computed for (x, y) only.
        """
        z = jnp.array(X[3], dtype=int)   # z in {0,1}
        zp = jnp.array(Xp[3], dtype=int) # zp in {0,1}
        w = jnp.array(X[4], dtype=int)
        wp = jnp.array(Xp[4], dtype=int)
        sign = (-1) ** (z + zp)

        # Extract spatial (x, y) and temporal (t) components
        X_spatial = X[:2]    # (x, y)
        Xp_spatial = Xp[:2]  # (x', y')
        X_time = X[2]        # t
        Xp_time = Xp[2]      # t'

        # Compute the Hessians for the spatial kernel (2 x 2 matrix)
        pot_hess = -jnp.array(hessian(self.potential_kernel)(X_spatial, Xp_spatial), dtype=jnp.float64)
        str_hess = -jnp.array(hessian(self.stream_kernel)(X_spatial, Xp_spatial), dtype=jnp.float64)

        # Compute the time kernel
        X_spacetime = X[:3]    # shape (3,) = (x, y, t)
        Xp_spacetime = Xp[:3]
        time_cov = self.time_kernel(X_spacetime, Xp_spacetime)
        grad_val = jnp.array(jax.grad(self.time_kernel, argnums=(0,1))(X_spacetime, Xp_spacetime))[:,2]
        hess_val = jnp.array(jax.hessian(self.time_kernel,argnums=(0,1))(X_spacetime, Xp_spacetime))[0,1,2,2]
        switch1 = (1-w)*(1-wp)
        switch2 = (w + wp) % 2
        switch3 = w*wp
        time_k = switch1 * time_cov + switch2 * grad_val[wp] +  switch3 * hess_val

        # Select the appropriate spatial derivatives
        potential_dvtve = pot_hess[z, zp] * time_k
        stream_dvtve = str_hess[1 - z, 1 - zp] * time_k

        # Combine spatial and temporal components
        return potential_dvtve + sign * stream_dvtve

#%%
data = jnp.array(np.loadtxt("suntans.csv", delimiter=","))
dataset_ground_truth = dataset_4d(data[0:3,:], data[3:5,:])


x1 = data[0,0:441]
x2 = data[1,0:441]
space_grid = jnp.vstack([x1,x2]).T

range_x = [float(jnp.min(x1)), float(jnp.max(x1))]
range_y = [float(jnp.min(x2)), float(jnp.max(x2))]

space_grid_3d = jnp.vstack([jnp.hstack([space_grid, jnp.repeat(0,space_grid.shape[0]).reshape(-1,1)]), jnp.hstack([space_grid, jnp.repeat(1,space_grid.shape[0]).reshape(-1,1)])])

N = 21
t_grid = jnp.unique(data[2,:])
dt = t_grid[2] - t_grid[1]
time_final = float(jnp.max(data[2,:]))
obs_noise = 0.1

def initialise_gp(kernel, mean, dataset):
    prior = gpx.gps.Prior(mean_function=mean, kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian(
        num_datapoints=dataset.n, obs_stddev=jnp.array([obs_noise], dtype=jnp.float64)
    )
    posterior = prior * likelihood
    return posterior

sigma2 = 15.0      # process variance
l = 1.0       # length-scale
lambda_val = jnp.sqrt(3.0) / l

# Define the state-space matrices (companion form)
F = jnp.array([[0.0, 1.0],
               [-lambda_val**2, -2.0 * lambda_val]])
L = jnp.array([[0.0],
               [1.0]])
H = jnp.array([[1.0, 0.0]])
P_inf = jnp.diag(jnp.array([sigma2, lambda_val**2 * sigma2]))

N = 21
t_grid = jnp.unique(data[2,:])
dt = t_grid[2] - t_grid[1]
time_final = float(jnp.max(data[2,:]))
obs_noise = 0.1

Phi = la.expm(F * dt)
Q = P_inf - Phi @ P_inf @ Phi.T
        
space_kernel = HelmholtzKernel(potential_kernel=gpx.kernels.RBF(variance=20,lengthscale=5), stream_kernel=gpx.kernels.RBF(variance=0.01,lengthscale=4))
cov_matrix = space_kernel.gram(space_grid_3d).to_dense() + jnp.eye(space_grid_3d.shape[0])*1e-10

# ----- precompute small/big factors -----
# Cholesky factors (use lower=True for conventional LL^T)
L_cov = la.cholesky(cov_matrix, lower=True)                # (2N^2, 2N^2)
L_Pinf = la.cholesky(P_inf, lower=True)                    # (2, 2)
L_Q = la.cholesky(Q, lower=True)                           # (2, 2)

n_blocks = cov_matrix.shape[0]                              # = 2*N^2

# ----- helper: sample vec ~ N(0, cov ⊗ S) WITHOUT building kron -----
# Using: vec(B * E * A^T) ~ N(0, (A A^T) ⊗ (B B^T))
def sample_kron_cov(A_lower, B_lower, subkey):
    # A_lower: (n x n) such that A_lower A_lower^T = cov
    # B_lower: (2 x 2) such that B_lower B_lower^T = S   
    E = jr.normal(subkey,(n_blocks, 2)).T# iid standard normal
    M = B_lower @ E @ A_lower.T                             # shape (2, n_blocks)
    return M                                               # return as 2 x n_blocks matrix

obs_noise = 0.1

def initialise_gp(kernel, mean, dataset):
    prior = gpx.gps.Prior(mean_function=mean, kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian(
        num_datapoints=dataset.n, obs_stddev=jnp.array([1.0e-3], dtype=jnp.float64)
    )
    posterior = prior * likelihood
    return posterior
#%%
# get drifter observations 
def find_loc_idx(position, target):
    differences = jnp.linalg.norm(position - target.flatten(), axis=1)
    closest_index = jnp.argmin(differences)
    return closest_index

def extract_velocity_via_time(dataset, t_grid, time):
    t = t_grid[jnp.isclose(t_grid, time)]
    return dataset.y[dataset.X[:,2] == t].reshape(-1,2)

def extract_space_via_time(dataset, t_grid, time):
    t = t_grid[jnp.isclose(t_grid, time)]
    return dataset.X[dataset.X[:,2] == t,0:2].reshape(-1,2)

def initialise_drifter_traj(seed, dataset_ground_truth, space_grid, t_grid, start_loc_idx, start_time_idx, term_time_idx, obs_noise, drifter_id):
    t_grid_sample = jnp.unique(dataset_ground_truth.X[:,2])

    start_time = t_grid[start_time_idx]
    start_loc = space_grid[start_loc_idx]
    start_vField = extract_velocity_via_time(dataset_ground_truth, t_grid_sample, start_time)
    start_vel = start_vField[start_loc_idx]

    curr_loc = start_loc
    curr_time = start_time
    curr_vel = start_vel

    traj_pos = jnp.concatenate([curr_loc, jnp.array([curr_time]),jnp.array([drifter_id])])
    traj_vel = jnp.concatenate([curr_vel + jr.normal(seed,(2,))*obs_noise, jnp.array([curr_time]),jnp.array([drifter_id])])

    for t_idx in range(start_time_idx+1, term_time_idx):
        seed, seed1 = jr.split(seed)

        new_loc = curr_loc + dt * curr_vel
        new_time = t_grid[t_idx]
        new_loc_idx = find_loc_idx(space_grid, new_loc)
        new_vel = extract_velocity_via_time(dataset_ground_truth, t_grid_sample, new_time)[new_loc_idx]
        if (range_x[0] <= new_loc[0] <= range_x[1] and range_y[0] <= new_loc[1] <= range_y[1]):    
            if (t_idx-start_time_idx) % 5 == 0:
                traj_pos = jnp.vstack([traj_pos, jnp.concatenate([new_loc, jnp.array([new_time]),jnp.array([drifter_id])])])
                traj_vel = jnp.vstack([traj_vel, jnp.concatenate([new_vel+ jr.normal(seed,(2,))*obs_noise, jnp.array([new_time]),jnp.array([drifter_id])])])

            curr_loc = new_loc
            curr_time = new_time 
            curr_vel = new_vel
        else:
            break

    return traj_pos, traj_vel

def build_velocity_lookup(dataset, t_grid):
    shape_v = dataset.y[jnp.isclose(dataset.X[:, 2], jnp.unique(dataset.X[:,2])[0])].reshape(-1, 2).shape
    lookup = []
    for t in t_grid:
        mask = jnp.isclose(dataset.X[:, 2], t)
        if jnp.sum(mask) > 0:
            velocities = dataset.y[mask].reshape(-1, 2)
        else:
            velocities = jnp.zeros(shape=shape_v)
        lookup.append(velocities)
    return jnp.stack(lookup)

def initialise_drifter_traj_optimized(
    seed, velocity_lookup, space_grid, t_grid,
    start_loc_idx, start_time_idx, term_time_idx,
    obs_noise, drifter_id
):
    start_time = t_grid[start_time_idx]
    start_loc = space_grid[start_loc_idx]
    start_vel = velocity_lookup[start_time_idx, start_loc_idx]
    
    max_steps = int((term_time_idx - start_time_idx)/5)
    
    init_pos = jnp.concatenate([start_loc, jnp.array([start_time]), jnp.array([drifter_id])])
    init_vel = jnp.concatenate([
        start_vel + jr.normal(seed, (2,)) * obs_noise,
        jnp.array([start_time]),
        jnp.array([drifter_id])
    ])

    def step(t_idx, carry):
        seed, curr_loc, curr_vel, step_count, traj_pos, traj_vel = carry

        seed, seed1 = jr.split(seed)
        new_time = t_grid[t_idx]
        new_loc = curr_loc + dt * curr_vel
        new_loc_idx = find_loc_idx(space_grid, new_loc)
        new_vel = velocity_lookup[t_idx, new_loc_idx]

        in_bounds = (range_x[0] <= new_loc[0]) & (new_loc[0] <= range_x[1]) & \
                    (range_y[0] <= new_loc[1]) & (new_loc[1] <= range_y[1])
        
        should_record = ((t_idx - start_time_idx) % 5 == 0)

        def update_traj(_):
            new_pos = jnp.concatenate([new_loc, jnp.array([new_time]), jnp.array([drifter_id])])
            new_noisy_vel = jnp.concatenate([
                new_vel + jr.normal(seed1, (2,)) * obs_noise,
                jnp.array([new_time]),
                jnp.array([drifter_id])
            ])
            updated_traj_pos = traj_pos.at[step_count].set(new_pos)
            updated_traj_vel = traj_vel.at[step_count].set(new_noisy_vel)
            return (seed1, new_loc, new_vel, step_count + 1, updated_traj_pos, updated_traj_vel)

        def skip_recording(_):
            return (seed1, new_loc, new_vel, step_count, traj_pos, traj_vel)

        def handle_bounds(_):
            return lax.cond(should_record, update_traj, skip_recording, operand=None)

        return lax.cond(in_bounds, handle_bounds, lambda _: carry, operand=None)

    # Preallocate memory
    traj_pos = jnp.zeros((max_steps, 4))
    traj_vel = jnp.zeros((max_steps, 4))
    traj_pos = traj_pos.at[0].set(init_pos)
    traj_vel = traj_vel.at[0].set(init_vel)

    init_state = (seed, start_loc, start_vel, 1, traj_pos, traj_vel)

    final_state = lax.fori_loop(start_time_idx + 1, term_time_idx, step, init_state)
    _, _, _, step_count, traj_pos, traj_vel = final_state

    return traj_pos, traj_vel, step_count

def continue_drifter_traj(seed, dataset_ground_truth, space_grid, t_grid, start_loc_idx, start_time_idx, term_time_idx, obs_noise, drifter_id):
    t_grid_sample = jnp.unique(dataset_ground_truth.X[:,2])

    start_time = t_grid[start_time_idx]
    start_loc = space_grid[start_loc_idx]
    start_vField = extract_velocity_via_time(dataset_ground_truth, t_grid_sample, start_time)
    start_vel = start_vField[start_loc_idx]

    curr_loc = start_loc
    curr_time = start_time
    curr_vel = start_vel

    traj_pos = jnp.concatenate([curr_loc, jnp.array([curr_time]),jnp.array([drifter_id])])
    traj_vel = jnp.concatenate([curr_vel + jr.normal(seed,(2,))*obs_noise, jnp.array([curr_time]),jnp.array([drifter_id])])

    # move one step and see if we are still within boundary
    new_loc = curr_loc + dt * curr_vel
    new_time = t_grid[start_time_idx+1]
    new_loc_idx = find_loc_idx(space_grid, new_loc)
    new_vel = extract_velocity_via_time(dataset_ground_truth, t_grid_sample, new_time)[new_loc_idx]

    if (range_x[0] <= new_loc[0] <= range_x[1] and range_y[0] <= new_loc[1] <= range_y[1]):
        curr_loc = new_loc
        curr_time = new_time 
        curr_vel = new_vel
    else:
        return None, None

    for t_idx in range(start_time_idx+2, term_time_idx):
        seed, seed1 = jr.split(seed)

        new_loc = curr_loc + dt * curr_vel
        new_time = t_grid[t_idx]
        new_loc_idx = find_loc_idx(space_grid, new_loc)
        new_vel = extract_velocity_via_time(dataset_ground_truth, t_grid_sample, new_time)[new_loc_idx]
        if (range_x[0] <= new_loc[0] <= range_x[1] and range_y[0] <= new_loc[1] <= range_y[1]):    
            if (t_idx-start_time_idx) % 5 == 0:
                traj_pos = jnp.vstack([traj_pos, jnp.concatenate([new_loc, jnp.array([new_time]),jnp.array([drifter_id])])])
                traj_vel = jnp.vstack([traj_vel, jnp.concatenate([new_vel+ jr.normal(seed,(2,))*obs_noise, jnp.array([new_time]),jnp.array([drifter_id])])])

            curr_loc = new_loc
            curr_time = new_time 
            curr_vel = new_vel
        else:
            break

    if traj_pos.reshape(-1,4).shape[0] > 1:
        return traj_pos[1::,:], traj_vel[1::,:]
    else:
        return None, None
    

def generate_5d_test_location(space_grid, time_now):
    extended_space_grid = jnp.repeat(space_grid, 2, axis=0)
    return jnp.vstack([jnp.hstack([extended_space_grid, jnp.repeat(time_now, extended_space_grid.shape[0]).reshape(-1,1),jnp.repeat(0, extended_space_grid.shape[0]).reshape(-1,1), jnp.tile(jnp.array([0,1]), int(extended_space_grid.shape[0]/2)).reshape(-1,1)]) , jnp.hstack([extended_space_grid, jnp.repeat(time_now, extended_space_grid.shape[0]).reshape(-1,1),jnp.repeat(1, extended_space_grid.shape[0]).reshape(-1,1), jnp.tile(jnp.array([0,1]), int(extended_space_grid.shape[0]/2)).reshape(-1,1)]) ])

#%%
N_space = space_grid.shape[0]
seednum= int(sys.argv[1])
seed = jr.key(seednum)
obs_noise = 0.1

drifter_num = 1
time_now = 0
deploy_gap = 0.5
t_choice = int(sys.argv[2])
deploy_num = int(t_choice / deploy_gap - 1)
start_time_idx = int(jnp.where(t_grid == time_now)[0][0])
term_time_idx = int(jnp.where(t_grid == (deploy_gap+time_now))[0][0])
initial_loc_indices = jr.choice(seed, N_space, (drifter_num,))

traj_pos_full = []
traj_vel_full = []
bias_list = []
uncertainty_list = []
deployment_data = []

for drifter_id in range(drifter_num):
    seed, seed1 = jr.split(seed)
    start_loc_idx = initial_loc_indices[drifter_id]
    traj_pos, traj_vel = initialise_drifter_traj(seed, dataset_ground_truth, space_grid, t_grid, start_loc_idx, start_time_idx, term_time_idx, obs_noise, drifter_id)
    if traj_pos is not None:
        traj_pos_full.append(traj_pos)
        traj_vel_full.append(traj_vel)

traj_pos_full = jnp.vstack(traj_pos_full)
traj_vel_full = jnp.vstack(traj_vel_full)

traj_data = dataset_4d(traj_pos_full.T[0:3,:], traj_vel_full.T[0:2,:])
traj_data_raw = jnp.column_stack([traj_pos_full[:,0:2], traj_vel_full]).T
time_now = time_now + deploy_gap


for deploy in range(deploy_num):
    seed, seed1 = jr.split(seed)

    # acquisition
    #acq_list = full_system_EIG_BALLAST(traj_pos_full, traj_vel_full, space_grid, time_now, BALLAST_samples=10)
    next_obs_loc_idx = jr.choice(seed, N_space, (1,))[0] #jnp.argmax(acq_list)

    # next deployment
    start_time_idx = int(time_now / dt)
    term_time_idx = int((time_now + deploy_gap) / dt)
    new_drifter_idx = drifter_num
    drifter_num = drifter_num + 1
    traj_pos, traj_vel = initialise_drifter_traj(seed, dataset_ground_truth, space_grid, t_grid, next_obs_loc_idx, start_time_idx, term_time_idx, obs_noise, new_drifter_idx)
    if traj_pos is not None:
        traj_pos_full = jnp.vstack([traj_pos_full,traj_pos])
        traj_vel_full = jnp.vstack([traj_vel_full,traj_vel])

    max_time = jnp.max(traj_data_raw[4])
    start_time_idx = int(max_time / dt)
    term_time_idx = int((time_now + deploy_gap) / dt)
    existing_drifter_data = traj_data_raw[:, traj_data_raw[4] == max_time]
    existing_drifter_indices = jnp.unique(existing_drifter_data[5])
    for existing_drifter_idx in existing_drifter_indices:
        seed, seed1 = jr.split(seed)
        curr_loc = existing_drifter_data[0:2,existing_drifter_data[5] == existing_drifter_idx].T[0]
        curr_loc_idx = find_loc_idx(space_grid, curr_loc)
        
        traj_pos, traj_vel = continue_drifter_traj(seed, dataset_ground_truth, space_grid, t_grid, curr_loc_idx, start_time_idx, term_time_idx, obs_noise, existing_drifter_idx)
        if traj_pos is not None:
            traj_pos_full = jnp.vstack([traj_pos_full,traj_pos])
            traj_vel_full = jnp.vstack([traj_vel_full,traj_vel])

    # pack up data
    traj_data = dataset_4d(traj_pos_full.T[0:3,:], traj_vel_full.T[0:2,:])
    traj_data_raw = jnp.column_stack([traj_pos_full[:,0:2], traj_vel_full]).T
    #print("At time", time_now,", a new drifter is placed at", space_grid[next_obs_loc_idx])
    deployment_data.append(jnp.concatenate([jnp.array([time_now]), space_grid[next_obs_loc_idx]]))
    time_now = time_now + deploy_gap

#%%
EIG_BALLAST_list = jnp.zeros(shape=space_grid.shape[0])

traj_data_5d = dataset_5d(traj_pos_full.T[0:3,:], traj_vel_full.T[0:2,:])
mean_fn = gpx.mean_functions.Zero()
kernel = HelmholtzKernelSpatioTemporal_derivative(potential_kernel=gpx.kernels.RBF(variance=20,lengthscale=5), stream_kernel=gpx.kernels.RBF(variance=0.01,lengthscale=4),time_kernel=Matern32new(active_dims=[2],lengthscale=1,variance=15))
posterior = initialise_gp(kernel, mean_fn, traj_data_5d)
opt_posterior = posterior
helmholtz_mean, helmholtz_cov = latent_distribution(
    opt_posterior,
    generate_5d_test_location(space_grid, time_now),
    traj_data_5d
)
helmholtz_cov_chol = scipy.linalg.cholesky(helmholtz_cov,lower=True)

#horizon = 5#int(sys.argv[2])
t_grid_sample = np.arange(time_now, time_final+dt, dt)#time_now + horizon+dt, dt)
n_steps_sample = t_grid_sample.shape[0]
start_time_idx_sample = int(np.where(np.isclose(t_grid, t_grid_sample[0]))[0][0])
term_time_idx_sample = int(np.where(np.isclose(t_grid, t_grid_sample[-1]))[0][0])

traj_data = dataset_4d(traj_pos_full.T[0:3,:], traj_vel_full.T[0:2,:])
X_n = traj_data.X

K = kernel.gram(X_n).to_dense()
L_base = jax.scipy.linalg.cho_factor(
    jnp.eye(X_n.shape[0], dtype=K.dtype) + obs_noise**2 * K, lower=True
)[0]  
log_base = 2.0 * jnp.sum(jnp.log(jnp.diag(L_base)))
def utility_comp(pos_sample):
    new_obs_points = jnp.hstack([jnp.vstack([pos_sample[:,0:3], pos_sample[:,0:3]]), jnp.repeat(jnp.array([0,1]),pos_sample.shape[0]).reshape(-1,1)])

    kxx = kernel.gram(new_obs_points).to_dense()
    kxn = kernel.cross_covariance(X_n, new_obs_points)

    alpha = obs_noise ** 2
    # Solve L W = Kxn  ->  W = L^{-1} Kxn
    W = solve_triangular(L_base, kxn, lower=True)      # shape (n, m)
    # S = I_m + alpha*kxx - alpha^2 * (W.T @ W)
    m = kxx.shape[0]
    S = jnp.eye(m, dtype=kxx.dtype) + alpha * kxx - (alpha**2) * (W.T @ W)
    s, log = jnp.linalg.slogdet(S)
    return s*log + log_base
utility_comp_jit = jax.jit(utility_comp)

pos_stacked_sample = []
for t in t_grid_sample:
    pos_stacked_sample.append(jnp.hstack([space_grid, jnp.repeat(t, space_grid.shape[0]).reshape(-1,1)]))
pos_stacked_sample = jnp.vstack(pos_stacked_sample).T

#%%
# --- helpers to keep memory tight ---

# 1) All-JAX posterior rollout with scan: returns (T, N^2, 2)
@jax.jit
def one_posterior_trajectory(rng_key, helmholtz_mean, helmholtz_cov_chol):
    key0, keyn = jr.split(rng_key)

    # initial draw; make sure helmholtz_mean is shape (2*n_blocks, 1)
    f0_noise  = jr.normal(key0, shape=helmholtz_mean.shape)
    f0_sample = (helmholtz_mean + helmholtz_cov_chol @ f0_noise).reshape(-1, 2).T  # (2, n_blocks)

    def step(carry_f, key):
        ek = sample_kron_cov(L_cov, L_Q, key)               # (2, n_blocks) -- must be pure JAX
        f_next = Phi @ carry_f + ek                          # (2, n_blocks)
        y_t = f_next[0]                                      # (n_blocks,)
        return f_next, y_t

    keys = jr.split(keyn, n_steps_sample - 1)
    y0 = f0_sample[0]                                        # (n_blocks,)
    _, ys = jax.lax.scan(step, f0_sample, keys)              # (T-1, n_blocks)
    ys = jnp.vstack([y0[None, :], ys])                       # (T, n_blocks)

    # reshape once to (T, N^2, 2) (first N^2 are u, next N^2 are v)
    ys = ys.reshape(n_steps_sample, 2, N**2).transpose(0, 2, 1)
    return ys                                                # velocity_lookup

# 2) continue-traj using velocity_lookup (no Dataset needed)
def continue_drifter_traj_velocity(
    seed, velocity_lookup, space_grid, t_grid,
    start_loc_idx, start_time_idx, term_time_idx,
    obs_noise, drifter_id
):
    curr_loc   = space_grid[start_loc_idx]
    curr_time  = t_grid[start_time_idx]
    curr_vel   = velocity_lookup[start_time_idx, start_loc_idx]   # (2,)

    # one step to test bounds
    new_loc  = curr_loc + dt * curr_vel
    new_time = t_grid[start_time_idx + 1]
    new_idx  = find_loc_idx(space_grid, new_loc)
    new_vel  = velocity_lookup[start_time_idx + 1, new_idx]

    if not ((range_x[0] <= new_loc[0] <= range_x[1]) and (range_y[0] <= new_loc[1] <= range_y[1])):
        return None, None

    curr_loc, curr_time, curr_vel = new_loc, new_time, new_vel

    # accumulate in lists; concat once (short trajectories)
    traj_pos, traj_vel = [], []
    for t_idx in range(start_time_idx + 2, term_time_idx):
        seed, seed1 = jr.split(seed)
        new_loc  = curr_loc + dt * curr_vel
        new_time = t_grid[t_idx]
        new_idx  = find_loc_idx(space_grid, new_loc)
        new_vel  = velocity_lookup[t_idx, new_idx]
        in_bounds = (range_x[0] <= new_loc[0] <= range_x[1]) and (range_y[0] <= new_loc[1] <= range_y[1])
        if not in_bounds:
            break
        if (t_idx - start_time_idx) % 5 == 0:
            traj_pos.append(jnp.concatenate([new_loc, jnp.array([new_time, drifter_id])]))
            traj_vel.append(jnp.concatenate([new_vel + jr.normal(seed1, (2,)) * obs_noise, jnp.array([new_time, drifter_id])]))
        curr_loc, curr_time, curr_vel = new_loc, new_time, new_vel

    if len(traj_pos) == 0:
        return None, None
    return jnp.stack(traj_pos, axis=0), jnp.stack(traj_vel, axis=0)

# 3) utility with fixed shapes (pad + mask) so it can be jitted & vmapped
def make_utility_comp_padded(X_n, kernel, L_base, log_base, obs_noise, max_m):
    @jax.jit
    def utility_comp_padded(pos_sample, count):
        # pos_sample: (max_m, 4), count: scalar in [0, max_m]
        # Build 2*max_m new_obs_points = [(x,y,t,label=0/1)] with padding masked out
        # base: (2*max_m, 3)
        base = jnp.vstack([pos_sample[:, :3], pos_sample[:, :3]])
        labels = jnp.repeat(jnp.array([0, 1]), max_m).reshape(-1, 1)
        new_obs_points = jnp.hstack([base, labels])                 # (2*max_m, 4)

        # mask
        valid  = (jnp.arange(max_m) < count).astype(X_n.dtype)      # (max_m,)
        valid2 = jnp.repeat(valid, 2)                                # (2*max_m,)

        # kernels
        kxx = kernel.gram(new_obs_points).to_dense()                 # (2M,2M)
        kxx = kxx * (valid2[:, None] * valid2[None, :])              # zero padded rows/cols
        kxn = kernel.cross_covariance(X_n, new_obs_points) * valid2  # (n, 2M)

        W = solve_triangular(L_base, kxn, lower=True)                # (n, 2M)
        alpha = obs_noise ** 2
        S = jnp.eye(kxx.shape[0], dtype=kxx.dtype) + alpha * kxx - (alpha**2) * (W.T @ W)
        s, log = jnp.linalg.slogdet(S)
        return s * log + log_base
    return utility_comp_padded

eig_ballast_list = jnp.zeros(space_grid.shape[0])
chosen_ids = []
sample_num = 300

# Prebuild padded utility with a fixed per-candidate max length.
# Use the same max_steps you already use in initialise_drifter_traj_optimized:
max_m = int((term_time_idx_sample - start_time_idx_sample) / 5) + 1
utility_comp_padded = make_utility_comp_padded(X_n, kernel, L_base, log_base, obs_noise, max_m)

for m in range(sample_num):
    # 1) Roll out one posterior trajectory and use it directly as the lookup: (T, N^2, 2)
    velocity_lookup = one_posterior_trajectory(jr.PRNGKey(m), helmholtz_mean, helmholtz_cov_chol)

    # 2) VMAP drifter sims (already preallocates internally and returns (max_steps,4), count)
    def sample_drifter(start_loc_idx):
        pos_sample, vel_sample, count = initialise_drifter_traj_optimized(
            seed,                 # if you want per-candidate randomness, mix start_loc_idx into seed
            velocity_lookup,      # (T, N^2, 2)
            space_grid,           # (N^2, 2)
            t_grid,               # (T,)
            start_loc_idx,
            start_time_idx_sample,
            term_time_idx_sample,
            obs_noise,
            drifter_id,
        )
        # pad to (max_m, 4) for fixed shape
        pad = max_m - pos_sample.shape[0]
        pos_padded = jnp.pad(pos_sample, ((0, pad), (0, 0)))
        return pos_padded, count

    start_loc_idxs = jnp.arange(space_grid.shape[0])
    pos_padded_batch, count_batch = jax.vmap(sample_drifter)(start_loc_idxs)  # shapes: (N^2, max_m, 4), (N^2,)

    existing_traj_pos = []
    existing_ids = jnp.unique(traj_pos_full[:, 3])
    for existing_id in existing_ids:
        seed, seed1 = jr.split(seed)
        curr_loc = traj_pos_full[traj_pos_full[:, 3] == existing_id, 0:2][-1]
        curr_loc_idx = find_loc_idx(space_grid, curr_loc)
        pos_ext, _ = continue_drifter_traj_velocity(
            seed1, velocity_lookup, space_grid, t_grid,
            curr_loc_idx, start_time_idx_sample, term_time_idx_sample,
            obs_noise, existing_id
        )
        if pos_ext is not None:
            existing_traj_pos.append(pos_ext)

    if len(existing_traj_pos) > 0:
        existing_block = jnp.vstack(existing_traj_pos)              # (E_tot, 4)
        # pad existing block once so we can fuse it to every candidate without changing shapes
        e_count = existing_block.shape[0]
        e_pad   = max(0, max_m - e_count)
        existing_padded = jnp.pad(existing_block[:max_m], ((0, e_pad), (0, 0)))  # (max_m, 4)
        e_count_clipped = jnp.array(min(e_count, max_m))
    else:
        existing_padded = jnp.zeros((max_m, 4))
        e_count_clipped = jnp.array(0)

    def candidate_utility(pos_padded, count):
        # fuse candidate + existing by "tiling" and masking via counts
        # We simply call utility twice and add; cheaper than actually concatenating to 2*max_m rows before the label duplication.
        u1 = utility_comp_padded(pos_padded, count)                  # candidate part
        u2 = jax.lax.cond(e_count_clipped > 0,
                          lambda _: utility_comp_padded(existing_padded, e_count_clipped),
                          lambda _: 0.0,
                          operand=None)
        return u1 + u2

    EIG_BALLAST_list_sample = jax.vmap(candidate_utility)(pos_padded_batch, count_batch)  # (N^2,)

    eig_ballast_list = eig_ballast_list + EIG_BALLAST_list_sample
    chosen_ids.append(int(jnp.argmax(eig_ballast_list)))

#%%
overall_mean = eig_ballast_list/sample_num
best_value = np.max(overall_mean)
utility_gaps = []

for m in range(sample_num):
    chosen_value = overall_mean[chosen_ids[m]]
    gap = best_value - chosen_value
    utility_gaps.append(gap)


metric = jnp.hstack([jnp.array([seednum]), jnp.array(utility_gaps)])            # shape (626,)

# Save as one row
with open(f'./m_ablation_suntans_utility_gap_t{t_choice}.csv', 'a') as myfile:
    np.savetxt(myfile, np.array(metric).reshape(1, -1), delimiter=',')

metric = jnp.hstack([jnp.array([seednum]), jnp.array(chosen_ids)])            # shape (626,)

# Save as one row
with open(f'./m_ablation_suntans_utility_id_t{t_choice}.csv', 'a') as myfile:
    np.savetxt(myfile, np.array(metric).reshape(1, -1), delimiter=',')

metric = jnp.hstack([jnp.array([seednum]), jnp.array(overall_mean)]) 
# Save as one row
with open(f'./m_ablation_suntans_utility_final_t{t_choice}.csv', 'a') as myfile:
    np.savetxt(myfile, np.array(metric).reshape(1, -1), delimiter=',')