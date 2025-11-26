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

F = jnp.array([[0.0, 1.0],
               [-lambda_val**2, -2.0 * lambda_val]])
H = jnp.array([[1.0, 0.0]])
P_inf = jnp.diag(jnp.array([sigma2, lambda_val**2 * sigma2]))
Phi = expm(F * dt)
Q = P_inf - Phi @ P_inf @ Phi.T
I_space = jnp.eye(space_grid.shape[0]*2,space_grid.shape[0]*2)
H_full = jnp.kron(I_space,H)

space_kernel = HelmholtzKernel(potential_kernel=gpx.kernels.RBF(variance=20,lengthscale=5), stream_kernel=gpx.kernels.RBF(variance=0.01,lengthscale=4))
cov_matrix = space_kernel.gram(space_grid_3d).to_dense() + jnp.eye(space_grid_3d.shape[0])*1e-10

n_blocks = cov_matrix.shape[0]  
@jax.jit
def sample_kron_cov(A_lower, B_lower, subkey):
    # A_lower: (n x n) such that A_lower A_lower^T = cov
    # B_lower: (2 x 2) such that B_lower B_lower^T = S   
    E = jr.normal(subkey,(n_blocks, 2)).T# iid standard normal
    M = B_lower @ E @ A_lower.T                             # shape (2, n_blocks)
    return M                                               # return as 2 x n_blocks matrix

L_cov = la.cholesky(cov_matrix, lower=True)                # (2N^2, 2N^2)
L_Pinf = la.cholesky(P_inf, lower=True)                    # (2, 2)
L_Q = la.cholesky(Q, lower=True)                           # (2, 2)

#%%
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

def full_system_EIG_BALLAST(seed, traj_pos_full, traj_vel_full, space_grid, time_now, BALLAST_samples):
    # ---- GP setup (no jitting here) ----
    traj_data_5d = dataset_5d(traj_pos_full.T[0:3, :], traj_vel_full.T[0:2, :])
    mean_fn = gpx.mean_functions.Zero()
    kernel = HelmholtzKernelSpatioTemporal_derivative(
        potential_kernel=gpx.kernels.RBF(variance=20, lengthscale=5),
        stream_kernel=gpx.kernels.RBF(variance=0.01, lengthscale=4),
        time_kernel=Matern32new(active_dims=[2], lengthscale=1, variance=15),
    )
    posterior = initialise_gp(kernel, mean_fn, traj_data_5d)
    opt_posterior = posterior

    # predictive mean/cov at current time
    helmholtz_mean, helmholtz_cov = latent_distribution(
        opt_posterior,
        generate_5d_test_location(space_grid, time_now),
        traj_data_5d,
    )
    helmholtz_cov_chol = jax.scipy.linalg.cholesky(helmholtz_cov, lower=True)

    # time indices for the rollout window
    t_grid_sample = np.arange(time_now, time_final + dt, dt)
    n_steps_sample = t_grid_sample.shape[0]
    start_time_idx_sample = int(np.where(np.isclose(t_grid, t_grid_sample[0]))[0][0])
    term_time_idx_sample  = int(np.where(np.isclose(t_grid, t_grid_sample[-1]))[0][0])

    # ---- acquisition pieces reused across the loop ----
    traj_data = dataset_4d(traj_pos_full.T[0:3, :], traj_vel_full.T[0:2, :])
    X_n = traj_data.X
    K = kernel.gram(X_n).to_dense()
    L_base = jax.scipy.linalg.cho_factor(
        jnp.eye(X_n.shape[0], dtype=K.dtype) + obs_noise**2 * K, lower=True
    )[0]
    log_base = 2.0 * jnp.sum(jnp.log(jnp.diag(L_base)))

    # fixed max length so shapes are static
    max_m = int((term_time_idx_sample - start_time_idx_sample) / 5) + 1

    # utility builder (compiled once)
    def make_utility_comp_padded(X_n, kernel, L_base, log_base, obs_noise, max_m):
        @jax.jit
        def utility_comp_padded(pos_sample, count):
            # pos_sample: (max_m, 4) [x,y,t,drifter_id]; count: scalar <= max_m
            base   = jnp.vstack([pos_sample[:, :3], pos_sample[:, :3]])   # (2M, 3)
            labels = jnp.repeat(jnp.array([0, 1]), max_m).reshape(-1, 1)  # (2M, 1)
            new_obs_points = jnp.hstack([base, labels])                    # (2M, 4)

            valid  = (jnp.arange(max_m) < count).astype(X_n.dtype)        # (M,)
            valid2 = jnp.repeat(valid, 2)                                  # (2M,)

            kxx = kernel.gram(new_obs_points).to_dense()
            kxx = kxx * (valid2[:, None] * valid2[None, :])
            kxn = kernel.cross_covariance(X_n, new_obs_points) * valid2

            W = solve_triangular(L_base, kxn, lower=True)
            alpha = obs_noise ** 2
            S = jnp.eye(kxx.shape[0], dtype=kxx.dtype) + alpha * kxx - (alpha**2) * (W.T @ W)
            s, log = jnp.linalg.slogdet(S)
            return s * log + log_base
        return utility_comp_padded

    utility_comp_padded = make_utility_comp_padded(X_n, kernel, L_base, log_base, obs_noise, max_m)
    vmapped_utility = jax.jit(jax.vmap(utility_comp_padded, in_axes=(0, 0)))  # (N^2, max_m, 4),(N^2,) -> (N^2,)

    # state-space rollout (compiled once)
    @jax.jit
    def one_posterior_trajectory(rng_key, helmholtz_mean, helmholtz_cov_chol):
        key0, keyn = jr.split(rng_key)
        f0_noise  = jr.normal(key0, shape=helmholtz_mean.shape)
        f0_sample = (helmholtz_mean + helmholtz_cov_chol @ f0_noise).reshape(-1, 2).T  # (2, n_blocks)

        def step(carry_f, key):
            ek = sample_kron_cov(L_cov, L_Q, key)  # (2, n_blocks)
            f_next = Phi @ carry_f + ek
            y_t = f_next[0]
            return f_next, y_t

        keys = jr.split(keyn, n_steps_sample - 1)
        y0 = f0_sample[0]
        _, ys = jax.lax.scan(step, f0_sample, keys)
        ys = jnp.vstack([y0[None, :], ys])                           # (T, n_blocks)
        ys = ys.reshape(n_steps_sample, 2, N**2).transpose(0, 2, 1)  # (T, N^2, 2)
        return ys

    # ---- vmapped sampler: CLOSE OVER Python ints; do NOT pass them as args ----
    sidx_py = int(start_time_idx_sample)
    eidx_py = int(term_time_idx_sample)
    did_py  = int(drifter_id)  # assumes drifter_id is defined in your outer scope

    def _sample_drifter_vmapped(idx, seed_in, vlookup):
        pos_sample, _, count = initialise_drifter_traj_optimized(
            seed_in, vlookup, space_grid, t_grid,
            idx, sidx_py, eidx_py, obs_noise, did_py
        )
        pad = max_m - pos_sample.shape[0]
        pos_padded = jnp.pad(pos_sample, ((0, pad), (0, 0)))
        return pos_padded, count

    vmapped_sample_drifter = jax.jit(
        jax.vmap(_sample_drifter_vmapped, in_axes=(0, None, None))
    )

    # continuation helper (pure Python/JAX mix; not jitted)
    def continue_drifter_traj_velocity(seed_in, velocity_lookup, space_grid, t_grid,
                                       start_loc_idx, start_time_idx, term_time_idx,
                                       obs_noise, drifter_id):
        curr_loc  = space_grid[start_loc_idx]
        curr_time = t_grid[start_time_idx]
        curr_vel  = velocity_lookup[start_time_idx, start_loc_idx]

        new_loc  = curr_loc + dt * curr_vel
        new_time = t_grid[start_time_idx + 1]
        new_idx  = find_loc_idx(space_grid, new_loc)
        new_vel  = velocity_lookup[start_time_idx + 1, new_idx]
        if not ((range_x[0] <= new_loc[0] <= range_x[1]) and (range_y[0] <= new_loc[1] <= range_y[1])):
            return None, None

        curr_loc, curr_time, curr_vel = new_loc, new_time, new_vel
        traj_pos, traj_vel = [], []
        for t_idx in range(start_time_idx + 2, term_time_idx):
            seed_in, seed1 = jr.split(seed_in)
            new_loc  = curr_loc + dt * curr_vel
            new_time = t_grid[t_idx]
            new_idx  = find_loc_idx(space_grid, new_loc)
            new_vel  = velocity_lookup[t_idx, new_idx]
            if not ((range_x[0] <= new_loc[0] <= range_x[1]) and (range_y[0] <= new_loc[1] <= range_y[1])):
                break
            if (t_idx - start_time_idx) % 5 == 0:
                traj_pos.append(jnp.concatenate([new_loc, jnp.array([new_time, drifter_id])]))
                traj_vel.append(jnp.concatenate([new_vel + jr.normal(seed1, (2,)) * obs_noise,
                                                 jnp.array([new_time, drifter_id])]))
            curr_loc, curr_time, curr_vel = new_loc, new_time, new_vel

        if len(traj_pos) == 0:
            return None, None
        return jnp.stack(traj_pos, axis=0), jnp.stack(traj_vel, axis=0)

    # ---- main MC loop (reuses compiled funcs) ----
    eig_ballast_list = jnp.zeros(space_grid.shape[0])
    start_loc_idxs = jnp.arange(space_grid.shape[0])

    for m in range(BALLAST_samples):
        # 1) one posterior rollout -> (T, N^2, 2)
        velocity_lookup = one_posterior_trajectory(jr.PRNGKey(m), helmholtz_mean, helmholtz_cov_chol)

        # 2) candidates (vmapped, compiled once)
        pos_padded_batch, count_batch = vmapped_sample_drifter(
            start_loc_idxs, seed, velocity_lookup
        )  # shapes: (N^2, max_m, 4), (N^2,)

        # 3) existing drifters continued (small, per-m)
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
            existing_block = jnp.vstack(existing_traj_pos)            # (E_tot, 4)
            e_count = existing_block.shape[0]
            e_pad   = max(0, max_m - e_count)
            existing_padded = jnp.pad(existing_block[:max_m], ((0, e_pad), (0, 0)))
            e_count_clipped = jnp.array(min(e_count, max_m))
            u_existing = utility_comp_padded(existing_padded, e_count_clipped)
        else:
            u_existing = 0.0

        # 4) utilities per candidate (compiled once); add existing part
        u_candidates = vmapped_utility(pos_padded_batch, count_batch) + u_existing
        eig_ballast_list = eig_ballast_list + u_candidates

    return eig_ballast_list

#%%
N_space = space_grid.shape[0]
seednum= int(sys.argv[1])
seed = jr.key(seednum)


drifter_num = 1
time_now = 0
deploy_gap = 0.5
deploy_num = 9
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
    acq_list = full_system_EIG_BALLAST(seed, traj_pos_full, traj_vel_full, space_grid, time_now, BALLAST_samples=20)
    next_obs_loc_idx = jnp.argmax(acq_list)

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


arr_np = jnp.hstack([traj_data.X, traj_data.y])
arr_np = jnp.hstack([arr_np, jnp.repeat(seednum,arr_np.shape[0]).reshape(-1,1)])
with open('./traj_data_BALLAST20_suntans.csv', 'a') as myfile:
    np.savetxt(myfile, np.array(arr_np), delimiter=',')



deployment_time = jnp.arange(deploy_gap*2,time_final+deploy_gap,deploy_gap)
metric_eval_time = jnp.arange(0, time_final + dt, deploy_gap)
metric_eval_time = (metric_eval_time/dt).astype(int)

mean_fn = gpx.mean_functions.Zero()
kernel = HelmholtzKernelSpatioTemporal(potential_kernel=gpx.kernels.RBF(variance=20,lengthscale=5), stream_kernel=gpx.kernels.RBF(variance=0.01,lengthscale=4),time_kernel=gpx.kernels.Matern32(lengthscale=1,variance=15))

bias_list_full = []
for d in deployment_time:
    curr_data = gpx.Dataset(traj_data.X[traj_data.X[:,2] < d], traj_data.y[traj_data.X[:,2] < d])

    bias_list = []
    for mt in metric_eval_time:
        mean, cov = latent_distribution(
            initialise_gp(kernel, mean_fn, curr_data),
            dataset_ground_truth.X[dataset_ground_truth.X[:,2] == t_grid[mt],:],   # shape (2N, 4)
            curr_data
        )
        true_field = dataset_ground_truth.y[dataset_ground_truth.X[:,2] == t_grid[mt]].flatten().reshape(-1,2)
        mean = mean.reshape(-1,2)
        bias_list.append(jnp.linalg.norm(mean - true_field, axis=1))

    bias_list_full.append(jnp.mean(jnp.array(bias_list).flatten()))

metric = jnp.array([jnp.array(bias_list_full)])#jnp.vstack([jnp.array(bias_list_full),jnp.array(deg_bias_list_full),jnp.array(vel_bias_list_full),jnp.array(entropy_list_full)])
metric = jnp.hstack([metric, jnp.repeat(seednum,metric.shape[0]).reshape(-1,1)])
with open('./metric_BALLAST20_suntans.csv', 'a') as myfile:
    np.savetxt(myfile, np.array(metric), delimiter=',')