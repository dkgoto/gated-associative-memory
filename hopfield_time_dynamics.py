# ==================== Imports ====================
import numpy as np
import matplotlib.pyplot as plt
import numba
from joblib import Parallel, delayed
from scipy.optimize import brentq

# ==================== Core Functions ====================
#@numba.njit
def hopfield_update(x, z, Xi, J, W, g, Tp, Ta, tau, dt, s_sigmoid, gate):
    N, P = Xi.shape
    alpha = P / N

    # Activation
    phi_x = np.tanh(x)
    proj = (1 / N) * (Xi.T @ phi_x)

    # 2-body force
    #two_body = -J @ s
    two_body = -(g / np.sqrt(alpha)) * (Xi @ proj)

    H_grad = x + two_body 

    # Gating
    if gate:
        sigma_z = 1.0 / (1.0 + np.exp(-s_sigmoid * z))
    else:
        sigma_z = np.ones(N)

    eta = np.random.randn(N)
    etaa = np.random.randn(N)

    dx_dt = sigma_z * (-H_grad + np.sqrt(2 * Tp / dt) * eta)
    dz_dt = (-z + W @ phi_x) / tau + np.sqrt(2 * Ta / dt) * etaa / tau

    # Update
    #x += dt * dx_dt
    #z += dt * dz_dt

    return x+dt*dx_dt, z+dt*dz_dt

#@numba.njit
def calc_overlap(s, Xi, g, mu=0):
    N = s.shape[0]
    xi = Xi[:, mu].copy()  # Ensure contiguous
    overlaps = (xi.T @ np.tanh(s)) / N
    return overlaps

#@numba.njit
def calc_normalized_overlap(s, Xi, mu=0):
    s_norm = np.sqrt(np.sum(s**2))
    xi = Xi[:, mu].copy()  # Ensure contiguous
    xi_norm = np.sqrt(np.sum(xi**2))
    dot = np.dot(s, xi)
    return dot / (s_norm * xi_norm)

#@numba.njit
def calc_qEA(s):
    return np.mean(s**2)

#@numba.njit
def calc_msd(s, Xi, g, mu=0):
    N = s.shape[0]
    diff = np.tanh(s) - Xi[:, mu].copy()  # Ensure contiguous
    msd = np.sum(diff * diff) / N
    return msd
"""
def generate_initial_state(xi, m0):
    N = len(xi)
    f = (1 + m0) / 2
    s = np.empty(N, dtype=int)

    # Create an array of indices and shuffle
    indices = np.arange(N)
    np.random.shuffle(indices)

    # Number of aligned spins
    N_align = int(np.round(f * N))

    # Assign aligned and anti-aligned spins
    s[indices[:N_align]] = xi[indices[:N_align]]
    s[indices[N_align:]] = -xi[indices[N_align:]]

    return s
"""
def create_initial_state(
    xi_target: np.ndarray, 
    m0: float, 
    sigma: float = 1.0
) -> np.ndarray:
    """
    Creates a random initial state s(0) for a Hopfield network that satisfies 
    a target overlap m0 with a specific pattern.

    This function uses the "Signal Plus Noise" method:
    s_i(0) = alpha * xi_i + eta_i
    
    where `alpha` is numerically calculated to meet the overlap condition.

    Args:
        xi_target (np.ndarray): The target pattern vector, shape (N,). 
                                Should contain -1s and 1s.
        m0 (float): The target initial overlap, must be in the range (-1, 1).
        sigma (float, optional): The standard deviation of the Gaussian noise (eta).
                                 Controls the "randomness" of the state. Defaults to 1.0.

    Returns:
        np.ndarray: The initial state vector s(0), shape (N,).
        
    Raises:
        ValueError: If m0 is not in the range (-1, 1).
    """
    if not -1.0 <= m0 <= 1.0:
        raise ValueError(f"Target overlap m0 must be in (-1, 1), but got {m0}.")

    N = xi_target.shape[0]

    # 1. Generate a random noise vector from a Gaussian distribution
    eta = np.random.normal(loc=0.0, scale=sigma, size=N)

    # 2. Define the objective function f(alpha) = m(alpha) - m0
    #    The root of this function is the alpha we need.
    def objective_function(alpha: float) -> float:
        """Calculates the difference between the current overlap and the target m0."""
        s_i = alpha * xi_target + eta
        tanh_s = np.tanh(s_i)
        
        # We use np.dot for an efficient summation: sum(xi_i * tanh(s_i))
        current_m = np.dot(xi_target, tanh_s) / N
        
        return current_m - m0

    # 3. Solve for alpha using a numerical root-finder.
    #    brentq is a robust and reliable choice. It requires a bracket [a, b] 
    #    where f(a) and f(b) have opposite signs.
    #    As alpha -> inf, m -> 1. As alpha -> -inf, m -> -1.
    #    A large bracket like [-100, 100] is almost always sufficient.
    try:
        alpha_solution = brentq(objective_function, a=-100.0, b=100.0)
    except ValueError:
        # This can happen if f(a) and f(b) have the same sign, which is
        # unlikely for m0 in (-1,1) but could occur with extreme noise.
        raise RuntimeError("Could not find a bracketing interval for the root. "
                           "Try adjusting sigma or the search interval [a, b].")

    # 4. Construct the initial state s(0) with the solved alpha
    s0 = alpha_solution * xi_target + eta
    
    return s0

def set_initial_state_by_flipping(N, m0: float, pattern):
        """
        Generates an initial state x(0) to achieve a target overlap m0 by flipping bits.
        
        Starting with a perfect pattern y = xi^0 (overlap = 1), we flip a calculated
        number of components from +1 to -1 (or -1 to +1) to reduce the overlap to m0.
        Each flip reduces the total sum in the overlap calculation by 2, thus reducing
        the overlap 'm' by 2/N.
        
        Number of flips K = (1 - m0) * N / 2.
        
        Args:
            m0 (float): The desired initial overlap, between 0 and 1.
            target_pattern_idx (int): The index of the target pattern xi^0.
            
        Returns:
            Tuple[np.ndarray, float]: A tuple containing the initial state vector x(0)
                                      and the actual overlap achieved.
        """
        if not (0.0 <= m0 <= 1.0):
            raise ValueError("Target overlap m0 must be between 0 and 1.")

        def _calculate_overlap(N, y: np.ndarray, pattern: np.ndarray) -> float:
            return np.dot(pattern, y) / N
            
        # Start with the output y being the perfect target pattern
        y0 = pattern.copy()
        
        # Calculate the number of flips needed
        num_flips = int(round(N * (1 - m0) / 2.0))
        
        # Randomly choose unique indices to flip
        flip_indices = np.random.choice(N, size=num_flips, replace=False)
        
        # Flip the signs at these indices
        for nu in  flip_indices:
            y0[nu] = -1.0 * y0[nu]
        
        # --- Validity Check ---
        actual_m0 = _calculate_overlap(N, y0, pattern)
        
        # Convert output y0 to state x0 using arctanh.
        # We must clip y0 to avoid infinity at arctanh(+-1).
        epsilon = 1e-4
        y0_clipped = np.clip(y0, -1.0 + epsilon, 1.0 - epsilon)
        x0 = np.arctanh(y0_clipped)
        
        return x0

# ==================== Time Dynamics ====================
def run_time_dynamics(
    N, P, g, Tp, Ta, tau, dt, t_steps, 
    s_sigmoid, gate, init_type="pattern", 
    target_mu=0, target_overlap=0.995, seed=0, record_every=1
):
    np.random.seed(seed)

    #Xi = np.random.randn(N, P)
    Xi = np.random.choice([-1.0, 1.0], (N, P))
    J = (Xi @ Xi.T)# / N
    np.fill_diagonal(J, 0.0)
    # Stored W includes the paper's 1/sqrt(N) prefactor (W_bar in the SI).
    W = np.random.randn(N, N) / np.sqrt(N)    # iid real random asymmetric matrix

    if init_type == "random":
        x = np.random.randn(N)
    elif init_type == "pattern":
        xi_target = Xi[:, target_mu].copy()  # Ensure contiguous
        #xi_target = xi_target / np.linalg.norm(xi_target) * np.sqrt(N)
        #eta = np.random.randn(N)
        #eta -= (np.dot(eta, xi_target) / np.dot(xi_target, xi_target)) * xi_target
        #eta *= np.sqrt(N / np.dot(eta, eta))
        #x = target_overlap * xi_target + np.sqrt(1 - target_overlap**2) * eta
        #x = generate_initial_state(xi_target, target_overlap)    
        x = create_initial_state(xi_target, target_overlap) # perpendicular noise
        #x = set_initial_state_by_flipping(N, target_overlap, xi_target) # random flip
    else:
        raise ValueError("Unknown init_type")

    #z = np.zeros(N)
    z = np.random.randn(N)

    steps_recorded = t_steps // record_every + 1
    m_t = np.zeros(steps_recorded)
    m_norm_t = np.zeros(steps_recorded)
    msd_t = np.zeros(steps_recorded)
    qea_t = np.zeros(steps_recorded)

    m_t[0] = calc_overlap(x, Xi, g, mu=target_mu)
    m_norm_t[0] = calc_normalized_overlap(x, Xi, mu=target_mu)
    msd_t[0] = calc_msd(x, Xi, g, mu=target_mu)
    qea_t[0] = calc_qEA(x)

    counter = 1
    for t in range(1, t_steps + 1):
        x, z = hopfield_update(x, z, Xi, J, W, g, Tp, Ta, tau, dt, s_sigmoid, gate)
        if t % record_every == 0:
            m_t[counter] = calc_overlap(x, Xi, g, mu=target_mu)
            m_norm_t[counter] = calc_normalized_overlap(x, Xi, mu=target_mu)
            msd_t[counter] = calc_msd(x, Xi, g, mu=target_mu)
            qea_t[counter] = calc_qEA(x)
            counter += 1

    return m_t, m_norm_t, msd_t, qea_t

# ==================== Plotting ====================
def plot_time_dynamics(m_t, m_norm_t, msd_t, qea_t=None, dt=0.01, record_every=1, save_path=None):
    t = np.arange(len(m_t)) * dt * record_every
    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.plot(t, m_t, label="Raw Overlap $m(t)$")
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Overlap", fontsize=12)
    plt.grid(True)
    plt.legend()

    plt.subplot(1,3,2)
    plt.plot(t, m_norm_t, label="Normalized Overlap $\\tilde{m}(t)$", color='green')
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Normalized Overlap", fontsize=12)
    plt.grid(True)
    plt.legend()

    plt.subplot(1,3,3)
    plt.plot(t, msd_t, label="MSD", color='orange')
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Mean Squared Deviation", fontsize=12)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

