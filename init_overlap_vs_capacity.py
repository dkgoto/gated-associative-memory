"""Sweep memory load and initial overlap using the supplied dynamics module."""

from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# === Single trial with fixed initial overlap ===
def run_single_trial_initial_overlap(N, P, g, tau, dt, s_sigmoid, gate,
                                     init_overlap, t_steps, seed, record_every=10, return_full=False):
    from hopfield_time_dynamics import run_time_dynamics
    m_t, m_norm_t, msd_t, _ = run_time_dynamics(
        N=N, P=P,
        g=g,
        Tp=0.0, Ta=0.0,
        tau=tau, dt=dt,
        t_steps=t_steps,
        s_sigmoid=s_sigmoid,
        gate=gate,
        init_type="pattern",
        target_mu=0,
        target_overlap=init_overlap,
        seed=seed,
        record_every=record_every
    )
    if return_full:
        return m_t, msd_t
    return m_t[-1], msd_t[-1]

# === Full sweep over initial overlaps and capacity (alpha = P/N) ===
def run_initial_overlap_vs_capacity_heatmaps(
    N, g, tau, dt, s_sigmoid, gate,
    alpha_list, init_overlap_list,
    t_steps=1000, num_ensemble=40, seed_base=0, record_every=10, n_jobs=-1,
    return_full=False
):
    if N < 1 or num_ensemble < 1 or t_steps < 1 or record_every < 1:
        raise ValueError("N, num_ensemble, t_steps, and record_every must be positive.")
    if not np.isfinite(dt) or not np.isfinite(tau) or dt <= 0 or tau <= 0:
        raise ValueError("dt and tau must be finite and positive.")
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be zero.")
    for name, values in (("alpha_list", alpha_list),
                         ("init_overlap_list", init_overlap_list)):
        array = np.asarray(values, dtype=float)
        if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
            raise ValueError(f"{name} must be a nonempty finite one-dimensional grid.")
    if np.any(np.abs(np.asarray(init_overlap_list)) > 1):
        raise ValueError("Initial overlaps must lie in [-1, 1].")
    # Retain the original integer conversion and seed schedule.
    P_list = (np.array(alpha_list) * N).astype(int)
    if np.any(P_list < 1):
        raise ValueError("Every memory load must give at least one stored pattern.")
    overlap_map = np.zeros((len(P_list), len(init_overlap_list)))
    msd_map = np.zeros((len(P_list), len(init_overlap_list)))

    if return_full:
        T = t_steps // record_every + 1  # Includes the initial state.
        overlap_time = np.zeros((len(P_list), len(init_overlap_list), T))
        msd_time = np.zeros_like(overlap_time)

    for j, init_overlap in enumerate(tqdm(init_overlap_list, desc="Init Overlap")):
        for i, P in enumerate(P_list):
            results = Parallel(n_jobs=n_jobs)(
                delayed(run_single_trial_initial_overlap)(
                    N, P, g, tau, dt, s_sigmoid, gate,
                    init_overlap, t_steps,
                    seed=seed_base + 10000*j + 100*i + k,
                    record_every=record_every,
                    return_full=return_full
                )
                for k in range(num_ensemble)
            )

            if return_full:
                m_t_all = np.stack([r[0] for r in results])
                msd_t_all = np.stack([r[1] for r in results])
                overlap_time[i, j, :] = np.mean(m_t_all, axis=0)
                msd_time[i, j, :] = np.mean(msd_t_all, axis=0)
                # Also save the final values
                overlap_map[i, j] = overlap_time[i, j, -1]
                msd_map[i, j] = msd_time[i, j, -1]
            else:
                overlap_map[i,j] = np.mean([r[0] for r in results])
                msd_map[i,j] = np.mean([r[1] for r in results])
    if return_full:
        return alpha_list, init_overlap_list, overlap_map, msd_map, overlap_time, msd_time
    else:
        return alpha_list, init_overlap_list, overlap_map, msd_map

# === Plotting ===
def plot_overlap_msd_vs_init_overlap(alpha_list, init_overlap_list, overlap_map, msd_map,
                                     clim1=(0, 1), clim2=None, title_suffix=""):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))

    im1 = axs[0].imshow(
        overlap_map.T, origin='lower', aspect='auto',
        extent=[alpha_list[0], alpha_list[-1], init_overlap_list[0], init_overlap_list[-1]],
        cmap='Blues'
    )
    if clim1:
        im1.set_clim(*clim1)
    axs[0].set_title(r"Final overlap $m$" + title_suffix)
    axs[0].set_xlabel(r"Capacity $\alpha = P/N$")
    axs[0].set_ylabel(r"Initial overlap $m_0$")
    fig.colorbar(im1, ax=axs[0])

    im2 = axs[1].imshow(
        msd_map.T, origin='lower', aspect='auto',
        extent=[alpha_list[0], alpha_list[-1], init_overlap_list[0], init_overlap_list[-1]],
        cmap='Greys'
    )
    if clim2:
        im2.set_clim(*clim2)
    axs[1].set_title(r"Final mean squared deviation (MSD)" + title_suffix)
    axs[1].set_xlabel(r"Capacity $\alpha = P/N$")
    axs[1].set_ylabel(r"Initial overlap $m_0$")
    fig.colorbar(im2, ax=axs[1])

    plt.tight_layout()
    plt.show()
