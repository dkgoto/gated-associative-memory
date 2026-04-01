# Fig. 1 data

- `gated_m_ss_vs_alpha_N1000_g1.5_dt0.2_T10000_ens100.npz`:
  steady-state overlaps for initial overlap m(0) vs memory load α for the gated model.

- `ungated_m_ss_vs_alpha_N1000_g1.5_dt0.2_T10000_ens100.npz`:
  steady-state overlaps for initial overlap m(0)  vs memory load α for the ungated model.

Parameters:
- N = 1000: network size
- g = 1.5: scalar gain
- dt = 0.2: timestep
- T = 10000: number of steps
- ensemble size = 100


## Contents of each `.npz` file

Each file contains:
- `alpha_vals`: array of memory-load values \(\alpha\)
- `init_ov_vals`: array of initial overlaps \(m(0)\)
- `overlap_map`: 2D array of steady-state overlaps \(m_{\mathrm{ss}}\)

## Example: loading the data in Python

```python
import numpy as np

data = np.load("gated_m_ss_vs_alpha_N1000_g1.5_dt0.2_T10000_ens100.npz")

alpha_vals = data["alpha_vals"]
init_ov_vals = data["init_ov_vals"]
overlap_map = data["overlap_map"]

print(alpha_vals.shape)
print(init_ov_vals.shape)
print(overlap_map.shape)
