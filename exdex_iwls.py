import math
import numpy as np
import statsmodels.api as sm

def fun(n_wls, N, S_1_sort, exp_qs, ws, nx):
    chi_i = S_1_sort[:n_wls]
    x_i = exp_qs[:n_wls]
    ws = ws[:n_wls]
    x_i = sm.add_constant(x_i)
    mod_wls = sm.WLS(chi_i, x_i, weights=ws)
    results = mod_wls.fit()
    ab = results.params
    theta = min(np.exp(ab[0] / ab[1]), 1)
    n_wls = math.floor(theta * (N - 1))
    return {"theta": theta, "n_wls": n_wls}

# fini




