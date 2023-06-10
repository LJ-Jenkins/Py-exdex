import numpy as np
import math

def stat(data, u, q_u=None, k=1, inc_cens=True):
    if q_u is None:
        q_u = np.mean(data > u)
    data = data[~np.isnan(data)]
    if not isinstance(u, (int, float)) or not isinstance(k, int) or k < 0:
        raise ValueError("Invalid arguments: u must be a numeric scalar, k must be a non-negative integer")
    if u >= np.nanmax(data):
        return {'N0': 0, 'N1': 0, 'sum_qs': 0, 'n_kgaps': 0}
    nx = len(data)
    exc_u = np.arange(1, nx + 1)[data > u]
    N_u = len(exc_u)
    T_u = np.diff(exc_u)
    S_k = np.maximum(T_u - k, 0)
    N1 = np.sum(S_k > 0)
    N0 = N_u - 1 - N1
    sum_qs = np.sum(q_u * S_k)
    n_kgaps = N0 + N1
    if inc_cens:
        T_u_cens = np.concatenate(([exc_u[0] - 1], [nx - exc_u[-1]]))
        S_k_cens = np.maximum(T_u_cens - k, 0)
        N1_cens = np.sum(S_k_cens > 0)
        n_kgaps += N1_cens
        S_k_cens = S_k_cens[S_k_cens > 0]
        sum_s_cens = np.sum(q_u * S_k_cens)
        N1 += N1_cens / 2
        sum_qs += sum_s_cens
    return {'N0': N0, 'N1': N1, 'sum_qs': sum_qs, 'n_kgaps': n_kgaps}

def quad_solve(N0, N1, sum_qs):
    aa = sum_qs
    bb = -(N0 + 2 * N1 + sum_qs)
    cc = 2 * N1
    qq = -(bb - np.sqrt(bb ** 2 - 4 * aa * cc)) / 2
    theta_mle = cc / qq
    return theta_mle

def exp_info(theta, ss, inc_cens):
    not_right_censored = ss['N0'] + ss['N1'] - (ss['n_kgaps'] - ss['N0'] - ss['N1'])
    term1 = 1 / (1 - theta)
    term2 = 2 / theta
    if inc_cens:
        term3 = 2 / theta
    else:
        term3 = 0
    val = not_right_censored * (term1 + term2) + term3
    return val

def loglik(theta, N0, N1, sum_qs, n_kgaps):
    if theta < 0 or theta > 1:
        return float('-inf')
    loglik = 0
    if N1 > 0:
        loglik += 2 * N1 * math.log(theta) - sum_qs * theta
    if N0 > 0:
        loglik += N0 * math.log(1 - theta)
    return loglik

# fini