import numpy as np
import math

def stat(data, u, q_u=None, D=1, inc_cens=True):
    if q_u is None:
        q_u = np.mean(data[~np.isnan(data)] > u)
    data = data[~np.isnan(data)]
    if not isinstance(u, (int, float)):
        raise ValueError("u must be a numeric scalar")
    if not isinstance(D, (int, float)) or D < 0:
        raise ValueError("D must be a non-negative scalar")
    if u >= np.nanmax(data):
        return {'N0': 0, 'N1': 0, 'sum_qtd': 0, 'n_dgaps': 0}
    nx = len(data)
    exc_u = np.where(data > u)[0] + 1
    N_u = len(exc_u)
    T_u = np.diff(exc_u)
    left_censored = T_u <= D
    N1 = np.sum(~left_censored)
    N0 = N_u - 1 - N1
    T_gt_D = T_u[~left_censored]
    sum_qtd = np.sum(q_u * T_gt_D)
    n_dgaps = N0 + N1
    if inc_cens:
        T_u_cens = np.array([exc_u[0] - 1, nx - exc_u[-1]])
        left_censored_cens = T_u_cens <= D
        N1_cens = np.sum(~left_censored_cens)
        n_dgaps += N1_cens
        T_gt_D_cens = T_u_cens[~left_censored_cens]
        sum_qtd_cens = np.sum(q_u * T_gt_D_cens)
        N1 += N1_cens / 2
        sum_qtd += sum_qtd_cens
    return {'N0': N0, 'N1': N1, 'sum_qtd': sum_qtd, 'n_dgaps': n_dgaps}


def loglik(theta, N0, N1, sum_qtd, n_dgaps, q_u, D):
    loglik = 0
    if N1 > 0:
        loglik += 2 * N1 * math.log(theta) - sum_qtd * theta
    if N0 > 0:
        loglik += N0 * math.log(1 - theta * math.exp(-theta * q_u * D))
    return loglik

def exp_info(theta, ss, inc_cens):
    not_right_censored = ss['N0'] + ss['N1'] - (ss['n_dgaps'] - ss['N0'] - ss['N1'])
    d = ss['q_u'] * ss['D']
    emtd = math.exp(-theta * d)
    term1 = (theta * d**2 - 2 * d + emtd) / (1 - theta * emtd)
    term2 = 2 / theta
    if inc_cens:
        term3 = 2 * emtd / theta
    else:
        term3 = 0
    val = not_right_censored * emtd * (term1 + term2) + term3
    return val

# fini