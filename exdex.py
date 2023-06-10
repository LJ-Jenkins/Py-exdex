import exdex_dgaps
import exdex_kgaps
import exdex_iwls
import exdex_int
import math
import numpy as np
from scipy.optimize import minimize_scalar

def dgaps(data, u, D=1, inc_cens=True):
    data = np.array(data)
    data = np.reshape(data, (data.size,1))
    if not isinstance(u, (int, float)):
        raise ValueError("u must be a numeric scalar")
    if u >= np.nanmax(data):
        raise ValueError("'u' must be less than 'max(data, na.rm = TRUE)'")
    if not isinstance(D, (int, float)) or D < 0:
        raise ValueError("D must be a non-negative scalar")
    if np.any(np.isnan(data)):
        data = exdex_int.split_by_nans(data)
    q_u = np.mean(data[~np.isnan(data)] > u)
    if data.shape[1] == 1:
        ss = exdex_dgaps.stat(data, u, q_u, D, inc_cens)
    else:
        stats_list = [exdex_dgaps.stat(col, u, q_u, D, inc_cens) for col in data.T]
        ss = {'N0' : sum([item['N0'] for item in stats_list]),
           'N1' : sum([item['N1'] for item in stats_list]),
           'n_dgaps' : sum([item['n_dgaps'] for item in stats_list]),
           'sum_qtd' : sum([item['sum_qtd'] for item in stats_list])}
    ss['q_u'] = q_u
    ss['D'] = D
    N0 = ss['N0']
    N1 = ss['N1']
    if N1 == 0:
        theta_mle = 0
    elif N0 == 0:
        theta_mle = 1
    else:
        def dgaps_negloglik(theta):
            return -exdex_dgaps.loglik(theta, **ss)
        #result = minimize(dgaps_negloglik, theta_init, method='bounded', bounds=[(0, 1)])
        result = minimize_scalar(dgaps_negloglik, bounds=(0, 1), method='bounded')
        theta_mle = result.x
    exp_info = exdex_dgaps.exp_info(theta_mle, ss=ss, inc_cens=inc_cens) if N1 > 0 else np.nan
    obs_info = 0
    if N0 > 0:
        if N1 > 0 or D == 0:
            obs_info -= N0 * exdex_int.gdd_theta(theta_mle, q_u=q_u, D=D)
        else:
            obs_info = np.nan
    if N1 > 0:
        obs_info += 2 * N1 / (theta_mle ** 2)
    if not np.isnan(obs_info) and obs_info <= 0:
        theta_se = np.nan
        se_exp = np.nan
    else:
        theta_se = np.sqrt(1 / obs_info)
        se_exp = 1 / np.sqrt(exp_info)
    max_loglik = exdex_dgaps.loglik(theta_mle, **ss)
    res = {
        'theta': theta_mle,
        'se': theta_se,
        'se_exp': se_exp,
        'N0': N0,
        'N1': N1,
        'sum_qtd': ss['sum_qtd'],
        'n_dgaps': ss['n_dgaps'],
        'q_u': ss['q_u'],
        'D': D,
        'u': u,
        'inc_cens': inc_cens,
        'max_loglik': max_loglik
    }
    return res

def kgaps(data, u, k=1, inc_cens=True):
    data = np.array(data)
    data = np.reshape(data, (data.size,1))
    if not isinstance(u, (int, float)) or not isinstance(k, int) or k < 0:
        raise ValueError("Invalid arguments: u must be a numeric scalar, k must be a non-negative integer")
    if u >= np.nanmax(data):
        raise ValueError("'u' must be less than 'max(data, na.rm = TRUE)'")
    if np.any(np.isnan(data)):
        data = exdex_int.split_by_nans(data)
    q_u = np.mean(data[~np.isnan(data)] > u)
    if data.shape[1] == 1:
        ss = exdex_kgaps.stat(data, u, q_u, k, inc_cens)
    else:
        stats_list = [exdex_kgaps.stat(col, u, q_u, k, inc_cens) for col in data.T]
        ss = {'N0' : sum([item['N0'] for item in stats_list]),
           'N1' : sum([item['N1'] for item in stats_list]),
           'n_kgaps' : sum([item['n_kgaps'] for item in stats_list]),
           'sum_qs' : sum([item['sum_qs'] for item in stats_list])}
    stats_list = [exdex_kgaps.stat(col, u, q_u, k, inc_cens) for col in data.T]
    ss = {}
    for key in stats_list[0].keys():
        ss[key] = np.sum([stats[key] for stats in stats_list])
    N0 = ss['N0']
    N1 = ss['N1']
    if N1 == 0:
        theta_mle = 0
    elif N0 == 0:
        theta_mle = 1
    else:
        sum_qs = ss['sum_qs']
        theta_mle = exdex_kgaps.quad_solve(N0, N1, sum_qs)
    if N1 > 0 and N0 > 0:
        exp_info = exdex_kgaps.exp_info(theta_mle, ss, inc_cens)
    else:
        exp_info = np.nan
    se_exp = 1 / np.sqrt(exp_info) if not np.isnan(exp_info) else np.nan
    obs_info = 0
    if N0 > 0:
        obs_info += N0 / (1 - theta_mle) ** 2
    if N1 > 0:
        obs_info += 2 * N1 / theta_mle ** 2
    theta_se = np.sqrt(1 / obs_info) if obs_info > 0 else 0
    if k == 0:
        theta_se = 0
    max_loglik = exdex_kgaps.loglik(theta_mle, **ss)
    res = {
        'theta': theta_mle,
        'se': theta_se,
        'se_exp': se_exp,
        'N0': N0,
        'N1': N1,
        'sum_qs': ss['sum_qs'],
        'n_kgaps': ss['n_kgaps'],
        'k': k,
        'u': u,
        'inc_cens': inc_cens,
        'max_loglik': max_loglik
    }
    return res

def iwls(data, u, maxit=100):
    data = np.array(data)
    data = np.reshape(data, (data.size,1))
    if not isinstance(u, (int, float)):
        raise ValueError("u must be a numeric scalar")
    if u >= np.max(data):
        raise ValueError("u must be less than max(data)")
    nx = len(data)
    exc_u = [i+1 for i, d in enumerate(data) if d > u]
    N = len(exc_u)
    T_u = [exc_u[i] - exc_u[i-1] for i in range(1, len(exc_u))]
    S_1 = [max(T-1, 0) for T in T_u]
    n_gaps = len(S_1)
    n_wls = len(np.array(S_1) > 0)
    S_1_sort = sorted(S_1, reverse=True)
    qhat = N / nx
    S_1_sort = [gap * qhat for gap in S_1_sort]
    exp_qs = [-math.log(i/N) for i in range(1, N)]
    indices = np.arange(N, 0, -1)
    ws = 1 / np.cumsum(1 / indices**2)
    ws = np.sort(ws)
    old_n_wls = n_wls
    diff_n_wls = 1
    niter = 1
    while diff_n_wls != 0 and niter < maxit:
        temp = exdex_iwls.fun(n_wls, N, S_1_sort, exp_qs, ws, nx)
        n_wls = temp["n_wls"]
        diff_n_wls = n_wls - old_n_wls
        old_n_wls = n_wls
        niter += 1
    conv = 1 if diff_n_wls > 0 else 0
    n_wls = temp["n_wls"]
    theta = temp["theta"]
    res = {"theta": theta, "conv": conv, "niter": niter, "n_gaps": n_gaps}
    return res

# fini