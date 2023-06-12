import exdex_dgaps
import exdex_kgaps
import exdex_iwls
import exdex_spm
import exdex_int
import math
import numpy as np
import pandas as pd
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

def spm(data, b, bias_adjust="BB3", constrain=True, varN=True, which_dj="last"):
    data = np.array(data)
    data = np.reshape(data, (data.size,1))
    if data is None or len(data) == 0:
        raise ValueError("'data' must be non-empty and >len(0)")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("'data' contains missing or infinite values")
    if not isinstance(b, int) or b < 1:
        raise ValueError("'b' must be a positive integer")
    if bias_adjust not in ["BB3", "BB1", "N", "none"]:
        raise ValueError("'bias_adjust' must be one of 'BB3', 'BB1', 'N', or 'none'")
    if not isinstance(constrain, bool):
        raise ValueError("'constrain' must be a logical value")
    if not isinstance(varN, bool):
        raise ValueError("'varN' must be a logical value")
    if which_dj not in ["last", "first"]:
        raise ValueError("'which_dj' must be either 'last' or 'first'")
    k_n = len(data) // b
    if k_n < 1:
        raise ValueError("b is too large: it is larger than length(data)")
    all_max = exdex_spm.all_max_rcpp(data, b, which_dj)
    res = exdex_spm.ests_sigmahat_dj(all_max, b, which_dj, bias_adjust)
    Fhaty = exdex_int.ecdf2(all_max['xs'], all_max['ys'])
    k_n_sl = len(all_max['ys'])
    m = len(all_max['xs'])
    const = -np.log(m - b + k_n_sl)
    if bias_adjust == "N":
        Fhaty = (m * Fhaty - b) / (m - b)
    res['theta_sl'] = np.array((-1 / np.mean(b * exdex_int.log0const(Fhaty, const)),
        1 / (b * np.mean(1 - Fhaty))))
    res['data_sl'] = np.hstack((-b * np.log(Fhaty), b * (1 - Fhaty)))
    res['sigma2sl'] = res['sigma2dj_for_sl'] - (3 - 4 * np.log(2)) / res['theta_sl'] ** 2
    res['sigma2sl'][res['sigma2sl'] <= 0] = np.nan
    index = [0, 1] if varN else [1, 1]
    res['se_dj'] = res['theta_dj'] ** 2 * np.sqrt(res['sigma2dj'][index] / k_n)
    res['se_sl'] = res['theta_sl'] ** 2 * np.sqrt(res['sigma2sl'][index] / k_n)
    res['raw_theta_dj'] = res['theta_dj']
    res['raw_theta_sl'] = res['theta_sl']
    if bias_adjust == "BB3":
        res['bias_dj'] = res['theta_dj'] / k_n + res['theta_dj'] ** 3 * res['sigma2dj'] / k_n
        res['theta_dj'] = res['theta_dj'] - res['bias_dj']
        BB3adj_sl = res['theta_sl'] / k_n + res['theta_sl'] ** 3 * res['sigma2sl'] / k_n
        BB1adj_sl = res['theta_sl'] / k_n
        res['bias_sl'] = np.where(np.isnan(res['se_sl']), BB1adj_sl, BB3adj_sl)
        res['theta_sl'] = res['theta_sl'] - res['bias_sl']
    elif bias_adjust == "BB1":
        res['bias_dj'] = res['theta_dj'] / k_n
        res['theta_dj'] = res['theta_dj'] - res['bias_dj']
        res['bias_sl'] = res['theta_sl'] / k_n
        res['theta_sl'] = res['theta_sl'] - res['bias_sl']
    else:
        res['bias_dj'] = res['bias_sl'] = np.array([0,0])
    res['theta_dj'] = np.append(res['theta_dj'], res['theta_dj'][1] - 1 / b)
    res['theta_sl'] = np.append(res['theta_sl'], res['theta_sl'][1] - 1 / b)
    if bias_adjust == "BB3" or bias_adjust == "BB1":
        res['bias_dj'] = np.append(res['bias_dj'], res['bias_dj'][1] + 1 / b)
        res['bias_sl'] = np.append(res['bias_sl'], res['bias_sl'][1] + 1 / b)
    else:
        res['bias_dj'] = np.append(res['bias_dj'], 1 / b)
        res['bias_sl'] = np.append(res['bias_sl'], 1 / b)
    res['se_dj'] = np.append(res['se_dj'], res['se_dj'][1])
    res['se_sl'] = np.append(res['se_sl'], res['se_sl'][1])
    res['uncon_theta_dj'] = res['theta_dj']
    res['uncon_theta_sl'] = res['theta_sl']
    if constrain:
        res['theta_dj'] = np.minimum(res['theta_dj'], 1)
        res['theta_sl'] = np.minimum(res['theta_sl'], 1)
    res['theta_dj'] = np.maximum(res['theta_dj'], 0)
    res['theta_sl'] = np.maximum(res['theta_sl'], 0)
    res['bias_adjust'] = bias_adjust
    res['b'] = b
    rn = ["N2015, sliding", "BB2018, sliding", "BB2018b, sliding", "N2015, disjoint", "BB2018, disjoint", "BB2018b, disjoint"]
    cn = ["Estimate", "Std. Error", "Bias Adjustment"]
    summary_table = pd.DataFrame(np.reshape((res['theta_sl'], res['theta_dj'], 
                                             res['se_sl'], res['se_dj'], res['bias_sl'], 
                                             res['bias_dj']),[3,6]).T,
                                 columns=cn, index=rn)
    res['summary'] = summary_table
    print(res['summary'])
    return res

# fini