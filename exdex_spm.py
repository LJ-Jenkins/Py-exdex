import numpy as np
import pandas as pd
import math
import exdex_int

def all_max_rcpp(x, b=1, which_dj='all'):
    df = pd.DataFrame(x) 
    df = df.rolling(b).max()
    ys = np.array(df[b-1:])
    n = len(x)
    n_max = math.floor(n / b)
    if which_dj == 'all':
        first_value = np.arange(1, n - n_max * b + 1)
        if first_value.size == 0 : first_value = [1]  
    elif which_dj == 'first':
        first_value = [1]
    elif which_dj == 'last':
        first_value = [n - n_max * b + 1]
    def get_maxima(first):
        s_ind = np.arange(first, b * n_max, b)
        return np.concatenate((ys[s_ind-1], x[first-1:first + n_max * b - 1]), axis=0)
    temp = np.vstack([get_maxima(first) for first in first_value])
    yd = temp[:n_max]
    xd = temp[n_max:]
    return {'ys': ys, 'xs': x, 'yd': yd, 'xd': xd}

def ests_sigmahat_dj(all_max, b, which_dj, bias_adjust='N'):
    k_n = all_max['yd'].shape[0]
    m = all_max['xd'].shape[0]
    const = -np.log(m - b + k_n)
    block = np.repeat(np.arange(1, k_n + 1), b)
    which_vals = np.arange(0, all_max['yd'].shape[1])
    def UsumN_fn(i):
        y = all_max['yd'][:, i]
        x = all_max['xd'][:, i]
        sum_fun = lambda q: [sum(x[block == q] <= yi) for yi in y]
        nums_mat = np.array([sum_fun(q) for q in np.unique(block)]).T
        Fhaty = np.sum(nums_mat, axis=1) / m
        logm = np.logical_not(np.eye(k_n))
        FhatjMni = np.array([np.sum(nums_mat[:, logm[i, :]], axis=1) for i in range(k_n)]).T / (m - b)
        UsumN = (-b * np.mean(exdex_int.log0const(FhatjMni, const), axis=0)).T
        Usum = (b * (1 - np.mean(FhatjMni, axis=0))).T
        out = [Fhaty, UsumN, Usum]
        return out
    temp = [UsumN_fn(in_val) for in_val in which_vals]
    Nhat = np.array([temp[i][0] for i in range(len(which_vals))]).T
    Zhat = b * (1 - Nhat)
    That = np.mean(Zhat, axis=0)
    Usum = np.array([temp[i][2] for i in range(len(which_vals))]).T
    Usum = (k_n * That - (k_n - 1) * (Usum).T).T
    Bhat = ((Zhat + Usum).T - 2 * That).T
    ZhatN = -b * np.log(Nhat)
    ThatN = np.mean(ZhatN, axis=0)
    UsumN = np.array([temp[i][1] for i in range(len(which_vals))]).T
    UsumN = (k_n * ThatN - (k_n - 1) * (UsumN).T).T
    BhatN = ((ZhatN + UsumN).T - 2 * ThatN).T
    BhatN = ((BhatN).T - np.mean(BhatN, axis=0)).T
    sigmahat2_dj = np.sum(Bhat ** 2, axis=0) / len(Bhat)
    sigmahat2_djN = np.sum(BhatN ** 2, axis=0) / len(BhatN)
    sigmahat2_dj_for_sl = np.sum(sigmahat2_dj) / len(sigmahat2_dj)
    sigmahat2_dj_for_slN = np.sum(sigmahat2_djN) / len(sigmahat2_djN)
    #sigma2dj_for_sl = [sigmahat2_dj_for_slN, sigmahat2_dj_for_sl]
    sigma2dj_for_sl = np.array((sigmahat2_dj_for_slN, sigmahat2_dj_for_sl))
    if which_dj == 'first':
        j = 0
    elif which_dj == 'last':
        j = len(sigmahat2_dj) - 1
    #sigma2dj = [sigmahat2_djN[j], sigmahat2_dj[j]]
    sigma2dj = np.array((sigmahat2_djN[j], sigmahat2_dj[j]))
    if bias_adjust == 'n':
        Nhat = (m * Nhat - b) / (m - b)
        That = np.mean(b * (1 - Nhat), axis=0).T
        ThatN = np.mean(-b * exdex_int.log0const(Nhat, const), axis=0).T
    theta_dj = 1 / np.array([ThatN[j], That[j]])
    res = {'sigma2dj': sigma2dj,
        'sigma2dj_for_sl': sigma2dj_for_sl,
        'theta_dj': theta_dj,
        'data_dj': np.array([-b * np.log(Nhat[:, j]), b * (1 - Nhat[:, j])]).T}
        #'data_dj': pd.DataFrame([-b * np.log(Nhat[:, j]), b * (1 - Nhat[:, j])]).T}
    return res

# fini