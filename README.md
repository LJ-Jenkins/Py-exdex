# Py-exdex
Python translation of the exdex R package (Estimation of the Extremal Index):
 
 
 https://github.com/paulnorthrop/exdex/tree/master
 
 https://paulnorthrop.github.io/exdex/
 
 
 All code has only been minimally tested (but results mirror that of the R package) - use with caution
 
 
At present, only the functions for the estimation of the extremal index have been completed, in the future I will look to translating the rest of the package. For full details about the theory, functions, and inputs, please refer to the original package.
 
 
Currently working functions are:

```python:Code
import exdex
import numpy as np

data = np.random.randn(100, 100)
u = np.nanpercentile(data, 99)

exdex.kgaps(data, u, k = 10)

{'theta': 0.9168945248423279, 'se': 0.02565280166846518, 
'se_exp': 0.026636993245887558, 'N0': 9, 'N1': 91.0, 
'sum_qs': 90.19999999999999, 'n_kgaps': 101, 'k': 10, 
'u': 2.2901046924963424, 'inc_cens': True, 
'max_loglik': -120.88352440589222}

exdex.dgaps(data, u, D = 10)

{'theta': 0.9999960671524637, 'se': 0.034820247675270484, 
'se_exp': 0.034222493177455515, 'N0': 9, 'N1': 91.0, 
'sum_qtd': 99.4, 'n_dgaps': 101, 'q_u': 0.01, 'D': 10, 
'u': 2.2901046924963424, 'inc_cens': True, 
'max_loglik': -120.56953811094117}

exdex.iwls(data, u)

{'theta': 0.9988527298931564, 'conv': 0, 
'niter': 3, 'n_gaps': 99}

res = exdex.spm(data, b = 5)

                   Estimate  Std. Error  Bias Adjustment
N2015, sliding     1.000000    0.010191         0.000608
BB2018, sliding    1.000000    0.004651         0.000623
BB2018b, sliding   1.000000    0.004651         0.200623
N2015, disjoint    0.997136    0.014483         0.000709
BB2018, disjoint   1.000000    0.013451         0.000750
BB2018b, disjoint  0.997665    0.013451         0.200750

```

All credit and thanks to authors of the original R package.

Dependencies:
numpy,
math,
statsmodels,
itertools,
scipy
