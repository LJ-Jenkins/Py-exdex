# Py-exdex
Python translation of the exdex R package (Estimation of the Extremal Index):
 
 
 https://github.com/paulnorthrop/exdex/tree/master
 
 https://paulnorthrop.github.io/exdex/
 
 
 All code has only been minimally tested (but results mirror that of the R package) - use with caution
 
 
At present, only the functions for the extremal index calculation have been completed, in the future I will look to translating the rest of the package, such as the diagnostic plots. For full details about the theory, functions, and inputs, please refer to the original package.
 
 
Syntax is exdex.function(), the currently working functions are:

```python:Code
import exdex as ex
import numpy as np

data = np.random.randn(100, 100)
u = np.nanpercentile(data, 99)
res = ex.dgaps(data, u, D=10, inc_cens=True)
print(res)
{'theta': 0.9999944125101555, 'se': 0.03818228483786259, 'se_exp': 0.034222672761982896, 'N0': 7, 'N1': 93.0, 'sum_qtd': 99.57000000000001, 'n_dgaps': 101, 'q_u': 0.01, 'D': 10, 'u': 2.349454470755292, 'inc_cens': True, 'max_loglik': -116.03532746020215}
```

All credit and thanks to authors of the original R package.
