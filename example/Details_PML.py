import numpy as np
import SVAR.MoG

## Simulate SVAR
np.random.seed(0)
n = 3
T = 5000
# Specitfy B_true
B_true = np.eye(n)
# Draw structural shocks
mu1, sigma1 = (-0.284, np.power(0.28, 2))
mu2, sigma2 = (0.409, np.power(1.43, 2))
lamb = 0.59
eps = np.empty([T, n])
for i in range(n):
    eps[:, i] = SVAR.MoG.MoG_rnd(np.array([[mu1, sigma1], [mu2, sigma2]]), lamb, T)
# Generate reduced form shocks
u = np.matmul(B_true, np.transpose(eps))
u = np.transpose(u)



## Non-Recrusive SVAR (with GMM)
# default options
SVAR_out = SVAR.SVARest(u, estimator='PML')

prepOptions = dict()
SVAR_out = SVAR.SVARest(u, estimator='PML', prepOptions=prepOptions)

prepOptions = dict()
options = SVAR.estimatorPML.prepareOptions(u=u, **prepOptions)
SVAR_out = SVAR.SVARest(u, estimator='PML', options=options, prepared=True)


# All available options
prepOptions = dict()
prepOptions['bstart'] = []  # optimizarion start value (default will be solution of Cholesky decomposition)
prepOptions['bstartopt'] ='Rec'
prepOptions['df_t'] = 7 # degrees of freedom of pseudo t-distribution (default 7)
prepOptions['n_rec'] = False
# options = SVAR.estimatorPML.prepareOptions(u=u, **prepOptions)
# SVAR_out = SVAR.SVARest(u, estimator='PML', options=options, prepared=True)
SVAR_out = SVAR.SVARest(u, estimator='PML', prepOptions=prepOptions)


# Fully-Recrusive SVAR
prepOptions = dict()
prepOptions['n_rec'] = n
SVAR_out = SVAR.SVARest(u, estimator='PML', prepOptions=prepOptions)

# Partly-Recrusive SVAR
prepOptions = dict()
prepOptions['n_rec'] = 1
SVAR_out = SVAR.SVARest(u, estimator='PML', prepOptions=prepOptions)


# Manually pass optimization start value
prepOptions = dict()
prepOptions['bstart'] = np.array([0,0,0])
prepOptions['bstartopt'] ='specific'
SVAR_out = SVAR.SVARest(u, estimator='PML', prepOptions=prepOptions)

# Manually pass degrees of freedom of pseudo t-distribution
prepOptions = dict()
prepOptions['df_t'] = 7
SVAR_out = SVAR.SVARest(u, estimator='PML', prepOptions=prepOptions)





