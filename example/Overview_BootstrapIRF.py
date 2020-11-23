import SVAR.MoG
import numpy as np

## Simulate SVAR
np.random.seed(0)
n = 2
T = 2500
# Specitfy B_true
B_true = np.eye(n)
# Draw structural shocks
mu1, sigma1 = (-0.284, np.power(0.28, 2))
mu2, sigma2 = (0.409, np.power(1.43, 2))
lamb = 0.59
eps = np.empty([T, n])
for i in range(n):
    eps[:, i] = SVAR.MoG.MoG_rnd(np.array([[mu1, sigma1], [mu2, sigma2]]), lamb, T)
# Generate u
u = np.matmul(B_true, np.transpose(eps))
u = np.transpose(u)
# Generate y
AR1 = np.array([[0.7, 0.5], [-0.5, 0]])
AR2 = np.array([[0.3, 0], [0, 0]])
AR3 = np.array([[-0.2, 0], [0, 0]])
AR4 = np.array([[0.1, 0], [0, 0]])
AR = np.dstack((AR1, AR2, AR3, AR4))
const = np.zeros(n)
trend = np.zeros(n)
trend2 = np.zeros(n)
y = SVAR.estSVARbootstrap.simulate_SVAR(u, AR, const, trend, trend2)

## Options
# Options Bootstrap
opt_bootstrap = dict()
opt_bootstrap['number_bootstrap'] = 50
opt_bootstrap['update_bstart'] = False
opt_bootstrap['find_perm'] = True
# Options reduced form
opt_redform = dict()
opt_redform['lags'] = 4
opt_redform['add_const'] = True
opt_redform['add_trend'] = False
opt_redform['add_trend2'] = False
# Options irf
opt_irf = dict()
opt_irf['irf_length'] = 14

## ----------------------------##
## Recursive SVAR (with cholesky)
## ----------------------------##
prepOptions = dict()
prepOptions['n_rec'] = n
out_irf, out_irf_bootstrap = SVAR.bootstrap_SVAR(y,   options_bootstrap=opt_bootstrap,
                                                 options_redform=opt_redform, options_irf=opt_irf, prepOptions=prepOptions)
SVAR.plot_IRF(out_irf, out_irf_bootstrap)

## ----------------------------##
## Non-Recrusive SVAR
## ----------------------------##
estimator='GMM' # GMM, GMM_W, GMM_WF, PML
prepOptions = dict() # see Overview_SVAR.py
out_irf, out_irf_bootstrap = SVAR.bootstrap_SVAR(y, estimator=estimator, options_bootstrap=opt_bootstrap,
                                                 options_redform=opt_redform, options_irf=opt_irf, prepOptions=prepOptions)
SVAR.plot_IRF(out_irf, out_irf_bootstrap)


estimator='GMM_WF' # GMM, GMM_W, GMM_WF, PML
prepOptions = dict() # see Overview_SVAR.py
out_irf, out_irf_bootstrap = SVAR.bootstrap_SVAR(y, estimator=estimator, options_bootstrap=opt_bootstrap,
                                                 options_redform=opt_redform, options_irf=opt_irf, prepOptions=prepOptions)
SVAR.plot_IRF(out_irf, out_irf_bootstrap)
