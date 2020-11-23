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

## Estiamte reduced form
opt_redform = dict()
opt_redform['add_const'] = True
opt_redform['add_trend'] = False
opt_redform['add_trend2'] = False
out_redform = SVAR.OLS_ReducedForm(y, lags = 1,  **opt_redform)
out_redform['u']
out_redform['AR']

## Estiamte reduced form with individual lag parameters
lags = np.array([2, 1])
out_redform = SVAR.OLS_ReducedForm(y, lags=lags, **opt_redform)

## Infocrit
maxLag = 2
out_info = SVAR.SVARbasics.infocrit(y, maxLag, **opt_redform)

## IRF (in an application B has to be estimated first)
# Calculate IRF
irf = SVAR.get_IRF(B_true, out_redform['AR'])

# # alternative with specific irf length
# opt_irf = dict()
# opt_irf['irf_length'] = 12
# irf = SVAR.get_IRF(B_true, out_redform['AR'], **opt_irf)

# Plot irf
SVAR.plot_IRF(irf, alpha=0.1)












