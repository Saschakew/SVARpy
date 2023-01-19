import numpy as np
import SVAR.MoG
import SVAR.SVARutilGMM

np.random.seed(0)
n = 3
T = 500
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



# default options
prepOptions = dict()
options = SVAR.estimatorGMMWF.prepareOptions(u=u, **prepOptions)
SVAR_out = SVAR.SVARest(u, estimator='GMM_WF', options=options, prepared=True)

# default options
prepOptions = dict()
SVAR_out = SVAR.SVARest(u, estimator='GMM_WF', prepOptions=prepOptions)


# All available options
prepOptions = dict()
prepOptions['addThirdMoments'] = False
prepOptions['addFourthMoments'] = True
prepOptions['moments'] = []  # moments used for gMM (default will be set to 3-4 Moments)
prepOptions['bstart'] = []  # optimizarion start value (default will be solution of Cholesky decomposition)
prepOptions['bstartopt'] ='Rec'
prepOptions['n_rec'] = []
prepOptions['printOutput'] = True
# options = SVAR.estimatorGMMWF.prepareOptions(u=u, **prepOptions)
# SVAR_out = SVAR.SVARest(u, estimator='GMM_WF', options=options, prepared=True)
SVAR_out = SVAR.SVARest(u, estimator='GMM_WF', prepOptions=prepOptions)


# Fully-Recrusive SVAR
prepOptions = dict()
prepOptions['n_rec'] = n
SVAR_out = SVAR.SVARest(u, estimator='GMM_WF', prepOptions=prepOptions)

# Partly-Recrusive SVAR
prepOptions = dict()
prepOptions['n_rec'] = 1
SVAR_out = SVAR.SVARest(u, estimator='GMM_WF', prepOptions=prepOptions)

# Manually pass optimization start value
prepOptions = dict()
prepOptions['bstart'] = np.array([0,0,1])
prepOptions['bstartopt'] ='specific'
SVAR_out = SVAR.SVARest(u, estimator='GMM_WF', prepOptions=prepOptions)


# Autonomatically generate moments (here  third)
prepOptions = dict()
prepOptions['addThirdMoments'] = True
prepOptions['addFourthMoments'] = False
SVAR_out = SVAR.SVARest(u, estimator='GMM_WF', prepOptions=prepOptions)





