import numpy as np
import SVAR.MoG
import SVAR.estimatorGMMW
import SVAR.SVARutilGMM

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



# default options
prepOptions = dict()
options = SVAR.estimatorGMMW.prepareOptions(u=u, **prepOptions)
SVAR_out = SVAR.SVARest(u, estimator='GMM_W', options=options, prepared=True)


# default options
SVAR_out = SVAR.SVARest(u, estimator='GMM_W')



# All available options
prepOptions = dict()
prepOptions['addThirdMoments'] = False
prepOptions['addFourthMoments'] = True
prepOptions['moments'] = []  # moments used for gMM (default will be set to 3-4 Co-Moments)
prepOptions['bstart'] = []  # optimizarion start value (default will be solution of Cholesky decomposition)
prepOptions['bstartopt'] = 'Rec'  # optimizarion start value (default will be solution of Cholesky decomposition)
prepOptions['n_rec'] = False
prepOptions['W'] = []  # weighting matrix of first step (default will be set to identity matrix)
prepOptions['Wstartopt'] = 'I'  # compute optimal weighting matrix at bstart during first step of k-step GMM (default false)
prepOptions['printOutput'] = True
# options = SVAR.estimatorGMMW.prepareOptions(u=u, **prepOptions)
# SVAR_out = SVAR.SVARest(u, estimator='GMM_W', options=options, prepared=True)
SVAR_out = SVAR.SVARest(u, estimator='GMM_W', prepOptions=prepOptions)

# Fully-Recrusive SVAR
prepOptions = dict()
prepOptions['n_rec'] = n
SVAR_out = SVAR.SVARest(u, estimator='GMM_W', prepOptions=prepOptions)

# Partly-Recrusive SVAR
prepOptions = dict()
prepOptions['n_rec'] = 1
SVAR_out = SVAR.SVARest(u, estimator='GMM_W', prepOptions=prepOptions)

# Manually pass optimization start value
prepOptions = dict()
prepOptions['bstart'] = np.array([0,0,1])
prepOptions['bstartopt'] = 'specific'
SVAR_out = SVAR.SVARest(u, estimator='GMM_W', prepOptions=prepOptions)


# Manually pass moments (here only  third co-moments)
moments = SVAR.SVARutilGMM.get_Cr(3, n)
prepOptions = dict()
prepOptions['moments'] = moments
SVAR_out = SVAR.SVARest(u, estimator='GMM_W', prepOptions=prepOptions)


# Autonomatically generate moments (here  fourth)
prepOptions = dict()
prepOptions['addThirdMoments'] = False
prepOptions['addFourthMoments'] = True
SVAR_out = SVAR.SVARest(u, estimator='GMM_W', prepOptions=prepOptions)


# Fast weighting matrix (estimator equal to GMM_WF)
prepOptions = dict()
prepOptions['Wstartopt'] = 'GMM_WF'
SVAR_out = SVAR.SVARest(u, estimator='GMM_W', prepOptions=prepOptions)


# Manually pass  weighting matrix for first-step GMM
moments = SVAR.SVARutilGMM.get_Cr(3, n)
prepOptions = dict()
prepOptions['moments'] = moments
prepOptions['W'] = np.eye(np.shape(moments)[0])
prepOptions['Wstartopt'] = 'specific'
SVAR_out = SVAR.SVARest(u, estimator='GMM_W', prepOptions=prepOptions)




