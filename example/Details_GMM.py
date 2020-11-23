import numpy as np
import SVAR.MoG
import SVAR.SVARutilGMM

## Simulate SVAR
np.random.seed(0)
n = 3
T = 150
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


prepOptions = dict()
options = SVAR.estimatorGMM.prepareOptions(u=u, **prepOptions)
SVAR_out = SVAR.SVARest(u, estimator='GMM', options=options, prepared=True)


## Non-Recrusive SVAR (with GMM)
# default options
SVAR_out = SVAR.SVARest(u, estimator='GMM')



# All available options
prepOptions = dict()
prepOptions['addThirdMoments'] = False
prepOptions['addFourthMoments'] = True
prepOptions['moments'] = [] # moments used for gMM (default will be set to 2-4 Moments)
prepOptions['bstart'] = [] # optimizarion start value (default will be solution of Cholesky decomposition)
prepOptions['bstartopt'] = 'Rec' #
prepOptions['n_rec'] = False
prepOptions['kstep'] = 2 # k-step GMM (default is two-step GMM)
prepOptions['W'] = [] # weighting matrix of first step (default will be set to identity matrix)
prepOptions['Wstartopt'] = 'WoptBstart' #'WoptparaNorm' # compute optimal weighting matrix at bstart during first step of k-step GMM (default false)
prepOptions['printOutput'] = True
# options = SVAR.estimatorGMM.prepareOptions(u=u, **prepOptions)
# SVAR_out = SVAR.SVARest(u, estimator='GMM', options=options, prepared=True)
SVAR_out = SVAR.SVARest(u, estimator='GMM', prepOptions=prepOptions)



# Fully-Recrusive SVAR
prepOptions = dict()
prepOptions['n_rec'] = n
SVAR_out = SVAR.SVARest(u, estimator='GMM', prepOptions=prepOptions)

# Partly-Recrusive SVAR
prepOptions = dict()
prepOptions['n_rec'] = 1
SVAR_out = SVAR.SVARest(u, estimator='GMM', prepOptions=prepOptions)

# Manually pass optimization start value
prepOptions = dict()
prepOptions['bstart'] = np.array([ 0.99572253,  0.0021608 ,  0.02561262,  0.0152424 ,  1.01568046,
       -0.00357898, -0.03177731, -0.01363025,  0.98135826])
prepOptions['bstartopt'] = 'specific'
SVAR_out = SVAR.SVARest(u, estimator='GMM', prepOptions=prepOptions)


# Start optimization at recursive solution
prepOptions = dict()
prepOptions['bstartopt'] = 'Rec'
SVAR_out = SVAR.SVARest(u, estimator='GMM', prepOptions=prepOptions)


# Start optimization at identity matrix
prepOptions = dict()
prepOptions['bstartopt'] = 'I'
SVAR_out = SVAR.SVARest(u, estimator='GMM', prepOptions=prepOptions)


# Start optimization at solution of GMM_WF
prepOptions = dict()
prepOptions['bstartopt'] = 'GMM_WF'
SVAR_out = SVAR.SVARest(u, estimator='GMM', prepOptions=prepOptions)


# Manually pass moments (here only second and third)
moments = SVAR.SVARutilGMM.get_Mr(2, n)
moments = np.append(moments, SVAR.SVARutilGMM.get_Cr(2, n), axis=0)
moments = np.append(moments, SVAR.SVARutilGMM.get_Cr(3, n), axis=0)
prepOptions = dict()
prepOptions['moments'] = moments
SVAR_out = SVAR.SVARest(u, estimator='GMM', prepOptions=prepOptions)


# Autonomatically generate moments (here   second and third)
prepOptions = dict()
prepOptions['addThirdMoments'] = True
prepOptions['addFourthMoments'] = False
SVAR_out = SVAR.SVARest(u, estimator='GMM', prepOptions=prepOptions)


# Set k-step of GMM
prepOptions = dict()
prepOptions['kstep'] = 1
SVAR_out = SVAR.SVARest(u, estimator='GMM', prepOptions=prepOptions)



# Manually pass optimal weighting matrix for first-step GMM
moments = SVAR.SVARutilGMM.get_Mr(2, n)
moments = np.append(moments, SVAR.SVARutilGMM.get_Cr(2, n), axis=0)
moments = np.append(moments, SVAR.SVARutilGMM.get_Cr(3, n), axis=0)
prepOptions = dict()
prepOptions['moments'] = moments
prepOptions['W'] = np.eye(np.shape(moments)[0])
prepOptions['kstep'] = 1
prepOptions['Wstartopt'] = 'specific'
SVAR_out = SVAR.SVARest(u, estimator='GMM', prepOptions=prepOptions)

# compute optimal weighting matrix at bstart during first step of k-step GMM (default false)
prepOptions = dict()
prepOptions['Wstartopt'] = 'WoptBstart'
prepOptions['kstep'] = 1 # k-step GMM (default is two-step GMM)
SVAR_out = SVAR.SVARest(u, estimator='GMM', prepOptions=prepOptions)








