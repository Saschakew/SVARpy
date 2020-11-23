import numpy as np
import SVAR.MoG




## Simulate SVAR
np.random.seed(0)
n = 4
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

# ## ----------------------------##
# ## Recursive SVAR (with cholesky)
# ## ----------------------------##
# # Note: use Non-Recrusive SVAR function with options_SVAR['n_rec']=n
# SVAR_out = SVAR.estSVAR.get_B_recursive(u)
# SVAR_out['B_est']


## ----------------------------##
## Non-Recrusive SVAR (with GMM)
## ----------------------------##
# Default options
SVAR_out = SVAR.SVARest(u, estimator='GMM')
SVAR_out['B_est']

# All available options (see Details_GMM.py)
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
SVAR_out = SVAR.SVARest(u, estimator='GMM', prepOptions=prepOptions)


## ----------------------------##
## Non-Recrusive SVAR (GMM White)
## ----------------------------##
# Default options
SVAR_out = SVAR.SVARest(u, estimator='GMM_W')


# All available options (see Details_GMM_W.py)
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
SVAR_out = SVAR.SVARest(u, estimator='GMM_W', prepOptions=prepOptions)
SVAR_out['B_est']

## ----------------------------##
## Non-Recrusive SVAR (GMM White Fast)
## ----------------------------##
# Default options
SVAR_out = SVAR.SVARest(u, estimator='GMM_WF')
SVAR_out['B_est']


# All available options (see Details_GMM_WF.py)
prepOptions = dict()
prepOptions['addThirdMoments'] = False
prepOptions['addFourthMoments'] = True
prepOptions['moments'] = []  # moments used for gMM (default will be set to 3-4 Moments)
prepOptions['bstart'] = []  # optimizarion start value (default will be solution of Cholesky decomposition)
prepOptions['bstartopt'] ='Rec'
prepOptions['n_rec'] = []
prepOptions['printOutput'] = True
SVAR_out = SVAR.SVARest(u, estimator='GMM_WF', prepOptions=prepOptions)
SVAR_out['B_est']



## ----------------------------##
## Non-Recrusive SVAR (with PML and t-distribution)
## ----------------------------##
# Default options
SVAR_out = SVAR.SVARest(u, estimator='PML')
SVAR_out['B_est']

# All available options (see Details_PML.py)
prepOptions = dict()
prepOptions['df_t'] = 7 # degrees of freedom of pseudo t-distribution
prepOptions['bstart'] = []
prepOptions['n_rec'] = []
SVAR_out = SVAR.SVARest(u, estimator='PML', prepOptions=prepOptions)
SVAR_out['B_est']

