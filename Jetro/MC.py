import numpy as np
import SVAR.MoG
np.random.seed(0)


# Number of Variables n and sample size T
n = 3
T = 5000
# Specitfy B_true
B_true = np.eye(n)
# Draw structural shocks from mixture of normal
mu1, sigma1 = (-0.284, np.power(0.28, 2))
mu2, sigma2 = (0.409, np.power(1.43, 2))
lamb = 0.59


MCsize = 1000
save = np.full([n,n,MCsize],np.nan)
for mc in range(MCsize):
    eps = np.empty([T, n])
    for i in range(n):
        eps[:, i] = SVAR.MoG.MoG_rnd(np.array([[mu1, sigma1], [mu2, sigma2]]), lamb, T)
    # Generate reduced form shocks
    u = np.matmul(B_true, np.transpose(eps))
    u = np.transpose(u)

    # Estimate SVAR with GMM-WF
    prepOptions = dict()
    prepOptions['printOutput'] = False
    SVAR_out = SVAR.SVARest(u, estimator='GMM_WF', prepOptions=prepOptions )
    save[:,:,mc] = np.sqrt(T) * (SVAR_out['B_est'] - B_true)



print(  np.var(save,axis=2) )