import numpy as np
import SVAR
import SVAR.SVARutilGMM
from SVAR.SVARutilGMM import get_Moments, get_Moments_powerindex, get_f, get_g, get_G_ana, get_S_ana, get_Wopt_ana
from SVAR.estimatorGMMW import get_W_fast, get_GMM_W_Avar_param


def loss(u, b, W, restrictions, moments, moments_powerindex, blocks=False):
    whiten = False
    g = get_g(b=b, u=u, restrictions=restrictions, moments=moments, moments_powerindex=moments_powerindex,
              whiten=whiten, blocks=blocks)
    Q = np.linalg.multi_dot([g, W, g])
    return Q


def get_W_opt(u, b, restrictions, moments, whiten=False, blocks=False):

    def get_W_optimal(u, b, restrictions, moments, whiten=False, blocks=False):
        moments_fast = get_Moments_powerindex(moments)
        f = get_f(u=u, b=b, restrictions=restrictions, moments=moments, moments_powerindex=moments_fast, whiten=whiten,
                  blocks=blocks)
        W = np.cov(f.T)
        W = np.linalg.inv(W)
        return W

    W = get_W_optimal(u, b, restrictions, moments, whiten=whiten, blocks=blocks)

    return W


def gradient(u, b, Jacobian, W, restrictions, moments, moments_powerindex):
    e = SVAR.innovation(u, b, restrictions=restrictions)
    Omega = SVAR.SVARutil.get_Omega(e)

    Jac_temp = Jacobian(u=u, b=b, restrictions=restrictions, Omega=Omega)
    this_g = get_g(b=b[:], u=u[:], restrictions=restrictions[:],
                   moments=moments[:], moments_powerindex=moments_powerindex)
    dGMM_temp = np.linalg.multi_dot([np.transpose(Jac_temp), W, this_g])
    return (dGMM_temp)


# Estimate Avar and tests
def get_GMM_Avar_nonparam(u, b, Omega, Jacobian, moments, restrictions, W=[]):
    n, n = np.shape(restrictions)
    G = Jacobian(u=u, b=b, restrictions=restrictions, Omega=Omega)
    moments_powerindex = get_Moments_powerindex(moments)
    f = get_f(u=u, b=b, restrictions=restrictions, moments=moments,
              moments_powerindex=moments_powerindex)
    S = np.cov(f.T)
    if np.array(W).size == 0:
        W = np.linalg.inv(S)
    M1 = np.linalg.inv(np.matmul(np.matmul(np.transpose(G), W), G))
    M2 = np.matmul(np.transpose(G), W)
    M = np.matmul(M1, M2)
    V_est = np.matmul(np.matmul(M, S), np.transpose(M))

    elementcounter = 0
    for i in range(n):
        for j in range(n):
            if not (np.isnan(restrictions[i, j])):
                V_est = np.insert(V_est, elementcounter, np.full(np.shape(V_est)[1], np.NaN), 0)
                V_est = np.insert(V_est, elementcounter, np.full(np.shape(V_est)[0], np.NaN), 1)
            elementcounter += 1
    return V_est





# Only for GMM (not w or wf)
def prepareOptions(u,
                   addThirdMoments=True, addFourthMoments=True, moments=[],
                   bstart=[], bstartopt='Rec',
                   restrictions=[], n_rec=False,
                   kstep=2, W=[],  Wstartopt='I',
                   printOutput=True):
    options = dict()

    blocks = False
    moments_blocks = True

    estimator = 'GMM'
    options['estimator'] = estimator

    T, n = np.shape(u)
    options['T'] = T
    options['n'] = n

    options['printOutput'] = printOutput
    options['whiten'] = False

    options['kstep'] = kstep

    restrictions, blocks = SVAR.estPrepare.prepare_blocks_restrictions(n, n_rec, blocks, restrictions)
    options['restrictions'] = restrictions
    options['blocks'] = blocks

    moments = SVAR.estPrepare.prepare_moments(estimator, moments, addThirdMoments, addFourthMoments, moments_blocks,
                                              blocks, n)
    options['moments'] = moments
    options['moments_powerindex'] = SVAR.SVARutilGMM.get_Moments_powerindex(moments)

    bstart = SVAR.estPrepare.prepare_bstart(estimator, bstart, u, restrictions, n_rec, blocks, bstartopt=bstartopt)
    options['bstart'] = bstart

    Jacobian = SVAR.SVARutilGMM.generate_Jacobian_function(moments=moments, restrictions=restrictions)
    options['Jacobian'] = Jacobian

    # Weighting
    W = SVAR.estPrepare.prepare_W(u, n, W, Wstartopt, moments, bstart, restrictions, blocks, options['whiten'] )
    options['W'] = W

    if np.shape(options['moments'])[0] < np.shape(options['bstart'])[0]:
        raise ValueError('Less moment conditions than parameters. The SVAR is not identified')

    return options


def SVARout(est_SVAR, options, u):
    out_SVAR = dict()
    out_SVAR['options'] = options

    b_est = est_SVAR['x']
    out_SVAR['b_est'] = b_est

    B_est = SVAR.get_BMatrix(b_est, restrictions=options['restrictions'], whiten=options['whiten'],
                             blocks=options['blocks'])
    out_SVAR['B_est'] = B_est

    e = SVAR.innovation(u, b_est, restrictions=options['restrictions'], whiten=options['whiten'],
                        blocks=options['blocks'])
    out_SVAR['e'] = e

    Omega_all = SVAR.SVARutil.get_Omega(e)
    out_SVAR['Omega_all'] = Omega_all
    omega = SVAR.SVARutil.get_Omega_Moments(e)
    out_SVAR['omega'] = omega

    out_SVAR['loss'] = est_SVAR['fun']

    V_est = get_GMM_Avar_nonparam(u, out_SVAR['b_est'], out_SVAR['Omega_all'], options['Jacobian'],
                                      options['moments'], options['restrictions'], options['W'])
    out_SVAR['Avar_est'] = V_est

    if options['printOutput']:
        print('Estimated B:')
        SVAR.estOutput.print_B(out_SVAR['B_est'])

        print('Estimated Avar of B:')
        SVAR.estOutput.print_Avar(out_SVAR['Avar_est'])

        print('Moments of unmixed innovations')
        SVAR.estOutput.print_Moments(out_SVAR['omega'])

    return out_SVAR
