import numpy as np
import SVAR
from SVAR.SVARutilGMM import get_Moments, get_Moments_powerindex, get_f, get_g, get_G_ana, get_S_ana, get_Wopt_ana






def loss(u, b, W, restrictions, moments, moments_powerindex, blocks=False):
    whiten = True
    g = get_g(b=b, u=u, restrictions=restrictions, moments=moments, moments_powerindex=moments_powerindex,
              whiten=whiten, blocks=blocks)
    Q = np.linalg.multi_dot([g, W, g])
    return Q

def get_W_fast(Moments):
    W = np.eye(np.shape(Moments)[0])

    counter = 0
    for moment in Moments:
        r = np.sum(moment)

        tmp = np.zeros(np.shape(moment))
        counter_in = 0
        for i in moment:
            tmp[counter_in] = np.math.factorial(moment[counter_in])
            counter_in += 1

        W[counter, counter] = np.math.factorial(r) / np.prod(tmp)
        counter += 1

    return W


def get_GMM_W_Avar_param(Moments, B, omega, restrictions=[], W=[]):
    # Calculates analytically the asymptotic variance of sqrt(T)(b_hat-b) with b_hat GMM estimator with optimal weighting

    n, n = np.shape(B)

    if np.array(restrictions).size == 0: restrictions = np.full([n, n], np.nan)
    number_of_restrictions = np.sum(restrictions == 0)

    Moments2 = SVAR.SVARutilGMM.get_Mr(2, n)
    Moments2 = np.append(Moments2, SVAR.SVARutilGMM.get_Cr(2, n), axis=0)

    G2 = get_G_ana(Moments2, B, omega, restrictions)
    GC = get_G_ana(Moments, B, omega, restrictions)

    S22 = get_S_ana(Moments2, Moments2, omega)
    S2C = get_S_ana(Moments2, Moments, omega)
    SC2 = get_S_ana(Moments, Moments2, omega)
    SCC = get_S_ana(Moments, Moments, omega)

    if np.array(W).size == 0:  W = np.linalg.inv(SCC)

    H11 = G2
    H12 = np.zeros([np.shape(Moments2)[0], np.shape(Moments2)[0]])
    H21 = np.matmul(np.matmul(np.transpose(GC), W), GC)
    H22 = - np.transpose(G2)

    H = np.hstack((np.vstack((H11, H21)), np.vstack((H12, H22))))
    H = np.linalg.inv(H)

    M2 = H[0:(n * n - number_of_restrictions), 0: np.shape(Moments2)[0]]
    MC_tmp1 = H[0: (n * n - number_of_restrictions), np.shape(Moments2)[0]:]
    MC_tmp2 = np.matmul(np.transpose(GC), W)
    MC = np.matmul(MC_tmp1, MC_tmp2)

    V1 = np.matmul(np.matmul(M2, S22), np.transpose(M2))
    V2 = np.matmul(np.matmul(M2, S2C), np.transpose(MC))
    V3 = np.matmul(np.matmul(MC, SC2), np.transpose(M2))
    V4 = np.matmul(np.matmul(MC, SCC), np.transpose(MC))

    V = V1 + V2 + V3 + V4

    elementcounter = 0
    for i in range(n):
        for j in range(n):
            if not (np.isnan(restrictions[i, j])):
                V = np.insert(V, elementcounter, np.full(np.shape(V)[1], np.NaN), 0)
                V = np.insert(V, elementcounter, np.full(np.shape(V)[0], np.NaN), 1)
            elementcounter += 1

    return V



def prepareOptions(u,
                   addThirdMoments=True, addFourthMoments=True, moments=[],
                   bstart=[], bstartopt='Rec',
                    n_rec=False,
                   W=[], Wstartopt='I',
                   printOutput=True
                   ):
    options = dict()

    blocks = False
    moments_blocks = True

    estimator = 'GMM_W'
    options['estimator'] = estimator

    T, n = np.shape(u)
    options['T'] = T
    options['n'] = n

    options['printOutput'] = printOutput

    options['whiten'] = True
    _, V = SVAR.do_whitening(u, white=True)
    options['V'] = V

    restrictions, blocks = SVAR.estPrepare.prepare_blocks_restrictions(n, n_rec, blocks, restrictions=[])
    options['restrictions'] = restrictions
    options['blocks'] = blocks

    moments = SVAR.estPrepare.prepare_moments(estimator, moments, addThirdMoments, addFourthMoments, moments_blocks,
                                              blocks, n)
    options['moments'] = moments
    options['moments_powerindex'] = SVAR.SVARutilGMM.get_Moments_powerindex(moments)

    bstart = SVAR.estPrepare.prepare_bstart(estimator, bstart, u, restrictions, n_rec, blocks, bstartopt=bstartopt)
    options['bstart'] = bstart

    # Weighting
    if Wstartopt == 'I':
        W = np.eye(np.shape(moments)[0])
    elif Wstartopt == 'GMM_WF':
        W = get_W_fast(moments)
    elif Wstartopt == 'specific':
        W = W
    else:
        raise ValueError('Unknown option Wstartopt')
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
    B_est = np.matmul(options['V'], B_est)
    out_SVAR['B_est'] = B_est

    e = SVAR.innovation(u, b_est, restrictions=options['restrictions'], whiten=options['whiten'],
                        blocks=options['blocks'])
    out_SVAR['e'] = e

    Omega_all = SVAR.SVARutil.get_Omega(e)
    out_SVAR['Omega_all'] = Omega_all
    omega = SVAR.SVARutil.get_Omega_Moments(e)
    out_SVAR['omega'] = omega

    out_SVAR['loss'] = est_SVAR['fun']

    V_est = get_GMM_W_Avar_param(options['moments'], B=out_SVAR['B_est'], omega=out_SVAR['omega'],
                                   restrictions=options['restrictions'], W=options['W'])
    out_SVAR['Avar_est'] = V_est

    if options['printOutput']:
        print('Estimated B:')
        SVAR.estOutput.print_B(out_SVAR['B_est'])

        print('Estimated Avar of B:')
        SVAR.estOutput.print_Avar(out_SVAR['Avar_est'])

        print('Moments of unmixed innovations')
        SVAR.estOutput.print_Moments(out_SVAR['omega'])

    return out_SVAR