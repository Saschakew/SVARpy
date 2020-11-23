import numpy as np
import SVAR
from SVAR.SVARutilGMM import get_Moments, get_Moments_powerindex, get_f, get_g, get_G_ana, get_S_ana, get_Wopt_ana
import SVAR.estimatorGMMW


def loss(u, b, restrictions, moments, moments_powerindex, blocks=False):
    whiten = True
    g = get_g(b=b, u=u, restrictions=restrictions, moments=moments, moments_powerindex=moments_powerindex,
              whiten=whiten, blocks=blocks)
    Q = - np.sum(np.power(g, 2))
    return Q


def get_GMM_WF_Avar_param(B, omega, restrictions=[], blocks=False, addThirdMoments=True, addFourthMoments=True):
    # ToDo: only works with default moments
    n, n = np.shape(B)
    comoments = get_Moments('GMM_W', n, blocks=blocks,
                            addThirdMoments=addThirdMoments,
                            addFourthMoments=addFourthMoments)
    Wfast = SVAR.estimatorGMMW.get_W_fast(comoments)
    V = SVAR.estimatorGMMW.get_GMM_W_Avar_param(comoments, B, omega, restrictions=restrictions, W=Wfast)

    return V


def prepareOptions(u,
                   addThirdMoments=True, addFourthMoments=True, moments=[],
                   bstart=[], bstartopt='Rec',
                   n_rec=False,
                   printOutput=True
                   ):
    options = dict()

    blocks = False
    moments_blocks = True

    estimator = 'GMM_WF'
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

    options['addThirdMoments'] = addThirdMoments
    options['addFourthMoments'] = addFourthMoments
    moments = SVAR.estPrepare.prepare_moments(estimator, moments, addThirdMoments, addFourthMoments, moments_blocks,
                                              blocks, n)
    options['moments'] = moments
    options['moments_powerindex'] = SVAR.SVARutilGMM.get_Moments_powerindex(moments)

    bstart = SVAR.estPrepare.prepare_bstart(estimator, bstart, u, restrictions, n_rec,blocks, bstartopt=bstartopt)
    options['bstart'] = bstart


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

    V_est = get_GMM_WF_Avar_param(out_SVAR['B_est'], out_SVAR['omega'], restrictions=options['restrictions'],
                                  blocks=options['blocks'],
                                  addThirdMoments=options['addThirdMoments'], addFourthMoments=options['addFourthMoments'])
    out_SVAR['Avar_est'] = V_est

    if options['printOutput']:
        print('Estimated B:')
        SVAR.estOutput.print_B(out_SVAR['B_est'])

        print('Estimated Avar of B:')
        SVAR.estOutput.print_Avar(out_SVAR['Avar_est'])

        print('Moments of unmixed innovations')
        SVAR.estOutput.print_Moments(out_SVAR['omega'])

    return out_SVAR
