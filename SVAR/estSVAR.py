from scipy import optimize as opt
from pyunlocbox import acceleration, solvers, functions
import numpy as np
import pandas as pd
import SVAR
import SVAR.estOutput
import SVAR.estimatorPML
import SVAR.estimatorGMM
import SVAR.estimatorGMMW
import SVAR.estimatorGMMWF
import SVAR.estimatorCholesky


def SVARest(u, estimator='GMM', options=dict(), prepOptions=dict(), prepared=False):
    if estimator == 'GMM':
        if not (prepared):
            options = SVAR.estimatorGMM.prepareOptions(u=u, **prepOptions)

        def optimize_this(options):
            this_loss = lambda b_vec: SVAR.estimatorGMM.loss(u, b_vec,
                                                             restrictions=options['restrictions'],
                                                             moments=options['moments'],
                                                             moments_powerindex=options['moments_powerindex'],
                                                             W=options['W'])
            this_grad = lambda b_vec: SVAR.estimatorGMM.gradient(u, b_vec,
                                                                 Jacobian=options['Jacobian'],
                                                                 W=options['W'],
                                                                 restrictions=options['restrictions'],
                                                                 moments=options['moments'],
                                                                 moments_powerindex=options['moments_powerindex'])
            optim_start = options['bstart']
            if np.shape(optim_start)[0] == 0:
                # raise ValueError('No free parameters.')
                ret = dict()
                ret['x'] = np.array([])
                ret['fun'] = this_loss(ret['x'])
            else:

                ret_tmp = opt.minimize(this_loss, optim_start, method='L-BFGS-B', jac=this_grad)

                ret = dict()
                ret['x'] = ret_tmp.x
                ret['fun'] = ret_tmp.fun
            return ret

        est_SVAR = optimize_this(options)

        for k in range(1, options['kstep']):
            bstart = est_SVAR['x']
            options['W'] = SVAR.estimatorGMM.get_W_opt(u, b=bstart, restrictions=options['restrictions'],
                                                       moments=options['moments'],
                                                       whiten=False, blocks=options['blocks'] )
            est_SVAR = optimize_this(options)

        out_SVAR = SVAR.estimatorGMM.SVARout(est_SVAR, options, u)




    elif estimator == 'GMM_W':
        if not (prepared):
            options = SVAR.estimatorGMMW.prepareOptions(u=u, **prepOptions)
        u, _ = SVAR.do_whitening(u, white=True)

        def optimize_this(options):
            this_loss = lambda b_vec: SVAR.estimatorGMMW.loss(u, b_vec,
                                                              restrictions=options['restrictions'],
                                                              moments=options['moments'],
                                                              moments_powerindex=options['moments_powerindex'],
                                                              W=options['W'],
                                                              blocks=options['blocks'])
            this_grad = []

            optim_start = options['bstart']
            if np.shape(optim_start)[0] == 0:
                # raise ValueError('No free parameters.')
                ret = dict()
                ret['x'] = np.array([])
                ret['fun'] = this_loss(ret['x'])
            else:
                ret_tmp = opt.minimize(this_loss, optim_start, method='L-BFGS-B', jac=this_grad)
                ret = dict()
                ret['x'] = ret_tmp.x
                ret['fun'] = ret_tmp.fun

            return ret

        est_SVAR = optimize_this(options)

        out_SVAR = SVAR.estimatorGMMW.SVARout(est_SVAR, options, u)

    elif estimator == 'GMM_WF':
        if not (prepared):
            options = SVAR.estimatorGMMWF.prepareOptions(u=u, **prepOptions)
        u, _ = SVAR.do_whitening(u, white=True)

        def optimize_this(options):
            this_loss = lambda b_vec: SVAR.estimatorGMMWF.loss(u, b_vec,
                                                               restrictions=options['restrictions'],
                                                               moments=options['moments'],
                                                               moments_powerindex=options['moments_powerindex'],
                                                               blocks=options['blocks'])
            this_grad = []

            optim_start = options['bstart']
            if np.shape(optim_start)[0] == 0:
                # raise ValueError('No free parameters.')
                ret = dict()
                ret['x'] = np.array([])
                ret['fun'] = this_loss(ret['x'])
            else:
                ret_tmp = opt.minimize(this_loss, optim_start, method='L-BFGS-B', jac=this_grad)
                ret = dict()
                ret['x'] = ret_tmp.x
                ret['fun'] = ret_tmp.fun
            return ret

        est_SVAR = optimize_this(options)

        out_SVAR = SVAR.estimatorGMMWF.SVARout(est_SVAR, options, u)

    elif estimator == 'PML':
        if not (prepared):
            options = SVAR.estimatorPML.prepareOptions(u=u, **prepOptions)
        u, _ = SVAR.do_whitening(u, white=True)

        def optimize_this(options):
            this_loss = lambda b_vec: SVAR.estimatorPML.LogLike_t(u, b_vec, options['df_t'], blocks=options['blocks'],
                                                                  whiten=options['whiten'])
            this_grad = []

            optim_start = options['bstart']
            if np.shape(optim_start)[0] == 0:
                # raise ValueError('No free parameters.')
                ret = dict()
                ret['x'] = np.array([])
                ret['fun'] = this_loss(ret['x'])
            else:
                ret_tmp = opt.minimize(this_loss, optim_start, method='L-BFGS-B', jac=this_grad)
                ret = dict()
                ret['x'] = ret_tmp.x
                ret['fun'] = ret_tmp.fun
            return ret

        est_SVAR = optimize_this(options)

        out_SVAR = SVAR.estimatorPML.SVARout(est_SVAR, options, u)


    elif estimator == 'Cholesky':
        out_SVAR = SVAR.estimatorCholesky.get_B_Cholesky(u)

    else:
        print('Unknown estimator')

    return out_SVAR
