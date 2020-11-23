import numpy as np
import SVAR.SVARbasics
import SVAR.estSVAR

import SVAR.estPrepare
from SVAR.estPrepare import prepare_bootstrapIRF


def bootstrap_estimation(y, scale=[], estimator='GMM', opt_bootstrap=dict(), opt_redform=dict(), prepOptions=dict(),
                         opt_SVAR=dict(), opt_irf=dict(), prepared=False):
    out_redform = SVAR.SVARbasics.OLS_ReducedForm(y, **opt_redform)

    # Estimate B
    if prepared:
        opt_SVAR['printOutput'] = False
        out_svar = SVAR.SVARest(out_redform['u'], estimator=estimator, options=opt_SVAR, prepared=True)
    else:
        out_svar = SVAR.SVARest(out_redform['u'], estimator=estimator, prepOptions=prepOptions)

    # Find permutation
    if opt_bootstrap['find_perm'] == True:
        # BestPerm = SVAR.K_GMM.get_BestSignPerm(e_first, e_boots, n_rec)
        BestPerm = np.diag(np.sign(np.diag(out_svar['B_est'])))
        out_svar['B_est'] = np.matmul(out_svar['B_est'], BestPerm)
        # ToDo: Update all outputs off out_SVAR
    #
    if scale == []:
        scale = np.linalg.inv(np.diag(np.diag(out_svar['B_est'])))

    # get IRF
    out_irf = SVAR.SVARbasics.get_IRF(out_svar['B_est'], out_redform['AR'], scale=scale, **opt_irf)

    return out_redform, out_svar, out_irf, out_svar['options']


def bootstrap_SVAR(y, estimator='GMM',
                   options_bootstrap=dict(), options_redform=dict(), prepOptions=dict(), options_irf=dict()):
    # Bundle inputs to options
    options_bootstrap, options_redform, options_irf = prepare_bootstrapIRF(**options_bootstrap, **options_redform,
                                                                           **options_irf)

    # First iteration
    out_redform, out_svar, out_irf, opt_SVAR = bootstrap_estimation(y, [], estimator,opt_bootstrap =  options_bootstrap,
                                                                    opt_redform= options_redform, prepOptions = prepOptions, opt_SVAR =[],opt_irf =options_irf,
                                                                    prepared=False)


    scale = np.linalg.inv(np.diag(np.diag(out_svar['B_est'])))

    # # ToDo: Add update values
    # if options_bootstrap['update_bstart']:
    #     options_SVAR['bstart'] = out_svar['b_est']
    #     options_SVAR['bstartopt'] = 'specific'
    #     # if options_SVAR['kstep'] > 1:
    #     #     options_SVAR['kstep']  = options_SVAR['kstep']  - 1
    #     # options_SVAR['Wstartopt'] = True

    # Bootstrap iterations
    def bootstrap_iter():
        y_sim = simulate_SVAR(**out_redform)
        out_redform_iter, out_svar_iter, out_irf_iter, _ = bootstrap_estimation(y_sim, scale, estimator,
                                                                                       options_bootstrap,
                                                                                       options_redform, prepOptions,
                                                                                       opt_SVAR, options_irf,
                                                                                       prepared=True)
        return out_irf_iter

    # Loop through iterations
    out_irf_bootstrap = [bootstrap_iter() for _ in range(options_bootstrap['number_bootstrap'])]

    return out_irf, out_irf_bootstrap


def simulate_SVAR(u, AR, const, trend, trend2, shuffle=True):
    T, n = np.shape(u)
    lags = np.shape(AR)[2]

    if shuffle:
        u_resample_index = np.random.choice(T, T, replace=True)
        u_resample = u[u_resample_index, :]
    else:
        u_resample = u

    y_new = np.zeros([T, n])
    for t in range(T):
        tmpsum = const + np.multiply(trend, t - lags) + np.multiply(trend2, t - lags)
        for j in range(lags):
            if t - j > 0:
                tmpsum = tmpsum + np.matmul(AR[:, :, j], y_new[t - j - 1])
        tmpsum = tmpsum + u_resample[t - lags]
        y_new[t, :] = tmpsum

    return y_new
