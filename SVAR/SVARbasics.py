import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt


# def var_statsmodel(y,lags):
#     model = VAR(y)
#     results = model.fit(lags)
#     print(results.summary())
#     return results
#
# def var_infocrit(y,maxlags,crit='aic'):
#     model = VAR(y)
#     model.select_order(maxlags)
#     results = model.fit(maxlags=maxlags, ic=crit)
#     print(results.summary())
#     return results





def get_rhs_flex(y, this_lags, maxLag, add_const=True, add_trend=False, add_trend2=False):
    # LagsM = np.array([2, 3])

    T, n = np.shape(y)
    Time = np.arange(T - maxLag)

    if add_const:
        rhs = np.ones([T - maxLag])
    else:
        rhs = np.zeros([T - maxLag])

    if add_trend:
        rhs = np.vstack((rhs, Time))
    else:
        rhs = np.vstack((rhs, np.zeros([T - maxLag])))

    if add_trend2:
        rhs = np.vstack((rhs, np.power(Time, 2)))
    else:
        rhs = np.vstack((rhs, np.zeros([T - maxLag])))

    rhs = np.transpose(rhs)

    for lag in range(maxLag):
        if lag < this_lags:
            rhs = np.append(rhs, y[(maxLag - lag - 1):(-lag - 1), :], axis=1)
        else:
            rhs = np.append(rhs, np.zeros([T - maxLag, n]), axis=1)

    return rhs


def get_lhs(y, lags):
    lhs = y[lags:, :]
    return lhs


def get_ARMAtrices(coefmat, n, lags):
    const = coefmat[:, 0]
    coefmat = coefmat[:, 1:]

    trend = coefmat[:, 0]
    coefmat = coefmat[:, 1:]

    trend2 = coefmat[:, 0]
    coefmat = coefmat[:, 1:]

    AR = np.zeros([n, n, lags])
    for i in range(lags):
        AR[:, :, i] = coefmat[:, :n]
        coefmat = coefmat[:, n:]

    return AR, const, trend, trend2


def OLS_ReducedForm(y, lags, add_const=True, add_trend=False, add_trend2=False):
    T, n = np.shape(y)
    if np.shape(lags) == ():
        lags = np.multiply(np.ones([n], dtype=int), np.int(lags), dtype=int)
    maxLags = np.int(np.max(lags))

    coefmat = np.empty([n, maxLags * n + 3])
    u = np.empty([T - maxLags, n])
    for i in range(n):
        rhs = get_rhs_flex(y, lags[i], maxLags, add_const=add_const, add_trend=add_trend, add_trend2=add_trend2)
        lhs = y[maxLags:, i]

        model = sm.OLS(lhs, rhs)
        results = model.fit()
        coefmat[i, :] = results.params
        u[:, i] = results.resid

    AR, const, trend, trend2 = get_ARMAtrices(coefmat, n, maxLags)
    out = dict()
    out['u'] = u
    out['AR'] = AR
    out['const'] = const
    out['trend'] = trend
    out['trend2'] = trend2
    return out


def infocrit(y, maxLag, add_const=True, add_trend=False, add_trend2=False):
    T, n = np.shape(y)
    T = T - maxLag

    par_plus = int(add_const) + int(add_trend) + int(add_trend2)

    AIC_crit = np.full([maxLag + 1, 2], 0.)
    AIC_crit_row = np.full([n, maxLag + 1, 2], 0.)
    BIC_crit = np.full([maxLag + 1, 2], 0.)
    BIC_crit_row = np.full([n, maxLag + 1, 2], 0.)
    for this_lag in range(maxLag + 1):
        thisY = y[maxLag - this_lag:, :]

        out_redform = OLS_ReducedForm(thisY, this_lag, add_const=add_const, add_trend=add_trend,
                                                      add_trend2=add_trend2)
        u = out_redform['u']
        this_aic = np.log(np.linalg.det(np.matmul(np.transpose(u), u) / T)) + 2 / T * (
                np.sum(this_lag) * np.power(n, 2) + par_plus * n)
        AIC_crit[this_lag, :] = np.array([this_lag, this_aic])

        this_BIC = np.log(np.linalg.det(np.matmul(np.transpose(u), u) / T)) + np.log(T) / T * (
                np.sum(this_lag) * np.power(n, 2) + par_plus * n)
        BIC_crit[this_lag, :] = np.array([this_lag, this_BIC])

        for i in range(n):
            this_aic_row = np.log(np.matmul(np.transpose(u[:, i]), u[:, i]) / T) + 2 / T * (
                    np.sum(this_lag) * n + par_plus)
            AIC_crit_row[i, this_lag, :] = np.array([this_lag, this_aic_row])

            this_bic_row = np.log(np.matmul(np.transpose(u[:, i]), u[:, i]) / T) + np.log(T) / T * (
                    np.sum(this_lag) * n + par_plus)
            BIC_crit_row[i, this_lag, :] = np.array([this_lag, this_bic_row])

    AIC_min_crit = np.argmin(AIC_crit[:, 1])
    BIC_min_crit = np.argmin(BIC_crit[:, 1])
    AIC_min_crit_row = np.zeros([n])
    BIC_min_crit_row = np.zeros([n])
    for i in range(n):
        AIC_min_crit_row[i] = np.argmin(AIC_crit_row[i, :, 1])
        BIC_min_crit_row[i] = np.argmin(BIC_crit_row[i, :, 1])

    out = dict()
    out['AIC_crit'] = AIC_crit
    out['AIC_crit_row'] = AIC_crit_row
    out['AIC_min_crit'] = AIC_min_crit
    out['AIC_min_crit_row'] = AIC_min_crit_row
    out['BIC_crit'] = BIC_crit
    out['BIC_crit_row'] = BIC_crit_row
    out['BIC_min_crit'] = BIC_min_crit
    out['BIC_min_crit_row'] = BIC_min_crit_row
    return out


def get_IRF(B, AR, irf_length=12, scale=[]):
    n = np.shape(B)[0]

    if np.ndim(AR) == 2:
        lags = 1
        AR_new = np.full([n, n, 1], np.nan)
        AR_new[:, :, 0] = AR
        AR = AR_new
    else:
        lags = np.shape(AR)[2]
    phi = np.full([n, n, irf_length], np.nan)

    phi[:, :, 0] = np.eye(n, n)

    # normalize impact
    if np.array(scale).size != 0:
        B = np.matmul(B,scale)

    for i in range(1, irf_length):
        tmpsum = np.zeros([n, n])
        for j in range(i):
            if j < lags:
                tmpsum = tmpsum + np.matmul(phi[:, :, i - j - 1], AR[:, :, j])
        phi[:, :, i] = tmpsum

    for i in range(irf_length):
        phi[:, :, i] = np.matmul(phi[:, :, i], B)

    irf = np.full([irf_length, n, n], np.nan)
    for i in range(n):
        for j in range(irf_length):
            irf[j, :, i] = phi[:, i, j]

    return irf


def plot_IRF(irf, irf_bootstrap=[], alpha=0.1, shocks=[], responses=[], shocknames=[], responsnames=[]):
    n = np.shape(irf)[1]
    if np.array(shocks).size == 0:
        shocks = range(n)
    if np.array(responses).size == 0:
        responses = range(n)
    if np.array(shocknames).size == 0:
        shocknames = range(n)
    if np.array(responsnames).size == 0:
        responsnames = range(n)

    plotcounter = 1
    for shock in shocks:
        for response in responses:
            ax1 = plt.subplot(np.shape(shocks)[0], np.shape(responses)[0], plotcounter)
            y = irf[:, response, shock]
            x = np.arange(np.shape(y)[0])
            ax1.plot(x, y, color="red")

            # Plot Bootstrap irf quantiles
            if np.array(irf_bootstrap).size != 0:
                irf_lower = np.quantile(irf_bootstrap, alpha, axis=0)
                irf_upper = np.quantile(irf_bootstrap, 1 - alpha, axis=0)
                lower = irf_lower[:, response, shock]
                upper = irf_upper[:, response, shock]
                ax1.fill_between(x, lower, upper)

            plt.title(
                r'$\epsilon^{ ' + str(shocknames[shock]) + '} \\rightarrow  y_{' + str(responsnames[response]) + '}$')
            plotcounter += 1
    plt.show()
