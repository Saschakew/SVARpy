import os
import numpy as np
import SVAR
import SVAR.estimatorCholesky
import SVAR.estSVAR
import SVAR.SVARbasics
import pickle
import pytest
np.random.seed(0)

if False:
    with open("tests\data\eps.data", 'rb') as filehandle:
        # read the data as binary data stream
        eps = pickle.load(filehandle)
    supply_eps = eps


#
# # path = os.path.join("MCResults", version)
# file_name = "grad.data"
# with open(file_name, 'wb') as filehandle:
#     pickle.dump(grad, filehandle)


@pytest.fixture
def supply_eps():
    with open("data/eps.data", 'rb') as filehandle:
        # read the data as binary data stream
        eps = pickle.load(filehandle)
    return eps


def test_get_B_recursive(supply_eps):
    u = supply_eps
    out = SVAR.estimatorCholesky.get_B_Cholesky(u)
    B = out['B_est']
    B_expected = np.array(
        [[0.997239987699063, 0.0, 0.0, 0.0, 0.0], [0.00995863527750116, 1.0175109830432003, 0.0, 0.0, 0.0],
         [0.003330608469677456, -0.033621407028708195, 0.9849721988243723, 0.0, 0.0],
         [-0.010995613002447795, -0.020220644924939964, 0.024015850825733976, 1.000823275601793, 0.0],
         [-0.0179660316466463, -0.0012376406712120119, 0.007032910730583536, 0.01646644814446428, 0.9883101427396063]]
    )
    assert B.tolist() == B_expected.tolist()


def test_OLS_ReducedForm(supply_eps):
    n = 2
    y = supply_eps[:10, :n]
    lags = 2

    out = SVAR.SVARbasics.OLS_ReducedForm(y, lags)
    const = np.round(out['const'], 14)
    trend = np.round(out['trend'], 14)
    trend2 = np.round(out['trend2'], 14)

    u_expected = np.array([[-0.7157164983214352, -0.3982566518284819], [-0.8818273160053619, -0.8399595636449608],
                           [-0.5167974005751311, -0.2116452764112019], [1.1038821156207432, 0.6051987290464562],
                           [0.16242018742610365, 0.2703959582292481], [0.1538266498561563, -0.053934400792861936],
                           [0.4052546007535178, 0.40443074168915405], [0.288957661245414, 0.22377046371264037]])
    AR_expected = np.array([[[-0.7950356983361764, -0.6698826737442521], [0.06390762125204139, -0.022567288497924554]],
                            [[1.0047088305436138, 0.3525220324939553], [0.7692214553972884, -0.7503757508809368]]])
    const_expected = np.array([1.73695125962916, -0.34765298181486])
    trend_expected = np.array([-0.0, -0.0])
    trend2_expected = np.array([0.0, 0.0])

    u = np.round(out['u'], decimals=10)
    u_expected = np.round(u_expected, decimals=10)
    AR = np.round(out['AR'], decimals=10)
    AR_expected = np.round(AR_expected, decimals=10)
    const = np.round(const, decimals=10)
    const_expected = np.round(const_expected, decimals=10)


    assert u.tolist() == u_expected.tolist()
    assert AR.tolist() == AR_expected.tolist()
    assert const.tolist() == const_expected.tolist()
    assert trend.tolist() == trend_expected.tolist()
    assert trend2.tolist() == trend2_expected.tolist()

    out = SVAR.SVARbasics.OLS_ReducedForm(y, lags, add_const=False, add_trend=True, add_trend2=True)
    const = np.round(out['const'], 14)
    trend = np.round(out['trend'], 14)
    trend2 = np.round(out['trend2'], 14)
    u_expected = np.array([[0.5673841024455379, -0.8359018460861763], [0.35768846264653964, -0.3769279489656512],
                           [0.10376898647983804, -0.2781892978113251], [0.10054761547107605, -0.014151740411663616],
                           [-0.22274382445488616, 0.19234520702692448], [0.05665572324817064, 0.03890534211647401],
                           [-0.38067056089228446, 0.4193476085599236], [0.2892643949875538, -0.35774734631801736]])
    AR_expected = np.array([[[-1.0544138539632832, -1.0349463050590209], [0.012470373799059964, -0.15069923098233978]],
                            [[0.8324156393093451, 0.20830013953158072], [0.6354092061636292, -0.9678105506392531]]])
    const_expected = np.array([0.0, 0.0])
    trend_expected = np.array([1.11514462234945, -0.16507937666834])
    trend2_expected = np.array([-0.10125382221811, 0.04990173192479])

    u = np.round(out['u'], decimals=10)
    u_expected = np.round(u_expected, decimals=10)
    AR = np.round(out['AR'], decimals=10)
    AR_expected = np.round(AR_expected, decimals=10)
    const = np.round(const, decimals=10)
    const_expected = np.round(const_expected, decimals=10)


    assert u.tolist() == u_expected.tolist()
    assert AR.tolist() == AR_expected.tolist()
    assert const.tolist() == const_expected.tolist()
    assert trend.tolist() == trend_expected.tolist()
    assert trend2.tolist() == trend2_expected.tolist()

def test_infocrit(supply_eps):
    maxLag = 2
    out_info = SVAR.SVARbasics.infocrit(supply_eps, maxLag)

    AIC_crit = np.round(out_info['AIC_crit'],5)
    AIC_crit_row = np.round(out_info['AIC_crit_row'],5)
    AIC_min_crit = np.round(out_info['AIC_min_crit'],5)
    AIC_min_crit_row = np.round(out_info['AIC_min_crit_row'],5)
    BIC_crit = np.round(out_info['BIC_crit'],5)
    BIC_crit_row = np.round(out_info['BIC_crit_row'],5)
    BIC_min_crit = np.round(out_info['BIC_min_crit'],5)
    BIC_min_crit_row = np.round(out_info['BIC_min_crit_row'],5)

    AIC_crit_expected = np.array([[ 0.     , -0.02272],
       [ 1.     , -0.02009],
       [ 2.     , -0.0145 ]])
    AIC_crit_row_expected = np.array([[[ 0.     , -0.00502],
        [ 1.     , -0.00608],
        [ 2.     , -0.00488]],
       [[ 0.     ,  0.03476],
        [ 1.     ,  0.03607],
        [ 2.     ,  0.03711]],
       [[ 0.     , -0.02854],
        [ 1.     , -0.02744],
        [ 2.     , -0.02659]],
       [[ 0.     ,  0.00242],
        [ 1.     ,  0.00345],
        [ 2.     ,  0.00467]],
       [[ 0.     , -0.02329],
        [ 1.     , -0.02326],
        [ 2.     , -0.02196]]])

    AIC_min_crit_expected = np.array(0)
    AIC_min_crit_row_expected = np.array([1., 0., 0., 0., 0.])
    BIC_crit_expected = np.array([[ 0.     , -0.0162 ],
       [ 1.     ,  0.01903],
       [ 2.     ,  0.05721]])
    BIC_crit_row_expected = np.array([[[ 0.000e+00, -3.710e-03],
        [ 1.000e+00,  1.740e-03],
        [ 2.000e+00,  9.460e-03]],
       [[ 0.000e+00,  3.606e-02],
        [ 1.000e+00,  4.389e-02],
        [ 2.000e+00,  5.145e-02]],
       [[ 0.000e+00, -2.724e-02],
        [ 1.000e+00, -1.962e-02],
        [ 2.000e+00, -1.225e-02]],
       [[ 0.000e+00,  3.720e-03],
        [ 1.000e+00,  1.128e-02],
        [ 2.000e+00,  1.901e-02]],
       [[ 0.000e+00, -2.199e-02],
        [ 1.000e+00, -1.544e-02],
        [ 2.000e+00, -7.610e-03]]])

    BIC_min_crit_expected = np.array(0)
    BIC_min_crit_row_expected = np.array([0., 0., 0., 0., 0.])

    assert AIC_crit.tolist() == AIC_crit_expected.tolist()
    assert AIC_crit_row.tolist() == AIC_crit_row_expected.tolist()
    assert AIC_min_crit.tolist() == AIC_min_crit_expected.tolist()
    assert AIC_min_crit_row.tolist() == AIC_min_crit_row_expected.tolist()
    assert BIC_crit.tolist() == BIC_crit_expected.tolist()
    assert BIC_crit_row.tolist() == BIC_crit_row_expected.tolist()
    assert BIC_min_crit.tolist() == BIC_min_crit_expected.tolist()
    assert BIC_min_crit_row.tolist() == BIC_min_crit_row_expected.tolist()


def test_get_IRF(supply_eps):
    n = 2
    y = supply_eps[:, :n]
    lags = 2

    out = SVAR.SVARbasics.OLS_ReducedForm(y, lags)
    B = np.eye(n)

    irf = SVAR.SVARbasics.get_IRF(B, out['AR'], irf_length=4)
    irf_expected = np.array([[[1.0, 0.0], [0.0, 1.0]], [[-0.014379509578365488, 0.008972561120846994],
                                                        [-0.010948139124280613, 0.019910829297800473]],
                             [[0.003921815824795623, -0.004345059269958192],
                              [-0.011541803672234528, -0.02479087342003774]],
                             [[-0.00016667272937455577, -0.00021324484225223691],
                              [0.00016702997596673908, -0.0010485971399126044]]])
    assert irf.tolist() == irf_expected.tolist()

