import os
import numpy as np

import SVAR.SVARutilGMM
import SVAR.estSVAR
import SVAR.SVARbasics
import pickle
import pytest

import SVAR.estimatorGMM
import SVAR.estimatorGMMW
import SVAR.SVARutil

np.random.seed(0)

if False:
    with open("tests\data\eps.data", 'rb') as filehandle:
        # read the data as binary data stream
        eps = pickle.load(filehandle)
    supply_eps = eps


#
# # path = os.path.join("MCResults", version)
# file_name = "W.data"
# with open(file_name, 'wb') as filehandle:
#     pickle.dump(grad, filehandle)


@pytest.fixture
def supply_eps():
    with open("data/eps.data", 'rb') as filehandle:
        # read the data as binary data stream
        eps = pickle.load(filehandle)
    return eps


@pytest.fixture
def supply_g():
    with open("data/g.data", 'rb') as filehandle:
        # read the data as binary data stream
        g = pickle.load(filehandle)
    return g


# @pytest.fixture
# def supply_W():
#     with open("data/W.data", 'rb') as filehandle:
#         # read the data as binary data stream
#         W = pickle.load(filehandle)
#     return W


@pytest.fixture
def supply_Jac():
    with open("data/Jac.data", 'rb') as filehandle:
        # read the data as binary data stream
        Jac = pickle.load(filehandle)
    return Jac


@pytest.fixture
def supply_grad():
    with open("data/grad.data", 'rb') as filehandle:
        # read the data as binary data stream
        grad = pickle.load(filehandle)
    return grad


@pytest.mark.parametrize("r,n,expected", [
    (3, 2, np.array([[1, 2], [2, 1]])),
    (2, 3, np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])),
    (3, 3, np.array([[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0], [1, 1, 1]])),
    (4, 3, np.array([[0, 1, 3], [0, 3, 1], [1, 0, 3], [1, 3, 0], [3, 0, 1], [3, 1, 0], [0, 2, 2],
                     [2, 0, 2], [2, 2, 0], [1, 1, 2], [1, 2, 1], [2, 1, 1]]))
])
def test_get_Cr(r, n, expected):
    assert SVAR.SVARutilGMM.get_Cr(r, n).tolist() == expected.tolist()


@pytest.mark.parametrize("r,n,expected", [
    (3, 2, np.array([[3, 0], [0, 3]])),
    (2, 3, np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]]))
])
def test_get_Mr(r, n, expected):
    assert SVAR.SVARutilGMM.get_Mr(r, n).tolist() == expected.tolist()


@pytest.mark.parametrize("moments,expected", [
    (SVAR.SVARutilGMM.get_Cr(3, 2), np.array([[False, True, False, False, False, False, False, True, False,
                                               False],
                                              [False, False, True, False, False, False, True, False, False,
                                         False]])),
    (SVAR.SVARutilGMM.get_Cr(2, 3),
     np.array([[True, False, False, False, False, False, True, False, False,
                False, False, True, False, False, False],
               [False, True, False, False, False, True, False, False, False,
                False, False, True, False, False, False],
               [False, True, False, False, False, False, True, False, False,
                False, True, False, False, False, False]])),
    (SVAR.SVARutilGMM.get_Cr(3, 3),
     np.array([[True, False, False, False, False, False, True, False, False,
                False, False, False, True, False, False],
               [True, False, False, False, False, False, False, True, False,
                False, False, True, False, False, False],
               [False, True, False, False, False, True, False, False, False,
                False, False, False, True, False, False],
               [False, True, False, False, False, False, False, True, False,
                False, True, False, False, False, False],
               [False, False, True, False, False, True, False, False, False,
                False, False, True, False, False, False],
               [False, False, True, False, False, False, True, False, False,
                False, True, False, False, False, False],
               [False, True, False, False, False, False, True, False, False,
                False, False, True, False, False, False]])),
    (SVAR.SVARutilGMM.get_Cr(4, 3),
     np.array([[True, False, False, False, False, False, True, False, False,
                False, False, False, False, True, False],
               [True, False, False, False, False, False, False, False, True,
                False, False, True, False, False, False],
               [False, True, False, False, False, True, False, False, False,
                False, False, False, False, True, False],
               [False, True, False, False, False, False, False, False, True,
                False, True, False, False, False, False],
               [False, False, False, True, False, True, False, False, False,
                False, False, True, False, False, False],
               [False, False, False, True, False, False, True, False, False,
                False, True, False, False, False, False],
               [True, False, False, False, False, False, False, True, False,
                False, False, False, True, False, False],
               [False, False, True, False, False, True, False, False, False,
                False, False, False, True, False, False],
               [False, False, True, False, False, False, False, True, False,
                False, True, False, False, False, False],
               [False, True, False, False, False, False, True, False, False,
                False, False, False, True, False, False],
               [False, True, False, False, False, False, False, True, False,
                False, False, True, False, False, False],
               [False, False, True, False, False, False, True, False, False,
                False, False, True, False, False, False]])),
    (SVAR.SVARutilGMM.get_Mr(3, 2), np.array([[False, False, False, True, False, True, False, False, False,
                                               False],
                                              [True, False, False, False, False, False, False, False, True,
                                         False]])),
    (SVAR.SVARutilGMM.get_Mr(2, 3),
     np.array([[False, False, True, False, False, True, False, False, False,
                False, True, False, False, False, False],
               [True, False, False, False, False, False, False, True, False,
                False, True, False, False, False, False],
               [True, False, False, False, False, True, False, False, False,
                False, False, False, True, False, False]])),
])
def test_get_Moments_powerindex(moments, expected):
    assert SVAR.SVARutilGMM.get_Moments_powerindex(moments).tolist() == expected.tolist()


@pytest.mark.parametrize("moment,expected", [
    (SVAR.SVARutilGMM.get_Cr(3, 2)[0], np.array([1, 2, 2])),
    (SVAR.SVARutilGMM.get_Cr(3, 2)[1], np.array([1, 1, 2])),
    (SVAR.SVARutilGMM.get_Cr(3, 3)[0], np.array([2, 3, 3])),
    (SVAR.SVARutilGMM.get_Cr(3, 3)[1], np.array([2, 2, 3])),
    (SVAR.SVARutilGMM.get_Cr(3, 3)[6], np.array([1, 2, 3])),
    (SVAR.SVARutilGMM.get_Cr(4, 3)[0], np.array([2, 3, 3, 3])),
    (SVAR.SVARutilGMM.get_Cr(4, 3)[5], np.array([1, 1, 1, 2])),
    (SVAR.SVARutilGMM.get_Mr(2, 2)[0], np.array([1, 1])),
    (SVAR.SVARutilGMM.get_Mr(2, 2)[1], np.array([2, 2])),
    (SVAR.SVARutilGMM.get_Mr(2, 3)[0], np.array([1, 1])),
    (SVAR.SVARutilGMM.get_Mr(2, 3)[2], np.array([3, 3])),
])
def test_get_Moment_transformed(moment, expected):
    assert SVAR.SVARutilGMM.get_Moment_transformed(moment).tolist() == expected.tolist()


@pytest.mark.parametrize("b,restrictions,whiten,expected", [

    ([1, 2, 3, 4, 5, 6, 7, 8, 9], [], False,
     np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])),

    ([1, 2, 3, 4, 5, 6, 7, 8, 9],
     np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]), False,
     np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])),

    ([2, 3, 4, 5, 6, 7, 8, 9],
     np.array([[0, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]), False,
     np.array([[0, 2, 3], [4, 5, 6], [7, 8, 9]])),

    ([2, 3, 4, 5, 6, 7, 8, 9],
     np.array([[10, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]), False,
     np.array([[10, 2, 3], [4, 5, 6], [7, 8, 9]])),

    ([1, 4, 5, 7, 8, 9],
     SVAR.SVARutil.getRestrictions_recursive(np.array([[1, 0, 0], [4, 5, 0], [7, 8, 9]])), False,
     np.array([[1, 0, 0], [4, 5, 0], [7, 8, 9]])),

    ([np.pi / 2], [], True,
     np.array([[0., -1.], [1., 0.]])),

    ([0, 0, np.pi / 2], [], True,
     np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])),
])
def test_get_BMatrix(b, restrictions, whiten, expected):
    assert SVAR.SVARutil.get_BMatrix(b=b, restrictions=restrictions, whiten=whiten).tolist() == expected.tolist()


@pytest.mark.parametrize("B,restrictions,expected", [
    (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), [], np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])),

    (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
     np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]),
     np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])),

    (np.array([[0, 2, 3], [4, 5, 6], [7, 8, 9]]),
     np.array([[0, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]),
     np.array([2, 3, 4, 5, 6, 7, 8, 9])),

    (np.array([[10, 2, 3], [4, 5, 6], [7, 8, 9]]),
     np.array([[10, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]),
     np.array([2, 3, 4, 5, 6, 7, 8, 9])),

    (np.array([[1, 0, 0], [4, 5, 0], [7, 8, 9]]),
     SVAR.SVARutil.getRestrictions_recursive(np.array([[1, 0, 0], [4, 5, 0], [7, 8, 9]])),
     np.array([1, 4, 5, 7, 8, 9]))
])
def test_get_bVector(B, restrictions, expected):
    assert SVAR.SVARutil.get_BVector(B=B, restrictions=restrictions).tolist() == expected.tolist()


def test_get_Omega():
    # eps = supply_eps[:,:2]
    eps = np.array([[2, 3], [2, 3]])
    Omega2, Omega3, Omega4 = SVAR.SVARutil.get_Omega(eps)
    Omega2_exp = np.array([[4., 6.], [np.nan, 9.]])
    Omega3_exp = np.array([[[8., 12.],
                            [np.nan, 18.]],
                           [[np.nan, np.nan],
                            [np.nan, 27.]]])
    Omega4_exp = np.array([[[[16., 24.],
                             [np.nan, 36.]],
                            [[np.nan, np.nan],
                             [np.nan, 54.]]],
                           [[[np.nan, np.nan],
                             [np.nan, np.nan]],
                            [[np.nan, np.nan],
                             [np.nan, 81.]]]])
    assert Omega2[~np.isnan(Omega2)].tolist() == Omega2_exp[~np.isnan(Omega2)].tolist()
    assert Omega3[~np.isnan(Omega3)].tolist() == Omega3_exp[~np.isnan(Omega3)].tolist()
    assert Omega4[~np.isnan(Omega4)].tolist() == Omega4_exp[~np.isnan(Omega4)].tolist()

    assert Omega2[~np.isnan(Omega2_exp)].tolist() == Omega2_exp[~np.isnan(Omega2_exp)].tolist()
    assert Omega3[~np.isnan(Omega3_exp)].tolist() == Omega3_exp[~np.isnan(Omega3_exp)].tolist()
    assert Omega4[~np.isnan(Omega4_exp)].tolist() == Omega4_exp[~np.isnan(Omega4_exp)].tolist()


def test_get_f():
    u = np.array([[2, 3], [2, 3]])
    b = np.array([1, 0, 0, 1])
    restrictions = []
    n = 2
    moments = SVAR.SVARutilGMM.get_Mr(2, n)
    moments = np.append(moments, SVAR.SVARutilGMM.get_Cr(2, n), axis=0)
    moments = np.append(moments, SVAR.SVARutilGMM.get_Cr(3, n), axis=0)
    moments = np.append(moments, SVAR.SVARutilGMM.get_Cr(4, n), axis=0)
    moments_powerindex = SVAR.SVARutilGMM.get_Moments_powerindex(moments)
    whiten = False
    f = SVAR.SVARutilGMM.get_f(u, b, restrictions, moments, moments_powerindex, whiten=whiten)
    expected = np.array([[3., 8., 6., 18., 12., 54., 24., 35.],
                         [3., 8., 6., 18., 12., 54., 24., 35.]])
    assert f.tolist() == expected.tolist()

    u = np.array([[2, 3], [2, 3]])
    b = np.array([0])
    restrictions = []
    n = 2
    moments = SVAR.SVARutilGMM.get_Mr(3, n)
    moments = np.append(moments, SVAR.SVARutilGMM.get_Mr(4, n), axis=0)
    moments_powerindex = SVAR.SVARutilGMM.get_Moments_powerindex(moments)
    whiten = True
    f = SVAR.SVARutilGMM.get_f(u, b, restrictions, moments, moments_powerindex, whiten=whiten)
    expected = np.array([[8., 27., 13., 78.],
                         [8., 27., 13., 78.]])
    assert f.tolist() == expected.tolist()


def test_get_g(supply_eps, supply_g):
    n = 5
    u = supply_eps[:, :n]
    restrictions = []
    B = np.eye(n)
    b = SVAR.SVARutil.get_BVector(B, restrictions=restrictions)
    moments = SVAR.SVARutilGMM.get_Mr(2, n)
    moments = np.append(moments, SVAR.SVARutilGMM.get_Cr(2, n), axis=0)
    moments = np.append(moments, SVAR.SVARutilGMM.get_Cr(3, n), axis=0)
    moments = np.append(moments, SVAR.SVARutilGMM.get_Cr(4, n), axis=0)
    moments_powerindex = SVAR.SVARutilGMM.get_Moments_powerindex(moments)
    whiten = False
    g = SVAR.SVARutilGMM.get_g(u, b, restrictions, moments, moments_powerindex, whiten)
    assert g.tolist() == supply_g.tolist()


def test_loss():
    u = np.array([[2, 3], [2, 3]])
    b = np.array([1, 0, 0, 1])
    restrictions = []
    n = 2
    moments = SVAR.SVARutilGMM.get_Mr(2, n)
    moments = np.append(moments, SVAR.SVARutilGMM.get_Cr(2, n), axis=0)
    moments = np.append(moments, SVAR.SVARutilGMM.get_Cr(3, n), axis=0)
    moments = np.append(moments, SVAR.SVARutilGMM.get_Cr(4, n), axis=0)
    moments_powerindex = SVAR.SVARutilGMM.get_Moments_powerindex(moments)
    whiten = False
    g = SVAR.SVARutilGMM.get_g(u, b, restrictions, moments, moments_powerindex, whiten)
    W = np.eye(np.shape(moments)[0])
    loss = SVAR.estimatorGMM.loss(u, b, W=W, moments=moments, moments_powerindex=moments_powerindex, restrictions=restrictions)
    assert loss == 5294.0

def test_GMM_avar(supply_eps):
    n = 2
    u = supply_eps[:, :n]
    restrictions = np.full([n, n], np.nan)
    B = np.eye(n)
    b = SVAR.SVARutil.get_BVector(B, restrictions=restrictions)
    moments = SVAR.SVARutilGMM.get_Mr(2, n)
    moments = np.append(moments, SVAR.SVARutilGMM.get_Cr(2, n), axis=0)
    moments = np.append(moments, SVAR.SVARutilGMM.get_Cr(3, n), axis=0)
    moments = np.append(moments, SVAR.SVARutilGMM.get_Cr(4, n), axis=0)
    W = np.eye(np.shape(moments)[0])
    k_step = 1
    # GMM_out = SVAR.K_GMM.GMM(u, b, Jacobian, W, restrictions, moments, kstep=k_step)
    opt_svar = dict()
    opt_svar['W'] = W
    opt_svar['Wstartopt'] = 'specific'
    opt_svar['kstep'] = k_step
    opt_svar['bstartopt'] = 'specific'
    opt_svar['bstart'] = b
    opt_svar['moments'] = moments
    opt_svar['restrictions'] = restrictions


    svar_out = SVAR.SVARest(u, estimator='GMM', prepOptions=opt_svar)
    V = svar_out['Avar_est']
    V_expected = np.array([[1.7409094051161254, -0.031698773570974025, 0.15592054887742773, 0.46321878755198453], [-0.031698773570974, 1.4952825734424806, -0.6335439601070609, -0.30049023834461774], [0.15592054887742776, -0.6335439601070612, 1.7048131709677288, 0.38794602532055444], [0.46321878755198453, -0.30049023834461774, 0.38794602532055433, 1.6913768246817211]])
    assert V.tolist() == V_expected.tolist()




def test_get_W_optimal(supply_eps):
    n = 2
    u = supply_eps[:, :n]
    restrictions = []
    B = np.eye(n)
    b = SVAR.SVARutil.get_BVector(B, restrictions=restrictions)
    moments = SVAR.SVARutilGMM.get_Mr(2, n)
    moments = np.append(moments, SVAR.SVARutilGMM.get_Cr(2, n), axis=0)
    moments = np.append(moments, SVAR.SVARutilGMM.get_Cr(3, n), axis=0)
    moments = np.append(moments, SVAR.SVARutilGMM.get_Cr(4, n), axis=0)
    whiten = False
    W = SVAR.estimatorGMM.get_W_opt(u, b, restrictions, moments )

    W_expected = np.array([[0.2587077339661509, 0.041514398432926584, 0.04344726539731498, -0.01613875245469707, 0.051345454473620655, -0.008895149579102816, 0.0013870105305476524, -0.0618954276036375], [0.041514398432926605, 0.23677500858967257, 0.022164761737540454, 0.03858422566844861, 0.029488014105424, 0.0014062951112191516, -0.009440714467586375, -0.053205566093385934], [0.04344726539731486, 0.02216476173754043, 4.440209172187058, 0.028062029568523782, -0.10547641211582394, -0.3193285045191833, -0.2806889505002043, -0.03578372196195379], [-0.016138752454697054, 0.038584225668448616, 0.028062029568523876, 0.26256685050113865, 0.05956870708093367, -0.027510874036621762, -0.02779386178409415, -0.03631812777388941], [0.051345454473620725, 0.029488014105423992, -0.10547641211582363, 0.059568707080933724, 0.3783426226459335, 0.017866272823187812, -0.06103760918527769, -0.0919732047394959], [-0.0088951495791028, 0.001406295111219152, -0.31932850451918326, -0.027510874036621745, 0.017866272823187895, 0.0560920527321429, 0.0027468312213206033, 0.007666804977153761], [0.0013870105305476416, -0.009440714467586379, -0.2806889505002043, -0.027793861784094154, -0.061037609185277654, 0.0027468312213206184, 0.060177001122095515, 0.011310916312183101], [-0.061895427603637504, -0.05320556609338593, -0.03578372196195385, -0.03631812777388941, -0.09197320473949584, 0.007666804977153777, 0.011310916312183089, 0.07759784792428635]])
    assert W.tolist() == W_expected.tolist()


def test_Jacobian(supply_eps, supply_Jac):
    n = 5
    u = supply_eps[:, :n]
    restrictions = np.full([n, n], np.nan)
    B = np.eye(n)
    b = SVAR.SVARutil.get_BVector(B, restrictions=restrictions)
    moments = SVAR.SVARutilGMM.get_Mr(2, n)
    moments = np.append(moments, SVAR.SVARutilGMM.get_Cr(2, n), axis=0)
    moments = np.append(moments, SVAR.SVARutilGMM.get_Cr(3, n), axis=0)
    moments = np.append(moments, SVAR.SVARutilGMM.get_Cr(4, n), axis=0)
    Jacobian = SVAR.SVARutilGMM.generate_Jacobian_function(moments=moments, restrictions=restrictions)
    Omega = SVAR.SVARutil.get_Omega(supply_eps)
    Jac = Jacobian(u=u, b=b, restrictions=restrictions, Omega=Omega)
    assert Jac.tolist() == supply_Jac.tolist()


def test_gradient(supply_eps):
    n = 5
    u = supply_eps[:, :n]
    restrictions = np.full([n, n], np.nan)
    B = np.eye(n)
    b = SVAR.SVARutil.get_BVector(B, restrictions=restrictions)
    moments = SVAR.SVARutilGMM.get_Mr(2, n)
    moments = np.append(moments, SVAR.SVARutilGMM.get_Cr(2, n), axis=0)
    moments = np.append(moments, SVAR.SVARutilGMM.get_Cr(3, n), axis=0)
    moments = np.append(moments, SVAR.SVARutilGMM.get_Cr(4, n), axis=0)
    moments_powerindex = SVAR.SVARutilGMM.get_Moments_powerindex(moments)
    whiten = False
    W = SVAR.estimatorGMM.get_W_opt(u, b, restrictions, moments, whiten )
    Jacobian = SVAR.SVARutilGMM.generate_Jacobian_function(moments=moments, restrictions=restrictions)
    grad = SVAR.estimatorGMM.gradient(u, b, Jacobian, W, restrictions, moments, moments_powerindex)
    grad = np.round(grad,8)
    grad_expected = np.array([ 0.02309447, -0.02224085, -0.01756799,  0.02714491,  0.03424183,
       -0.01757593,  0.02913043,  0.04166165,  0.03150388,  0.01295188,
        0.00919941,  0.02080365,  0.04692599, -0.01913157, -0.0005934 ,
        0.02539723,  0.04048191, -0.03109488,  0.01201416, -0.04215912,
       -0.01284358,  0.03418466,  0.00109825, -0.04083541,  0.03672443])

    assert grad.tolist() == grad_expected.tolist()


def test_GMM(supply_eps):
    n = 2
    u = supply_eps[:, :n]
    restrictions = np.full([n, n], np.nan)
    B = np.eye(n)
    b = SVAR.SVARutil.get_BVector(B, restrictions=restrictions)
    moments = SVAR.SVARutilGMM.get_Mr(2, n)
    moments = np.append(moments, SVAR.SVARutilGMM.get_Cr(2, n), axis=0)
    moments = np.append(moments, SVAR.SVARutilGMM.get_Cr(3, n), axis=0)
    moments = np.append(moments, SVAR.SVARutilGMM.get_Cr(4, n), axis=0)
    W = np.eye(np.shape(moments)[0])
    k_step = 1
    # GMM_out = SVAR.K_GMM.GMM(u, b, Jacobian, W, restrictions, moments, kstep=k_step)
    prepOptions = dict()
    prepOptions['W'] = W
    prepOptions['Wstartopt'] = 'specific'
    prepOptions['kstep'] = k_step
    prepOptions['bstartopt'] = 'specific'
    prepOptions['bstart'] = b
    prepOptions['moments'] = moments
    prepOptions['restrictions'] = restrictions
    svar_out = SVAR.SVARest(u, estimator='GMM', prepOptions=prepOptions)
    B_est = svar_out['B_est']
    loss = svar_out['loss']
    avar = svar_out['Avar_est']

    loss_expected = 0.0010397156910913927
    B_est_expected = np.array([[0.9935435303893686, -0.008753877534001258], [0.010957876833087816, 1.013418217704791]])
    assert loss == loss_expected
    assert svar_out['options']['bstart'].tolist() == b.tolist()
    assert svar_out['options']['moments'].tolist() == moments.tolist()
    assert svar_out['options']['kstep'] == k_step
    assert B_est.tolist() == B_est_expected.tolist()
    assert svar_out['options']['W'].tolist() == W.tolist()

    prepOptions = dict()
    prepOptions['Wstartopt'] = 'I'
    svar_out = SVAR.SVARest(u, estimator='GMM', prepOptions=prepOptions)
    B_est = svar_out['B_est']
    loss = svar_out['loss']

    loss_expected = 0.0005985542242835934
    B_est_expected = np.array([[0.9940721813730994, -0.005111466693945372], [0.017745900777426223, 1.0131512637458946]])

    assert loss == loss_expected
    assert B_est.tolist() == B_est_expected.tolist()


def test_GMM_white(supply_eps):
    n = 2
    u = supply_eps[:, :n]
    b = np.array([0])
    moments = SVAR.SVARutilGMM.get_Cr(3, n)
    moments = np.append(moments, SVAR.SVARutilGMM.get_Cr(4, n), axis=0)
    W = np.eye(np.shape(moments)[0])

    prepOptions = dict()
    prepOptions['W'] = W
    prepOptions['Wstartopt'] = 'specific'
    prepOptions['bstart'] = b
    prepOptions['bstartopt'] = 'specific'
    prepOptions['moments'] = moments
    svar_out = SVAR.SVARest(u, estimator='GMM_W', prepOptions=prepOptions)
    B_est = svar_out['B_est']
    loss = svar_out['loss']
    Omega2 = svar_out['Omega_all'][0]
    Omega2 = np.round(Omega2, 15)

    loss_expected = 0.003865450152482355
    B_est_expected = np.array([[0.997230283623227, -0.004399373906083158], [0.014447338782992897, 1.017457148695816]])
    Omega2_expected = np.array([[1.0, -0.0], [np.nan, 1.0]])
    assert loss == loss_expected
    assert B_est.tolist() == B_est_expected.tolist()
    assert Omega2[~np.isnan(Omega2)].tolist() == Omega2_expected[~np.isnan(Omega2)].tolist()



def test_GMM_fast(supply_eps):
    n = 2
    u = supply_eps[:, :n]
    b = np.array([0])

    prepOptions = dict()
    prepOptions['bstart'] = b
    prepOptions['bstartopt'] = 'specific'
    svar_out = SVAR.SVARest(u, estimator='GMM_WF', prepOptions=prepOptions)
    B_est = svar_out['B_est']
    loss = svar_out['loss']
    Omega2 = svar_out['Omega_all'][0]
    Omega2 = np.round(Omega2, 15)

    loss_expected = -19.7494828332592
    B_est_expected = np.array([[0.997231567472602, -0.004098036379113462], [0.014139888759238346, 1.0174614678581229]])
    Omega2_expected = np.array([[1.0, -0.0], [np.nan, 1.0]])
    assert loss == loss_expected
    assert B_est.tolist() == B_est_expected.tolist()
    assert Omega2[~np.isnan(Omega2)].tolist() == Omega2_expected[~np.isnan(Omega2)].tolist()


def test_GMM_PartlyRecurisve(supply_eps):
    n = 5
    u = supply_eps[:, :n]

    n_rec = 2
    prepOptions = dict()
    prepOptions['n_rec'] = n_rec
    prepOptions['Wstartopt'] = 'I'
    SVAR_out = SVAR.SVARest(u, estimator='GMM_W', prepOptions=prepOptions)
    B_est_PC = SVAR_out['B_est']

    prepOptions = dict()
    prepOptions['n_rec'] = 5
    prepOptions['Wstartopt'] = 'I'
    Rec_out = SVAR.SVARest(u, estimator='GMM', prepOptions=prepOptions)
    B_est_rec = Rec_out['B_est']

    assert B_est_PC[:n_rec,:n_rec].tolist() == B_est_rec[:n_rec,:n_rec].tolist()

    expected = np.array([[0.003330608469677456, -0.033621407028708195],
 [-0.010995613002447795, -0.020220644924939964],
 [-0.0179660316466463, -0.0012376406712120119]])
    assert B_est_PC[n_rec:,:n_rec].tolist() == expected.tolist()

    expected = np.array([[0.9845972613833915, 0.02681168147749875, 0.004427083644211701],
 [-0.003259823923889076, 1.001091873682849, 0.00533142301218077],
 [0.0022816522163611945, 0.011374323197186675, 0.988404250711615]])
    assert B_est_PC[n_rec:, n_rec:].tolist() == expected.tolist()


def test_fast_weighting(supply_eps):
    n = 4
    u = supply_eps[:, :n]

    def check_fast_weighting(b):
        # GMM_W with Wfast
        # block1 = np.array([1, 2])
        # block2 = np.array([3, 4])
        # blocks = list()
        # blocks.append(block1)
        # blocks.append(block2)
        # moments = SVAR.K_GMM.get_Moments('GMM_W', n, blocks=blocks)
        # moments_powerindex = SVAR.K_GMM.get_Moments_powerindex(moments)
        # W = SVAR.K_GMM.get_W_fast(moments)
        # J2 = SVAR.K_GMM.loss(u, b, W, restrictions=[], moments=moments, moments_powerindex=moments_powerindex, whiten=True, blocks=False)
        # #print('J2:', J2)

        # GMM_W with Wfast
        moments = SVAR.SVARutilGMM.get_Cr(4, 4)
        moments_powerindex = SVAR.SVARutilGMM.get_Moments_powerindex(moments)
        W = SVAR.estimatorGMMW.get_W_fast(moments)
        J = SVAR.estimatorGMMW.loss(u, b, W, restrictions=[], moments=moments, moments_powerindex=moments_powerindex,
                                     blocks=False)
        # print('J:', J)

        # GMM fast
        moments = SVAR.SVARutilGMM.get_Mr(4, 4)
        moments_powerindex = SVAR.SVARutilGMM.get_Moments_powerindex(moments)
        W = np.eye(np.shape(moments)[0])
        H = SVAR.estimatorGMMW.loss(u, b, W, restrictions=[], moments=moments, moments_powerindex=moments_powerindex,
                                      blocks=False)
        # print('H:', H)

        # print('J+H',J+H)
        # print('J2+H',J2+H)
        # print(" ")
        return J, H

    b = np.array([1., 0, 0., 0., 0., 0.])
    J1, H1 = check_fast_weighting(b)

    b = np.array([0., 1, 0., 0., 0., 0.])
    J2, H2 = check_fast_weighting(b)

    b = np.array([0., 1, 1., 1., 0., 1.])
    J3, H3 = check_fast_weighting(b)

    assert np.round(J1 + H1,5) == np.round(J2 + H2,5)
    assert np.round(J1 + H1,5) == np.round(J3 + H3,5)
    assert np.round(J2 + H2,5) == np.round(J3 + H3,5)
