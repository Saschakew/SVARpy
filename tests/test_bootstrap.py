import os
import numpy as np
import SVAR.estSVARbootstrap
import pickle
import pytest
np.random.seed(0)

if False:
    with open("tests\data\eps.data", 'rb') as filehandle:
        # read the data as binary data stream
        eps = pickle.load(filehandle)


    with open("tests\data/irf_bootstrap_typeGMM.data", 'rb') as filehandle:
        # read the data as binary data stream
        irf_bootstrap_typeGMM_expected = pickle.load(filehandle)

    with open("tests\data/irf_bootstrap_typeGMM_W.data", 'rb') as filehandle:
        # read the data as binary data stream
        irf_bootstrap_typeGMM_W_expected = pickle.load(filehandle)

    with open("tests\data/irf_bootstrap_typeGMM_WF.data", 'rb') as filehandle:
        # read the data as binary data stream
        irf_first_bootstrap_WF_expected = pickle.load(filehandle)

    with open("tests\data/irf_first_typeGMM.data", 'rb') as filehandle:
        # read the data as binary data stream
        irf_first_typeGMM_expected = pickle.load(filehandle)

    with open("tests\data/irf_first_typeGMM_W.data", 'rb') as filehandle:
        # read the data as binary data stream
        irf_first_typeGMM_W_expected = pickle.load(filehandle)

    with open("tests\data/irf_first_typeGMM_WF.data", 'rb') as filehandle:
        # read the data as binary data stream
        irf_first_typeGMM_WF_expected = pickle.load(filehandle)


#
# file_name = "data/irf_bootstrap_typeGMM_WF.data"
# with open(file_name, 'wb') as filehandle:
#     pickle.dump(irf_bootstrap, filehandle)
#
# file_name = "data/irf_first_typeGMM_WF.data"
# with open(file_name, 'wb') as filehandle:
#     pickle.dump(irf_first, filehandle)


@pytest.fixture
def supply_y():
    np.random.seed(0)
    with open("data/eps.data", 'rb') as filehandle:
        # read the data as binary data stream
        eps = pickle.load(filehandle)
    n=2
    eps = eps[:,:n]
    B = np.eye(n)
    B[1,0] = 0.5
    u = eps
    # Generate y
    AR = np.array([[[0.5], [0.2]], [[-0.3], [-0.1]]])
    const = np.zeros(n)
    trend = np.zeros(n)
    trend2 = np.zeros(n)
    supply_y = SVAR.estSVARbootstrap.simulate_SVAR(u, AR, const, trend, trend2)
    return supply_y

@pytest.fixture
def irf_bootstrap_typeGMM_expected():
    with open("data/irf_bootstrap_typeGMM.data", 'rb') as filehandle:
        # read the data as binary data stream
        irf_bootstrap_typeGMM_expected = pickle.load(filehandle)
    return irf_bootstrap_typeGMM_expected

@pytest.fixture
def irf_first_typeGMM_expected():
    with open("data/irf_first_typeGMM.data", 'rb') as filehandle:
        # read the data as binary data stream
        irf_first_typeGMM_expected = pickle.load(filehandle)
    return irf_first_typeGMM_expected

@pytest.fixture
def irf_bootstrap_typeGMM_W_expected():
    with open("data/irf_bootstrap_typeGMM_W.data", 'rb') as filehandle:
        # read the data as binary data stream
        irf_bootstrap_typeGMM_W_expected = pickle.load(filehandle)
    return irf_bootstrap_typeGMM_W_expected

@pytest.fixture
def irf_first_typeGMM_W_expected():
    with open("data/irf_first_typeGMM_W.data", 'rb') as filehandle:
        # read the data as binary data stream
        irf_first_typeGMM_W_expected = pickle.load(filehandle)
    return irf_first_typeGMM_W_expected

@pytest.fixture
def irf_bootstrap_typeGMM_WF_expected():
    with open("data/irf_bootstrap_typeGMM_WF.data", 'rb') as filehandle:
        # read the data as binary data stream
        irf_bootstrap_typeGMM_WF_expected = pickle.load(filehandle)
    return irf_bootstrap_typeGMM_WF_expected

@pytest.fixture
def irf_first_typeGMM_WF_expected():
    with open("data/irf_first_typeGMM_WF.data", 'rb') as filehandle:
        # read the data as binary data stream
        irf_first_typeGMM_WF_expected = pickle.load(filehandle)
    return irf_first_typeGMM_WF_expected

def test_get_IRF_bootstrap(supply_y, irf_first_typeGMM_expected, irf_bootstrap_typeGMM_expected):
    np.random.seed(0)
    y = supply_y

    ## Options
    # Options Bootstrap
    opt_bootstrap = dict()
    opt_bootstrap['number_bootstrap'] = 50
    opt_bootstrap['update_bstart'] = False
    opt_bootstrap['find_perm'] = True
    # Options reduced form
    opt_redform = dict()
    opt_redform['lags'] = 4
    opt_redform['add_const'] = True
    opt_redform['add_trend'] = False
    opt_redform['add_trend2'] = False
    # Options irf
    opt_irf = dict()
    opt_irf['irf_length'] = 12

    prepOptions = dict()
    prepOptions['n_rec'] = 2
    prepOptions['bstartopt'] = 'Rec'
    prepOptions['Wstartopt'] = 'I'
    prepOptions['printOutput'] = False
    out_irf, out_irf_bootstrap = SVAR.bootstrap_SVAR(y, options_bootstrap=opt_bootstrap,
                                                     options_redform=opt_redform, options_irf=opt_irf,
                                                     prepOptions=prepOptions)




    irf_first = np.round(out_irf, decimals=10)
    irf_first_typeGMM_expected = np.round( np.array([[[1.0, 0.0], [0.0086198252, 1.0]],
         [[0.4957324078, 0.2125646709], [-0.2985050531, -0.0925653134]],
         [[0.1817973586, 0.091658236], [-0.1243511447, -0.0564572495]],
         [[0.060665982, 0.0405916696], [-0.0672523401, -0.0145869251]],
         [[0.0254065154, 0.020342693], [0.0047198797, -0.0208507933]],
         [[0.0171719384, 0.0073896522], [0.003270915, 0.0001293565]],
         [[0.0104589816, 0.0043491219], [-0.0013248457, -0.0002395841]],
         [[0.0054131228, 0.0023869241], [-0.0014555017, -0.0006247698]],
         [[0.0027265981, 0.0012122409], [-0.0011197021, -0.0001223374]],
         [[0.0013233565, 0.000661925], [-0.0004615208, -0.0002321932]],
         [[0.0006650192, 0.0003264742], [-0.0001676857, -0.0001047662]],
         [[0.0003438223, 0.0001642951], [-8.53716e-05, -4.22166e-05]]]), decimals=10)
    irf_bootstrap = np.round(out_irf_bootstrap, decimals=10)
    irf_bootstrap_0_expected = np.round([[[1.0278999186, 0.0], [0.0019752128, 1.0054092435]], [[0.5158474256, 0.2080603266], [-0.2932645563, -0.0772780357]], [[0.2062711726, 0.1218802839], [-0.1340343201, -0.0464183466]], [[0.0603456738, 0.0628066099], [-0.0798629917, -0.0238182854]], [[-0.0032792658, 0.0146073688], [0.009064825, -0.0269806854]], [[-0.0059333974, -0.0007761191], [0.0131302788, 0.0018093269]], [[-0.0015430486, -0.0019697387], [0.006709697, 0.0027244263]], [[0.0014831514, -0.0009851499], [0.0027740296, 0.0021797835]], [[0.001686682, 0.0002075437], [-0.0004606077, 0.000829135]], [[0.0008733231, 0.0003812727], [-0.0006065711, -6.05218e-05]], [[0.0002822198, 0.0002317861], [-0.0003116873, -0.0001386321]], [[1.05576e-05, 8.37926e-05], [-6.88094e-05, -0.0001008998]]], decimals=10)

    assert irf_first.tolist() == irf_first_typeGMM_expected.tolist()
    assert irf_bootstrap[0].tolist() == irf_bootstrap_0_expected.tolist()
