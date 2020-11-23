import copy
import itertools
from types import FunctionType

import numpy as np

import SVAR


def get_Cr(r, n):
    # Generates indices of co-moments

    Cr_tmp = list(itertools.combinations_with_replacement(range(r), n))
    Cr = list()
    for cr in Cr_tmp:
        if sum(cr) == r:
            for this_cr in np.unique(list(itertools.permutations(cr)), axis=0):
                Cr.append(this_cr)
    Cr = np.asarray(Cr)
    return Cr


def get_Mr(r, n):
    # Generates indices of moments (e.g. variance)

    Mr = np.dot(np.eye(n), r)
    Mr = Mr.astype(int)
    return Mr


def get_Moments(estimator, n, blocks=False, addThirdMoments=False, addFourthMoments=True):
    if blocks == False:
        blocks = list()
        blocks.append(np.array([1, n]))
    moments = np.full([1, n], 0)

    if estimator == 'GMM' or estimator == 'LassoB':
        moments = np.append(moments, get_Mr(2, n), axis=0)
        moments = np.append(moments, get_Cr(2, n), axis=0)

    for block in blocks:
        n_min = block[0]
        n_max = block[1]
        block_length = 1 + n_max - n_min
        if block_length > 1:
            moments_this = np.full([1, block_length], 0)
            if addThirdMoments:
                if estimator == 'GMM' or estimator == 'LassoB':
                    moments_this = np.append(moments_this, get_Cr(3, block_length), axis=0)
                elif estimator == 'GMM_W':
                    moments_this = np.append(moments_this, get_Cr(3, block_length), axis=0)
                elif estimator == 'GMM_WF':
                    moments_this = np.append(moments_this, get_Mr(3, block_length), axis=0)

            if addFourthMoments:
                if estimator == 'GMM' or estimator == 'LassoB':
                    moments_this = np.append(moments_this, get_Cr(4, block_length), axis=0)
                elif estimator == 'GMM_W':
                    moments_this = np.append(moments_this, get_Cr(4, block_length), axis=0)
                elif estimator == 'GMM_WF':
                    moments_this = np.append(moments_this, get_Mr(4, block_length), axis=0)
            moments_this = moments_this[1:, :]

            if np.shape(moments_this)[0] != 0:
                number_moments_this = np.shape(moments_this)[0]
                moments_this = np.hstack([np.zeros([number_moments_this, n_min - 1], dtype=int), moments_this])
                moments_this = np.hstack([moments_this, np.zeros([number_moments_this, n - n_max], dtype=int)])
                moments = np.append(moments, moments_this, axis=0)

    moments = moments[1:, :]
    return moments


def get_Moments_powerindex(moments):
    moments_fast = np.zeros([np.shape(moments)[0], np.shape(moments)[1] * 5], dtype=bool)
    counter = 0
    for moment in moments:
        for i in range(np.shape(moments)[1]):
            moments_fast[counter, i * 5 + moment[i]] = 1
        counter += 1
    return moments_fast


def get_Moment_transformed(moment):
    moment_trans = np.array([], dtype=int)
    for z in range(np.size(moment)):
        moment_trans = np.append(moment_trans, np.ones(moment[z], dtype=int) * (z + 1))
    return moment_trans


def get_f(u, b, restrictions, moments, moments_powerindex, whiten=False, blocks=False):
    T, n = np.shape(u)
    e = SVAR.innovation(u, b, restrictions=restrictions, whiten=whiten, blocks=blocks)
    e_power = np.empty([np.shape(e)[0], n * 5])
    e_power[:, range(0, 5 * n, 5)] = np.ones([np.shape(e)[0], n])
    for i in range(n):
        for j in range(1, 5):
            this_entry = (i) * 5 + j
            e_power[:, this_entry] = np.multiply(e_power[:, this_entry - 1], e[:, i])

    def calc_f(moments, moments_powerindex):
        counter = 0
        f = np.empty([T, np.shape(moments)[0]])
        for mom_fast in moments_powerindex:
            f[:, counter] = np.prod(e_power[:, mom_fast], axis=1)
            if ~np.isin(1, moments[counter, :]):
                if np.isin(2, moments[counter, :]):
                    f[:, counter] = np.subtract(f[:, counter], 1)
                if np.isin(4, moments[counter, :]):
                    f[:, counter] = np.subtract(f[:, counter], 3)
            counter += 1
        return f

    f = calc_f(moments, moments_powerindex)

    return f


def get_g(u, b, restrictions, moments, moments_powerindex, whiten=False, blocks=False):
    return np.mean(
        get_f(b=b[:], u=u[:], restrictions=restrictions[:], moments=moments[:], moments_powerindex=moments_powerindex,
              whiten=whiten, blocks=blocks), axis=0)


def generate_Jacobian_function(moments, restrictions):
    n = np.shape(moments)[1]
    function_string = str("def f(u,b,restrictions,Omega): ")
    function_string = function_string + str('B = SVAR.get_BMatrix(b, restrictions=restrictions);')
    function_string = function_string + str('A = np.linalg.inv(B);')

    function_string = function_string + str('Omega2, Omega3, Omega4 = Omega;')

    function_string = function_string + str('G=np.array([')
    counter = 0
    for moment in moments:
        counter_inner = 0
        if counter != 0:
            function_string = function_string + str(',')
        # Calculate Jacboian of this moment
        this_moment_string = str('[ ')
        for i in range(0, n):
            for j in range(0, n):
                # only if B element is not restricted
                if np.isnan(restrictions[i, j]):
                    tmp_string = str(' ')
                    for idx in range(0, n):
                        if moment[idx] != 0:
                            moment_new = copy.deepcopy(moment)
                            moment_new[j] = moment_new[j] + 1
                            moment_new[idx] = moment_new[idx] - 1

                            moment_new_trans = get_Moment_transformed(moment_new)

                            if sum(moment_new) == 2:
                                omega_str = 'Omega2[' + str(int(moment_new_trans[0]) - 1) + ', ' + str(
                                    int(moment_new_trans[1]) - 1) + ']'
                            if sum(moment_new) == 3:
                                omega_str = 'Omega3[' + str(int(moment_new_trans[0]) - 1) + ', ' + str(
                                    int(moment_new_trans[1]) - 1) + ', ' + str(int(moment_new_trans[2]) - 1) + ']'
                            if sum(moment_new) == 4:
                                omega_str = 'Omega4[' + str(int(moment_new_trans[0]) - 1) + ', ' + str(
                                    int(moment_new_trans[1]) - 1) + ', ' + str(
                                    int(moment_new_trans[2]) - 1) + ', ' + str(int(moment_new_trans[3]) - 1) + ']'

                            if moment[idx] != 0:
                                tmp_string = tmp_string + str(
                                    '-A[' + str(idx) + ',' + str(i) + '] * ' + str(moment[idx]) + ' * ' + omega_str)

                    if counter_inner != 0:
                        this_moment_string = this_moment_string + str(',')
                    counter_inner += 1
                    this_moment_string = this_moment_string + tmp_string
        this_moment_string = this_moment_string + str(']')

        # Append to function_string
        function_string = function_string + this_moment_string
        counter += 1

    function_string = function_string + str(']);')
    function_string = function_string + str('return G')
    # print(function_string)

    function_code = compile(function_string, "<string>", "exec")
    function_func = FunctionType(function_code.co_consts[0], globals(), "f")

    return function_func


def get_G_ana(Moments, B, omega, B_restrictions=[]):
    # Calculates analytical G = E[ partial f(u,B_0) / partial B ]

    # ToDo: Check if i j loop is correct
    n, n = np.shape(B)

    if np.array(B_restrictions).size == 0: B_restrictions = np.full([n, n], np.nan)

    A = np.linalg.inv(B)
    n = np.shape(B)[1]
    # add one row to omega for moment e^0=1
    omega = np.concatenate((np.ones([n, 1]), omega), axis=1)
    # empty G array
    G = np.empty([np.shape(Moments)[0], np.sum(np.isnan(B_restrictions))])
    moment_counter = 0
    for moment in Moments:
        elementcounter = 0
        for i in range(0, n):
            for j in range(0, n):
                # only if B element is not restricted
                if np.isnan(B_restrictions[i, j]):
                    G_this = 0
                    for idx in range(0, n):
                        moment_new = copy.deepcopy(moment)
                        moment_new[j] = moment_new[j] + 1
                        moment_new[idx] = moment_new[idx] - 1
                        G_this = G_this - A[idx, i] * moment[idx] * np.prod(omega[np.array([range(0, n)]), moment_new])
                    G[moment_counter, elementcounter] = G_this
                    elementcounter = elementcounter + 1
        moment_counter = moment_counter + 1
    return G


def get_S_ana(Moments_1, Moments_2, omega):
    # Calculates analytically S = E[ f_[Moments_1](u,B_0) f_[Moments_2](u,B_0)' ]

    # empty S array
    S = np.empty([np.shape(Moments_1)[0], np.shape(Moments_2)[0]])
    n = np.shape(Moments_1)[1]
    # add one row to omega for moment e^0=1
    omega = np.concatenate((np.ones([n, 1]), omega), axis=1)
    moments_1_counter = 0
    moments_2_counter = 0
    for moment_1 in Moments_1:
        for moment_2 in Moments_2:
            v_1 = - int(~np.isin(1, moment_1))
            v_2 = - int(~np.isin(1, moment_2))
            S[moments_1_counter, moments_2_counter] = np.prod(
                omega[np.array([range(0, n)]), moment_1 + moment_2]) + v_1 * np.prod(
                omega[np.array([range(0, n)]), moment_2]) + v_2 * np.prod(
                omega[np.array([range(0, n)]), moment_1]) + v_1 * v_2
            moments_2_counter = moments_2_counter + 1
        moments_2_counter = 0
        moments_1_counter = moments_1_counter + 1
    return S


def get_Wopt_ana(Moments, omega):
    # Analytical calculation of the asymptotically optimal weighting matrix
    S = get_S_ana(Moments, Moments, omega)
    W_opt = np.linalg.inv(S)
    return W_opt