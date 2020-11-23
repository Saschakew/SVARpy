import copy
import itertools
import numpy as np
import scipy
from pyunlocbox import acceleration, solvers
from scipy.optimize import LinearConstraint



def innovation(u, b, restrictions=[], whiten=False, blocks=False):
    # Calculates unmixed innovations according to: e = B^{-1} u

    B = get_BMatrix(b, restrictions=restrictions, whiten=whiten, blocks=blocks)
    A = np.linalg.inv(B)
    e = np.matmul(A, np.transpose(u))
    e = np.transpose(e)
    return e


def get_BMatrix(b, restrictions=[], whiten=False, blocks=False):
    # Transforms vectorized B into B
    if whiten:
        # ToDo: Add white restrictions
        B = get_Orthogonal(b, blocks)
    else:
        if blocks:
            restrictions = getRestrictions_blocks(blocks)

        if np.array(restrictions).size == 0:
            b_length = np.sqrt(np.size(b))
            n = b_length.astype(int)
            restrictions = np.full([n, n], np.nan)

        B = copy.deepcopy(restrictions)
        B[np.isnan(B)] = b

    return B


def get_BVector(B, restrictions=[], whiten=False):
    # inverse of get_BMatrix
    # ToDo: Error if element of B!=restrictions
    if whiten:
        # ToDo: Add restrictions
        b = get_Skewsym(B)
    else:
        if np.array(restrictions).size == 0:
            restrictions = np.full(np.shape(B), np.nan)
        b = B[np.isnan(restrictions) == 1]
    return b


def get_block_rec(n_rec, n):
    if not (n_rec):
        n_rec = 0
    blocks = list()
    for i in range(1, n_rec + 1):
        blocks.append(np.array([i, i]))
    if n_rec + 1 <= n:
        blocks.append(np.array([n_rec + 1, n]))
    return blocks


def getRestrictions_recursive(B):
    # restricts upper right triangular matrix
    myrestrictions = np.full(np.shape(B), np.nan)
    iu1 = np.triu_indices(np.shape(myrestrictions)[1], 1)
    myrestrictions[iu1] = B[iu1]
    return myrestrictions


def getRestrictions_blocks(blocks):
    myrestrictions = np.full([blocks[-1][-1], blocks[-1][-1]], np.nan)
    for block in blocks:
        myrestrictions[block[0] - 1:block[1], block[1]:] = 0
    return myrestrictions

def getRestrictions_none(n):
    myrestrictions = np.full([n,n], np.nan)
    return myrestrictions

def do_whitening(u, white):
    if white:
        T, n = np.shape(u)
        Sigma = np.dot(np.transpose(u), u) / T
        V = np.linalg.cholesky(Sigma)
    else:
        V = np.eye(np.size(u))
    Vinv = np.linalg.inv(V)
    z = np.matmul(Vinv, np.transpose(u))
    z = np.transpose(z)
    return z, V


def get_Skewsym(B):
    n = np.shape(B)[0]

    S = scipy.linalg.logm(B)
    il1 = np.tril_indices(n, k=-1)
    s = S[il1]
    return s


def get_Orthogonal(b, blocks=False):
    if not blocks:
        n = int(np.ceil(np.sqrt(2 * np.size(b))))
        blocks = list()
        blocks.append(np.array([1, n]))
    B = np.eye(blocks[-1][-1])

    b_counter = 0
    for block in blocks:
        n = block[1] - block[0] + 1
        S = np.full([n, n], 0.0)
        il1 = np.tril_indices(n, k=-1)
        n_s = np.int(np.size(il1) / 2)
        S[il1] = b[b_counter:b_counter + n_s]
        S = S - np.transpose(S)
        B_this = scipy.linalg.expm(S)
        B[block[0] - 1:block[1], block[0] - 1:block[1]] = B_this
        b_counter = b_counter + n_s
    if not (b_counter == np.shape(b)[0]):
        raise ValueError('Specified b value does not match B restrictions.')
    return B


## Permutation
def PermInTheta(B):
    n = np.shape(B)[0]
    for i in range(n):
        ind = np.argmax(np.abs(B[i, i:]))
        thisPerm = np.arange(n)
        thisPerm[i] = (i) + ind
        thisPerm[(i) + ind] = i
        B = B[:, thisPerm]
    S = np.diag(np.diag(np.sign(B)))
    B = np.matmul(B, S)
    return B


def get_AllSignPerm(n):
    Perms = []

    permutations = list(itertools.permutations(range(n)))
    num_permutations = np.shape(permutations)[0]

    signs = list(itertools.product(np.array([1, -1]), repeat=n))
    num_signs = np.shape(signs)[0]

    I = np.eye(n)
    for p in range(num_permutations):
        perm_this = I[permutations[p], :]
        for s in range(num_signs):
            sign_this = np.matmul(I, np.diag(signs[s]))
            sign_perm_this = np.matmul(perm_this, sign_this)
            Perms.append(sign_perm_this)

    return Perms


def get_BestSignPerm(eps, e, n_rec=0):
    n = np.shape(eps)[1]
    n_nonrec = n - n_rec

    Perm_rec = np.eye(n_rec)
    Perm_ur = np.zeros([n_rec, n_nonrec])
    Perm_ll = np.zeros([n_nonrec, n_rec])

    # Drop recursive part
    eps = eps[:, n_rec:]
    e = e[:, n_rec:]
    n = np.shape(eps)[1]
    Perms = get_AllSignPerm(n)

    score = np.zeros([np.shape(Perms)[0], n])
    for i, perm in enumerate(Perms):
        e_perm = np.matmul(e, perm)
        for j in range(n):
            _, p_val = scipy.stats.ks_2samp(eps[:, j], e_perm[:, j])
            score[i, j] = p_val

    score_sum = np.sum(score, axis=1)
    idx = np.argmax(score_sum)
    Perm_nonrec = Perms[idx]

    BestPerm = np.hstack((np.vstack((Perm_rec, Perm_ll)), np.vstack((Perm_ur, Perm_nonrec))))

    return BestPerm


def get_Omega(e):
    n = np.shape(e)[1]
    Omega2 = np.full([n, n], np.nan)
    Omega3 = np.full([n, n, n], np.nan)
    Omega4 = np.full([n, n, n, n], np.nan)
    for pi in range(n):
        for pj in range(pi, n):
            save_ij = np.array([np.prod(e[:, np.array([pi, pj])], axis=1)]).T
            Omega2[pi, pj] = np.mean(save_ij)
            for pk in range(pj, n):
                save_ijk = np.array([np.prod(np.append(save_ij, e[:, np.array([pk])], axis=1), axis=1)]).T
                Omega3[pi, pj, pk] = np.mean(save_ijk)
                for pl in range(pk, n):
                    save_ijkl = np.prod(np.append(save_ijk, e[:, np.array([pl])], axis=1), axis=1)
                    Omega4[pi, pj, pk, pl] = np.mean(save_ijkl)
    return Omega2, Omega3, Omega4


def get_Omega_Moments(e):
    n = np.shape(e)[1]
    Omega = np.full([n, 6], np.nan)
    Omega[:, 0] = np.mean(np.power(e, 1), axis=0)
    Omega[:, 1] = np.mean(np.power(e, 2), axis=0)
    Omega[:, 2] = np.mean(np.power(e, 3), axis=0)
    Omega[:, 3] = np.mean(np.power(e, 4), axis=0)
    Omega[:, 4] = np.mean(np.power(e, 5), axis=0)
    Omega[:, 5] = np.mean(np.power(e, 6), axis=0)

    return Omega

