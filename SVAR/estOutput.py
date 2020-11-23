import numpy as np
import pandas as pd


def idx_Best(n):
    idx = np.array([])
    for i in np.arange(n):
        string = "B(" + str(i + 1) + ',:)'
        idx = np.append(idx, string)
    return idx


def print_Avar(V):
    n = int(np.sqrt(np.shape(V)[0]))
    print(pd.DataFrame(data=np.reshape(np.diag(V), [n, n]),
                       index=idx_Best(n),
                       columns=col_Best(  n)))


def print_Moments(omega):
    n = np.shape(omega)[0]
    print(pd.DataFrame(data=omega, index=idx_moments(n),
                       columns=np.array(['E[e]', 'E[e^2]', 'E[e^3]', 'E[e^4]', 'E[e^5]', 'E[e^6]'])))


def print_B(B):
    n = np.shape(B)[0]
    print(pd.DataFrame(data=B, index=idx_Best(n),
                       columns=col_Best(n)))


def idx_moments(n):
    idx = np.array([])
    for i in np.arange(n):
        string = "e" + str(i + 1)
        idx = np.append(idx, string)
    return idx


def col_Best(n):
    idx = np.array([])
    for i in np.arange(n):
        string = "B(:," + str(i + 1) + ')'
        idx = np.append(idx, string)
    return idx