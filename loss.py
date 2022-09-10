import mindspore.nn as nn
from mindspore import ops
import numpy as np

class MaximalCodingRateReduction(nn.Cell):
    def __init__(self, eps=0.1, gam=1.):
        super(MaximalCodingRateReduction, self).__init__()
        self.eps = eps
        self.gam = gam

    def forward(self, Z, y):
        m, d = Z.shape
        I = ops.eye(d)
        c = d / (m * self.eps)
        loss_expd = logdet(c * covariance(Z) + I) / 2.
        loss_comp = 0.
        for j in y.unique():
            Z_j = Z[(y == int(j))[:, 0]]
            m_j = Z_j.shape[0]
            c_j = d / (m_j * self.eps)
            logdet_j = logdet(I + c_j * Z_j.T @ Z_j)
            loss_comp += logdet_j * m_j / (2 * m)
        loss_expd, loss_comp = loss_expd.item(), loss_comp.item()
        return loss_expd - loss_comp
        
def covariance(X):
    return  np.einsum('ji...,jk...->ik...', X, X.conj())

def logdet(X):
    sgn, logdet = np.linalg.slogdet(X)
    return sgn * logdet