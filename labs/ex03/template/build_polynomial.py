# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    _x = np.ones((np.shape(x)[0], degree+1))
    for i in range(1, degree+1):
        _x[:, i] = x**i
    return _x