import pytest
import sys, os
import numpy as np
import sys
try:
    sys.path.append('../AutoDiff')
    sys.path.append('../Implementation')
    import AD_numpy as anp
    from Optimizer import minimize, PRECISION
except:
    import AutoDiff.AD_numpy as anp
    from Implementation.Optimizer import minimize, PRECISION

def no_minimum(method):
    f = lambda x,y: x+y
    x = [0,0]
    r = minimize(f, x, method)
    assert r.converge == False

def parabola(method):
    f = lambda x,y: x**2+y**2
    x = [20, 12]
    r = minimize(f, x, method)
    distance_from_zero = np.linalg.norm(r.x)
    assert distance_from_zero < PRECISION/2

def at_minimum(method):
    f = lambda x,y: x**2+y**2
    x = [0, 0]
    r = minimize(f, x, method)
    distance_from_zero = np.linalg.norm(r.x)
    print(r.val_rec)
    assert distance_from_zero < PRECISION/2

def parabola_univariate(method):
    f = lambda x: x**2
    x = [-20]
    r = minimize(f, x, method)
    assert np.linalg.norm(r.x) < PRECISION/2

def saddle(method):
    f = lambda x,y: x**3+y**3
    x = [1, 15]
    r = minimize(f, x, method)

def start_at_max(method):
    f = lambda x: anp.cos(x)
    x = [0]
    r = minimize(f, x, method)

def test_BFGS():
    m = 'BFGS'
    no_minimum(m)
    parabola(m)
    at_minimum(m)
    parabola_univariate(m)
    saddle(m)
    start_at_max(m)

def test_gradient_descent():
    m = 'Gradient Descend'
    no_minimum(m)
    parabola(m)
    at_minimum(m)
    parabola_univariate(m)
    saddle(m)
    start_at_max(m)

def test_conjugate_gradient():
    m = 'Conjugate Gradient'
    no_minimum(m)
    parabola(m)
    at_minimum(m)
    parabola_univariate(m)
    saddle(m)
    start_at_max(m)

def test_steepest_descent():
    m = 'Steepest Descend'
    no_minimum(m)
    parabola(m)
    at_minimum(m)
    parabola_univariate(m)
    saddle(m)
    start_at_max(m)


if __name__ == "__main__":
    test_optimization()
