import pytest
import pandas as pd
import sys, os
import numpy as np
import sys
from ..automin.autodiff import AD_numpy as anp
from ..automin.optimizer import minimize, PRECISION, Model, minimize_over_data

def rosenbrock(method):
    a = 1
    b = 100
    f = lambda x,y: (a-x)**2+b*(y-x**2)**2
    x = [2,2]
    r = minimize(f,x,method)
    distance_from_min = np.linalg.norm(r.x-np.array([1,1]))
    print(r.converge)
    if method == 'Gradient Descend':
        p = 1e-2
    else:
        p = PRECISION
    assert distance_from_min < p

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
    rosenbrock(m)
    no_minimum(m)
    parabola(m)
    at_minimum(m)
    parabola_univariate(m)
    saddle(m)
    start_at_max(m)

def test_gradient_descent():
    m = 'Gradient Descend'
    rosenbrock(m)
    no_minimum(m)
    parabola(m)
    at_minimum(m)
    parabola_univariate(m)
    saddle(m)
    start_at_max(m)

def test_conjugate_gradient():
    m = 'Conjugate Gradient'
    rosenbrock(m)
    no_minimum(m)
    parabola(m)
    at_minimum(m)
    parabola_univariate(m)
    saddle(m)
    start_at_max(m)

def test_steepest_descent():
    m = 'Steepest Descend'
    rosenbrock(m)
    no_minimum(m)
    parabola(m)
    at_minimum(m)
    parabola_univariate(m)
    saddle(m)
    start_at_max(m)

def test_minimize_over_data():
    indep_var = np.random.normal(size = (100,2))
    data = pd.DataFrame(data = indep_var, columns = ['indep_var1','indep_var2'])
    data['dep_var'] = 2*data.indep_var1 + 3*data.indep_var2

    class MSE_Regression(Model):
        def __init__(self, data):
            super().__init__(data)

        def predict(self, beta1, beta2):
            return self.data['indep_var1']*beta1 + self.data['indep_var2']*beta2

        def loss(self, beta1, beta2):
            prediction = self.predict(beta1, beta2)
            return np.sum((prediction-self.data['dep_var'])**2)

    model = MSE_Regression(data)

    r_all = minimize_over_data(model, [10,10], 'Gradient Descend', 2000, stochastic = False, lr = 1e-4)
    r_all.x
    assert np.linalg.norm(r_all.x - np.array([2,3])) < PRECISION/2
    r_stoch = minimize_over_data(model, [10,10], 'Gradient Descend', 100, stochastic = True, lr = 1e-3)
    r_stoch.x
    assert np.linalg.norm(r_stoch.x - np.array([2,3])) < 0.1


if __name__ == "__main__":
    test_optimization()
