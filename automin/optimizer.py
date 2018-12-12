import sys
import os
import warnings
import time
import numpy as np
from .autodiff.variables import Variable

# sys.path.append('../AutoDiff')
#base_dir = os.path.dirname(__file__) or '.'
#package_dir_a = os.path.join(base_dir, '../AutoDiff')
#sys.path.insert(0, package_dir_a)



PRECISION = 1e-3
MAXITER = 5000
NORM = 2


class Result:
    def __init__(self, x, val_rec, time_rec, converge, convergence_warning = True):
        """Record the optimization results and performance

        INPUTS
        =======
        x array: optimization value, can be either Variable or value. //Or just store the value, since there is no need for its derivatives
        val_rec array: stores the function inputs at each iteration, save for plotting the accuracy results.
        time_rec array: stores the cumulative time at each iteration.
        converge boolean: did the optimization procedure converge
        """
        self.x = x
        self.val_rec = val_rec
        self.time_rec = time_rec
        self.converge = converge
        if (not converge) and convergence_warning:
            warnings.warn("optimization did not converge")
        # throw warning if not convergent


def minimize(fun, x0, method=None, **kwargs):
    """Minimization of scalar or vector function of scalar or vector variables.

    INPUTS
    =======
    fun: callable object. The opjective function to be minimized.
    x0: variable inputs or normal value tuple. Initial guess.
    args: tuple (optional). Extra arguments passed to the opjective function.
    method: string (optional). Type of different optimizer. Should be one of

        - 'BFGS'                        :ref:`(see here) <optimizer.min_BFGS>`
        - 'Gradient Descent'            :ref:`(see here) <optimizer.min_gradient_descent>`
        - 'Conjugate Gradient'          :ref:`(see here) <optimizer.min_conjugate_gradient>`
        - 'Steepest Descent'            :ref:`(see here) <optimizer.min_steepestdescent>`

        If not specified, it will automatically choose 'Newton Method'.



    RETURNS
    ========
    res: OptimizationResult. Maybe a Variable or a normal value tuple, depends on the input object.

    NOTES
    =====
    PRE:
         - fun is normal function.
         - x0 are initial guess of the results

    POST:
         - fun and x0 are not changed by this function
         - if initial guess x0 is a Variable instance,
         returns a new Variable instance
         - if x0 is numeric, returns numeric
    """
    if method == "Conjugate Gradient":
        return min_conjugate_gradient(fun, x0, **kwargs)
    elif method == "Steepest Descent":
        return min_steepestdescent(fun, x0, **kwargs)
    elif method == "BFGS" or method=="None":
        return min_BFGS(fun, x0, **kwargs)
    elif method == "Gradient Descent":
        return min_gradientdescent(fun, x0, **kwargs)
    else:
        raise ValueError(
            "{} is not a valid optimization method".format(method))


class Model(object):
    def __init__(self, data):
        self.data = data
        self.all_data = data
        self.row_idx = 0

    def make_stochastic(self):
        self.stochastic = True
        self.row_idx = 0
        self.data = self.all_data.iloc[self.row_idx, :]

    def make_deterministic(self):
        self.data = self.all_data

    def step(self):
        if self.row_idx < len(self.all_data) - 1:
            self.row_idx += 1
        else:
            self.row_idx = 0
            self.data = self.data.sample(frac=1)
        self.data = self.all_data.iloc[self.row_idx, :]

    def predict(self):
        raise NotImplementedError

    def loss(self):
        raise NotImplementedError

def minimize_over_data(model, init_param, method, epochs, stochastic = False, **kwargs):
    supported_stochastic_methods = ['Gradient Descent']
    if stochastic:
        if method not in supported_stochastic_methods:
            raise ValueError("""{} is not supported for stochastic optimization.
                                Supported methods are {}"""
                                .format(method, supported_stochastic_methods))
        x = np.array(init_param)
        val_rec = [x]
        model.make_stochastic()
        time_rec = [0]
        time_total = 0
        for epoch in range(epochs):
            start_time = time.time()
            for rows in range(len(model.all_data)):
                r = minimize(model.loss, x, method=method,
                             max_iter=1, convergence_warning = False, **kwargs)
                x = r.x
                val_rec.append(r.x)
                model.step()
                time_total = time_total + time.time()-start_time
                time_rec.append(time_total)
        model.make_deterministic()
        r = Result(x, val_rec, time_rec, None, convergence_warning = False)
    else:
        x = init_param
        val_rec = [x]
        r = minimize(model.loss, x, method=method, max_iter=epochs, **kwargs)
    return r


def min_conjugate_gradient(fn, x0, precision=PRECISION, max_iter=10000, sigma=0.01, norm=NORM, **kwargs):
    # create initial variables
    x = np.array(x0)

    # create initial variables
    var_names = ['x'+str(idx) for idx in range(len(x))]

    # initial gradient and steepest descent
    val_rec = [x.copy()]
    time_rec = [0]
    time0 = time.time()
    sgrad0 = -_get_grad(fn, x, var_names)

    if np.linalg.norm(sgrad0, norm) <= precision:
        # reshape val_rec
        return Result(x, np.array(val_rec), time_rec, True)

    gradsigma = _get_grad(fn, x+sigma*sgrad0, var_names)
    # check this
    alpha = (-sigma*sgrad0@sgrad0) / (gradsigma@sgrad0 - sgrad0@sgrad0)
    x = x + alpha*sgrad0
    conj_direct = sgrad0

    val_rec.append(x.copy())
    time_rec.append(time.time()-time0)
    init_time = time.time()

    for i in range(max_iter-1):
        sgrad1 = -_get_grad(fn, x, var_names)

        beta = min(0, (sgrad1 @ (sgrad0-sgrad1)) / (sgrad0 @ sgrad0))
        conj_direct = sgrad1 + beta*conj_direct
        gradsigma = _get_grad(fn, x+sigma*conj_direct, var_names)
        # secant method
        alpha = (sigma*sgrad1 @ conj_direct) / (gradsigma@conj_direct +  sgrad1@conj_direct)
        x = x + alpha*conj_direct

        # store history of values
        val_rec.append(x.copy())
        time_rec.append(time.time()-init_time)

        # update grad
        sgrad0 = sgrad1

        if np.linalg.norm(sgrad1, norm) <= precision:
            # reshape val_rec
            return Result(x, np.array(val_rec), time_rec, True)

    return Result(x, np.array(val_rec), time_rec, False)

def min_steepestdescent(fn, x0, precision=PRECISION, max_iter=MAXITER, sigma=0.01, norm=NORM, **kwargs):
     # create initial variables
    x = np.array(x0, dtype=float)
    var_names = ['x'+str(idx) for idx in range(len(x))]

    val_rec = [x.copy()]
    time_rec = [0]
    init_time = time.time()

    for i in range(max_iter):
        grad1 = _get_grad(fn, x, var_names)
        # threshold stopping condition
        # maximum norm

        if np.linalg.norm(grad1, norm) <= precision:
            # reshape val_rec
            return Result(x, np.array(val_rec), time_rec, True)
        s = -grad1
        # secant method line search
        eta = (-sigma*grad1 @ s) / (_get_grad(fn, x+sigma*s, var_names)@s - grad1@s)
        #eta = scmin(lambda eta: fn(*(x+eta*s)), 0)

        dx = eta*s
        x += dx

        val_rec.append(x.copy())
        time_rec.append(time.time()-init_time)


    return Result(x, np.array(val_rec), time_rec, False)


def _get_grad(fn, x, var_names):
    variables = [Variable(var_names[idx], x_n) for idx, x_n in enumerate(x)]
    out = fn(*variables)
    jacobian = out.jacobian()
    grad = np.array([jacobian[name] for name in var_names])
    return grad


def get_gradient(fn, x, var_names,**kwargs):
    return _get_grad(fn, x, var_names,**kwargs)

def min_BFGS(fn, x0, precision=PRECISION, max_iter=MAXITER, beta=0.9, c=0.9, alpha_init=1, norm=NORM, **kwargs):
    approx_hessian = np.eye(len(x0))

    x = np.array(x0, dtype=np.float)

    var_names = ['x'+str(idx) for idx in range(len(x))]

    val_rec = [x.copy()]
    time_rec = [0]
    init_time = time.time()

    for i in range(max_iter):
        grad_now = _get_grad(fn, x, var_names)
        s = np.linalg.solve(approx_hessian, -grad_now)
        x += s
        val_rec.append(x.copy())

        # update matrix Hessian
        grad1 = _get_grad(fn, x, var_names)
        y = grad1-grad_now
        dH1 = np.outer(y, y)/np.dot(y, s)
        Hs = np.dot(approx_hessian, s)
        dH2 = -np.outer(Hs, Hs)/np.dot(Hs, s)
        approx_hessian += dH1+dH2

        time_rec.append(time.time()-init_time)

        if np.linalg.norm(grad1, norm) <= precision:
            # reshape val_rec
            return Result(x, np.array(val_rec), time_rec, True)

    return Result(x, np.array(val_rec), time_rec, False)


def min_gradientdescent(fn, x0, precision=1e-2, max_iter=30000, lr=1e-3, norm=NORM, **kwargs):
    x = np.array(x0)

    var_names = ['x'+str(idx) for idx in range(len(x))]

    val_rec = [x.copy()]
    time_rec = [0]
    initial_time = time.time()
    g = _get_grad(fn, x, var_names)

    for i in range(max_iter):
        x = x - lr*g

        # store history of values
        val_rec.append(x)
        time_rec.append(time.time()-initial_time)
        g = _get_grad(fn, x, var_names)

        # threshold stopping condition
        if np.linalg.norm(g, norm) <= precision:
            return Result(x, np.array(val_rec), time_rec, True)

        # iteration stopping condition
    return Result(x, np.array(val_rec), time_rec, False)
