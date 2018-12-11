import sys,os
#sys.path.append('../AutoDiff')
base_dir = os.path.dirname(__file__) or '.'

package_dir_a = os.path.join(base_dir, '../AutoDiff')
sys.path.insert(0, package_dir_a)

try:
    from variables import Variable
except:
    from AutoDiff.variables import Variable


import time
import numpy as np

PRECISION = 1e-5
MAXITER = 1000

class Result:
    def __init__(self, x, val_rec, time_rec, converge):
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
        - 'Gradient Descend'            :ref:`(see here) <optimizer.min_gradient_descend>`
        - 'Conjugate Gradient'          :ref:`(see here) <optimizer.min_conjugate_gradient>`
        - 'Steepest Descend'            :ref:`(see here) <optimizer.min_steepestdescent>`

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
    elif method == "Steepest Descend":
        return min_steepestdescent(fun, x0, **kwargs)
    elif method =="BFGS":
        return min_BFGS(fun,x0, **kwargs)
    elif method == "Gradient Descend":
        return min_gradientdescent(fun, x0, **kwargs)
    else:
        raise ValueError("{} is not a valid optimization method".format(method))

class Model(object):
    def __init__(self, data):
        self.data = data
        self.all_data = data
        self.row_idx = 0

    def make_stochastic(self):
        self.stochastic = True
        self.row_idx = 0
        self.data = self.all_data.iloc[self.row_idx,:]

    def make_deterministic(self):
        self.data = self.all_data

    def step(self):
        if self.row_idx < len(self.all_data) - 1:
            self.row_idx += 1
        else:
            self.row_idx = 0
        self.data = self.all_data.iloc[self.row_idx,:]

    def predict(self):
        raise NotImplementedError

    def loss(self):
        raise NotImplementedError

def minimize_over_data(model, init_param, method, epochs, stochastic = False, **kwargs):
    if stochastic:
        x = np.array(init_param)
        val_rec = [x]
        model.make_stochastic()
        for epoch in range(epochs):
            for rows in range(len(model.all_data)):
                r = minimize(model.loss, x, method = method, max_iter = 1, **kwargs)
                x = r.x
                val_rec.append(r.x)
                model.step()
        model.make_deterministic()
        r = Result(x, val_rec, None, None)
    else:
        x = init_param
        val_rec = [x]
        r = minimize(model.loss, x, method = method, max_iter = epochs, **kwargs)
    return r

def min_conjugate_gradient(fn, x0, precision=PRECISION, max_iter=MAXITER, sigma=0.01, norm=np.inf):
    # create initial variables
    x = np.array(x0)

    # create initial variables
    var_names = ['x'+str(idx) for idx in range(len(x))]

    # initial gradient and steepest descent
    # recreate new variables with new values
    grad0 = _get_grad(fn, x, var_names)
    x = -grad0
    conj_direct = x

    nums_iteration = 0
    val_rec = [np.array(x0)]
    time_rec = [0]
    time_total = 0
    while np.linalg.norm(grad0, norm) > precision and nums_iteration < max_iter:
        start_time = time.time()
        # secant method line search
        alpha = (-sigma*grad0 @ conj_direct) / (_get_grad(fn, x+sigma*conj_direct, var_names)@conj_direct -  grad0@conj_direct)
        x = x + alpha*conj_direct
        grad1 = _get_grad(fn, x, var_names)
        beta = (grad1 @ grad1) / (grad0 @ grad0)
        conj_direct = -grad1 + beta*conj_direct

        # store history of values
        val_rec.append(x)
        time_total = time_total + time.time()-start_time
        time_rec.append(time_total)

        # update grad
        grad0 = grad1

        # threshold stopping condition
        # maximum norm
        nums_iteration += 1

    if np.linalg.norm(grad0, norm) < precision:
        # reshape val_rec
        val_rec = np.concatenate(val_rec).reshape(-1, len(x))
        time_rec = np.array(time_rec)
        return Result(x, val_rec, time_rec, True)

    # iteration stopping condition
    if nums_iteration >= max_iter:
        # reshape val_rec
        val_rec = np.concatenate(val_rec).reshape(-1, len(x))
        time_rec = np.array(time_rec)
        return Result(x, val_rec, time_rec, False)


def min_steepestdescent(fn, x0, precision=PRECISION, max_iter=MAXITER, sigma=0.01, norm=2):
     # create initial variables
    x = np.array(x0,dtype=float)
    var_names = ['x'+str(idx) for idx in range(len(x))]

    val_rec = [x.copy()]
    time_rec = [0]
    init_time = time.time()

    for i in range(max_iter):
        s = -_get_grad(fn, x, var_names)
        # secant method line search
        eta = (-sigma*s @ s) / (_get_grad(fn, x+sigma*s, var_names)@s -  s@s)

        dx = eta*s
        x += dx

        val_rec.append(x.copy())
        
        grad1 = _get_grad(fn,x,var_names)
        time_rec.append(time.time()-init_time)

        # threshold stopping condition
        # maximum norm

        if np.linalg.norm(grad1, norm) < precision:
            # reshape val_rec
            val_rec = np.array(val_rec)
            time_rec = np.array(time_rec)
            return Result(x, val_rec, time_rec, True)

    converge = (np.linalg.norm(grad1, norm) <= precision)

    return Result(x, np.array(val_rec), np.array(time_rec), converge)


def _get_grad(fn, x, var_names):
    variables = [Variable(var_names[idx], x_n) for idx, x_n in enumerate(x)]
    out = fn(*variables)
    jacobian = out.jacobian()
    grad = np.array([jacobian[name] for name in var_names])
    return grad


# def _line_search(fn, x, search_direction, grad, beta = 0.9, c = 0.9, alpha_init = 1):
#     """approximately minimizes f along search_direction
#     https://en.wikipedia.org/wiki/Backtracking_line_search
#     """
#     m = search_direction.T.dot(grad)
#     alpha = alpha_init
#     while (fn(*(x)) - fn(*(x+alpha*search_direction))) < -c*alpha*m:
#         alpha = alpha * beta
#     return alpha


# def _update_hessian(approx_hessian, d_grad, step):
#     return (approx_hessian
#             + 1/(d_grad.T.dot(step))*d_grad.dot(d_grad.T)
#             - 1/(step.T.dot(approx_hessian).dot(step))*(approx_hessian.dot(step).dot(step.T).dot(approx_hessian.T))
#            )


def min_BFGS(fn, x0, precision=PRECISION, max_iter=MAXITER, beta=0.9, c=0.9, alpha_init=1, norm=np.inf):
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
            val_rec = np.array(val_rec)
            time_rec = np.array(time_rec)
            return Result(x, val_rec, time_rec, True)

    converge = (np.linalg.norm(grad1, norm) <= precision)
    return Result(x, np.array(val_rec), time_rec, converge)

def min_gradientdescent(fn, x0, precision = PRECISION, max_iter = MAXITER, lr=1e-2, norm=2):
    x = np.array(x0)

    var_names = ['x'+str(idx) for idx in range(len(x))]

    nums_iteration = 0
    val_rec = [x]
    time_rec = [0]
    time_total = 0
    while True:
        start_time = time.time()
        g = _get_grad(fn, x, var_names)
        x = x - lr*g

        # store history of values
        val_rec.append(x)
        time_total = time_total + time.time()-start_time
        time_rec.append(time_total)

        # threshold stopping condition
        if np.linalg.norm(g, norm) < precision:
            return Result(x, val_rec, time_rec, True)

        # iteration stopping condition
        if nums_iteration >= max_iter:
            return Result(x, val_rec, time_rec, False)
        nums_iteration +=1
