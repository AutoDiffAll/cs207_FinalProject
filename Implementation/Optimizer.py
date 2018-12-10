import sys
try:
    sys.path.append('../AutoDiff')
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

        - 'Newton Method'               :ref:`(see here) <optimizer.min_newton>`
        - 'BFGS'                        :ref:`(see here) <optimizer.min_BFGS>`
        - 'Stochastic Gradient Descend' :ref:`(see here) <optimizer.min_SGD>`
        - 'Gradient Descend'            :ref:`(see here) <optimizer.min_gradient_descend>`
        - 'Conjugate Gradient'          :ref:`(see here) <optimizer.min_conjugate_gradient>`
        - 'Steepest Descend'            :ref:`(see here) <optimizer.min_steepestdescent>`
        \\ To add
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

    EXAMPLES
    =========
    >>> try:
    ...     from variables import Variable
    ... except:
    ...     from AutoDiff.variables import Variable
    >>> try:
    ...     from Optimizer import minimize
    ... except:
    ...     from AutoDiff.Optimizer import minimize
    >>> a = Variable('a', 2)
    >>> myfunc = lambda x: x**2
    >>> res=minimize(myfunc,a)
    >>> res.x.val # Remeber to change this to res.x if we finally decides store x as numerical value!!!!!!!!
    0
    """
    if method == "Conjugate Gradient":
        result = min_conjugate_gradient(fun, x0, **kwargs)
    elif method == "Steepest Descend":
        result = min_steepestdescent(fun, x0, **kwargs)
    elif method =="BFGS":
        result = min_BFGS(fun,x0, **kwargs)
    return result



ddef min_conjugate_gradient(fn, x0, precision=1e-5, max_iter=10000, alpha_init=0, norm=np.inf):
    # create initial variables
    from scipy.optimize import minimize
    import time
    import numpy as np

    x = np.array(x0)

    # create initial variables
    var_names = ['x'+str(idx) for idx in range(len(x))]

    # initial gradient and steepest descent
    # recreate new variables with new values
    grad0 = _get_grad(fn, x, var_names)
    x = -grad0
    conj_direct = x

    nums_iteration = 0
    val_rec = []
    time_rec = []
    time_total = 0
    while True:
        start_time = time.time()
        argmin_fn = lambda alpha: fn(*[i + alpha*j for i, j in zip(x, conj_direct)])
        alpha = minimize(argmin_fn, alpha_init).x
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
        if np.linalg.norm(grad1, norm) < precision:
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
        nums_iteration += 1




def min_newton():
    pass

def min_steepestdescent(fn, x0, precision = PRECISION, max_iter = MAXITER):
     # create initial variables
    # right now we only test with the 26 alphabets
    from string import ascii_lowercase
    import time
    import numpy as np
    from scipy.optimize import fmin

    name_ls = iter(ascii_lowercase)

    # create initial variables
    var_names = []
    for i in x0:
        name = next(name_ls)
        var_names.append(name)

    x = np.array(x0)
    s = 0 # initialize as 0 works to ensure that s=g in 1st iteration

    nums_iteration = 0
    val_rec = []
    time_rec = []
    init_time = time.time()
     # initial guess of n = 0.01
    n = 0.01
    while True:
        # recreate new variables with new values
        x_var = []
        for i, v in enumerate(x):
            x_var.append(Variable(var_names[i], v))
        # obtain values and jacobian to find delta_f
        val_vector = np.array([value.val for value in x_var])
        jacobian = np.array([fn(x_var).der.get(i) for i in var_names])
        delta_f = jacobian*val_vector


        find_min = fmin(fn, val_vector-n*delta_f, maxiter = 1, disp=False)
        n = (find_min - x)/delta_f

        # update x
        old_x = x
        x = x + n*delta_f
        print(x)
        # threshold stopping condition
        if max(abs(x-old_x)) < precision:
            return Result(x, val_rec, time_rec, True)

        # store history of values
        val_rec.append(x)

        time_rec.append(time.time()-init_time)

        # iteration stopping condition
        if nums_iteration >= max_iter:
            return Result(x, val_rec    , time_rec, False)
        nums_iteration +=1

def _get_grad(fn, x, var_names):
    variables = [Variable(var_names[idx], x_n) for idx, x_n in enumerate(x)]
    out = fn(*variables)
    jacobian = out.jacobian()
    grad = np.array([jacobian[name] for name in var_names])
    return grad

def _line_search(fn, x, search_direction, grad, beta = 0.9, c = 0.9, alpha_init = 1):
    """approximately minimizes f along search_direction
    https://en.wikipedia.org/wiki/Backtracking_line_search
    """
    m = search_direction.T.dot(grad)
    alpha = alpha_init
    while (fn(*(x)) - fn(*(x+alpha*search_direction))) < -c*alpha*m:
        alpha = alpha * beta
    return alpha

def _update_hessian(approx_hessian, d_grad, step):
    return (approx_hessian
            + 1/(d_grad.T.dot(step))*d_grad.dot(d_grad.T)
            - 1/(step.T.dot(approx_hessian).dot(step))*(approx_hessian.dot(step).dot(step.T).dot(approx_hessian.T))
           )

def min_BFGS(fn, x0, precision = PRECISION, max_iter = MAXITER, beta = 0.9, c = 0.9, alpha_init = 1):
    time_rec = [time.time()]
    approx_hessian = np.identity(len(x0))
    x = np.array(x0).reshape(-1,1)
    var_names = ['x'+str(idx) for idx in range(len(x))]
    new_grad = _get_grad(fn, x, var_names)
    iter = 0
    val_rec = [x.flatten()]
    while np.linalg.norm(new_grad) > precision and iter < max_iter:
        # get new x values
        grad = new_grad
        search_direction = -np.linalg.pinv(approx_hessian).dot(grad)
        stepsize = _line_search(fn, x, search_direction, grad, beta = beta, c = c, alpha_init = alpha_init)
        step = stepsize * search_direction
        x = x + step
        val_rec.append(x.flatten())

        # update hessian approximation
        new_grad = _get_grad(fn, x, var_names)
        d_grad = new_grad - grad
        approx_hessian = _update_hessian(approx_hessian, d_grad, step)

        iter += 1
        time_rec.append(time.time())

    converge = (np.linalg.norm(new_grad) <= precision)
    return Result(x, np.array(val_rec), time_rec, converge)

def min_gradientdescent(fn, x0, precision = PRECISION, max_iter = MAXITER, lr=0.01):
     # create initial variables
    # right now we only test with the 26 alphabets
    from string import ascii_lowercase
    import time
    import numpy as np

    name_ls = iter(ascii_lowercase)

    # create initial variables
    var_names = []
    for i in x0:
        name = next(name_ls)
        var_names.append(name)

    x = np.array(x0)
    s = 0 # initialize as 0 works to ensure that s=g in 1st iteration

    nums_iteration = 0
    val_rec = []
    time_rec = []
    init_time = time.time()
     # initial guess of n = 0.01
    n = 0.01
    while True:
        # recreate new variables with new values
        x_var = []
        for i, v in enumerate(x):
            x_var.append(Variable(var_names[i], v))
        # obtain values and jacobian to find delta_f
        val_vector = np.array([value.val for value in x_var])
        jacobian = np.array([fn(*x_var).der.get(i) for i in var_names])
        delta_f = jacobian*val_vector


        # update x
        old_x = x
        x = x - lr*delta_f
        print(x)
        # threshold stopping condition
        if max(abs(x-old_x)) < precision:
            return Result(x, val_rec, time_rec, True)

        # store history of values
        val_rec.append(x)

        time_rec.append(time.time()-init_time)

        # iteration stopping condition
        if nums_iteration >= max_iter:
            return Result(x, val_rec    , time_rec, False)
        nums_iteration +=1


def findroot(fun, x0, method=None, **kwargs):
    """Find the roots of a function.
    Return the roots of the (non-linear) equations defined by
    ``func(x) = 0`` given a starting estimate.

    INPUTS
    =======
    fun: callable object. The opjective function to be solved.
    x0: variable inputs or normal value tuple. Initial guess.
    args: tuple (optional). Extra arguments passed to the opjective function.
    method: string (optional). Type of different optimizer. Should be one of

        - 'Newton Method'               :ref:`(see here) <optimizer.root_newton>`
        - 'BFGS'                        :ref:`(see here) <optimizer.root_BFGS>`
        - 'Stochastic Gradient Descend' :ref:`(see here) <optimizer.root_SGD>`
        - 'Gradient Descend'            :ref:`(see here) <optimizer.root_gradientdescend>`
        \\ To add
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

    EXAMPLES
    =========
    >>> try:
    ...     from variables import Variable
    ... except:
    ...     from AutoDiff.variables import Variable
    >>> try:
    ...     from Optimizer import findroot
    ... except:
    ...     from AutoDiff.Optimizer import findroot
    >>> a = Variable('a', 2)
    >>> myfunc = lambda x: x**2
    >>> res=findroot(myfunc,a)
    >>> res.x.val # Remeber to change this to res.x if we finally decides store x as numerical value!!!!!!!!
    0
    """
    pass


def root_secant_method(fun, x0, precision=PRECISION, max_iter=MAXITER):
    # choose initial guess of x0, and use finite difference to approximate the derivatives
    import time
    import numpy as np
    begin = time.time()
    time_arr = [0]
    val_arr = [x0]
    converge=False

    x1=x0-1 # randomly assigned
    i=0

    f_der_inv=lambda x1,x0:(x1-x0)/(fun(x1)-fun(x0))
    while True:

        i+=1
        x0,x1=x1,x1-fun(x1)*f_der_inv(x1,x0)
        time_arr.append(time.time()-begin)
        val_arr.append(x1)
        if abs(fun(x1)-fun(x0))<=precision:
            converge=True
            break
        if i>max_iter:
            converge=False
            break
    return Result(x1,np.array(val_arr),np.array(time_arr),converge)


def root_BFGS():
    pass


def root_gradientdescend():
    pass


def root_newton():
    pass


def root_SGD():
    pass
