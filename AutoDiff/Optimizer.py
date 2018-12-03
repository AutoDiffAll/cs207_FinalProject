try:
    from variables import Variable
except:
    from AutoDiff.variables import Variable


class Result:
    def __init__(self, x, step_rec, val_rec, time, true_value=None):
        """Record the optimization results and performance

        INPUTS
        =======
        x: optimization value, can be either Variable or value. //Or just store the value, since there is no need for its derivatives
        step_rec: stores the iteration steps, save for plotting the accuracy results.
        val_rec: stores the value of each iteration, save for plotting the optimization paths.
        time: stores the total time for converging.
        true_value: if it is not None, then we can calculate the accuracy rate of each iteration results, and plot the error rate plots.
        """
        self.x = x
        self.step_rec = step_rec
        self.val_rec = val_rec
        self.time = time
        self.acc_rec=None

        if true_value:
            self.acc_rec=self.val_rec-true_value
        


def minimize(fun, x0, args=(), method=None):
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
        - 'Gradient Descend'            :ref:`(see here) <optimizer.min_gradientdescend>`
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
    pass


def min_SGD():
    pass


def min_newton():
    pass


def min_gradientdescend():
    pass


def min_BFGS():
    pass


def findroot(fun, x0, args=(), method=None):
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


def root_BFGS():
    pass


def root_gradientdescend():
    pass


def root_newton():
    pass


def root_SGD():
    pass
