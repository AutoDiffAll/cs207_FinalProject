try:
    from vector_variables import vector_Variable
except:
    from AutoDiff.vector_variables import vector_Variable

def vectorize_variable(fn):
    """Given a vector function of variables, returns a function that
    wraps the original function to return a new vector_Variable class
    that can be used to extract the values and the jacobian of the vector
    easily

    INPUTS
    =======
    fn: predefined vector function that can take in any number of inputs
       Vector function that user wants to apply on variables

    RETURNS
    ========
    fn_wrapper: wrapped function that takes in any number of inputs

    EXAMPLES
    =========
    >>> try:
    ...     from variables import Variable
    ... except:
    ...     from AutoDiff.variables import Variable
    >>> try:
    ...     from vectorize_func import vectorize_variable
    ... except:
    ...     from AutoDiff.vectorize_func import vectorize_variable
    >>> try:
    ...     import AD_numpy as anp
    ... except:
    ...     import AutoDiff.AD_numpy as anp
    >>> @vectorize_variable
    >>> def vec_fn(x, y, z):
    ...     f1 = x * y + anp.sin(y) + anp.cos(z)
    ...     f2 = x + y + anp.sin(x*y)
    ...     return [f1,f2]
    >>> a = Variable('a', 3)
    >>> b = Variable('b', 1)
    >>> c = Variable('c', 2)
    >>> f = vec_fn(a, b, c)
    >>> f.jacobian().values
    array([[ 1.        ,  3.54030231, -0.90929743],
           [ 0.0100075 , -1.96997749,  0.        ]])
    >>> f.val
    array([3.42532415, 4.14112001])
    """
    def fn_wrapper(*args):
        variable_vec = fn(*args)

        return vector_Variable(variable_vec)
    return fn_wrapper
