from variables import variable

def user_function(fn, fn_der):
    """Given a function and its derivative, returns an original function that
    can be applied to the variable class, keeping track of the actual value,
    and the derivative of the function to use in auto-differentiation

    INPUTS
    =======
    fn: function that takes in one input
       Mathematical function that user wants to apply on a variable
    fn_der: function that takes in one input
       Mathematical function that is the derivative of fn

    RETURNS
    ========
    AD_fn: function that takes in one input
       Mathematical function that can be applied to the variable class

    NOTES
    =====
    PRE:
         - fn and fn_der are function types with single inputs
         - fn is a mathematical function and fn_der is the mathematical
         derivative of the function
    POST:
         - fn and fn_der are not changed by this function
         - returns a function AD_fn that has a single input
         - AD_fn should work on numeric types as well as on variable class

    EXAMPLES
    =========
    >>> from variables import Variable
    >>> import numpy as np
    >>> sec = lambda x: 1/np.cos(x)
    >>> sec_der = lambda x: sec(x)*np.tan(x)
    >>> ad_sec = user_function(sec, sec_der)
    >>> a = Variable(2)
    >>> x = ad_sec(a)
    >>> x.val
    -2.4029979617223809
    >>> x.der
    5.25064633769958
    """
    def AD_fn(x):
        try:
            return variable(fn(x.val), {k:v*fn_der(x.val) for (k,v) in x.der.items()})
        except AttributeError:
            return fn(x)
    return AD_fn
