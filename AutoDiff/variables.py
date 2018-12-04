import numpy as np

class Variable(object):
    def __init__(self, name, val, der = None, primitive = True):
        self.val = val
        if primitive:
            self.name = name
            try:
                self.der = {name : np.ones(val.shape)}
            except AttributeError:
                self.der = {name : 1}
        else:
            self.name = name
            self.der = der

    def __repr__(self):
        """
        EXAMPLES
        =========
        >>> try:
        ...     from variables import Variable
        ... except:
        ...     from AutoDiff.variables import Variable
        >>> a = Variable('a', 2)
        >>> a
        Variable name: a, Value: 2, Derivatives: {'a': 1}
        """
        return ("Variable name:\n{}\nValue:\n{}\nDerivatives:\n{}"
                .format(self.name, self.val, self.der)
                )

    def partial_der(self, dep_var):
        """Returns partial derivative of a variable w.r.t. another variable.

        INPUTS
        =======
        dep_var: Variable

        RETURNS
        ========
        value: numeric, element-wise for lists, arrays, or similar structures

        NOTES
        =====
        POST:
            - if this variable is not a function of dep_var, return 0
            - if dep_var is not a variable, raise AttributeError

        EXAMPLES
        =========
        >>> try:
        ...     from variables import Variable
        ... except:
        ...     from AutoDiff.variables import Variable
        >>> a = Variable('a', 2)
        >>> x = 4*a
        >>> x.partial_der(a)
        4
        >>> b = 3
        >>> x.partial_der(b)
        input is not a Variable
        """
        try:
            return self.der.get(dep_var.name,0)
        except AttributeError:
            print("input is not a Variable")

    def jacobian(self):
        """Returns jacobian of variable


        RETURNS
        ========
        value: dict

        NOTES
        =====
        POST:
            - returns jacobian as a dictionary where keys are dependent variable

        EXAMPLES
        =========
        >>> try:
        ...     from variables import Variable
        ... except:
        ...     from AutoDiff.variables import Variable
        >>> import pprint
        >>> a = Variable('a', 2)
        >>> b = Variable('b', 3)
        >>> x = a*b
        >>> pprint.pprint(x.jacobian())
        {'a': 3, 'b': 2}
        """
        return self.der

    def __pos__(self):
        return Variable(self.name, self.val, der, False)

    def __neg__(self):
        neg = unary_user_function(lambda x: -x, lambda x: -1)
        return neg(self)

    def __add__(self, other):
        add = binary_user_function(lambda x,y: x+y, lambda x,y: 1, lambda x,y: 1)
        return add(self, other)

    __radd__ = __add__

    def __sub__(self, other):
        sub = binary_user_function(lambda x,y: x-y, lambda x,y: 1, lambda x,y: -1)
        return sub(self, other)

    def __rsub__(self, other):
        sub = binary_user_function(lambda x,y: x-y, lambda x,y: 1, lambda x,y: -1)
        return sub(other, self)

    def __mul__(self, other):
        mul = binary_user_function(lambda x,y: x*y, lambda x,y: y, lambda x,y: x)
        return mul(self, other)
    __rmul__ = __mul__

    def __truediv__(self, other):
        div = binary_user_function(lambda x,y: x/y, lambda x,y: 1/y, lambda x,y: -x/(y**2))
        return div(self, other)

    def __rtruediv__(self, other):
        div = binary_user_function(lambda x,y: x/y, lambda x,y: 1/y, lambda x,y: -x/(y**2))
        return div(other,self)

    def __pow__(self, other):
        pow = binary_user_function(lambda x,y: x**y, lambda x,y: y*(x**(y-1)), lambda x,y: x**y*np.log(x))
        return pow(self, other)

    def __rpow__(self, other):
        pow = binary_user_function(lambda x,y: x**y, lambda x,y: y*(x**(y-1)), lambda x,y: x**y*np.log(x))
        return pow(other,self)

def unary_user_function(fn, fn_der):
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
    >>> try:
    ...     from variables import Variable
    ... except:
    ...     from AutoDiff.variables import Variable
    >>> try:
    ...     from user_func import user_function
    ... except:
    ...     from AutoDiff.user_func import user_function
    >>> import numpy as np
    >>> sec = lambda x: 1/np.cos(x)
    >>> sec_der = lambda x: sec(x)*np.tan(x)
    >>> ad_sec = user_function(sec, sec_der)
    >>> a = Variable('a', 2)
    >>> x = ad_sec(a)
    >>> x.val
    -2.402997961722381
    >>> x.der
    {'a': 5.25064633769958}
    >>> ad_sec(2)
    -2.402997961722381
    """
    def AD_fn(x):
        try:
            name = 'f('+','.join(x.der.keys())+')'
            return Variable(name, fn(x.val), {k:v*fn_der(x.val) for (k,v) in x.der.items()}, False)
        except AttributeError:
            return fn(x)
    return AD_fn

def binary_user_function(fn, fn_der_x1, fn_der_x2):
    """Given a function and its derivative, returns an original function that
    can be applied to the variable class, keeping track of the actual value,
    and the derivative of the function to use in auto-differentiation

    INPUTS
    =======
    fn: function that takes in two inputs
       Mathematical function that user wants to apply on a variable
    fn_der_x1: function that takes in two inputs
       Mathematical function that is the derivative of fn with respect to x1
   fn_der_x2: function that takes in two inputs
      Mathematical function that is the derivative of fn with respect to x2

    RETURNS
    ========
    AD_fn: function that takes in two inputs
       Mathematical function that can be applied to the variable class

    NOTES
    =====
    PRE:
         - fn, fn_der_x1,  fn_der_x2 are function types with two inputs
         - fn is a mathematical function and fn_der is the mathematical
         derivative of the function
    POST:
         - fn and fn_der are not changed by this function
         - returns a function AD_fn that has a single input
         - AD_fn should work on numeric types as well as on variable class

    EXAMPLES
    =========
    >>> from variables import Variable
    >>> mult = binary_user_function(lambda x,y: x*y, lambda x,y: y, lambda x,y: x)
    >>> x = Variable('x', 3)
    >>> y = Variable('y', 2)
    >>> z = Variable('z', 4)
    >>> print(mult(mult(x,y),z))

    Variable name: f(y,x,z), Value: 24, Derivatives: {'y': 12, 'x': 8, 'z': 6}
    """
    def AD_fn(x1, x2):
        # get dep variables and variables
        x1_val, x2_val, der1, der2 = _unpack_vars(x1, x2)
        dep_vars = set(der1).union(der2)
        #calculate derivative
        der={dep_var:
            fn_der_x1(x1_val,x2_val) * der1.get(dep_var, 0) + fn_der_x2(x1_val,x2_val) * der2.get(dep_var, 0) # chain rule
            for dep_var in dep_vars
        }
        # get function name
        name = 'f('+','.join(der.keys())+')'
        # return new variable
        return Variable(name, fn(x1_val,x2_val), der, False)
    return AD_fn

def _unpack_vars(x1, x2):
    """gets the dependent variables and values of variables"""
    try:
        der1 = x1.der
        x1_val = x1.val
    except AttributeError:
        der1 = {}
        x1_val = x1
    try:
        der2 = x2.der
        x2_val = x2.val
    except AttributeError:
        der2 = {}
        x2_val = x2
    return x1_val, x2_val, der1, der2
