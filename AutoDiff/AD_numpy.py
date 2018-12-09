# look in numpy for functions
# functions should be able to handle inputs of Variable and regular python
# numbers
import numpy as np
try:
    from variables import Variable, unary_user_function
except:
    from AutoDiff.variables import Variable, unary_user_function

# arithmetic
def add(x, y):
    """Returns sum of two values x and y, can be used to sum Variable instances

    INPUTS
    =======
    x: numeric or Variable, element-wise for lists, arrays, or similar structures
       left element of sum
    y: numeric or Variable, element-wise for lists, arrays, or similar structures
       right element of sum

    RETURNS
    ========
    value: numeric or Variable, element-wise for lists, arrays, or similar structures

    NOTES
    =====
    PRE:
         - x and y are either numeric or Variable types

    POST:
         - x and y are not changed by this function
         - if either x and y are Variable instances,
         returns a new Variable instance
         - if both x and y are numeric, returns numeric

    EXAMPLES
    =========
    >>> try:
    ...     from variables import Variable
    ... except:
    ...     from AutoDiff.variables import Variable
    >>> try:
    ...     import AD_numpy as np
    ... except:
    ...     import AutoDiff.AD_numpy as np
    >>> import pprint
    >>> a = Variable('a', 2)
    >>> b = Variable('b', 3)
    >>> x = np.add(a,b)
    >>> x.val
    5
    >>> pprint.pprint(x.der)
    {'a': 1, 'b': 1}
    """
    return x+y

def negative(x):
    """Returns negative of x, can be used to calculate negative of
    Variable instance

    INPUTS
    =======
    x: numeric or Variable, element-wise for lists, arrays, or similar structures

    RETURNS
    ========
    value: numeric or Variable, element-wise for lists, arrays, or similar structures

    NOTES
    =====
    PRE:
         - x is either numeric or Variable types

    POST:
         - x is not changed by this function
         - if x is Variable instance, returns a new Variable instance
         - if x is numeric, returns numeric

    EXAMPLES
    =========
    >>> try:
    ...     from variables import Variable
    ... except:
    ...     from AutoDiff.variables import Variable
    >>> try:
    ...     import AD_numpy as np
    ... except:
    ...     import AutoDiff.AD_numpy as np
    >>> a = Variable('a', 0)
    >>> x = np.negative(a)
    >>> x.val
    0
    >>> x.der
    {'a': -1}
    """
    return -x

def multiply(x, y):
    """Returns product of two values x and y, can be used to multiply Variable instances

    INPUTS
    =======
    x: numeric or Variable, element-wise for lists, arrays, or similar structures
       left element of product
    y: numeric or Variable, element-wise for lists, arrays, or similar structures
       right element of product

    RETURNS
    ========
    value: numeric or Variable, element-wise for lists, arrays, or similar structures

    NOTES
    =====
    PRE:
         - x and y are either numeric or Variable types

    POST:
         - x and y are not changed by this function
         - if either x and y are Variable instances,
         returns a new Variable instance
         - if both x and y are numeric, returns numeric

    EXAMPLES
    =========
    >>> try:
    ...     from variables import Variable
    ... except:
    ...     from AutoDiff.variables import Variable
    >>> try:
    ...     import AD_numpy as np
    ... except:
    ...     import AutoDiff.AD_numpy as np
    >>> import pprint
    >>> a = Variable('a', 2)
    >>> b = Variable('b', 3)
    >>> x = np.multiply(a,b)
    >>> x.val
    6
    >>> pprint.pprint(x.der)
    {'a': 3, 'b': 2}
    """
    return x*y

def divide(x, y):
    """Returns division of two values x and y, can be used to divide Variable instances

    INPUTS
    =======
    x: numeric or Variable, element-wise for lists, arrays, or similar structures
       left element of divide
    y: numeric or Variable, element-wise for lists, arrays, or similar structures
       right element of divide

    RETURNS
    ========
    value: numeric or Variable, element-wise for lists, arrays, or similar structures

    NOTES
    =====
    PRE:
         - x and y are either numeric or Variable types

    POST:
         - x and y are not changed by this function
         - if either x and y are Variable instances,
         returns a new Variable instance
         - if both x and y are numeric, returns numeric

    EXAMPLES
    =========
    >>> try:
    ...     from variables import Variable
    ... except:
    ...     from AutoDiff.variables import Variable
    >>> try:
    ...     import AD_numpy as np
    ... except:
    ...     import AutoDiff.AD_numpy as np
    >>> import pprint
    >>> a = Variable('a', 6)
    >>> b = Variable('b', 2)
    >>> x = np.divide(a,b)
    >>> x.val
    3.0
    >>> pprint.pprint(x.der)
    {'a': 0.5, 'b': -1.5}
    """
    return x/y

def power(x, y):
    """Returns power of x to y, can be used to calculate power of Variable instances

    INPUTS
    =======
    x: numeric or Variable, element-wise for lists, arrays, or similar structures
       left element of product
    y: numeric or Variable, element-wise for lists, arrays, or similar structures
       right element of product

    RETURNS
    ========
    value: numeric or Variable, element-wise for lists, arrays, or similar structures

    NOTES
    =====
    PRE:
         - x and y are either numeric or Variable types

    POST:
         - x and y are not changed by this function
         - if either x and y are Variable instances,
         returns a new Variable instance
         - if both x and y are numeric, returns numeric

    EXAMPLES
    =========
    >>> try:
    ...     from variables import Variable
    ... except:
    ...     from AutoDiff.variables import Variable
    >>> try:
    ...     import AD_numpy as np
    ... except:
    ...     import AutoDiff.AD_numpy as np
    >>> import pprint
    >>> a = Variable('a', 2)
    >>> b = Variable('b', 3)
    >>> x = np.power(a,b)
    >>> x.val
    8
    >>> pprint.pprint(x.der)
    {'a': 12.0, 'b': 5.545177444479562}
    """
    return x**y

def subtract(x, y):
    """Returns difference of two values x and y, can be used to subtract Variable instances

    INPUTS
    =======
    x: numeric or Variable, element-wise for lists, arrays, or similar structures
       left element of subtract
    y: numeric or Variable, element-wise for lists, arrays, or similar structures
       right element of subtract

    RETURNS
    ========
    value: numeric or Variable, element-wise for lists, arrays, or similar structures

    NOTES
    =====
    PRE:
         - x and y are either numeric or Variable types

    POST:
         - x and y are not changed by this function
         - if either x and y are Variable instances,
         returns a new Variable instance
         - if both x and y are numeric, returns numeric

    EXAMPLES
    =========
    >>> try:
    ...     from variables import Variable
    ... except:
    ...     from AutoDiff.variables import Variable
    >>> try:
    ...     import AD_numpy as np
    ... except:
    ...     import AutoDiff.AD_numpy as np
    >>> import pprint
    >>> a = Variable('a', 2)
    >>> b = Variable('b', 3)
    >>> x = np.subtract(a,b)
    >>> x.val
    -1
    >>> pprint.pprint(x.der)
    {'a': 1, 'b': -1}
    """
    return x-y

# trigonometric functions
def sin(x):
    """Returns trigonometric sin of x, can be used to calculate sin of
    Variable instance

    INPUTS
    =======
    x: numeric or Variable, element-wise for lists, arrays, or similar structures

    RETURNS
    ========
    value: numeric or Variable, element-wise for lists, arrays, or similar structures

    NOTES
    =====
    PRE:
         - x is either numeric or Variable types

    POST:
         - x is not changed by this function
         - if x is Variable instance, returns a new Variable instance
         - if x is numeric, returns numeric
         - should return value between -1 and 1

    EXAMPLES
    =========
    >>> try:
    ...     from variables import Variable
    ... except:
    ...     from AutoDiff.variables import Variable
    >>> try:
    ...     import AD_numpy as np
    ... except:
    ...     import AutoDiff.AD_numpy as np
    >>> a = Variable('a', 0)
    >>> x = np.sin(a)
    >>> x.val
    0.0
    >>> x.der
    {'a': 1.0}
    >>> np.sin(0)
    0.0
    """
    return unary_user_function(lambda x: np.sin(x), lambda x: np.cos(x))(x)

def cos(x):
    """Returns trigonometric cos of x, can be used to calculate cos of
    Variable instance

    INPUTS
    =======
    x: numeric or Variable, element-wise for lists, arrays, or similar structures

    RETURNS
    ========
    value: numeric or Variable, element-wise for lists, arrays, or similar structures

    NOTES
    =====
    PRE:
         - x is either numeric or Variable types

    POST:
         - x is not changed by this function
         - if x is Variable instance, returns a new Variable instance
         - if x is numeric, returns numeric
         - should return value between -1 and 1

    EXAMPLES
    =========
    >>> try:
    ...     from variables import Variable
    ... except:
    ...     from AutoDiff.variables import Variable
    >>> try:
    ...     import AD_numpy as np
    ... except:
    ...     import AutoDiff.AD_numpy as np
    >>> a = Variable('a', 0)
    >>> x = np.cos(a)
    >>> x.val
    1.0
    >>> x.der
    {'a': -0.0}
    >>> np.cos(0)
    1.0
    """
    return unary_user_function(lambda x: np.cos(x), lambda x: -np.sin(x))(x)

def tan(x):
    """Returns trigonometric tan of x, can be used to calculate tan of
    Variable instance

    INPUTS
    =======
    x: numeric or Variable, element-wise for lists, arrays, or similar structures

    RETURNS
    ========
    value: numeric or Variable, element-wise for lists, arrays, or similar structures

    NOTES
    =====
    PRE:
         - x is either numeric or Variable types

    POST:
         - x is not changed by this function
         - if x is Variable instance, returns a new Variable instance
         - if x is numeric, returns numeric

    EXAMPLES
    =========
    >>> try:
    ...     from variables import Variable
    ... except:
    ...     from AutoDiff.variables import Variable
    >>> try:
    ...     import AD_numpy as np
    ... except:
    ...     import AutoDiff.AD_numpy as np
    >>> a = Variable('a', 0)
    >>> x = np.tan(a)
    >>> x.val
    0.0
    >>> x.der
    {'a': 1.0}
    >>> np.tan(0)
    0.0
    """
    return unary_user_function(lambda x: np.tan(x), lambda x: 1/np.cos(x)**2)(x)

def arcsin(x):
    """Returns trigonometric arcsin of x, can be used to calculate arcsin of
    Variable instance

    INPUTS
    =======
    x: numeric or Variable, element-wise for lists, arrays, or similar structures

    RETURNS
    ========
    value: numeric or Variable, element-wise for lists, arrays, or similar structures

    NOTES
    =====
    PRE:
         - x is either numeric or Variable types
         - x is between -1 and 1

    POST:
         - x is not changed by this function
         - if x is Variable instance, returns a new Variable instance
         - if x is numeric, returns numeric
         - raises RuntimeWarning if x is not between -1 and 1
         - should return value between -pi/2 and pi/2

    EXAMPLES
    =========
    >>> try:
    ...     from variables import Variable
    ... except:
    ...     from AutoDiff.variables import Variable
    >>> try:
    ...     import AD_numpy as np
    ... except:
    ...     import AutoDiff.AD_numpy as np
    >>> a = Variable('a', 0)
    >>> x = np.arcsin(a)
    >>> x.val
    0.0
    >>> x.der
    {'a': 1.0}
    >>> b = Variable('b', 2)
    >>> try:
    ...     np.arcsin(b)
    ... except ValueError as e:
    ...     print(e)
    math domain error
    >>> np.arcsin(0)
    0.0
    >>> try:
    ...     np.arcsin(2)
    ... except ValueError as e:
    ...     print(e)
    math domain error
    """
    _check_input(x, lower = -1, upper = 1)
    return  unary_user_function(lambda x: np.arcsin(x), lambda x: 1/np.sqrt(1-x**2))(x)

def arccos(x):
    """Returns trigonometric arccos of x, can be used to calculate arccos of
    Variable instance

    INPUTS
    =======
    x: numeric or Variable, element-wise for lists, arrays, or similar structures

    RETURNS
    ========
    value: numeric or Variable, element-wise for lists, arrays, or similar structures

    NOTES
    =====
    PRE:
         - x is either numeric or Variable types
         - x should be between -1 and 1

    POST:
         - x is not changed by this function
         - if x is Variable instance, returns a new Variable instance
         - if x is numeric, returns numeri
         - raises RuntimeWarning if x is not between -1 and 1
         - should return value between 0 and pi

    EXAMPLES
    =========
    >>> try:
    ...     from variables import Variable
    ... except:
    ...     from AutoDiff.variables import Variable
    >>> try:
    ...     import AD_numpy as np
    ... except:
    ...     import AutoDiff.AD_numpy as np
    >>> a = Variable('a', 0)
    >>> x = np.arccos(a)
    >>> x.val
    1.5707963267948966
    >>> x.der
    {'a': -1.0}
    >>> b = Variable('b', 2)
    >>> try:
    ...     np.arccos(b)
    ... except ValueError as e:
    ...     print(e)
    math domain error
    >>> np.arccos(1)
    0.0
    >>> try:
    ...     np.arccos(2)
    ... except ValueError as e:
    ...     print(e)
    math domain error
    """
    _check_input(x, lower = -1, upper = 1)
    return unary_user_function(lambda x: np.arccos(x), lambda x: -1/np.sqrt(1-x**2))(x)

def arctan(x):
    """Returns trigonometric arctan of x, can be used to calculate arctan of
    Variable instance

    INPUTS
    =======
    x: numeric or Variable, element-wise for lists, arrays, or similar structures

    RETURNS
    ========
    value: numeric or Variable, element-wise for lists, arrays, or similar structures

    NOTES
    =====
    PRE:
         - x is either numeric or Variable types

    POST:
         - x is not changed by this function
         - if x is Variable instance, returns a new Variable instance
         - if x is numeric, returns numeric
         - should return value between -pi/2 and pi/2

    EXAMPLES
    =========
    >>> try:
    ...     from variables import Variable
    ... except:
    ...     from AutoDiff.variables import Variable
    >>> try:
    ...     import AD_numpy as np
    ... except:
    ...     import AutoDiff.AD_numpy as np
    >>> a = Variable('a', 0)
    >>> x = np.arctan(a)
    >>> x.val
    0.0
    >>> x.der
    {'a': 1.0}
    >>> np.arctan(0)
    0.0
    """
    return unary_user_function(lambda x: np.arctan(x), lambda x: 1.0/(1+x**2))(x)

# hyperbolic functions
def sinh(x):
    """Returns hyberbolic sinh of x, can be used to calculate sinh of
    Variable instance

    INPUTS
    =======
    x: numeric or Variable, element-wise for lists, arrays, or similar structures

    RETURNS
    ========
    value: numeric or Variable, element-wise for lists, arrays, or similar structures

    NOTES
    =====
    PRE:
         - x is either numeric or Variable types

    POST:
         - x is not changed by this function
         - if x is Variable instance, returns a new Variable instance
         - if x is numeric, returns numeric

    EXAMPLES
    =========
    >>> try:
    ...     from variables import Variable
    ... except:
    ...     from AutoDiff.variables import Variable
    >>> try:
    ...     import AD_numpy as np
    ... except:
    ...     import AutoDiff.AD_numpy as np
    >>> a = Variable('a', 1)
    >>> x = np.sinh(a)
    >>> x.val
    1.1752011936438014
    >>> x.der
    {'a': 1.5430806348152437}
    >>> np.sinh(0)
    0.0
    """
    return unary_user_function(lambda x: np.sinh(x), lambda x: np.cosh(x))(x)

def cosh(x):
    """Returns hyberbolic cosh of x, can be used to calculate cosh of
    Variable instances

    INPUTS
    =======
    x: numeric or Variable, element-wise for lists, arrays, or similar structures

    RETURNS
    ========
    value: numeric or Variable, element-wise for lists, arrays, or similar structures

    NOTES
    =====
    PRE:
         - x is either numeric or Variable types

    POST:
         - x is not changed by this function
         - if x is Variable instance, returns a new Variable instance
         - if x is numeric, returns numeric
         - should return value greater than 1

    EXAMPLES
    =========
    >>> try:
    ...     from variables import Variable
    ... except:
    ...     from AutoDiff.variables import Variable
    >>> try:
    ...     import AD_numpy as np
    ... except:
    ...     import AutoDiff.AD_numpy as np
    >>> a = Variable('a', 1)
    >>> x = np.cosh(a)
    >>> x.val
    1.5430806348152437
    >>> x.der
    {'a': 1.1752011936438014}
    >>> np.cosh(0)
    1.0
    """
    return unary_user_function(lambda x: np.cosh(x), lambda x: np.sinh(x))(x)

def tanh(x):
    """Returns hyberbolic tanh of x, can be used to calculate tanh of
    Variable instance

    INPUTS
    =======
    x: numeric or Variable, element-wise for lists, arrays, or similar structures

    RETURNS
    ========
    value: numeric or Variable, element-wise for lists, arrays, or similar structures

    NOTES
    =====
    PRE:
         - x is either numeric or Variable types

    POST:
         - x is not changed by this function
         - if x is Variable instance, returns a new Variable instance
         - if x is numeric, returns numeric
         - should return value between -1 and 1

    EXAMPLES
    =========
    >>> try:
    ...     from variables import Variable
    ... except:
    ...     from AutoDiff.variables import Variable
    >>> try:
    ...     import AD_numpy as np
    ... except:
    ...     import AutoDiff.AD_numpy as np
    >>> a = Variable('a', 1)
    >>> x = np.tanh(a)
    >>> x.val
    0.7615941559557649
    >>> x.der
    {'a': 0.4199743416140261}
    >>> np.tanh(0)
    0.0
    """
    return unary_user_function(lambda x: np.tanh(x), lambda x: 1/np.cosh(x)**2)(x)

def arcsinh(x):
    """Returns hyberbolic inverse arcsinh of x, can be used to calculate
    arcsinh of Variable instance

    INPUTS
    =======
    x: numeric or Variable, element-wise for lists, arrays, or similar structures

    RETURNS
    ========
    value: numeric or Variable, element-wise for lists, arrays, or similar structures

    NOTES
    =====
    PRE:
         - x is either numeric or Variable types

    POST:
         - x is not changed by this function
         - if x is Variable instance, returns a new Variable instance
         - if x is numeric, returns numeric

    EXAMPLES
    =========
    >>> try:
    ...     from variables import Variable
    ... except:
    ...     from AutoDiff.variables import Variable
    >>> try:
    ...     import AD_numpy as np
    ... except:
    ...     import AutoDiff.AD_numpy as np
    >>> a = Variable('a', 1)
    >>> x = np.arcsinh(a)
    >>> x.val
    0.881373587019543
    >>> x.der
    {'a': 0.7071067811865475}
    >>> np.arcsinh(0)
    0.0
    """
    return unary_user_function(lambda x: np.arcsinh(x), lambda x: 1/np.sqrt(x**2+1))(x)

def arccosh(x):
    """Returns hyberbolic inverse arccosh of x, can be used to calculate
    arccosh of Variable instance

    INPUTS
    =======
    x: numeric or Variable, element-wise for lists, arrays, or similar structures

    RETURNS
    ========
    value: numeric or Variable, element-wise for lists, arrays, or similar structures

    NOTES
    =====
    PRE:
         - x is either numeric or Variable types
         - x must be greater than 1

    POST:
         - x is not changed by this function
         - if x is Variable instance, returns a new Variable instance
         - if x is numeric, returns numeric
         - should return ValueError if x lesser than 1
         - should return value greater than 0

    EXAMPLES
    =========
    >>> try:
    ...     from variables import Variable
    ... except:
    ...     from AutoDiff.variables import Variable
    >>> try:
    ...     import AD_numpy as np
    ... except:
    ...     import AutoDiff.AD_numpy as np
    >>> a = Variable('a', 2)
    >>> x = np.arccosh(a)
    >>> x.val
    1.3169578969248166
    >>> x.der
    {'a': 0.5773502691896258}
    >>> b = Variable('b', 0)
    >>> try:
    ...     np.arccosh(b)
    ... except ValueError as e:
    ...     print(e)
    math domain error
    >>> np.arccosh(1)
    0.0
    >>> try:
    ...     np.arccosh(0)
    ... except ValueError as e:
    ...     print(e)
    math domain error
    """
    _check_input(x, lower = 1)
    return unary_user_function(lambda x: np.arccosh(x), lambda x: 1/np.sqrt(x**2-1))(x)

def arctanh(x):
    """Returns hyberbolic inverse arccosh of x, can be used to calculate
    arccosh of Variable instance

    INPUTS
    =======
    x: numeric or Variable, element-wise for lists, arrays, or similar structures

    RETURNS
    ========
    value: numeric or Variable, element-wise for lists, arrays, or similar structures

    NOTES
    =====
    PRE:
         - x is either numeric or Variable types
         - x must be between -1 and 1

    POST:
         - x is not changed by this function
         - if x is Variable instance, returns a new Variable instance
         - if x is numeric, returns numeric
         - should return ValueError if x not within domain

    EXAMPLES
    =========
    >>> try:
    ...     from variables import Variable
    ... except:
    ...     from AutoDiff.variables import Variable
    >>> try:
    ...     import AD_numpy as np
    ... except:
    ...     import AutoDiff.AD_numpy as np
    >>> a = Variable('a', 0)
    >>> x = np.arctanh(a)
    >>> x.val
    0.0
    >>> x.der
    {'a': 1.0}
    >>> b = Variable('b', 2)
    >>> try:
    ...     np.arctanh(b)
    ... except ValueError as e:
    ...     print(e)
    math domain error
    >>> np.arctanh(0)
    0.0
    >>> try:
    ...     np.arctanh(2)
    ... except ValueError as e:
    ...     print(e)
    math domain error
    """
    _check_input(x, lower = -1, upper = 1)
    return unary_user_function(lambda x: np.arctanh(x), lambda x: 1.0/(1-x**2))(x)

# exponentials and logarithms
def exp(x):
    """Returns exponential of x, can be used to calculate
    exponential of Variable instance

    INPUTS
    =======
    x: numeric or Variable, element-wise for lists, arrays, or similar structures

    RETURNS
    ========
    value: numeric or Variable, element-wise for lists, arrays, or similar structures

    NOTES
    =====
    PRE:
         - x is either numeric or Variable types

    POST:
         - x is not changed by this function
         - if x is Variable instance, returns a new Variable instance
         - if x is numeric, returns numeric
         - should return value greater than 0

    EXAMPLES
    =========
    >>> try:
    ...     from variables import Variable
    ... except:
    ...     from AutoDiff.variables import Variable
    >>> try:
    ...     import AD_numpy as np
    ... except:
    ...     import AutoDiff.AD_numpy as np
    >>> a = Variable('a', 0)
    >>> x = np.exp(a)
    >>> x.val
    1.0
    >>> x.der
    {'a': 1.0}
    >>> np.exp(0)
    1.0
    """
    return unary_user_function(lambda x: np.exp(x), lambda x: np.exp(x))(x)

def log(x, base=np.exp(1)):
    """Returns logarithm of x with any base (defaults to natural logarithm),
    can be used to calculate natural logarithm of Variable instance

    INPUTS
    =======
    x: numeric or Variable, element-wise for lists, arrays, or similar structures
    base: numeric

    RETURNS
    ========
    value: numeric or Variable, element-wise for lists, arrays, or similar structures

    NOTES
    =====
    PRE:
         - x is either numeric or Variable types
         - x should be greater than 0
         - base should a numeric type

    POST:
         - x is not changed by this function
         - if x is Variable instance, returns a new Variable instance
         - if x is numeric, returns numeric
         - should raise ValueError if x less than 0

    EXAMPLES
    =========
    >>> try:
    ...     from variables import Variable
    ... except:
    ...     from AutoDiff.variables import Variable
    >>> try:
    ...     import AD_numpy as np
    ... except:
    ...     import AutoDiff.AD_numpy as np
    >>> a = Variable('a', 1)
    >>> x = np.log(a)
    >>> x.val
    0.0
    >>> x.der
    {'a': 1.0}
    >>> b = Variable('b', -1)
    >>> try:
    ...     np.log(b)
    ... except ValueError as e:
    ...     print(e)
    math domain error
    >>> np.log(1)
    0.0
    >>> try:
    ...     np.log(-1)
    ... except ValueError as e:
    ...     print(e)
    math domain error
    >>> try:
    ...     np.log(-1, base='a')
    ... except ValueError as e:
    ...     print(e)
    Base must be numeric!!
    """
    _check_input(x, lower = 0)
    if not isinstance(base, (numeric, float)):
        raise ValueError('Base must be numeric!!')
    return unary_user_function(lambda x: np.log(x)/np.log(base), lambda x: 1/(x*np.log(base)))(x)

def exp2(x):
    """Returns 2 to the power of x, can be used to calculate
    2 to the power of Variable instance

    INPUTS
    =======
    x: numeric or Variable, element-wise for lists, arrays, or similar structures

    RETURNS
    ========
    value: numeric or Variable, element-wise for lists, arrays, or similar structures

    NOTES
    =====
    PRE:
         - x is either numeric or Variable types

    POST:
         - x is not changed by this function
         - if x is Variable instance, returns a new Variable instance
         - if x is numeric, returns numeric

    EXAMPLES
    =========
    >>> try:
    ...     from variables import Variable
    ... except:
    ...     from AutoDiff.variables import Variable
    >>> try:
    ...     import AD_numpy as np
    ... except:
    ...     import AutoDiff.AD_numpy as np
    >>> a = Variable('a', 1)
    >>> x = np.exp2(a)
    >>> x.val
    2.0
    >>> x.der
    {'a': 1.3862943611198906}
    >>> np.exp2(0)
    1.0
    """
    return 2.0**x

def log10(x):
    """Returns logarithm to the base 10 of x, can be used to calculate
    logarithm to the base 10 of Variable instance

    INPUTS
    =======
    x: numeric or Variable, element-wise for lists, arrays, or similar structures

    RETURNS
    ========
    value: numeric or Variable, element-wise for lists, arrays, or similar structures

    NOTES
    =====
    PRE:
         - x is either numeric or Variable types
         - x should be greater than 0

    POST:
         - x is not changed by this function
         - if x is Variable instance, returns a new Variable instance
         - if x is numeric, returns numeric
         - should raise ValueError if x less than 0
         - should raise ZeroDivisionError if x equals 0

    EXAMPLES
    =========
    >>> try:
    ...     from variables import Variable
    ... except:
    ...     from AutoDiff.variables import Variable
    >>> try:
    ...     import AD_numpy as np
    ... except:
    ...     import AutoDiff.AD_numpy as np
    >>> a = Variable('a', 1)
    >>> x = np.log10(a)
    >>> x.val
    0.0
    >>> x.der
    {'a': 0.43429448190325176}
    >>> b = Variable('b', -1)
    >>> try:
    ...     np.log10(b)
    ... except ValueError as e:
    ...     print(e)
    math domain error
    >>> np.log10(10)
    1.0
    >>> try:
    ...     np.log10(-1)
    ... except ValueError as e:
    ...     print(e)
    math domain error
    """
    _check_input(x, lower = 0)
    return unary_user_function(lambda x: np.log10(x), lambda x: 1/(x*np.log(10)))(x)

def log2(x):
    """Returns logarithm to the base 2 of x, can be used to calculate
    logarithm to the base 2 of Variable instance

    INPUTS
    =======
    x: numeric or Variable, element-wise for lists, arrays, or similar structures

    RETURNS
    ========
    value: numeric or Variable, element-wise for lists, arrays, or similar structures

    NOTES
    =====
    PRE:
         - x is either numeric or Variable types
         - x should be greater than 0

    POST:
         - x is not changed by this function
         - if x is Variable instance, returns a new Variable instance
         - if x is numeric, returns numeric
         - should raise ValueError if x less than 0
         - should raise ZeroDivisionErrorif x equals 0

    EXAMPLES
    =========
    >>> try:
    ...     from variables import Variable
    ... except:
    ...     from AutoDiff.variables import Variable
    >>> try:
    ...     import AD_numpy as np
    ... except:
    ...     import AutoDiff.AD_numpy as np
    >>> a = Variable('a', 1)
    >>> x = np.log2(a)
    >>> x.val
    0.0
    >>> x.der
    {'a': 1.4426950408889634}
    >>> b = Variable('b', -1)
    >>> try:
    ...     np.log2(b)
    ... except ValueError as e:
    ...     print(e)
    math domain error
    >>> np.log2(2)
    1.0
    >>> try:
    ...     np.log2(-1)
    ... except ValueError as e:
    ...     print(e)
    math domain error
    """
    _check_input(x, lower = 0)
    return unary_user_function(lambda x: np.log2(x), lambda x: 1/(x*np.log(2)))(x)

# miscellaneous
def sqrt(x):
    """Returns square root of x, can be used to calculate
    square root of Variable instance

    INPUTS
    =======
    x: numeric or Variable, element-wise for lists, arrays, or similar structures

    RETURNS
    ========
    value: numeric or Variable, element-wise for lists, arrays, or similar structures

    NOTES
    =====
    PRE:
         - x is either numeric or Variable types
         - x should be greater than or equal to 0

    POST:
         - x is not changed by this function
         - if x is Variable instance, returns a new Variable instance
         - if x is numeric, returns numeric
         - should raise ValueError if x is less than 0

    EXAMPLES
    =========
    >>> try:
    ...     from variables import Variable
    ... except:
    ...     from AutoDiff.variables import Variable
    >>> try:
    ...     import AD_numpy as np
    ... except:
    ...     import AutoDiff.AD_numpy as np
    >>> a = Variable('a', 1)
    >>> x = np.sqrt(a)
    >>> x.val
    1.0
    >>> x.der
    {'a': 0.5}
    >>> b = Variable('b', -1)
    >>> try:
    ...     np.sqrt(b)
    ... except ValueError as e:
    ...     print(e)
    math domain error
    >>> np.sqrt(4)
    2.0
    >>> try:
    ...     np.sqrt(-1)
    ... except ValueError as e:
    ...     print(e)
    math domain error
    """
    _check_input(x, lower = 0, lower_inclusive = True)
    return unary_user_function(lambda x: np.sqrt(x), lambda x: 1/(2*sqrt(x)))(x)

def logistic(x):
    """Returns square root of x, can be used to calculate
    square root of Variable instance

    INPUTS
    =======
    x: numeric or Variable, element-wise for lists, arrays, or similar structures

    RETURNS
    ========
    value: numeric or Variable, element-wise for lists, arrays, or similar structures

    NOTES
    =====
    PRE:
         - x is either numeric or Variable types
         - x should be greater than or equal to 0

    POST:
         - x is not changed by this function
         - if x is Variable instance, returns a new Variable instance
         - if x is numeric, returns numeric
         - should return value between 0 and 1

    EXAMPLES
    =========
    >>> try:
    ...     from variables import Variable
    ... except:
    ...     from AutoDiff.variables import Variable
    >>> try:
    ...     import AD_numpy as np
    ... except:
    ...     import AutoDiff.AD_numpy as np
    >>> a = Variable('a', 0)
    >>> x = np.logistic(a)
    >>> x.val
    0.5
    >>> x.der
    {'a': 0.25}
    """
    return unary_user_function(lambda x: 1/(1+np.exp(-x)), lambda x: np.exp(x)/(1+np.exp(x))**2)(x)

def _check_input(x, lower = None, upper = None, lower_inclusive = False, upper_inclusive = False):
    try:
        val = x.val
    except AttributeError:
        val = x
    if lower is not None:
        if lower_inclusive:
            if val <= lower:
                raise ValueError('math domain error')
        else:
            if val < lower:
                raise ValueError('math domain error')
    if upper is not None:
        if upper_inclusive:
            if val >= upper:
                raise ValueError('math domain error')
        else:
            if val > upper:
                raise ValueError('math domain error')


if __name__ == "__main__":
    import doctest
    doctest.testmod()
