# look in numpy for functions
# functions should be able to handle inputs of Variable and regular python
# numbers
import numpy as np
from variables import Variable

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
    >>> from variables import Variable
    >>> import AD_numpy as np
    >>> a = Variable('a', 2)
    >>> b = Variable('b', 3)
    >>> x = np.add(a,b)
    >>> x.val
    5.0
    >>> x.der
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
    >>> from variables import Variable
    >>> import AD_numpy as np
    >>> a = Variable('a', 0)
    >>> x = np.negative(a)
    >>> x.val
    0.0
    >>> x.der
    -1.0
    """
    try:
        return Variable(x.name, np.negative(x.val), {k:np.negative(v) for (k,v) in x.der.items()}, False)
    except AttributeError:
        return np.negative(x)

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
    >>> from variables import Variable
    >>> import AD_numpy as np
    >>> a = Variable('a', 2)
    >>> b = Variable('b', 3)
    >>> x = np.multiply(a,b)
    >>> x.val
    6.0
    >>> x.der
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
    >>> from variables import Variable
    >>> import AD_numpy as np
    >>> a = Variable('a', 6)
    >>> b = Variable('b', 2)
    >>> x = np.divide(a,b)
    >>> x.val
    3.0
    >>> x.der
    {'b': -1.5, 'a': 0.5}
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
    >>> from variables import Variable
    >>> import AD_numpy as np
    >>> a = Variable('a', 2)
    >>> b = Variable('b', 3)
    >>> x = np.power(a,b)
    >>> x.val
    8.0
    >>> x.der
    {'a': 12.0, 'b': 5.545177444479562}
    """
    try:
        # calculate new derivative
        new_der = {k: x.der.get(k, 0)*y.val*np.power(x.val, y.val-1) + y.der.get(k, 0)*np.log(x.val)*np.power(x.val, y.val) for k in set(x.der).union(y.der)}

        new_name = "f({},{})".format(x.name, y.name)
        new_val = np.power(x.val, y.val)

        return Variable(new_name, new_val, new_der, False)
    except AttributeError:
        # only x is variable
        if isinstance(x, Variable):
            return Variable(x.name, np.power(x.val, y), {k:v*y*np.power(x.val, y-1) for (k,v) in x.der.items()}, False)
        # only y is variable
        elif isinstance(y, Variable):
            return Variable(y.name, {k:v*np.log(x)*np.power(x, y.val) for (k,v) in y.der.items()})
        # both not variable
        else:
            return np.power(x, y)

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
    >>> from variables import Variable
    >>> import AD_numpy as np
    >>> a = Variable('a', 2)
    >>> b = Variable('b', 3)
    >>> x = np.subtract(a,b)
    >>> x.val
    -1.0
    >>> x.der
    {'a': 1, 'b': 1}
    """
    return x+(-y)

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
    >>> from variables import Variable
    >>> import AD_numpy as np
    >>> a = Variable('a', 0)
    >>> x = np.sin(a)
    >>> x.val
    0.0
    >>> x.der
    {'a': 1.0}
    """
    try:
        return Variable(x.name, np.sin(x.val), {k:v*np.cos(x.val) for (k,v) in x.der.items()}, False)
    except AttributeError:
        return np.sin(x)

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
    >>> from variables import Variable
    >>> import AD_numpy as np
    >>> a = Variable('a', 0)
    >>> x = np.cos(a)
    >>> x.val
    1.0
    >>> x.der
    {'a': 0.0}
    """
    try:
        return Variable(x.name, np.cos(x.val), {k:-v*np.sin(x.val) for (k,v) in x.der.items()}, False)
    except AttributeError:
        return np.cos(x)

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
    >>> from variables import Variable
    >>> import AD_numpy as np
    >>> a = Variable('a', 0)
    >>> x = np.tan(a)
    >>> x.val
    0.0
    >>> x.der
    {'a': 1.0}
    """
    try:
        return Variable(x.name, np.tan(x.val), {k:v/(np.cos(x.val)**2) for (k,v) in x.der.items()}, False)
    except AttributeError:
        return np.tan(x)

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
    >>> from variables import Variable
    >>> import AD_numpy as np
    >>> a = Variable('a', 0)
    >>> x = np.arcsin(a)
    >>> x.val
    0.0
    >>> x.der
    {'a': 1.0}
    """

    try:
        if x.val < -1 or x.val > 1:
            raise ValueError('math domain error')
        return Variable(x.name, np.arcsin(x.val), {k:v/np.sqrt(1-x.val**2) for (k,v) in x.der.items()}, False)
    except AttributeError:
        if x < -1 or x > 1:
            raise ValueError('math domain error')
        return np.arcsin(x)

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
    >>> from variables import Variable
    >>> import AD_numpy as np
    >>> a = Variable('a', 0)
    >>> x = np.arccos(a)
    >>> x.val
    1.5707963267948966
    >>> x.der
    {'a': -1.0}
    """
    try:
        if x.val < -1 or x.val > 1:
            raise ValueError('math domain error')
        return Variable(x.name, np.arccos(x.val), {k:-v/np.sqrt(1-x.val**2) for (k,v) in x.der.items()}, False)
    except AttributeError:
        if x < -1 or x > 1:
            raise ValueError('math domain error')
        return np.arccos(x)

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
    >>> from variables import Variable
    >>> import AD_numpy as np
    >>> a = Variable('a', 0)
    >>> x = np.arctan(a)
    >>> x.val
    0.0
    >>> x.der
    {'a': 1.0}
    """
    try:
        return Variable(x.name, np.arctan(x.val), {k:v/(1+x.val**2) for (k,v) in x.der.items()}, False)
    except AttributeError:
        return np.arctan(x)

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
    >>> from variables import Variable
    >>> import AD_numpy as np
    >>> a = Variable(1)
    >>> x = np.sinh(a)
    >>> x.val
    1.1752011936438014
    >>> x.der
    {'a': 1.5430806348152437}
    """
    try:
        return Variable(x.name, np.sinh(x.val), {k:v*np.cosh(x.val) for (k,v) in x.der.items()}, False)
    except AttributeError:
        return np.sinh(x)

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
    >>> from variables import Variable
    >>> import AD_numpy as np
    >>> a = Variable('a', 1)
    >>> x = np.cosh(a)
    >>> x.val
    1.1752011936438014
    >>> x.der
    {'a': 1.5430806348152437}
    """
    try:
        return Variable(x.name, np.cosh(x.val), {k:v*np.sinh(x.val) for (k,v) in x.der.items()}, False)
    except AttributeError:
        return np.cosh(x)

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
    >>> from variables import Variable
    >>> import AD_numpy as np
    >>> a = Variable('a', 1)
    >>> x = np.tanh(a)
    >>> x.val
    0.76159415595576485
    >>> x.der
    {'a': 0.41997434161402608}
    """
    try:
        return Variable(x.name, np.tanh(x.val), {k:v/(np.cosh(x.val)**2) for (k,v) in x.der.items()}, False)
    except AttributeError:
        return np.tanh(x)

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
    >>> from variables import Variable
    >>> import AD_numpy as np
    >>> a = Variable('a', 1)
    >>> x = np.arcsinh(a)
    >>> x.val
    0.88137358701954305
    >>> x.der
    {'a': 0.70710678118654746}
    """
    try:
        return Variable(x.name, np.arcsinh(x.val), {k:v/np.sqrt(1+x.val**2) for (k,v) in x.der.items()}, False)
    except AttributeError:
        return np.arcsinh(x)

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
    >>> from variables import Variable
    >>> import AD_numpy as np
    >>> a = Variable('a', 2)
    >>> x = np.arccosh(a)
    >>> x.val
    1.3169578969248166
    >>> x.der
    {'a': 0.57735026918962584}
    """
    if x.val <= 1:
        raise ValueError('math domain error')
    try:
        return Variable(x.name, np.arccosh(x.val), {k:v/np.sqrt(x.val**2 - 1) for (k,v) in x.der.items()}, False)
    except AttributeError:
        if x < 1:
            raise ValueError('math domain error')
        return np.arccosh(x)

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
    >>> from variables import Variable
    >>> import AD_numpy as np
    >>> a = Variable('a', 0)
    >>> x = np.arctanh(a)
    >>> x.val
    0.0
    >>> x.der
    {'a': 1.0}
    """
    try:
        if x.val <= -1 or x.val >= 1:
            raise ValueError('math domain error')
        return Variable(x.name, np.arctanh(x.val), {k:v/(1 - x.val**2) for (k,v) in x.der.items()}, False)
    except AttributeError:
        if x <= -1 or x >= 1:
            raise ValueError('math domain error')
        return np.arctanh(x)

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
    >>> from variables import Variable
    >>> import AD_numpy as np
    >>> a = Variable('a', 0)
    >>> x = np.exp(a)
    >>> x.val
    1.0
    >>> x.der
    {'a': 1.0}
    """
    try:
        return Variable(x.name, np.exp(x.val), {k:v*np.exp(x.val) for (k,v) in x.der.items()}, False)
    except AttributeError:
        return np.exp(x)

def log(x):
    """Returns natural logarithm of x, can be used to calculate
    natural logarithm of Variable instance

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

    EXAMPLES
    =========
    >>> from variables import Variable
    >>> import AD_numpy as np
    >>> a = Variable('a', 1)
    >>> x = np.log(a)
    >>> x.val
    0.0
    >>> x.der
    {'a': 1.0}
    """
    try:
        if x.val <= 0:
            raise ValueError('math domain error')
        return Variable(x.name, np.log(x.val), {k:v/x.val for (k,v) in x.der.items()}, False)
    except AttributeError:
        if x <= 0:
            raise ValueError('math domain error')
        return np.log(x)

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
    >>> from variables import Variable
    >>> import AD_numpy as np
    >>> a = Variable('a', 1)
    >>> x = np.exp2(a)
    >>> x.val
    2.0
    >>> x.der
    {'a': 1.3862943611198906}
    """
    try:
        return Variable(x.name, np.exp2(x.val), {k:v*np.log(2)*np.exp2(x.val) for (k,v) in x.der.items()}, False)
    except AttributeError:
        return np.exp2(x)

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
    >>> from variables import Variable
    >>> import AD_numpy as np
    >>> a = Variable('a', 1)
    >>> x = np.log10(a)
    >>> x.val
    0.0
    >>> x.der
    {'a': 0.43429448190325182}
    """
    try:
        if x.val <= 0:
            raise ValueError('math domain error')
        return Variable(x.name, np.log10(x.val), {k:v*np.log10(np.exp(1))/x.val for (k,v) in x.der.items()}, False)
    except AttributeError:
        if x <= 0:
            raise ValueError('math domain error')
        return np.log10(x)

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
    >>> from variables import Variable
    >>> import AD_numpy as np
    >>> a = Variable('a', 1)
    >>> x = np.log2(a)
    >>> x.val
    0.0
    >>> x.der
    {'a': 1.4426950408889634}
    """
    try:
        if x.val <= 0:
            raise ValueError('math domain error')
        return Variable(x.name, np.log2(x.val), {k:v*np.log2(np.exp(1))/x.val for (k,v) in x.der.items()}, False)
    except AttributeError:
        if x <= 0:
            raise ValueError('math domain error')
        return np.log2(x)

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
    >>> from variables import Variable
    >>> import AD_numpy as np
    >>> a = Variable('a', 1)
    >>> x = np.sqrt(a)
    >>> x.val
    1.0
    >>> x.der
    {'a': 0.5}
    """
    try:
        if x.val < 0:
            raise ValueError('math domain error')
        return Variable(x.name, np.sqrt(x.val), {k:v*0.5/np.sqrt(x.val) for (k,v) in x.der.items()}, False)
    except AttributeError:
        if x < 0:
            raise ValueError('math domain error')
        return np.sqrt(x)
