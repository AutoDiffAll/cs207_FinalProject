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
    >>> from Variables import Variables
    >>> from AD_numpy import add
    >>> a = Variable(2)
    >>> b = Variable(3)
    >>> x = add(a,b)
    >>> x.val
    5
    >>> x.der
    [[1,0], [0,1]] # or something liddat
    """
    try:
        return Variable(np.add(x.val, y.val), np.nan) # wrong, needs to have derx dery
    except AttributeError:
        try:
            return Variable(np.add(x.val, y), x.der)
        except AttributeError:
            try:
                return Variable(np.add(x, y.val), y.der)
            except AttributeError:
                return np.add(x,y)

def negative(x):
    try:
        return Variable(np.negative(x.val), np.negative(x.der))
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
    >>> from Variables import Variables
    >>> from AD_numpy import multiply
    >>> a = Variable(2)
    >>> b = Variable(3)
    >>> x = multiply(a,b)
    >>> x.val
    6
    >>> x.der
    [[1,0], [0,1]] # or something liddat
    """
    try:
        return Variable(np.multiply(x.val, y.val), np.nan) # wrong, needs to have derx dery
    except AttributeError:
        try:
            return Variable(np.multiply(x.val, y), np.multiply(x.der, y))
        except AttributeError:
            try:
                return Variable(np.multiply(x, y.val), np.multiply(x, y.der))
            except AttributeError:
                return np.multiply(x, y)

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
    >>> from Variables import Variables
    >>> from AD_numpy import divide
    >>> a = Variable(6)
    >>> b = Variable(3)
    >>> x = divide(a,b)
    >>> x.val
    2
    >>> x.der
    [[1/3,0], [0,-2/3]] # or something liddat
    """
    try:
        return Variable(np.divide(x.val, y.val), np.nan) # wrong, needs to have derx dery
    except AttributeError:
        try:
            return Variable(np.divide(x.val, y), np.divide(x.der, y))
        except AttributeError:
            try:
                return Variable(np.divide(x, y.val), np.divide(x, y.der))
            except AttributeError:
                return np.divide(x, y)

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
    >>> from Variables import Variables
    >>> from AD_numpy import power
    >>> a = Variable(2)
    >>> b = Variable(3)
    >>> x = power(a,b)
    >>> x.val
    8
    >>> x.der
    [[1,0], [0,1]] # or something liddat
    """
    try:
        return Variable(np.power(x.val, y.val), np.nan) # wrong, needs to have derx dery
    except AttributeError:
        try:
            return Variable(np.power(x.val, y), y*x.der*np.power(x.val, y-1))
        except AttributeError:
            try:
                return Variable(np.power(x, y.val), np.log(x)*y.der*np.power(x, y.val))
            except AttributeError:
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
    >>> from Variables import Variables
    >>> from AD_numpy import subtract
    >>> a = Variable(2)
    >>> b = Variable(3)
    >>> x = subtract(a,b)
    >>> x.val
    -1
    >>> x.der
    [[1,0], [0,1]] # or something liddat
    """
    try:
        return Variable(np.subtract(x.val, y.val), np.nan) # wrong, needs to have derx dery
    except AttributeError:
        try:
            return Variable(np.subtract(x.val, y), x.der)
        except AttributeError:
            try:
                return Variable(np.subtract(x, y.val), y.der)
            except AttributeError:
                return np.subtract(x,y)

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
    >>> from Variables import Variables
    >>> from AD_numpy import sin
    >>> a = Variable(0)
    >>> x = sin(a)
    >>> x.val
    0.0
    >>> x.der
    1.0
    """
    try:
        return Variable(np.sin(x.val), x.der*np.cos(x.val))
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
    >>> from Variables import Variables
    >>> from AD_numpy import cos
    >>> a = Variable(0)
    >>> x = cos(a)
    >>> x.val
    1.0
    >>> x.der
    0.0
    """
    try:
        return Variable(np.cos(x.val), -x.der*np.sin(x.val))
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
    >>> from Variables import Variables
    >>> from AD_numpy import tan
    >>> a = Variable(0)
    >>> x = tan(a)
    >>> x.val
    0.0
    >>> x.der
    1.0
    """
    try:
        return Variable(np.tan(x.val), x.der/(np.cos(x.val))**2)
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

    POST:
         - x is not changed by this function
         - if x is Variable instance, returns a new Variable instance
         - if x is numeric, returns numeric
         - raises RuntimeWarning if x is not between -1 and 1
         - should return value between -pi/2 and pi/2

    EXAMPLES
    =========
    >>> from Variables import Variables
    >>> from AD_numpy import arcsin
    >>> a = Variable(0)
    >>> x = arcsin(a)
    >>> x.val
    0.0
    >>> x.der
    1.0
    """
    try:
        return Variable(np.arcsin(x.val), x.der/np.sqrt(1-x.val**2))
    except AttributeError:
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
    >>> from Variables import Variables
    >>> from AD_numpy import arccos
    >>> a = Variable(0)
    >>> x = arccos(a)
    >>> x.val
    1.5707963267948966
    >>> x.der
    -1.0
    """
    try:
        return Variable(np.arccos(x.val), -x.der/np.sqrt(1-x.val**2))
    except AttributeError:
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
    >>> from Variables import Variables
    >>> from AD_numpy import arctan
    >>> a = Variable(0)
    >>> x = arctan(a)
    >>> x.val
    0.0
    >>> x.der
    1.0
    """
    try:
        return Variable(np.arctan(x.val), x.der/(1+x.val**2))
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
    >>> from Variables import Variables
    >>> from AD_numpy import sinh
    >>> a = Variable(1)
    >>> x = sinh(a)
    >>> x.val
    1.1752011936438014
    >>> x.der
    1.5430806348152437
    """
    try:
        return Variable(np.sinh(x.val), x.der*np.cosh(x))
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
    >>> from Variables import Variables
    >>> from AD_numpy import cosh
    >>> a = Variable(1)
    >>> x = cosh(a)
    >>> x.val
    1.1752011936438014
    >>> x.der
    1.5430806348152437
    """
    try:
        return Variable(np.cosh(x.val), x.der*np.sinh(x))
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
    >>> from Variables import Variables
    >>> from AD_numpy import tanh
    >>> a = Variable(1)
    >>> x = tanh(a)
    >>> x.val
    0.76159415595576485
    >>> x.der
    0.41997434161402608
    """
    try:
        return Variable(np.tanh(x.val), x.der/(np.cosh(x.val))**2)
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
    >>> from Variables import Variables
    >>> from AD_numpy import arcsinh
    >>> a = Variable(1)
    >>> x = arcsinh(a)
    >>> x.val
    0.88137358701954305
    >>> x.der
    0.70710678118654746
    """
    try:
        return Variable(np.arcsinh(x.val), x.der/np.sqrt(1+x.val**2))
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
         - should return RuntimeWarning if x lesser than 1
         - should return ZeroDivisionErrorif if x equal 1
         - should return value greater than 0

    EXAMPLES
    =========
    >>> from Variables import Variables
    >>> from AD_numpy import arccosh
    >>> a = Variable(2)
    >>> x = arccosh(a)
    >>> x.val
    1.3169578969248166
    >>> x.der
    0.57735026918962584
    """
    try:
        return Variable(np.arccosh(x.val), x.der/np.sqrt(x.val**2 - 1))
    except AttributeError:
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
         - should return RuntimeWarning if x not within domain

    EXAMPLES
    =========
    >>> from Variables import Variables
    >>> from AD_numpy import arctanh
    >>> a = Variable(0)
    >>> x = arctanh(a)
    >>> x.val
    0.0
    >>> x.der
    1.0
    """
    try:
        return Variable(np.arctanh(x.val), x.der/(1 - x.val**2))
    except AttributeError:
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
    >>> from Variables import Variables
    >>> from AD_numpy import exp
    >>> a = Variable(0)
    >>> x = exp(a)
    >>> x.val
    1.0
    >>> x.der
    1.0
    """
    try:
        return Variable(np.exp(x.val), x.der*np.exp(x.val))
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
         - should raise RuntimeWarning if x not greater than 0

    EXAMPLES
    =========
    >>> from Variables import Variables
    >>> from AD_numpy import log
    >>> a = Variable(1)
    >>> x = log(a)
    >>> x.val
    0.0
    >>> x.der
    1.0
    """
    try:
        return Variable(np.log(x.val), x.der/x.val)
    except AttributeError:
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
    >>> from Variables import Variables
    >>> from AD_numpy import exp2
    >>> a = Variable(1)
    >>> x = exp2(a)
    >>> x.val
    2.0
    >>> x.der
    1.3862943611198906
    """
    try:
        return Variable(np.exp2(x.val), np.log(2)*x.der*np.exp2(x.val))
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
         - x should be greater than 0

    POST:
         - x is not changed by this function
         - if x is Variable instance, returns a new Variable instance
         - if x is numeric, returns numeric
         - should raise RuntimeWarning if x not greater than 0
         - should raise ZeroDivisionErrorif x equals 0

    EXAMPLES
    =========
    >>> from Variables import Variables
    >>> from AD_numpy import log10
    >>> a = Variable(1)
    >>> x = log10(a)
    >>> x.val
    0.0
    >>> x.der
    0.43429448190325182
    """
    try:
        return Variable(np.log10(x.val), x.der*np.log10(np.exp(1))/x.val)
    except AttributeError:
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
         - x should be greater than 0

    POST:
         - x is not changed by this function
         - if x is Variable instance, returns a new Variable instance
         - if x is numeric, returns numeric
         - should raise RuntimeWarning if x not greater than 0
         - should raise ZeroDivisionErrorif x equals 0

    EXAMPLES
    =========
    >>> from Variables import Variables
    >>> from AD_numpy import log2
    >>> a = Variable(1)
    >>> x = log2(a)
    >>> x.val
    0.0
    >>> x.der
    1.4426950408889634
    """
    try:
        return Variable(np.log2(x.val), x.der*np.log2(np.exp(1))/x.val)
    except AttributeError:
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
         - x should be greater than or equal to 0

    POST:
         - x is not changed by this function
         - if x is Variable instance, returns a new Variable instance
         - if x is numeric, returns numeric
         - should raise RuntimeWarning if x not greater than or equal to 0

    EXAMPLES
    =========
    >>> from Variables import Variables
    >>> from AD_numpy import sqrt
    >>> a = Variable(1)
    >>> x = sqrt(a)
    >>> x.val
    1.0
    >>> x.der
    0.5
    """
    try:
        return Variable(np.sqrt(x.val), 0.5*x.der/np.sqrt(x.val))
    except AttributeError:
        return np.sqrt(x)
