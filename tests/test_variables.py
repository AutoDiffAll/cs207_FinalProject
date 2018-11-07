
import numpy as np
import pytest
import sys, os
sys.path.append('../AutoDiff')
from variables import Variable
import numpy as anp


def test_variable_scalar_add_minus():
    '''
    object.__add__(), object.__sub__(), object.__pos__(), object.__neg__(),
    object.__iadd__(), object.__isub__()
    '''
    # case 0: build variables
    x = Variable('x',2)
    y = Variable('y',-5)

    # case 1: right add
    f1 = x+2    
    assert (f1.val == 4)
    assert (f1.grad() == {'x':1})
    assert (f1.partial_der(x) == 1)
    #assert (f1.partial_der(y) == 0)

    # case 2: left add
    f2 = 10+2*8+y
    assert (f2.val == 21)
    assert (f2.grad() == {'y':1})
    #assert (f2.partial_der(x) == 0)
    assert (f2.partial_der(y) == 1)

    # case 3: variable+variable
    f3 = 3+x+y-2
    assert (f3.val == -2)
    assert (f3.grad() == {'x':1, 'y':1})
    assert (f3.partial_der(x) == 1)
    assert (f3.partial_der(y) == 1)

    # case 4: right minus
    f1 = x-2
    assert (f1.val == 0)
    assert (f1.grad() == {'x':1})
    assert (f1.partial_der(x) == 1)
    #assert (f1.partial_der(y) == 0)

    # case 5: left minus
    f2 = 10+2*8-y
    assert (f2.val == 31)
    assert (f2.grad() == {'y':-1})
    assert (f2.partial_der(x) == 0)
    assert (f2.partial_der(y) == -1)

    # case 6: variable+variable
    f3 = 3-x-y-2
    assert (f3.val == 2)
    assert (f3.grad() == [-1, -1])
    assert (f3.partial_der(x) == -1)
    assert (f3.partial_der(y) == -1)

    # case 7: positive
    f4 = +y
    assert (f4.val == -5)
    assert (f4.grad() == [1])
    assert (f4.partial_der(x) == 0)
    assert (f4.partial_der(y) == 1)

    f4 = +x
    assert (f4.val == 2)
    assert (f4.grad() == [1])
    assert (f4.partial_der(x) == 1)
    assert (f4.partial_der(y) == 0)

    # case 8: negative
    f5 = -y
    assert (f5.val == 5)
    assert (f5.grad() == [-1])
    assert (f5.partial_der(x) == 0)
    assert (f5.partial_der(y) == -1)

    f5 = -x
    assert (f5.val == -2)
    assert (f5.grad() == [-1])
    assert (f5.partial_der(x) == -1)
    assert (f5.partial_der(y) == 0)

    # case 9: iadd
    # f5 was -x
    f5 += y  # f5 =-x+y
    assert (f5.val == -7)
    assert (f5.grad() == [-1, 1])
    assert (f5.partial_der(x) == -1)
    assert (f5.partial_der(y) == 1)

    f5 += x  # f5=y
    assert (f5.val == -5)
    assert (f5.grad() == [0, 1])
    assert (f5.partial_der(x) == 0)
    assert (f5.partial_der(y) == 1)

    f5 += 2*x  # f5=2x+y
    assert (f5.val == -1)
    assert (f5.grad() == [2, 1])
    assert (f5.partial_der(x) == 2)
    assert (f5.partial_der(y) == 1)

    # case 9: isub
    f5 -= y  # f5=2x
    assert (f5.val == 4)
    assert (f5.grad() == [2, 0])
    assert (f5.partial_der(x) == 2)
    assert (f5.partial_der(y) == 0)

    f5 -= 2*y  # f5=2x-2y
    assert (f5.val == 14)
    assert (f5.grad() == [2, -2])
    assert (f5.partial_der(x) == 2)
    assert (f5.partial_der(y) == -2)


def test_variable_scalar_multiple_divide():
    '''
    object.__mul__(), object.__truediv__(), object.__imul__(), object.__idiv__()
    '''
    # case 0: build variables
    x = Variable(2)
    y = Variable(-5)
    z = Variable(0)

    # case 1: right multiple
    f1 = 2*x
    assert (f1.val == 4)
    assert (f1.grad() == [2])
    assert (f1.partial_der(x) == 2)
    assert (f1.partial_der(y) == 0)

    # case 2: left add
    f2 = y*3
    assert (f2.val == -15)
    assert (f2.grad() == [3])
    assert (f2.partial_der(x) == 0)
    assert (f2.partial_der(y) == 3)

    # case 3: variable+variable
    f3 = x*y
    assert (f3.val == -10)
    assert (f3.grad() == [-5, 2])
    assert (f3.partial_der(x) == -5)
    assert (f3.partial_der(y) == 2)

    # case 4: right divide
    f1 = x/2
    assert (f1.val == 1)
    assert (f1.grad() == [1/2])
    assert (f1.partial_der(x) == 1/2)
    assert (f1.partial_der(y) == 0)

    # case 5: left divide
    f2 = 10+2*5/y
    assert (f2.val == 8)
    assert (f2.grad() == [-0.4])
    assert (f2.partial_der(x) == 0)
    assert (f2.partial_der(y) == -0.4)

    # case 6: variable/variable
    f3 = x/y
    assert (f3.val == -0.4)
    assert (f3.grad() == [-0.2, -0.08])
    assert (f3.partial_der(x) == -0.2)
    assert (f3.partial_der(y) == -0.08)

    # case 7: divide 0:
    with pytest.raises(ValueError):
        f4 = x/0

    with pyest.raises(ValueError):
        f5 = 2/z

    with pytest.raises(ValueError):
        f6 = x/z

    with pytest.raises(ValueError):
        f5 = 29/(0*x)

    # case8: imul
    # f3 was x/y
    f3 *= x  # f3=x**2/y
    assert (f3.val == -0.8)
    assert (f3.grad() == [-0.8, -0.08])
    assert (f3.partial_der(x) == -0.8)
    assert (f3.partial_der(y) == -0.08)

    f3 *= y  # f3=x**2
    assert (f3.val == 4)
    assert (f3.grad() == [2, 0])
    assert (f3.partial_der(x) == 2)
    assert (f3.partial_der(y) == 0)

    f3 *= 0  # f3=0*x**2
    assert (f3.val == 0)
    assert (f3.grad() == [0])
    assert (f3.partial_der(x) == 0)
    assert (f3.partial_der(y) == 0)


def test_variable_scalar_pow():
    '''object.__pow__(), object.__ipow__()'''
    x = Variable(2)
    y = Variable(-5)
    z = Variable(0)

    # case 1:
    f1 = x**2
    assert (f1.val == 4)
    assert (f1.grad() == [4])
    assert (f1.partial_der(x) == 4)
    assert (f1.partial_der(y) == 0)

    # case 2:
    f2 = x**0
    assert (f2.val == 0)
    assert (f2.grad() == 0)
    assert (f2.partial_der(x) == 0)
    assert (f2.partial_der(y) == 0)
    # notice 0**0 not define, should throws error
    with pytest.raises(ValueError):
        f3 = z**0
    with pytest.raises(ValueError):
        f4 = z**(-2)

    # case 3:
    f3 = (2*x)**1/2
    assert (f3.val == 2)
    assert (f3.grad() == [1/2])
    assert (f3.partial_der(x) == 1/2)
    assert (f3.partial_der(y) == 0)

    # case 4:
    f4 = 2**x
    assert (f4.val == 4)
    assert (f4.grad() == [4*np.log(2)])
    assert (f4.partial_der(x) == 4*np.log(2))
    assert (f4.partial_der(y) == 0)

    # case 5:
    f5 = x**x
    assert (f5.val == 4)
    assert (f5.grad() == [np.log(2)*(2**(np.log(2)))])

    # case 6:
    f4 **= y  # f4=(2x)^y
    assert (f4.val == 1/1024)
    assert (f4.grad() == [-5/4096, 1/1024*np.log(4)])


test_variable_scalar_add_minus()