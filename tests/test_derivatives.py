from AutoDiff import variables
import AutoDiff.numpy as anp
import numpy as np

# test for the successfulness of building a variable class and call a function

# test for scalar function with scalar values
def test_variable_scalar_add_minus():
    # case 0: build variables
    x = variables(2)
    y = variables(-5)

    # case 1: right add
    f1 = x+2
    assert (f1.val == 4)
    assert (f1.grad() == [1])
    assert (f1.der(x) == 1)
    assert (f1.der(y) == 0)

    # case 2: left add
    f2 = 10+2*8+y
    assert (f2.val == 21)
    assert (f2.grad() == [1])
    assert (f2.der(x) == 0)
    assert (f2.der(y) == 1)

    # case 3: variable+variable
    f3 = 3+x+y-2
    assert (f3.val == -2)
    assert (f3.grad() == [1, 1])
    assert (f3.der(x) == 1)
    assert (f3.der(y) == 1)

    # case 4: right minus
    f1 = x-2
    assert (f1.val == 0)
    assert (f1.grad() == [-1])
    assert (f1.der(x) == -1)
    assert (f1.der(y) == 0)

    # case 5: left minus
    f2 = 10+2*8-y
    assert (f2.val == 31)
    assert (f2.grad() == [-1])
    assert (f2.der(x) == 0)
    assert (f2.der(y) == -1)

    # case 6: variable+variable
    f3 = 3-x-y-2
    assert (f3.val == 2)
    assert (f3.grad() == [-1, -1])
    assert (f3.der(x) == -1)
    assert (f3.der(y) == -1)


def test_variable_scalar_multiple_divide():
       # case 0: build variables
    x = variables(2)
    y = variables(-5)
    z = variables(0)

    # case 1: right multiple
    f1 = 2*x
    assert (f1.val == 4)
    assert (f1.grad() == [2])
    assert (f1.der(x) == 2)
    assert (f1.der(y) == 0)

    # case 2: left add
    f2 = y*3
    assert (f2.val == -15)
    assert (f2.grad() == [3])
    assert (f2.der(x) == 0)
    assert (f2.der(y) == 3)

    # case 3: variable+variable
    f3 = x*y
    assert (f3.val == -10)
    assert (f3.grad() == [-5, 2])
    assert (f3.der(x) == -5)
    assert (f3.der(y) == 2)

    # case 4: right divide
    f1 = x/2
    assert (f1.val == 1)
    assert (f1.grad() == [1/2])
    assert (f1.der(x) == 1/2)
    assert (f1.der(y) == 0)

    # case 5: left divide
    f2 = 10+2*5/y
    assert (f2.val == 8)
    assert (f2.grad() == [-0.4])
    assert (f2.der(x) == 0)
    assert (f2.der(y) == -0.4)

    # case 6: variable/variable
    f3 = x/y
    assert (f3.val == -0.4)
    assert (f3.grad() == [-0.2, -0.08])
    assert (f3.der(x) == -0.2)
    assert (f3.der(y) == -0.08)

    # case 7: divide 0:
    with pytest.raises(ValueError):
        f4 = x/0

    with pytest.raises(ValueError):
        f5 = 2/z

    with pytest.raises(ValueError):
        f6 = x/z

    with pytest.raises(ValueError):
        f5 = 29/(0*x)
    
