from AutoDiff.variables import Variable
import AutoDiff.numpy as anp
import numpy as np
import pytest

def test_construct_scalar():
    # test for scalar variables
    x = Variable('x', 2)
    y = Variable('y', 3)
    z = Variable('z', 10)
    f = 6*x
    print(f)

test_construct_scalar()
