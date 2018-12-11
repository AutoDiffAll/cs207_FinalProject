import numpy as np
import pandas as pd
try:
    from variables import Variable
except:
    from AutoDiff.variables import Variable

class vector_Variable(object):
    def __init__(self, variable_vec):

        # should convert to array?
        variable_vec = np.array(variable_vec)

        self.variables = variable_vec
        self.val = np.array([i.val for i in variable_vec])
        self.der = pd.concat([pd.DataFrame(i.jacobian(), index=[i.name]) for i in variable_vec],
         sort=True, ignore_index=True).fillna(0)

    def jacobian(self):
        """Returns jacobian of variable


        RETURNS
        ========
        value: pandas dataframe

        NOTES
        =====
        POST:
            - returns jacobian as a dataframe where columns are dependent variable

        EXAMPLES
        =========
        >>> try:
        ...     from variables import Variable
        ... except:
        ...     from AutoDiff.variables import Variable
        >>> try:
        ...     from vector_variables import vectorize_variable
        ... except:
        ...     from AutoDiff.vector_variables import vectorize_variable
        >>> try:
        ...     import AD_numpy as anp
        ... except:
        ...     import AutoDiff.AD_numpy as anp
        >>> @vectorize_variable
        ... def vec_fn(x, y, z):
        ...     f1 = x * y + anp.sin(y) + anp.cos(z)
        ...     f2 = x + y + anp.sin(x*y)
        ...     return [f1,f2]
        >>> a = Variable('a', 2)
        >>> b = Variable('b', 3)
        >>> c = Variable('c', 2)
        >>> f = vec_fn(a, b, c)
        >>> f.jacobian().values
        array([[ 3.        ,  1.0100075 , -0.90929743],
           [ 3.88051086,  2.92034057,  0.        ]])
        """
        return self.der

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
        >>> try:
        ...     from vector_variables import vectorize_variable
        ... except:
        ...     from AutoDiff.vector_variables import vectorize_variable
        >>> try:
        ...     import AD_numpy as anp
        ... except:
        ...     import AutoDiff.AD_numpy as anp
        >>> @vectorize_variable
        ... def vec_fn(x, y, z):
        ...     f1 = x * y + anp.sin(y) + anp.cos(z)
        ...     f2 = x + y + anp.sin(x*y)
        ...     return [f1,f2]
        >>> a = Variable('a', 2)
        >>> b = Variable('b', 3)
        >>> c = Variable('c', 2)
        >>> f = vec_fn(a, b, c)
        >>> x = partial_der(a)
        array([1.       , 0.0100075])
        >>> d = 3
        >>> x.partial_der(b)
        input is not a Variable
        >>> e = Variable('e', 3)
        >>> x.partial_der(e)
        array([0., 0.])
        """
        try:
            if dep_var.name not in self.der.columns.values:
                return np.zeros(self.der.shape[0])
            else:
                return self.der[dep_var.name].values
        except AttributeError:
            print('input is not a Variable')

    def __add__(self, other):
        # check that vector functions are of the same length
        if len(self.val) != len(other.val):
            raise (ValueError('operands could not be broadcast together with shapes {} {}'
                              .format(self.val.shape, other.val.shape)))

        # if both are vector variables
        try:
            return vector_Variable(self.variables + other.variables)
        # when other is not a vector of variables
        except AttributeError:
            return vector_Variable(self.variables + other)
    __radd__ = __add__

    def __sub__(self, other):
        other = -other
        return self+other
    __rsub__ = __sub__

    def __mul__(self, other):
        # check that vector functions are of the same length
        if len(self.val) != len(other.val):
            raise (ValueError('operands could not be broadcast together with shapes {} {}'
                              .format(self.val.shape, other.val.shape)))

        # if both are vector variables
        try:
            return vector_Variable(self.variables*other.variables)
        # when other is not a vector of variables
        except AttributeError:
            return vector_Variable(self.variables*other)
    __rmul__ = __mul__

    def __truediv__(self, other):
        # check that vector functions are of the same length
        if len(self.val) != len(other.val):
            raise (ValueError('operands could not be broadcast together with shapes {} {}'
                              .format(self.val.shape, other.val.shape)))

        # if both are vector variables
        try:
            return vector_Variable(self.variables/other.variables)
        # when other is not a vector of variables
        except AttributeError:
            return vector_Variable(self.variables/other)
    __rtruediv__ = __truediv__

    def __pos__(self):
        return vector_Variable(self.variables)

    def __neg__(self):
        return vector_Variable(np.negative(self.variables))

    def __eq__(self, other):
        # check that vector functions are of the same length
        try:
            if np.array_equal(self.variables, other.variables):
                return True
            else:
                return False
        except AttributeError:
            return False

    def __ne__(self, other):
        if self == other:
            return False
        else:
            return True

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
    ...     from vector_variables import vectorize_variable
    ... except:
    ...     from AutoDiff.vector_variables import vectorize_variable
    >>> try:
    ...     import AD_numpy as anp
    ... except:
    ...     import AutoDiff.AD_numpy as anp
    >>> @vectorize_variable
    ... def vec_fn(x, y, z):
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

        # if every object in vector is just numeric
        if all([isinstance(i, (int, float, complex)) for i in variable_vec]):
            return variable_vec

        # check that it is not single variable
        if isinstance(variable_vec, Variable):
            raise TypeError('Function is not a vector function!')

        # check that every object in vector is a variable
        # OR SHOULD WE ALLOW SUCH THAT NOT ALL OBJECT IS A VARIABLE?
        if not all([isinstance(i, Variable) for i in variable_vec]):
            raise TypeError('Every object in a vector function should be a Variable!')

        return vector_Variable(variable_vec)
    return fn_wrapper


if __name__ == "__main__":
    import doctest
    doctest.testmod()
