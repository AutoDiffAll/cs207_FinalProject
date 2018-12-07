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
        return self.der

    def partial_der(self, dep_var):
        # note: we should raise KeyError for variable class as well, rather than try and except?
        # raise error if input is not a variable
        if dep_var not in self.der.columns.values:
            raise KeyError('Input is not a Variable')

        return self.der[dep_var].values

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

    def __pow__(self, other):
        # check that vector functions are of the same length
        if len(self.val) != len(other.val):
            raise (ValueError('operands could not be broadcast together with shapes {} {}'
                              .format(self.val.shape, other.val.shape)))

        # if both are vector variables
        try:
            return vector_Variable(self.variables**other.variables)
        # when other is not a vector of variables
        except AttributeError:
            return vector_Variable(self.variables**other)
    __rpow__ = __pow__

    def __pos__(self):
        return vector_Variable(self.variables)

    def __neg__(self):
        return vector_Variable(np.negative(self.variables))


if __name__ == "__main__":
    import doctest
    doctest.testmod()
