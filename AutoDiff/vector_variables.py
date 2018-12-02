import numpy as np
import pandas as pd
try:
    from variables import Variable
except:
    from AutoDiff.variables import Variable

class vector_Variable(object):
    def __init__(self, variable_vec):

        # check that it is not single variable
        if isinstance(variable_vec, Variable):
            raise TypeError('Function is not a vector function!')

        # check that every object in vector is a variable
        # OR SHOULD WE ALLOW SUCH THAT NOT ALL OBJECT IS A VARIABLE?
        if not all([isinstance(i, Variable) for i in variable_vec]):
            raise TypeError('Every object in a vector function should be a Variable!')

        self.variables = variable_vec
        self.val = np.array([i.val for i in variable_vec])
        self.der = pd.concat([pd.DataFrame(i.jacobian(), index=[i.name]) for i in variable_vec], sort=True).fillna(0)

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
