import numpy as np

class variable():

    def __init__(self, var, der=1.0):
        self.val = var
        if isinstance(der, list):
            self.der = der
        else:
            self.der = [der]

    def __mul__(self, other):
        try:
            return variable(self.val*other.val,
            [der*other.val for der in self.der] + [self.val*der for der in other.der])
        except AttributeError:
            return variable(self.val*other, [der*other for der in self.der])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        try:
            return variable(self.val+other.val, self.der+other.der)
        except AttributeError:
            return variable(self.val+other, self.der)

    def __radd__(self, other):
        return self.__add__(other)

    def __pow__(self, other):
        return variable(self.val**other, other*self.der*sel.val**(other-1))

    def sin(self):
        return variable(np.sin(self.val), self.der*np.cos(self.val))

    def cos(self):
        return variable(np.cos(self.val), -self.der*np.sin(self.val))

    def tan(self):
        return variable(np.tan(sel.val), self.der/(np.cos(self.val))**2)

    def exp(self):
        return variable(np.exp(self.val), self.der*np.exp(self.val))

    def log(self):
        return variable(np.log(self.val), self.der/self.val)

    def sqrt(self):
        return variable(np.sqrt(self.val), 0.5*self.der/np.sqrt(self.val))
