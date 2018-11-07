import numpy as np
class Variable(object):    
    def __init__(self, name, val, der = None, primitive = True):
        self.val = val
        if primitive:
            self.name = name 
            self.der = {name : 1}
            
        else:
            self.name = name
            self.der = der

    def __repr__(self):
        return ("Variable name: {}, Value: {}, Derivatives: {}"
                .format(self.name, self.val, self.der)
                )

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
        >>> from variables import Variable
        >>> a = Variable('a', 2)
        >>> x = 4*a
        >>> x.partial_der(a)
        4
        """
        try:
            return self.der.get(dep_var.name,0)
        except AttributeError:
            print("input is not a Variable")

    def jacobian(self):
        """Returns jacobian of variable
        
        
        RETURNS
        ========
        value: dict
        
        NOTES
        =====
        POST:
            - returns jacobian as a dictionary where keys are dependent variable
        
        EXAMPLES
        =========
        >>> from variables import Variable
        >>> a = Variable('a', 2)
        >>> b = Variable('b', 3)
        >>> x = a*b
        >>> x.jacobian()
        {'a': 3, 'b': 2}
        """
        return self.der

    def __pos__(self):
        return Variable(self.name, self.val, der, False)

    def __neg__(self):
        try:
            return Variable(self.name, np.negative(self.val), {k:np.negative(v) for (k,v) in self.der.items()}, False)
        except AttributeError:
            return np.negative(self)

    def __add__(self, other):
        der1=self.der
        # when other is an instance of Variable. Ex) derivative(x+y) -> (y, x)
        try:
            der2=other.der
            der={x: der1.get(x, 0) + der2.get(x, 0) for x in set(der1).union(der2)}
            return Variable('f({},{})'.format(self.name, other.name), self.val + other.val, der, False)
        # when other is not an instance of Variable. Ex) derivative(x*6) -> 6
        except AttributeError:
            return Variable('f({})'.format(self.name), self.val + other, der1, False)
    __radd__ = __add__ 
    
    def __sub__(self, other):
        other = -other
        return self+other
    
    def __rsub__(self, other):
        var = -self
        return var+other
        
    def __mul__(self, other):
        der1=self.der
        # when other is an instance of Variable. Ex) derivative(x*y) -> (y, x)
        try:
            der2=other.der
            der={x: other.val * der1.get(x, 0) + self.val * der2.get(x, 0) for x in set(der1).union(der2)}
            return Variable('f({},{})'.format(self.name, other.name),
                            self.val * other.val, der, False)
        # when other is not an instance of Variable. Ex) derivative(x*6) -> 6
        except AttributeError:
            der={x: other * der1.get(x, 0) for x in set(der1)}
            return Variable('f({})'.format(self.name), self.val * other, der, False)
    __rmul__ = __mul__ 

    # a function for left division
    def __truediv__(self, other):
        der1 = self.der
        # when other is an instance of Variable. Ex) derivative(x/y) -> (1/y, x/(y**2))
        try:
            der2 = other.der
            der={x: 1/other.val * der1.get(x, 0) - self.val/other.val**2*der2.get(x,0) for x in set(der1).union(der2)}
            return Variable('f({},{})'.format(self.name, other.name), 
                            self.val / other.val, der, False)
        # when other is not an instance of Variable. Ex) derivative(x/6) -> 1/6
        except:
            der = {x: der1.get(x, 0) / other for x in set(der1)}
            return Variable('f({})'.format(self.name), self.val / other, der, False)
    # a function for right division. Ex) derivative(6/x) -> -6/(x**2)
    def __rtruediv__(self, other):
        der1 = self.der
        der = {x: -other/self.val**2*der1.get(x, 0) for x in set(der1)}
        return Variable('f({})'.format(self.name), other/self.val, der, False)

    def __pow__(self, other):
        der1 = self.der
        try:
            der2 = other.der
            # calculate new derivative
            new_der = {}
            for k in set(self.der).union(other.der):
                partial_self = der1.get(k, 0)*other.val*np.power(self.val, other.val-1)
                partial_other = der2.get(k, 0)*np.log(self.val)*np.power(self.val, other.val)
                new_der[k] = partial_self + partial_other
            new_name = "f({},{})".format(self.name, other.name)
            new_val = np.power(self.val, other.val)
            return Variable(new_name, new_val, new_der, False)      
        except AttributeError:
            new_der = {}
            for k in self.der:
                new_der[k] = self.der[k]*other*np.power(self.val, other-1)
            new_name = "f({})".format(self.name)
            new_val = np.power(self.val, other)
            return Variable(new_name, new_val, new_der, False)
            
            
    def __rpow__(self, other):
        new_der = {}
        for k in self.der:
            new_der[k] = self.der[k]*np.log(other)*np.power(other, self.val)
        new_name = "f({})".format(self.name)
        new_val = np.power(other, self.val)
        return Variable(new_name, new_val, new_der, False)
    

    






    # implement other dunder methods for numbers
    # https://www.python-course.eu/python3_magic_methods.php

#if __name__ == "__main__":
#   x = Variable('x', 2)
#    y = Variable('y', 3)
#    z = Variable('z', 10)
#    f = 12+x+y+z+y+5
#    print(f)
#    print(f.partial_der(y))
#    print(f.grad())
#    bad_x = Variable('x', 10)

#if __name__ == "__main__":
#    x = Variable('x', 2)
#    y = Variable('y', 3)
#   z = Variable('z', 10)
#    f = 6*x
#    print(f)
#    print(f.partial_der(x))
#    print(f.grad())
#    bad_x = Variable('x', 10)
