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

    def partial_der(self, dep_var_name):
        try:
            return self.der.get(dep_var_name.name,0)
        except AttributeError:
            print("input is not a Variable")
        
<<<<<<< HEAD
    def grad(self):
=======
    
    
    def jacobian(self):
>>>>>>> e26d6a72b8b7253ea3e7114adce9a14dc364bd0c
        return self.der
    # unary operation of Variable instance.

    def __pos__(self):
        return Variable(self.name, self.val, der, False)

    def __neg__(self):
<<<<<<< HEAD
        der1 = self.der
        new_der = {x: -der1.get(x,0) for x in set(der1)}
        return Variable(self.name, -self.val, new_der, False)
=======
        var = Variable('f({})'.format(self.name), -self.val, self.der, False)    
        for key in self.der:
            var.der[key] = -self.der[key]
        return var
>>>>>>> e26d6a72b8b7253ea3e7114adce9a14dc364bd0c

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
        der1=self.der
        # when other is an instance of Variable. Ex) derivative(x-y) -> (y, x)
        try:
            der2=other.der
            der={x: der1.get(x, 0) - der2.get(x, 0) for x in set(der1).union(der2)}
            return Variable('f({},{})'.format(self.name, other.name), self.val - other.val, der, False)
        # when other is not an instance of Variable. Ex) derivative(x-6) -> 6
        except AttributeError:
            return Variable('f({})'.format(self.name), self.val - other, der1, False)
    
    def __rsub__(self, other):
        der1=self.der
        # when other is an instance of Variable. Ex) derivative(y-x) -> (y, x)
        try:
            der2=other.der
            der={x: der2.get(x, 0) - der1.get(x, 0) for x in set(der1).union(der2)}
            return Variable('f({},{})'.format(self.name, other.name), 
                            other.val - self.val, der, False)
        # when other is not an instance of Variable. Ex) derivative(y-x) -> 6
        except AttributeError:
            for key in self.der:
                der1[key] = -der1[key]
            return Variable('f({})'.format(self.name), other - self.val, der1, False)
        
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

<<<<<<< HEAD
    def __pow__(self, other):
        try:
            # calculate new derivative
            new_der = {k: self.der.get(k, 0)*other.val*np.power(self.val, other.val-1) + other.der.get(k, 0)*np.log(self.val)*np.power(self.val, other.val) for k in set(self.der).union(other.der)}

            new_name = "f({},{})".format(self.name, other.name)
            new_val = np.power(self.val, other.val)

            return Variable(new_name, new_val, new_der, False)
        except AttributeError:
            # only x is variable
            if isinstance(self, Variable):
                return Variable(self.name, np.power(self.val, other), {k:v*other*np.power(self.val, other-1) for (k,v) in self.der.items()}, False)
            # only y is variable
            elif isinstance(other, Variable):
                return Variable(other.name, {k:v*np.log(self)*np.power(self, other.val) for (k,v) in other.der.items()})
            # both not variable
            else:
                return np.power(self, other)
    
    def jacobian(self):
        der1 = self.der
        jacobian = {key: self.der[key] for key in set(der1)}
        return jacobian
=======
>>>>>>> e26d6a72b8b7253ea3e7114adce9a14dc364bd0c

    






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
