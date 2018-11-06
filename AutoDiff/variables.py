class Variable(object):
    primitive_names = []
    
    def __init__(self, name, val, der = None, primitive = True):
        self.val = val
        if primitive:
            if name in Variable.primitive_names:
                raise ValueError("name {} already in use".format(name))
            self.name = name 
            self.der = {name : 1}
            Variable.primitive_names.append(name)
        else:
            self.name = name
            self.der = der

    def __repr__(self):
        return ("Variable name: {}, Value: {}, Derivatives: {}"
                .format(self.name, self.val, self.der)
                )

    def partial_der(self, dep_var_name):
        return self.der[dep_var_name.name]
    
    def grad(self):
        return self.der
    # unary operation of Variable instance.
    def __neg__(self):
        var = Variable(self.name, -self.val, self.der, False)    
        for key in self.der:
            var.der[key] = -self.der[key]
        return var

    def __add__(self, other):
        try:
            # get new dependent variables
            new_dep_vars = set(self.der.keys()).union(set(other.der.keys()))
            new_der = {}
            # calculate partial derivatives
            for dep_var in new_dep_vars:
                # get partial derivatives for self and other
                if dep_var in self.der.keys():
                    partial_der_1 = self.der[dep_var]
                else:
                    partial_der_1 = 0
                if dep_var in other.der.keys():
                    partial_der_2 = other.der[dep_var]
                else:
                    partial_der_2 = 0
                # calculate new partial
                new_der[dep_var] = partial_der_1 + partial_der_2
            new_name = "f({},{})".format(self.name, other.name)
            new_val = self.val+other.val
        except AttributeError:
            new_der = self.der
            new_name = self.name
            new_val = self.val + other
        return Variable(new_name, new_val, new_der, False)

    def __radd__(self, other):
        new_der = self.der
        new_name = self.name
        new_val = self.val + other
        return Variable(new_name, new_val, new_der, False)
    
    def __mul__(self, other):
        der1=self.der
        # when other is an instance of Variable. Ex) derivative(x*y) -> (y, x)
        try:
            der2=other.der
            der={x: other.val * der1.get(x, 0) + self.val * der2.get(x, 0) for x in set(der1).union(der2)}
            return Variable(self.name, self.val * other.val, der, False)
        # when other is not an instance of Variable. Ex) derivative(x*6) -> 6
        except AttributeError:
            der={x: other * der1.get(x, 0) for x in set(der1)}
            return Variable(self.name, self.val * other, der, False)
    __rmul__ = __mul__ 

    # a function for left division
    def __truediv__(self, other):
        der1 = self.der
        # when other is an instance of Variable. Ex) derivative(x/y) -> (1/y, x/(y**2))
        try:
            der2 = other.der
            der={x: 1/other.val * der1.get(x, 0) - self.val/other.val**2*der2.get(x,0) for x in set(der1).union(der2)}
            return Variable(self.name, self.val / other.val, der, False)
        # when other is not an instance of Variable. Ex) derivative(x/6) -> 1/6
        except:
            der = {x: der1.get(x, 0) / other for x in set(der1)}
            return Variable(self.name, self.val / other, der, False)
    # a function for right division. Ex) derivative(6/x) -> -6/(x**2)
    def __rtruediv__(self, other):
        der1 = self.der
        der = {x: -other/self.val**2*der1.get(x, 0) for x in set(der1)}
        return Variable(self.name, other/self.val, der, False)

    def jacobian(self):
        der1 = self.der
        jacobian = {key: self.der[key] for key in set(der1)}
        return jacobian

    # implement other dunder methods for numbers
    # https://www.python-course.eu/python3_magic_methods.php


if __name__ == "__main__":
    x = Variable('x', 2)
    y = Variable('y', 3)
    z = Variable('z', 10)
    f = 12+x+y+z+y+5
    print(f)
    print(f.partial_der(y))
    print(f.grad())
    bad_x = Variable('x', 10)

if __name__ == "__main__":
    x = Variable('x', 2)
    y = Variable('y', 3)
    z = Variable('z', 10)
    f = 6*x*x*y
    print(f)
    print(f.partial_der(x))
    print(f.grad())
    #bad_x = Variable('x', 10)
