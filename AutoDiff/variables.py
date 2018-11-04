class Variable(object):
    def __init__(self, name, val, der = None):
        self.val = val
        self.name = name
        if der is None:
            self.der = {name : 1}
        else:
            self.der = der
    
    def __repr__(self):
        return ("Variable name: {}, Value: {}, Derivatives: {}"
                .format(self.name, self.val, self.der)
                )

    def partial_der(self, dep_var_name):
        return self.der[dep_var_name.name]
    
    def grad(self):
        return self.der
    
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
        return Variable(new_name, new_val, new_der)

    def __radd__(self, other):
        pass
    def __mul__(self, other):
        pass
    def __rmul__(self, other):
        pass
    
    # implement other dunder methods for numbers
    # https://www.python-course.eu/python3_magic_methods.php


if __name__ == "__main__":
    x = Variable('x', 2)
    y = Variable('y', 3)
    z = Variable('z', 10)
    f = x+y+z+y+5
    print(f)
    print(f.partial_der(y))
    print(f.grad())