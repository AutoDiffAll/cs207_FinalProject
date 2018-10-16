# Milestone 1: *AutoDiffAll*

## Table of Contents
1. [Introduction](#introduction)
2. [Background](#background)
3. [User API](#API)
4. [Software Organization](#SoftwareOrganization)
5. [Implementation](#implementation)


#### Test Math Input

$x=1$


## Introduction <a name="introduction"></a>
>Todo: Describe problem the software solves and why it's important to solve that problem

## Background <a name="background"></a>
>To do: Describe (briefly) the mathematical background and concepts as you see fit.  You **do not** need to
give a treatise on automatic differentation or dual numbers.  Just give the essential ideas (e.g.
the chain rule, the graph structure of calculations, elementary functions, etc).

## User API <a name="API"></a>
>To do: How do you envision that a user will interact with your package?  What should they import?  How can
they instantiate AD objects?

The user would import the main AD class, as well as the numpy elementary functions that we have overriden. The usage of our elementary would use the same np short form that users are typically used to.

from AutoDiff import AD
import AutoDiff.numpy as np

In the case for a scalar function, the implementation is simple. The user calls initializes AD instances on the independent variable x at a given value. The user can then use their function on the AD instance just as they would previously on a given float/integer. They can then use the der function that basically just returns the der instance attribute.  

test_fn1 = lambda x: x - np.exp(-2*np.sin(4*x))

x = AD(2.0)
ad_res = test_fn1(tst1)
der(ad_res)

For a scalar function of vectors with length n with multiple independent variables d, the implementation is largely similar. The user has to initialize 2 separate AD instances for each independent variable. Moreover, the user has to indicate the number of independent variables d at initialization. The rest of the implementation is exactly the same as before. However, we note that the der function would return a n*d array of derivative with respect to all the independent variables at each value in the vector function.

test_fn2 = lambda x, y: x*y + np.sin(x)

x = AD([2.0, 5.0, 7.0], n_dim=2)
y = AD([1.0, 2.0, 3.0], n_dim=2)
ad_res = test_fn2(x, y)
der(ad_res) # returns a 3x2 array of derivatives with respect to both x and y at each value

Even if we have a vector function of vectors with lengths m and n respectively, with multiple independent variables d, the implementation remains the same. However, when the user calls the vector function on the AD instances, he/she gets returned a vector of AD instances. As such, the user will have to call the der function separately on each AD instance.

test_fn3 = lambda x, y: (x*y + np.sin(x), x+y+np.sin(x*y))

ad_res = test_fn3(x,y)
der(ad_res[0])
der(ad_res[1])

>Todo: **Note: This section should be a mix of pseudo code and text.  It should not include any actual
operations yet.**

## Software Organization <a name="SoftwareOrganization"></a>
>Todo: Discuss how you plan on organizing your software package.
* What will the directory structure look like?  
* What modules do you plan on including?  What is their basic functionality?
* Where will your test suite live?  Will you use `TravisCI`? `Coveralls`?
* How will you distribute your package (e.g. `PyPI`)?


## Implementation <a name="implementation"></a>
### Core Data Structure
We want to follow the computational graph and construct the "node" class as our major data structure `AutoDiff` class.

### Major Class
The main class that we will implement is the `AutoDiff` class that takes as input the values of the "independent variables"(either a scaler or a vector) at the function that we are calculating the derivative on. The "independent variables" can be seen as a node in the computational graph. 

![comp-graph](figs/Computational-Graph.png)

### Method and Name Attributes in AutoDiff Class
* Name Attributes

The `AutoDiff` class will have two main instance variables, the value of the AutoDiff instance, and the derivatives of the instance. The derivatives of the instance is initialized with the relevant `n_dim` seed vectors, where `n_dim` is the number of independent variables.

1. `AutoDiff.n_dim`: number of the variables in the target function. Use it to decide the dimension of derivatives.
2. `AutoDiff.val`: value of variable(nodes). the shape is the same as the input variable. So if input is scalar, it will be scalar, while if input is vector, it is vector.
3. `AutoDiff.der`: value of derivatives in this nodes. It is `m*n_dim` array, which store the value of the nodes(`m` value in a vector) to `n_dim` different variables.

* Overide the four basic operations. 

In order to override the four basic operations of elementary arithmetic (addition, subtraction, multiplication, and division), we use dunder methods within our `AutoDiff` class. The dunder methods return new `AutoDiff` instances with the updated value and derivatives.

* Define elementary differentiation function. 

In order to deal with the other elementary functions (exponential, logarithm, powers, roots, trigonometric functions, inverse trigonometric functions, hyperbolic functions, etc.), we will override the numpy elementary functions such that we can use it for our AutoDiff class. 
>For example, we will override the `np.sin` function such that if you use it on an AutoDiff instance `x` at a given value, it will return another AutoDiff instance with the value of `sin(x)`, and the calculated derivative of `$t{x}cos(x)$` at the given value. Similarly, we will override the `np.exp` function such that if you use it on an AutoDiff instance `x` at a given value, it will return another AutoDiff instance with the value of `exp(x)`, and the calculated derivative of `\dot{x}exp(x)` at the given value.

* Define non-differentiable function.

We can even handle some function which is non-differentiable at certain points, such as Zigzig function like Brownian Motion, or like `f(x)=1/x` at `x=0`. This is our extension for the `AutoDiff` class. We will employ



* What external dependencies will you rely on?
* How will you deal with elementary functions like `sin` and `exp`?

> Be sure to consider a variety of use cases.  For example, don't limit your design to scalar
> functions of scalar values.  Make sure you can handle the situations of vector functions of vectors and
scalar functions of vectors.  Don't forget that people will want to use your library in algorithms
like Newton's method (among others).

>Try to keep your report to a reasonable length.  It will form the core of your documentation, so you
want it to be a length that someone will actually want to read.

> Our core data structure is our AutoDiff class and the overriden elementary operations functions.
> 
> I
> In order to implement this, we will rely on the numpy and math external libraries, which will be specified as our external dependencies in our setup.py file.
> As such, after the user initializes the AutoDiff class on the indepndent variables, he/she will be able to use the usual elementary functions on the AutoDiff instance in order to calculate both the value of the function and the value of the derivative.
