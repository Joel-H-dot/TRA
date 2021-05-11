This package provides two trust region algorithms (TRA) for finding the minimum of some function, Levenberg-Marquardt and Powell's dogleg. 
# Levenberg-Marquardt
## Example
An example is included within the package, simply call:

```
import TRA as TRA
def forward_model(x):
    y = np.array(x[0] ** 2 + x[1] ** 2)
    y = y.reshape((1, 1))
    return y

def compute_gradient(x):
    g = np.array(([2 * x[0]], [2 * x[1]]))
    g = g.reshape((2, 1))
    return g

def compute_hessian(x):
    h = np.array(([2, 0], [0, 2]))
    h = h.reshape((2, 2))
    return h


initial_prediction = np.array([5, 2.7])

LM_algorithm = TRA.Levenberg_Marquart(initial_prediction, compute_hessian, compute_gradient,
                                                  forward_model, d_param=1e-50,
                                                  lower_constraint=-np.inf,
                                                  upper_constraint=np.inf,
                                                  num_iterations=5)

minimum = LM_algorithm.optimisation_main()



```
![image](https://user-images.githubusercontent.com/60707891/115858459-e4d7ea00-a426-11eb-8dd5-fdaa93e9c574.png)

This is a simple example, but shows how to use the Levenberg_Marquart class. 


## Function calls and arguments

There are a number of default values within the Levenberg_Marqaurdt class, including constraints on the solution, the number of iterations amd the damping parameter corresponding to the trust region. Three functions are required when instantiating a class object, one for computing the gradient, one for the Hessian and one for the mapping of the input to ouput (forward model). 
```
:
def forward_model(x)
    :
    return f(x)
def compute_gradient(x):
    :
return grad

def compute_hessian(x):
    :
return hessian

initial_prediction = x0

LM_object = TRA.Levenberg_Marquart(initial_prediction, compute_hessian, compute_gradient,
                                                  forward_model, d_param=1e-50,
                                                  lower_constraint=-np.inf,
                                                  upper_constraint=np.inf,
                                                  num_iterations=5)
                                                  
                                                  
    
```

# Theory 

 For the theory behind the code see [[1]](#1) and [[2]](#2). 

## References
<a id="1">[1]</a> 
Jorge Nocedal and Stephen J. Wright  (2006). 
Numerical Optimization. 

<a id="2">[2]</a> 
 Andrew R. Conn, Nicholas I. M. Gould, and P.L. Toint (2000). 
Trust Region Methods. 

