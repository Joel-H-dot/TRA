This package provides trust region algorithms (TRA) for finding the minimum of some function. At the minute it contains only Levenberg-Marquart, but will be expanded to include NL2SOL and Powell's dogleg. 
# Levenberg-Marquardt
## Example
An example is included within the package, simply call:

```
Import TRA as TRA
example_problem = TRA.example()
minimum = example_problem.find_minimum()

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

