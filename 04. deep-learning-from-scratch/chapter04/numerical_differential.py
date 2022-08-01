import numpy as np

def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2 * h)


def function_a(x):
    return 0.01* x ** 2 + 0.1*x
    

def function_b(x):
    return x[0] ** 2 + x[1] ** 2


def function_tmp(x0):
    return x0*x0 + 4.0 ** 2.0


def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_value = x[idx]
        print (tmp_value)
        # calculate f(x+h)
        x[idx] = tmp_value + h
        fxh1 = f(x)
        print (fxh1)
        
        # calcaulte f(x-h)
        x[idx] = tmp_value - h
        fxh2 = f(x)
        print (fxh2)
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        print (grad[idx])
        x[idx] = tmp_value
    
    return grad        
    

# print (numerical_diff(function_a, 5))
# print (numerical_diff(function_a, 10))
# print (numerical_diff(function_tmp, 3.0))

grad = numerical_gradient(function_b, np.array([3.0, 4.0]))
print (grad)