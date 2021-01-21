import numpy as np

def linesearch(x,p,g,fun,eta,gam,jmax):
    # Apply the backtracking linesearch algorithm
    # Inputs:
    #  x - starting point
    #  p - descent direction
    #  g - gradient at point
    #  fun - function being evaluated
    #  eta - stopping bounds
    #  gamma - slope coefficient
    #  jmax - max number of iterations
    a = 1
    f0 = fun(x)
    aux = eta*np.dot(g,p)
    for j in range(jmax+1):
        xtry = x + a*p
        f1 = fun(xtry)
        if f1 < (f0 + a*aux):
            break
        else:
            a = a*gam
    print(j)
    return a, j