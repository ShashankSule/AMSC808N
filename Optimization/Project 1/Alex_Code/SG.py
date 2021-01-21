import numpy as np
from linesearch import linesearch
import time

def SG(fun, gfun, Y, x, afun, batchsize = 64, itermax = 10000):
    # Minimize function using stochastic gradient descent
    # Inputs:
    #  fun - loss function
    #  gfun - gradient of loss function
    #  Y - data matrix
    #  x - initial weights
    #  afun - step size scheduling function
    #  batchsize - number of data points to evaluate function, gradient at
    #  itermax - maximum number of iterations
    
    # Initialize arrays
    if afun is 'linesearch':
        gam = 0.9
        eta = 0.5
        jmax = round(np.ceil(np.log(1e-14)/np.log(gam)))
    xiter = np.zeros((itermax+1,x.size))
    f = np.zeros(itermax+1)
    runtime = np.zeros(itermax)
    n = Y.shape[0] # Number of data points
    Iall = np.arange(n)
    tic = time.perf_counter()
    f[0] = fun(Iall,Y,x)
    xiter[0] = x
    grad_norm = np.zeros(itermax+1)
    bsz = np.minimum(n,batchsize) # If not enought data points for batchsize, set to maximum
    I = np.random.permutation(n)[:bsz] # Get random sample of data points
    g = gfun(I,Y,x)
    grad_norm[0] = np.linalg.norm(g)
    toc = time.perf_counter()
    runtime[0] = toc - tic # Time computation
    itr = 0
    while itr < itermax - 1:
        if afun is 'linesearch':
            linefun = lambda x: fun(I,Y,x)
            a,j = linesearch(xiter[itr],-g,g,linefun,eta,gam,jmax)
            xiter[itr + 1] = xiter[itr] - a*g
        else:
            xiter[itr+1] = xiter[itr] - afun(itr)*g # Update weights
        f[itr+1] = fun(Iall,Y,xiter[itr+1])
        itr += 1
        I = np.random.permutation(n)[:bsz] # Get a new sample
        g = gfun(I,Y,xiter[itr]) # Compute gradient
        grad_norm[itr] = np.linalg.norm(g)
        toc = time.perf_counter()
        runtime[itr] = toc - tic # Time computation
    if afun is 'linesearch':
        linefun = lambda x: fun(I,Y,x)
        a,j = linesearch(xiter[itr],-g,g,linefun,eta,gam,jmax)
        xiter[itr + 1] = xiter[itr] - a*g
    else:
        xiter[itr+1] = xiter[itr] - afun(itr)*g # Update weights
    f[itr+1] = fun(Iall,Y,xiter[itr+1])
    return xiter, f, grad_norm, runtime # Return final weights, results vectors, and runtime
    