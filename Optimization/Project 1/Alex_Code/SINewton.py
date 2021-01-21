import numpy as np
from CG import CG
import time

def SINewton(fun, gfun, Hvec, Y, w, batchsize = 64,kmax = 1e3):
    # SINewton applies the subsampled inexact Newton's method
    # Inputs:
    #  fun - function
    #  gfun - gradient function
    #  Hvec - Hessiant*vector function
    #  Y - data matrix for classification problem
    #  w - Initial guess for hyperplane weights + bias
    #  batchsize - size of batch for computing gradient, Hessian
    #  kmax - maximum number of iterations
    
    # Set parameters for linesearch algorithm
    rho = 0.1
    gam = 0.9
    jmax = round(np.ceil(np.log(1e-14)/np.log(gam)))
    eta = 0.5
    CGimax = 20 # CG maximum number of iterations
    n = Y.shape[0] # Number of data points
    bsz = np.minimum(n,batchsize) # If batchsize it too large, set it to max
    I = np.arange(n) # Set I for function evaluation at each step
    # Initialize vectors for storing function value and gradient norm
    f = np.zeros(int(kmax+1))
    f[0] = fun(I,Y,w)
    normgrad = np.zeros(int(kmax))
    runtime = np.zeros(int(kmax))
    tic = time.perf_counter() # Start runtime counter
    nfail = 0
    nfailmax = 5*np.ceil(n/bsz)
    # For testing purposes, set a maximum number of iterations, don't allow algorithm to fail
    # nfailmax = kmax+1
    for k in range(int(kmax)):
        # Generate random samples for given batch sizes
        Ig = np.random.permutation(I)[:bsz]
        IH = np.random.permutation(I)[:bsz]
        Mvec = lambda v: Hvec(IH,Y,w,v)
        b = gfun(Ig,Y,w)
        normgrad[k] = np.linalg.norm(b)
        s = CG(Mvec, -b, -b, CGimax, rho) #Using CG, find step direction
        # Using linesearch, find step size 
        a = 1
        f0 = fun(Ig,Y,w)
        aux = eta*np.dot(b,s)
        for j in range(jmax+1):
            wtry = w + a*s
            f1 = fun(Ig, Y, wtry)
            if f1 < (f0 + a*aux):
                """
                print('Linesearch: j=%d, f1 =%e, f0=%e' % (j,f1,f0))
                """
                break
            else:
                a = a*gam
        if j<jmax:
            w = wtry
        else:
            nfail += 1
        # Store  function evaluation & runtime
        f[k+1] = fun(I,Y,w)
        toc = time.perf_counter()
        runtime[k] = toc - tic
        """
        print('k = %d, a = %e, f = %e' % (k,a,f[k+1]))
        """
        if nfail > nfailmax:
            f = f[:k+2]
            normgrad = normgrad[:k+1]
            break
    # print('k = %d, a = %e, f = %e' % (k,a,f[k+1]))
    return w,f,normgrad,runtime # Return final w, values of f and grad norm, and runtime