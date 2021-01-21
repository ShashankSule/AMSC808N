import numpy as np
from finddirection import finddirection
from linesearch import linesearch
import time

def SLBFGS(fun, gfun, Y, x0, stepmethod = 'linesearch', m = 5, stepsperHupdate = 10, batchsizeg = 64, batchsizeH = 100, maxiter = 500, Hevalmethod = 'onestep'):
    # Stochastic L-BFGS routing for computing minimum
    #  fun - loss function
    #  gfun - loss function gradient
    #  Y - Data array
    #  x0 - initial guess for weights
    #  stepmethod - method for computing step size, default is linesearch, otherwise is set to a function 
    #  m - size of memory for Hessian approximation
    #  stepsperHupdate - number of iterations between each Hessian update
    #  batchsizeg/H - batchsizes for computation of gradient/Hessian
    #  maxiter - maximum number of iterations
    #  Hevalmethod - method for computing H update, can either be over onestep or stepsperHupdate
    
    n = Y.shape[0]
    # Set parameters for line search
    gam = 0.9
    eta = 0.5
    tol = 1e-10
    jmax = round(np.ceil(np.log(1e-14)/(np.log(gam))))
    
    # Initialize arrays for storing s, y, rho, f, gradnorm, x at each iteration, and runtime
    s = np.zeros((x0.size, m))
    y = np.zeros((x0.size, m))
    rho = np.zeros(m)
    f = np.zeros(maxiter+1)
    gradnorm  = np.zeros(maxiter+1)
    xiter = np.zeros((maxiter+1,x0.size))
    runtime = np.zeros(maxiter+1)
    tic = time.perf_counter()
    Iall = np.arange(n) #Array for all indices
    
    # Evaluate initial function values
    f[0] = fun(Iall,Y,x0)
    xiter[0] = x0
    bszg = np.minimum(batchsizeg, n) #Set batchsizes to max if batchsizes are too large for data
    bszH = np.minimum(batchsizeH, n)
    
    # Set initial samples for gradient and Hessian computation
    Ig = np.random.permutation(n)[:bszg]
    IH = np.random.permutation(n)[:bszH]
    g = gfun(Ig,Y,x0) # Compute grad for initial step
    gH = gfun(IH,Y,x0) # Compute grad for initial Hessian
    gradnorm[0] = np.linalg.norm(g)
    # Get initial step size
    if stepmethod is 'linesearch':
        linefun = lambda x: fun(Ig, Y, x)
        a, j = linesearch(x0, -g, g, linefun, eta, gam, jmax)
    else:
        a = stepmethod(0)
    # Compute first step
    xnew = x0 - a*g
    toc = time.perf_counter() # Store runtime
    runtime[0] = toc - tic
    gnew = gfun(Ig, Y, xnew) # Compute new gradient
    gnewH = gfun(IH, Y, xnew) # Compute new gradient for Hessian and store s, y and rho
    s[:,0] = xnew - x0
    y[:,0] = gnewH - gH
    rho[0] = 1/np.dot(s[:,0],y[:,0])
    # Begin next iteration and store values
    x = xnew
    g = gnew
    nor = np.linalg.norm(g)
    xiter[1] = x
    gradnorm[1] = nor
    f[1] = fun(Iall, Y,x)
    itr = 1
    mitr = 1
    # while nor > tol and itr < maxiter: # End iteration if norm(g) < tol
    while itr < maxiter:
        if mitr < m: #If we have yet to fill up s, y, and rho to size m
            p = finddirection(g, s[:,:mitr], y[:,:mitr],rho[:mitr])
        else:
            p = finddirection(g,s,y,rho)
        # Find step size
        if stepmethod is 'linesearch':
            linefun = lambda x: fun(Ig, Y, x)
            a,j = linesearch(x,p,g,linefun,eta,gam,jmax)
            if j == jmax:
                p = -g
                a,j = linesearch(x,p,g,linefun,eta,gam,jmax)
        else:
            a = stepmethod(itr)
        step = a*p # Compute next step
        xnew = x + step
        xiter[itr + 1] = xnew
        toc = time.perf_counter()
        runtime[itr] = toc - tic
        f[itr + 1] = fun(Iall,Y,xnew)
        if itr % stepsperHupdate == 0: # Every stepsperHupdate iterations, we update H
            IH = np.random.permutation(n)[:bszH] # Get a new, larger sample
            # Evaluate H either over one step or multiple
            if Hevalmethod is 'onestep':
                snew = step
                ynew = gfun(IH, Y, xnew) - gfun(IH, Y, x)
            else:
                snew = xnew - xiter[itr + 1 - stepsperHupdate]
                ynew = gfun(IH, Y, xnew) - gfun(IH,Y,xiter[itr + 1 - stepsperHupdate])
            # Add new value to s,y, and rho and remove oldest column
            s = np.append(snew.reshape((-1,1)), s[:,:m-1], axis = 1)
            y = np.append(ynew.reshape((-1,1)), y[:,:m-1], axis = 1)
            rho = np.append(1/np.dot(snew,ynew), rho[:m-1])
            mitr+=1
        x = xnew
        # Find new random sample and recalculate gradient
        Ig = np.random.permutation(n)[:bszg]
        g = gfun(Ig, Y, x)
        nor = np.linalg.norm(g)
        gradnorm[itr + 1] = nor
        itr += 1
    # If tolerance was reached, truncate results arrays
    if itr < maxiter:
        xiter = xiter[:itr+1]
        gradnorm = gradnorm[:itr+1]
        f = f[:itr+1]
        runtime = runtime[:itr+1]
       
    return xiter, f, gradnorm, runtime
        
        