import numpy as np
from scipy.linalg import solve, norm
from scipy.sparse.linalg import eigs
from scipy.optimize import lsq_linear
import time

def ASM(x, gfun, Hfun, A, b, W, TOL = 1e-10, itermax = 1000):
    ## Minimization using the active set method (Nocedal & Wright, Section 16.5)
    ## The objective function must be positive definite
    # Solves f(x) --> min s.t. Ax >=b
    # x = initial guess, a column vector
    dim = len(x)
    g = gfun(x)
    H = Hfun(x)
    itr = 0
    m = A.shape[0] #the number of constraints
    # W = working set, the set of active constraints
    I = np.arange(m)
    Wc = np.copy(I) # the compliment of W
    xiter = np.copy(x)
    runtime = np.zeros(itermax)
    tic = time.perf_counter()
    while itr < itermax:
        # compute step p: solve 0.5*p'*H*p + g'*p --> min subject to A(W,:)*p = 0
        if W.size != 0:
            AW = A[W,:]
        else:
            AW = np.array([])
        # fix H if it is not positive definite
        ee = eigs(H, k = 1, which = 'SM', return_eigenvectors=False)
        if np.linalg.norm(ee) < 1e-10:
            lam = -ee + 1
        else:
            lam = 0
        H = H + lam*np.eye(dim)
        # Form the KKT system and solve
        if W.size != 0:
            HmAW = np.concatenate((H, -AW.T),axis = 1)
            AWZ = np.concatenate((AW, np.zeros((W.shape[0],W.shape[0]))),axis = 1)
            M = np.concatenate((HmAW, AWZ))
            RHS = np.concatenate((-g,np.zeros(W.shape[0])))
        else:
            M = np.copy(H)
            RHS = -g
        
        aux = solve(M, RHS)
        p = aux[:dim]
        lm = aux[dim:]
        if norm(p) < TOL: # if step == 0
            if W.size != 0:
                if AW.shape[0] == AW.shape[1]:
                    lm = solve(AW.T, g) # find Lagrange multipliers
                else:
                    result = lsq_linear(AW.T, g)
                    lm = result.x
                
                if np.min(np.real(lm)) >= 0: # if Lagrange multipliers are positive, we are done
                    # the minimizer is one of the corners
                    
                    print('A local solution is found, iter = %d\n' % itr)
                    """
                    print('x = [\n')
                    for xi in x:
                        print('%e\n' % xi)
                    print(']\n')
                    """
                    break
                else: #remove the index of the most negative multiplier from W
                    imin = np.argmin(lm)
                    W = np.setdiff1d(W, W[imin])
                    Wc = np.setdiff1d(I, W)
                
            else:
                print('A local solution is found, iter = %d\n' % itr)
                """
                print('x = [\n')
                for xi in x:
                    print('%e\n' % xi)
                print(']\n')
                """
                break
        else: # if step is nonzero
            alp = 1
            # check for blocking constraints
            Ap = A[Wc,:] @ p
            icand = np.where(Ap < -TOL)[0]
            if icand.size != 0:
                # find step lengths to all possible blocking constraints
                al = (b[Wc[icand]] - A[Wc[icand],:] @ x)/Ap[icand]
                # Find minimal step length that does not exceed 1
                kmin = np.argmin(al)
                alp = np.minimum(1.0,np.amin(al))
            
            x = x + alp*p
            g = gfun(x);
            H = Hfun(x);
            if alp < 1:
                W = np.append(W, Wc[icand[kmin]])
                Wc = np.setdiff1d(I,W)
        
        itr += 1
        xiter = np.vstack((xiter,x))
        toc = time.perf_counter()
        runtime[itr-1] = toc - tic
    if itr == itermax:
        print('Stopped because the max number of iterations %d is performed\n' % itr)
    elif itr < itermax:
        runtime = runtime[:itr]
    return xiter, lm, runtime
        