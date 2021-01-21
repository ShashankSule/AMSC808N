import numpy as np

def CG(Mvec, x,b,kmax,rho):
    r = Mvec(x) - b
    p = -r
    k = 0
    rerr = 1
    normb = np.linalg.norm(b)
    while k < kmax and rerr > rho:
        Ap = Mvec(p)
        a = np.dot(r,r)/np.dot(Ap,p)
        x = x + a*p
        rr = np.dot(r,r)
        r = r + a*Ap
        bet = np.dot(r,r)/rr
        p = -r + bet*p
        k += 1
        rerr = np.linalg.norm(r)/normb
    return x