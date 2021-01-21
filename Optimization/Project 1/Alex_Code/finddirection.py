import numpy as np

def finddirection(g,s,y,rho):
    # Find the descent direction given an approximate Hessian
    # input: g = gradient dim-by-1
    # s = matrix dim-by-m, s[:,i] = x_{k-i+1} - x_{k-i}
    # y = marix dim-by-m, y[:,i] = g_{k-i+1}-g_{k-i}
    # rho is 1-by-m, rho[i] = 1/(s[:,i]'*y[:,i])
    m = s.shape[1]
    a = np.zeros(m)
    for i in range(m):
        a[i] = rho[i]*np.dot(s[:,i],g)
        g = g - a[i]*y[:,i]
    
    gam = np.dot(s[:,0],y[:,0])/(np.dot(y[:,0],y[:,0]))
    g = g*gam
    for i in range(m-1, -1, -1):
        aux = rho[i]*np.dot(y[:,i],g)
        g = g + (a[i] - aux)*s[:,i]
    
    return -g