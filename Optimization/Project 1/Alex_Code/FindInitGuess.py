import numpy as np

def FindInitGuess(w,A,b,itermax = 10000, step = 0.1):
    relu = lambda w: np.maximum(w,0)
    drelu = lambda w: np.ones((w.size,w.size)) * np.sign(relu(w))
    l = np.sum(relu(b - A @ w))
    itr = 0
    while l > 0 and itr < itermax:
        dl = np.sum(-drelu(b - A @ w) @ A, axis = 0)
        if np.linalg.norm(dl) > 1:
            dl = dl/np.linalg.norm(dl)
        w = w - step * dl
        l = np.sum(relu(b-A@w))
        itr+=1
    lcomp = relu(b - A @ w)
    return w, l, lcomp