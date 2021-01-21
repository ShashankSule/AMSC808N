import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
from numpy.linalg import solve, pinv
from scipy.sparse.linalg import svds

def kmeans(data, k, plot = False, num_trials = 1000, method = 'Random'):
    n,d = data.shape
    best_score = 1e12
    for p in range(num_trials):
        if method is 'Random':
            random_clusters = np.random.randint(k, size = n)
            means = np.zeros((k,d))
            for i in range(k):
                means[i] = np.mean(data[random_clusters == i],axis = 0)
        elif method is 'Forgy':
            means = data[np.random.choice(n, size = k, replace = False)]
        cluster = np.zeros(n)
        new_cluster = np.ones(n)
        itr = 0
        while not np.all(new_cluster == cluster):
            cluster = np.copy(new_cluster)
            for i in range(n):
                mean_dist = np.sum((means - data[i])**2, axis = 1)
                new_cluster[i] = np.argmin(mean_dist)

            for j in range(k):
                if data[new_cluster == j].size == 0:
                    break
                means[j] = np.mean(data[new_cluster == j],axis = 0)
            itr +=1
        score = 0
        for i in range(k):
            score += np.sum((means[i] - data[cluster == i])**2)
        if score < best_score:
            best_score = np.copy(score)
            best_cluster = np.copy(cluster)
            best_means = np.copy(means)
    if plot:
        plot_pca = PCA(n_components = 2)
        plot_pca.fit(data)
        print(plot_pca.explained_variance_ratio_)
        plot_data = plot_pca.transform(data)
        plot_means = plot_pca.transform(best_means)
        fig = plt.figure(figsize = (10,10))
        colors = cm.Set1(np.linspace(0,1,k))
        for i in range(k):
            plt.scatter(plot_data[best_cluster == i,0], plot_data[best_cluster == i,1], color = colors[i])
            plt.plot(plot_means[i,0],plot_means[i,1], '*', markersize = 10, color = colors[i])
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.show()
        
    return best_means, best_cluster, itr

def projectedGD(k, A, stepfun, maxiter = 500):
    n,d = A.shape
    W, S, H = svds(A, k = k)
    W = np.abs(W)
    H = np.abs(H)
    W_zeros = np.zeros((n,k))
    H_zeros = np.zeros((k,d))
    resnorm = np.zeros(maxiter+1)
    runtime = np.zeros(maxiter)
    gradnorm = np.zeros(maxiter)
    tic = time.perf_counter()
    res = A - W @ H
    resnorm[0] = np.linalg.norm(res)
    for itr in range(maxiter):
        Wnew = np.maximum(W + stepfun(itr) * res @ H.T, W_zeros)
        Hnew = np.maximum(H + stepfun(itr) * W.T @ res, H_zeros)
        new_res = A - Wnew @ Hnew
        gradnorm[itr] = np.linalg.norm(res - new_res)
        res = np.copy(new_res)
        resnorm[itr + 1] = np.linalg.norm(res)
        W = Wnew
        H = Hnew
        toc = time.perf_counter()
        runtime[itr] = toc - tic
    
    return W, H, resnorm, runtime, gradnorm
        
def LeeSeung(k, A, maxiter = 50):
    n,d = A.shape
    W, S, H = svds(A, k = k)
    W = np.abs(W)
    H = np.abs(H)
    # W = A[:,k]
    # H = np.ones((k,d))
    resnorm = np.zeros(maxiter + 1)
    runtime = np.zeros(maxiter)
    tic = time.perf_counter()
    res = A - W @ H
    resnorm[0] = np.linalg.norm(res)
    for itr in range(maxiter):
        Hnew = (H * (W.T @ A))/(W.T @ W @ H)
        Wnew = (W * (A @ Hnew.T))/(W @ Hnew @ Hnew.T)
        res = A - Wnew @ Hnew
        resnorm[itr+1] = np.linalg.norm(res)
        W = np.copy(Wnew)
        H = np.copy(Hnew)
        toc = time.perf_counter()
        runtime[itr] = toc - tic
    return W, H, resnorm, runtime

def PGD_LS(k, A, stepfun, maxiter = 2000, PG_iter = 1000):
    n,d = A.shape
    W, S, H = svds(A, k = k)
    W = np.abs(W)
    H = np.abs(H)
    W_zeros = np.zeros((n,k))
    H_zeros = np.zeros((k,d))
    resnorm = np.zeros(maxiter+1)
    runtime = np.zeros(maxiter)
    gradnorm = np.zeros(maxiter)
    tic = time.perf_counter()
    res = A - W @ H
    resnorm[0] = np.linalg.norm(res)
    for itr in range(PG_iter):
        Wnew = np.maximum(W + stepfun(itr) * res @ H.T, W_zeros)
        Hnew = np.maximum(H + stepfun(itr) * W.T @ res, H_zeros)
        new_res = A - Wnew @ Hnew
        gradnorm[itr] = np.linalg.norm(res - new_res)
        res = np.copy(new_res)
        resnorm[itr + 1] = np.linalg.norm(res)
        W = Wnew
        H = Hnew
        toc = time.perf_counter()
        runtime[itr] = toc - tic
    for itr in range(PG_iter, maxiter):
        Hnew = (H * (W.T @ A))/(W.T @ W @ H)
        Wnew = (W * (A @ Hnew.T))/(W @ Hnew @ Hnew.T)
        new_res = A - Wnew @ Hnew
        gradnorm[itr] = np.linalg.norm(res - new_res)
        res = np.copy(new_res)
        resnorm[itr+1] = np.linalg.norm(res)
        W = np.copy(Wnew)
        H = np.copy(Hnew)
        toc = time.perf_counter()
        runtime[itr] = toc - tic
    
    return W, H, resnorm, runtime, gradnorm

def LRF_alt(k, A, reg, maxiter = 100):
    n,d = A.shape
    rows, columns = np.where(np.isnan(A))
    p_omega = np.ones((n,d))
    p_omega[rows, columns] = 0
    A[rows, columns] = 0
    X, S, Y = svds(A, k = k)
    X = np.abs(X)
    Y = np.abs(Y.T)
    resnorm = np.zeros(maxiter + 1)
    runtime = np.zeros(maxiter)
    res = p_omega*(A - X @ Y.T)
    resnorm[0] = np.linalg.norm(res)
    tic = time.perf_counter()
    for itr in range(maxiter):
        Xnew = np.zeros((n,k))
        for i in range(n):
            p_i = p_omega[i]
            p_Y = p_i.reshape(-1,1) * Y
            Xnew[i] = solve(p_Y.T @ p_Y + reg, p_Y.T @ A[i])
            
        Ynew = np.zeros((d,k))
        for j in range(d):
            p_j = p_omega[:,j]
            p_X = p_j.reshape(-1,1) * Xnew
            Ynew[j] = solve(p_X.T @ p_X + reg, p_X.T @ A[:,j])
            
        res = p_omega*(A - Xnew @ Ynew.T)
        resnorm[itr+1] = np.linalg.norm(res)
        toc = time.perf_counter()
        runtime[itr] = toc - tic
        X = np.copy(Xnew)
        Y = np.copy(Ynew)
        
    return X, Y, resnorm, runtime

def LR_nuclear(A, reg, maxiter = 100):
    n,d = A.shape
    rows, columns = np.where(np.isnan(A))
    p_omega = np.ones((n,d))
    p_omega[rows, columns] = 0
    A[rows, columns] = 0
    X, S, Y = svds(A, k = 3)
    M = np.abs(X) @ np.abs(Y)
    resnorm = np.zeros(maxiter + 1)
    runtime = np.zeros(maxiter)
    res = p_omega*(A - M)
    resnorm[0] = np.linalg.norm(res)
    tic = time.perf_counter()
    for itr in range(maxiter):
        U = np.array([])
        Vt = np.array([])
        S = np.array([])
        update_red = M + p_omega*(A - M)
        u, sigma, vt = svds(update_red, k = 1)
        if sigma[0] - reg <= 0:
            break
        while sigma[0] - reg > 0:
            if U.size == 0:
                U = np.copy(u)
                Vt = np.copy(vt)
            else:
                U = np.append(U, u, axis = 1)
                Vt = np.append(Vt, vt, axis = 0)
            S = np.append(S,sigma[0])
            update_red += -np.outer(u*sigma[0], vt)
            u, sigma, vt = svds(update_red, k = 1)
        Mnew = (U * S) @ Vt
        res = p_omega*(A - Mnew)
        resnorm[itr+1] = np.linalg.norm(res)
        toc = time.perf_counter()
        runtime[itr] = toc - tic
        M = np.copy(Mnew)
    if itr != maxiter - 1:
        resnorm = resnorm[:itr+1]
        runtime = runtime[:itr]
    return M, resnorm, runtime

def ColumnSelect(A, k, c):
    U, S, Vt = svds(A, k = k)
    ptemp = c*np.sum(Vt**2,axis = 0)/k
    eta = np.random.rand(ptemp.size)
    ind = np.where(eta < ptemp)
    C = A[:, ind[0]]
    return C

def CUR(A, k, a):
    c = a*k
    C = ColumnSelect(A,k,c)
    R = ColumnSelect(A.T,k,c)
    U = pinv(C) @ A @ pinv(R.T)
    return C, U, R
    
    
    
    
        