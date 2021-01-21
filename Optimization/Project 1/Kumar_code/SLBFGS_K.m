function [point, slope, ngz] = SLBFGS_K(init, s, y, rho,m, Ng, Nh, M, func,...
                                 grad, hess, grank, hrank)

gam = 0.9; % line search step factor
jmax = ceil(log(1e-14)/log(gam)); % max # of iterations in line search
eta = 0.1; % backtracking stopping criterion factor
xi_g = randi(grank, Ng, 1);
g = grad(xi_g,init);
tol = 1e-10;
nor = norm(g);
iter = 1;
x = init;
point = x; 
slope = g; 
ngz = nor;
itermax=1e4
Ng = min(Ng,64);
Nh = min(Nh,64);
while (nor > tol) && (iter < itermax)
    % We compute the direction via the LBFGS routine 
    if iter < m 
        I = 1 : iter;
        p = finddirection(g,s(:,I),y(:,I),rho(I));
    else
        p = finddirection(g,s,y,rho);
    end
    % We compute the step size via lsearch 
    [a,j] = lsearch(x,p,g,func,eta,gam,jmax);
    if j == jmax
        p = -g;
        [a,j] = lsearch(x,p,g,func,eta,gam,jmax);
    end
    % 
    step = a*p;
    xnew = x + step; %xnew is x_k+1; x is x_k 
    snew = step; %snew is sk_k+1 
    %Stochastic step for Hessian 
    if mod(iter,M) == 0
        xi_h = randi(hrank, Nh, 1);
                             %generate random hessian
        ynew = hess(xi_h, xnew, snew); 
    else
        ynew = grad(xi_g,xnew) - g;
        snew = xnew - x;
        rhonew = 1/((snew(:,1)')*ynew(:,1));
    end
    % updating s, y and rho
    s = [s snew];
    y = [y ynew];
    rho = [rho rhonew];
    if size(s,2) > m
        s = s(:, 2:end);
        y = y(:, 2:end);
        rho = rho(:,2:end);
    end
    % preparing for next iteration 
    x = xnew; %x_k+1
    xi_g = randi(grank, Ng, 1); 
    g = grad(xi_g,x); % Random approximation to the gradient at the next
                      % iterate
    nor = norm(g);
    point = [point x];
    slope = [slope g];
    ngz = [ngz nor];
    iter = iter + 1;
end