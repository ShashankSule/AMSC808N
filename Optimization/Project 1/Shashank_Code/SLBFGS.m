function [evals, ngz, stepsizes] = SLBFGS(init, s, y, rho,...
                                 m, Ng, Nh, M, func,...
                                 grad, hess, grank, hrank)
%   init--initial guess
%   m--limited memory constant (default is 5)
%   Ng--batch size for gradient
%   Nh--batch size for hessian
%   S,Y--nonempty dim by m matrices whose columns are s_i and y_i
%   Rho--dim by 1 matrix whose entries are 1/s_i'y_i
%   grad--computing gradient
%   hess--computing hessian 
%   grank--rank of gradient
%   hrank--rank of hessian
%   M--frequency of hessian update
%   stepsizes--stepsizes 
gam = 0.9; % line search step factor
jmax = ceil(log(1e-14)/log(gam)); % max # of iterations in line search
eta = 0.1; % backtracking stopping criterion factor
xi_g = randi(grank, Ng, 1);
g = grad(xi_g,init);
tol = 1e-10;
nor = norm(g);
iter = 1;
x = init;
evals = func(x);
ngz = nor;
stepsizes = 1;
Ng = min(Ng,64);
Nh = min(Nh,64);
NN = 15;
while ((nor > tol) && (iter < 1e4))
    % We compute the direction via the LBFGS routine 
    if iter < m 
        I = 1 : iter;
        p = finddirection(g,s(:,I),y(:,I),rho(I));
    else
        p = finddirection(g,s,y,rho);
    end
    % We compute the step size via ls 
    [a,j] = ls(x,p,g,func,eta,gam,jmax);
    if j == jmax
        p = -g;
        [a,j] = ls(x,p,g,func,eta,gam,jmax);
    end
    % 
    step = a*p;
    xnew = x + step; %xnew is x_k+1; x is x_k 
     %snew is sk_k+1 
    %Stochastic step for Hessian 
    if mod(iter,M) == 0
        xi_h = randperm(hrank, Nh);
        ynew = hess(xi_h, xnew, snew);
                                             
        for k=2:100
                xi_h = randperm(hrank, Nh);
                yz = hess(xi_h, xnew, snew); 
                ynew = yz + ynew; 
        end
        ynew = 0.01*ynew;
        snew = step;
        % updating s, y and rho
        s = [s snew];
        y = [y ynew];
        rho = [rho rhonew];
    end
    
    
    if size(s,2) > m
        s = s(:, 2:end);
        y = y(:, 2:end);
        rho = rho(:,2:end);
    end
    % preparing for next iteration 
    x = xnew; %x_k+1
    xi_g = randi(grank, Ng, 1); 
    g = grad(xi_g,x); 
    for i=2:100
        xi_g = randi(grank, Ng, 1); 
        g = g + grad(xi_g,x);
    end% Random approximation to the gradient at the next
    g = 0.01*g;                  % iterate
    nor = norm(g);
    evals = [evals func(x)];
    ngz = [ngz nor];
    stepsizes = [stepsizes a];
    iter = iter + 1;
 end
sstep = 1;
N = 5;
NN = 10;
% for ii = 1:NN
%     ss = sstep/(2^ii);
%     nsteps = ceil(N*2^ii/ii);
%                 for i = 1 : nsteps
%                     if iter < m 
%                         I = 1 : iter;
%                         p = finddirection(g,s(:,I),y(:,I),rho(I));
%                     else
%                         p = finddirection(g,s,y,rho);
%                     end
%                     
%                     step = ss*p;
%                     xnew = x + step; %xnew is x_k+1; x is x_k 
%                     snew = step; %snew is sk_k+1 
%                     
%                     %Stochastic step for Hessian 
%                     if mod(iter,M) == 0
%                         xi_h = randperm(hrank, Nh);
%                         ynew = hess(xi_h, xnew, snew);
%                                              
%                         for k=2:100
%                             xi_h = randperm(hrank, Nh);
%                             yz = hess(xi_h, xnew, snew); 
%                             ynew = yz + ynew; 
%                         end
%                         ynew = 0.01*ynew;
%                         
%                     else
%                         xi_g = randperm(grank, Ng);
%                         gnew = grad(xi_g, xnew);
%                         for k=2:100
%                             xi_g = randperm(grank, Ng);
%                             gz = grad(xi_g, xnew);
%                             gnew = gnew + gz;
%                         end
%                         gnew = 0.01*gnew;
%                         ynew = gnew - g;
%                         snew = xnew - x;
%                         rhonew = 1/((snew(:,1)')*ynew(:,1));
%                     end
%                     % updating s, y and rho
%                     s = [s snew];
%                     y = [y ynew];
%                     rho = [rho rhonew];
%                     if size(s,2) > m
%                         s = s(:, 2:end);
%                         y = y(:, 2:end);
%                         rho = rho(:,2:end);
%                     end
%                     
%                     % preparing for next iteration 
%                     x = xnew; %x_k+1
%                     g = gnew; % Random approximation to the gradient at the next
%                                       % iterate
%                     nor = norm(g);
%                     evals = [evals func(x)];
%                     ngz = [ngz nor];
%                     stepsizes = [stepsizes ss];
%                     iter = iter + 1;
%                 end
% end

end
