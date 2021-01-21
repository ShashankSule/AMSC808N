function [point, slope, memory,gnorm] = lbfgs_K(x0, gfun, func,m)
gam = 0.9; % line search step factor
itermax=1e4;
jmax = ceil(log(1e-14)/log(gam)); % max # of iterations in line search
eta = 0.5;
x = x0;
g = gfun(x);
gnorm(1) = norm(g);
a = lsearch(x,-g,g,func,eta,gam,jmax);
xnew = x - a*g;
gnew = gfun(xnew);
s(:,1) = xnew - x;
y(:,1) = gnew - g;
rho(1) = 1/(s(:,1)'*y(:,1));
x = xnew;
g = gnew;
nor = norm(g);
gnorm(2) = nor;
iter = 1;
limit = 1e-4;
point = x;
slope = g; 
while (nor > limit) && (iter < itermax)
    if iter < m
        I = 1 : iter;
        p = finddirection(g,s(:,I),y(:,I),rho(I));
    else
        p = finddirection(g,s,y,rho);
    end
    [a,j] = lsearch(x,p,g,func,eta,gam,jmax);
    if j == jmax
        p = -g;
        [a,j] = lsearch(x,p,g,func,eta,gam,jmax);
    end
    step = a*p;
    xnew = x + step;
    gnew = gfun(xnew);
    ynew = gnew - g;
    rhonew = 1/(step'*y(:,1));
    s = [s step];
    y = [y ynew];
    rho = [rho rhonew];
    if size(s,2) > m
        s = s(:, 2:end);
        y = y(:, 2:end);
        rho = rho(:,2:end);
    end
    point = [point xnew];
    slope = [slope gnew];
    memory = [s;y];
    x = xnew;
    g = gnew;

    nor = norm(g);
    iter = iter + 1;
    gnorm(iter+1) = nor;
end