function [c,d1A,h,hp,rhs,exact_sol] = setup()
% f0 = @(y)y.^3;
% f1 = @(y)(1+y.^3).*exp(-1);
% g0 = @(x)x.*exp(-x);
% g1 = @(x)(x+1).*exp(-x);
c = @(x)x + (1 + 3*(x.^2))./(1 + x + (x.^3));
% differential operator is d/dx + c(x) 
% differential operator applied to A(x) = 1, the bdry term
% d2f0 = @(y)6*y;
% d2f1 = @(y)6*y.*exp(-1);
% d2g0 = @(x)(x-2).*exp(-x);
% d2g1 = @(x)(x-1).*exp(-x);
d1A = @(x)0; 
% differential operator applied to B(x) = xNN(x,y,v,W,u)
h = @(x)x;
hp = @(x)1;
% right-hand side
rhs = @(x)x.^3 + 2.*x + (x.^2).*(1 + 3.*(x.^2))./(1 + x + (x.^3));
exact_sol = @(x)x.^2 + (exp(((-x).^2)/2))./(1 + x + (x.^3));
end