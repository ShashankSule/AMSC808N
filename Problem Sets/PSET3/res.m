function [r,dr] = res(x,v,W,u,fun,dfun,d2fun,d3fun)
% THIS MODULE OUTPUTS r, dr EVALUATED AT X! 
% % BVP for the Poisson equation is setup here
% %
% % residual functions r and theor derivatives w.r.t. parameters dr 
% % are evaluated in this function
% %
% % computer diff_operator(Psi(x)) - RHS(x)
% % boundary functions
% [~,~,~,~,d2f0,d2f1,d2g0,d2g1,h,hp,hpp,rhs,~] = setup();
% % differential operator is d^2/dx^2 + d^2/dy^2
% % differential operator applied to A(x,y), the bdry term
% d2A = @(x)(1-x(2))*d2g0(x(1)) + x(2)*d2g1(x(1)) + ...
%     (1-x(1))*d2f0(x(2)) + x(1)*d2f1(x(2));
% % differential operator applied to B(x,y) = x(1-x)y(1-y)NN(x,y,v,W,u)
% h1 = h(x(1));
% h2 = h(x(2));
% hp1 = hp(x(1));
% hp2 = hp(x(2));
% [f,fx,fy,fxx,fyy,df,dfx,dfy,dfxx,dfyy] = NN(x,v,W,u,fun,dfun,d2fun,d3fun);
% d2B = hpp*(h1+h2)*f + 2*(hp1*h2*fx + hp2*h1*fy) + h1*h2*(fxx + fyy);
% % residual r = d2A + d2B - RHS
% r = d2A(x) + d2B - rhs(x(1),x(2));
% % % derivative of r w.r.t. parameters
% % dr = hpp*(h1+h2)*df + 2*(hp1*h2*dfx + hp2*h1*dfy) + h1*h2*(dfxx + dfyy);
% %
% IVP for Problem 1 is setup here
%
% residual functions r and theor derivatives w.r.t. parameters dr 
% are evaluated in this function
%
% computer diff_operator(Psi(x)) - RHS(x)
% boundary functions
[c,~,h,hp,rhs, ~] = setup();
% The function is 1 + xNN = A + B 
% differential operator is d/dx + c(x) 
% differential operator applied to A(x) = 1 is 0 + c(x)
cx = c(x);
dA = cx;
% differential operator applied to B(x) = xNN(x,y,v,W,u)
% h'NN + xNN_x + chNN 
h1 = h(x);
hp1 = hp(x);
[f,fx,df,dfx] = NN(x,v,W,u,fun,dfun,d2fun,d3fun);
%d2B = hpp*(h1+h2)*f + 2*(hp1*h2*fx + hp2*h1*fy) + h1*h2*(fxx + fyy);
d1B = hp1*f + h1*fx + cx*h1*f;
% residual r = dA + d1B - RHS
r = dA + d1B - rhs(x);
% derivative of r w.r.t. parameters
dr = hp1*df + h1*dfx + cx*h1*df;



end


