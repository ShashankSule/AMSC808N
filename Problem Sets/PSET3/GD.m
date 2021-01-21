%function [fall,norg] = GD(nt,N,tol,iter_max)
fsz = 16; % fontsize
%%
eta = 0.01;
gam = 0.9;
iter_max = 1000;
tol = 5e-3;
jmax = ceil(log(1e-14)/log(gam)); % max # of iterations in line search
%% setup training mesh
% % Mesh for Problem 5
% % nt = 5;
% t = linspace(0,1,nt+2);
% [xm,ym] = meshgrid(t,t);
% I = 2:(nt+1);
% xaux = xm(I,I);
% yaux = ym(I,I);
% xy = [xaux(:),yaux(:)]';
% Mesh for Problem 1 
nt = 10;
N = 10;
tol = 1e-4;
%iter_max = 1e4; 
t = linspace(0,1,nt+2);
I = 2:(nt+1);
xy = t(I);
%% initial guess for parameters
% N = 10; % the number of hidden nodes
% % Number of params for problem 5
% npar = 4*N;
% 
% Number of params for problem 1
npar = 3*N; % for problem 1 this is 3 because W \in R^{N x 1} 
w = ones(npar,1);
%%
[r,J] = Res_and_Jac(w,xy); % Res & Jacques compute the residue vector 
                           % and the Jacobian at the parameter values of
                           % v = 1_{N x 1}, W = 1_{N x 2}, u = 1_{N x 1}
f = F(r); % F is half norm squared  
g = J'*r;
nor = norm(g);

fprintf('Initially: f = %d, nor(g) = %d\n',f,nor);
%% The trust region BFGS method
tic

iter = 0;
I = eye(length(w));
% quadratic model: m(p) = (1/2)||r||^2 + p'*J'*r + (1/2)*p'*J'*J*p;
norg = zeros(iter_max+1,0);
fall = zeros(iter_max+1,0);
norg(1) = nor;
fall(1) = f;
while nor > tol && iter < iter_max
    % solve the constrained minimization problem using dogleg strategy
    p = -g;
    a = 1;
    aux = eta*g'*p;
    for j = 0 : jmax
        wtry = w + a*p;
        [rtry, Jtry] = Res_and_Jac(wtry,xy);
        f1 = F(rtry);
        if f1 < f + a*aux
            break;
        else
            a = a*gam;
        end
    end
    w = w + a*p;
    [r,J] = Res_and_Jac(w,xy);
    f = F(r);
    g = J'*r;
    nor = norm(g);
    fprintf('iter %d: line search: j = %d, a = %d, f = %d, norg = %d\n',iter,j,a,f,nor);
    iter = iter + 1;
    norg(iter+1) = nor;
    fall(iter+1) = f;
end
fprintf('iter # %d: f = %.14f, |df| = %.4e\n',iter,f,nor);
cputime = toc;
fprintf('CPUtime = %d, iter = %d\n',cputime,iter);
%% visualize the solution
nt = 101;
t = linspace(0,1,nt);
[fun,~,~,~] = ActivationFun();
[v,W,u] = param(w);
[c,~,h,~,~,exact_sol] = setup();
%A = @(x)1;
B = h(t);
NNfun = zeros(nt);
for ii = 1 : nt
        NNfun(ii) = v'*fun(W*t(ii) + u);
end
NNfun = v'*fun(W*t + u);
sol = ones(1,nt) + B.*NNfun;
esol = exact_sol(t);
err = sol - esol;
fprintf('max|err| = %d, L2 err = %d\n',max(max(abs(err))),norm(err(:)));

%%
close all 

figure(1);
plot(t,sol,'r-','DisplayName','Neural Network Solution'); 
hold on;
plot(t,esol,'b-','DisplayName','Actual Solution');
legend();
%set(gca,'Fontsize',fsz);
xlabel('x','Fontsize',fsz);
ylabel('y','Fontsize',fsz);

%
figure(2);
plot(t,err);
%set(gca,'Fontsize',fsz);
xlabel('x','Fontsize',fsz);
ylabel('y','Fontsize',fsz);
%
figure(3);
subplot(2,1,1);
fall(iter+2:end) = [];
plot((1:iter+1)',fall,'Linewidth',2,'Marker','.','Markersize',20);
%grid;
set(gca,'YScale','log','Fontsize',fsz);
xlabel('k','Fontsize',fsz);
ylabel('f','Fontsize',fsz);
subplot(2,1,2);
norg(iter+2:end) = [];
plot((1:iter+1)',norg,'Linewidth',2,'Marker','.','Markersize',20);
%grid;
set(gca,'YScale','log','Fontsize',fsz);
xlabel('k','Fontsize',fsz);
ylabel('|| grad f||','Fontsize',fsz);
%end


%%
function p = cauchy_point(B,g,R)
    ng = norm(g);
    ps = -g*R/ng;
    aux = g'*B*g;
    if aux <= 0
        p = ps;
    else
        p = min(ng^3/(R*aux),1);
    end
end
%%
function f = F(r)
    f = 0.5*r'*r;
end
