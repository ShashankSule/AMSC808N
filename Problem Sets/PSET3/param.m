function [v,W,u] = param(par)
N = length(par)/3;
v = par(1 : N);
W = reshape(par(N+1 : 2*N),[N,1]);
u = par(2*N+1 : end);
end