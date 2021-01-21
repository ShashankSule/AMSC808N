function f = fun0(I,Y,w,lam) %Loss function with tikhonov regularization
f = sum(log(1 + exp(-Y(I,:)*w)))/length(I) + 0.5*lam*w'*w; 
end