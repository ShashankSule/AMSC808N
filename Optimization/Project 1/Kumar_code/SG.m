function [witer,f,gnorm]= SG(bsz,Y,gfun,fun,w,ss,N,NN)
itermax=10000;
limit=0.005;
ssz=0.1;
[n,~] = size(Y);
f=fun(1:n,Y,w)
g=gfun(1:n,Y,w);
witer=w;
gnorm=norm(g);
f=fun(1:n,Y,w);
iter=0;
%fixed size
if ss==0
    while norm(g)>limit && iter<itermax
        g=gfun(randperm(n,bsz),Y,w);
        w=w-g*ssz;
        f=[f,fun(1:n,Y,w)];
        witer=[witer,w];
        gnorm=[gnorm,norm(g)];
        iter=iter+1;
    end
end
%decreasing step size
if ss==1
     for ii = 1 : NN
            s = ssz/(2^ii);
            nsteps = ceil(N*2^ii/ii);
            for i = 1 : nsteps
                 g=gfun(randperm(n,bsz),Y,w);
                    w=w-g*s;
                    f=[f,fun(1:n,Y,w)];
                witer=[witer,w];
                gnorm=[gnorm,norm(g)];
                
            end
      end
end

