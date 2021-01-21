function [a,j] = ls(x,p,g,func,eta,gam,jmax)
    a = 1;
    f0 = func(x);
    aux = eta*g'*p;
    for j = 0 : jmax
        xtry = x + a*p;
        f1 = func(xtry);
        if f1 < f0 + a*aux
            break;
        else
            a = a*gam;
        end
    end
end