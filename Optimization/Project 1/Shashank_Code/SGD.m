function [iters,gradz,ngz,steps] = SGD(init,grad,step,rank,...
                                 averaging, batchsize, routine,...
                                 func, m0,schedule)
    % init--initial guess
    % grad--function that evaluates gradient of f at current iterate
    % step--initial step size set to 0.3
    % rank--number of terms in the expansion of f
    % batchsize--batch size 
    % m0--length of first item in schedule
    % schedule--length of schedule
    % Default m0 = 10 and default schedule = 15
    % iters--stores x_k 
    % gradz--stores g(xi_k, x_k)
    % averaging--runs expectation on each step
    % This is a decreasing stepsize routine
    n = rank;
    N = m0;
    NN = schedule;
    x = init; 
    iters = func(x);
    k = randperm(n,batchsize);
    g = grad(k,x);
    gradz = g;
    nor = norm(g);
    ngz = nor; 
    tol = 1e-10;
    iter = 1;
    steps = step;
    if averaging 
        if (routine == 1)
            % Constant stepsize routine 
            while ((nor > tol) && (iter < 10000))
                    k = randperm(n,batchsize);
                    g = grad(k,x);
                    for j=2:1000
                        k = randperm(n,batchsize);
                        gr = grad(k,x);
                        g = g+gr;
                    end
                x = x - step*0.001*g;
                iters = [iters, func(x)];
                gradz = [gradz, g];
                steps = [steps step];
                nor = 0.001*norm(g);
                ngz = [ngz nor];
                iter = iter + 1;
            end
        elseif (routine == 2)
            % Decreasing stepsize routine 
            for ii = 1 : NN
             s = step/(2^ii);
             nsteps = ceil(N*2^ii/ii);
                for i = 1 : nsteps
                    g = grad(randperm(n,batchsize),x);
                    for j=2:1000
                        k = randperm(n,batchsize);
                        gr = grad(k,x);
                        g = g+gr;
                    end
                    x = x - s*0.001*g;
                    iters = [iters, func(x)];
                    gradz = [gradz, g];
                    steps = [steps s];
                    ngz = [ngz 0.001*norm(g)];
                end
            end
        else
                error('\a Enter 1 for constant stepsize and 2 for decreasing');
        end
        
    else
        if routine == 1
                    % Constant stepsize routine 
            while ((nor > tol) && (iter < 10000))
                    k = randperm(n,batchsize);
                    g = grad(k,x);                     
                x = x - step*g;
                iters = [iters, func(x)];
                gradz = [gradz, g];
                steps = [steps step];
                nor = norm(g);
                ngz = [ngz nor];
                iter = iter + 1;
            end
        elseif routine == 2
            % Decreasing stepsize routine 
            for ii = 1 : NN
             s = step/(2^ii);
             nsteps = ceil(N*2^ii/ii);
                for i = 1 : nsteps
                    k = randperm(n,batchsize);
                    g = grad(k,x);
                    x = x - s*g;
                    iters = [iters, func(x)];
                    gradz = [gradz, g];
                    ngz = [ngz norm(g)];
                    steps = [steps s];
                end
            end
        else
            error('\a Error in stepsize routine');
        end
    end
    
end