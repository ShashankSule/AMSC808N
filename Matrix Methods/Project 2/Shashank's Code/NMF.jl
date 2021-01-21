module  NMF

using LinearAlgebra
using Plots
using Statistics
using DataFrames
using CSV
using Clustering
using Images
using BenchmarkTools
using LowRankApprox

export PGD, LS

struct pgd
    W::Array{Float64,2} # W 
    H::Array{Float64,2} # H
    err::Array{Float64,1} # vector of errors
    grad::Array{Float64,1} # vector of norms of gradients 
    time::Array{Float64,1} # vector of time taken 
end

struct ls
    W::Array{Float64,2} # W 
    H::Array{Float64,2} # H
    err::Array{Float64,1} # vector of errors
    steperrW::Array{Float64,1} # vector of norms of stepsize of W 
    steperrH::Array{Float64,1} # vector of norms of stepsize of H
    #grad::Array{Float64,1} # vector of norms of gradients 
    iter::Int64 # number of iterations taken
    time::Array{Float64,1} # time taken
end

function PGD(A, 
    W::Array{Float64,2}, 
    H::Array{Float64,2}, 
    α::Float64, itermax=5000, 
    tol::Float64=1e-4,
    diagnostics::Bool=false)
# Input: 
# A--Data matrix
# W,H--initial W,H s.t A ≈ WH 
# α--Step Size 
# itermax--maximum iterations (default set to 100)
# tol--error tolerance (default set to 10⁻⁶)
err = Array{Float64,1}();
grad = Array{Float64,1}();
time = Array{Float64,1}();
R = A - W*H;
for i=1:itermax

t = @elapsed  begin
        if 0.5*norm(R) < tol 
            break;
    end
    # PGD Update
    W = max.(W + α*R*H',0);
    H = max.(H + α*(W')*R,0);
    R = A - W*H;
    end  

#Diagnostics
push!(err, norm(R));
push!(grad, norm(R*H') + norm(W'*R));
push!(time, t);

if diagnostics
    print("Iteration = ",i,"\n Error = ",err,"\n Gradient norm = ",grad,
    "Time taken =", t)
    println("\n------------------------------")
end


end
return pgd(W, H, err,grad,time)
end

function LS(A, 
    W_init::Array{Float64,2}, 
    H_init::Array{Float64,2}, 
    itermax=1e6, 
    tol::Float64=1e-4,
    diagnostics::Bool=false)
# Input: 
# A--Data matrix
# W,H--initial W,H s.t A ≈ WH 
# itermax--maximum iterations (default set to 100)
# tol--error tolerance (default set to 10⁻⁶)

#initialization
W = W_init;
H = H_init; 
err = Array{Float64,1}();
steperrW = Array{Float64,1}();
steperrH = Array{Float64,1}();
time = Array{Float64,1}();
iter = 1;
R = A - W*H;
while norm(R) > tol && iter < itermax

    t = @elapsed begin
       #LS Update
    W = (W.*(A*H'))./(W*H*H');
    H = (H.*(W'*A))./(W'*W*H);
    R = A - W*H; 
    end 

    #Diagnostics 
    push!(err, 0.5*norm(R));
    push!(steperrW, norm(W./(W*H*H')));
    push!(steperrH, norm(H./((W')*W*H)));
    push!(time, t);
    iter = iter+1; 

    if diagnostics
        print("Iteration = ",iter,
              "\n Norm of W step = ", norm(W./(W*H*H')),
              "\n Norm of H step = ", norm(H./((W')*W*H)),
              "\n Time taken = ",t);
        println("\n----------------------------------------")
    end

end

return ls(W, H, err, steperrW, steperrH, iter,time) 
end

end