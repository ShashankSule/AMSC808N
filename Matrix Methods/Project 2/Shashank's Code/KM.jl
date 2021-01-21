module KM

using LinearAlgebra
using Plots
using Statistics
using DataFrames
using CSV
using Clustering
using Images
using BenchmarkTools
using LowRankApprox

export kmnz

struct KMz
    Assigns::Array{Float64,2} # Assignment matrix
    Centers::Array{Float64,2} # matrix of centroids
    assignments::Array{Float64,1} # vector of assignments
    time::Array{Float64,1} # vector of execution times
    err::Array{Float64,1} # vector of errors
    A_int::Array{Float64,1} # initial assignment vector 
    C_init::Array{Float64,2} # final assignment matrix 
end

function kmnz(Data::Array{Float64,2},
              k::Int64,
              diagnostics::Bool=true)
    #Inputs
    #Data--n x d data matrix of n points in Rᵈ
    #k--prescribed number of clusters 

    #Outputs 
    #A -- n x k matrix of assignments 
    #C -- k x d matrix of centers 
    n, d = size(Data);
    C = zeros(k,d);
    A = zeros(n,k);
    ind = Int64.(sample(Float64.(1:size(Data)[1]),k,replace=false));
    Cnew = Data[ind,:];
    iter = 0;
    times = Array{Float64,1}();
    errs = Array{Float64,1}();

    #computing the initial assignments 
    as_init = zeros(n,k);
    for i = 1:n
            
        #compute distances 
        distances = zeros(k)
        for j=1:k
            distances[j] = norm(Cnew[j,:]-Data[i,:]);
        end
        
        #compute assignment
        ass = argmin(distances);
        as_init[i,ass] = 1.0;
    end
    a_init = zeros(n)
    for i=1:n
        a = argmax(as_init[i,:]);
        a_init[i] = a;
    end

    c_init = deepcopy(Cnew);
    while norm(C - Cnew) > 1e-5
        iter = iter + 1;
        C = Cnew; 

        #Compute assignments
        t = @elapsed begin 
            for i = 1:n
            
                #compute distances 
                distances = zeros(k)
                for j=1:k
                    distances[j] = norm(C[j,:]-Data[i,:]);
                end
                
                #compute assignment
                ass = argmin(distances);
                A[i,ass] = 1.0;
            end

                #Update centroids 
                for j=1:k 
                    numpts = sum(A[:,j]);
                    if numpts !=0 
                        Cnew[j,:] = Float64.((1/numpts)*A[:,j]'*Data);
                    end
                end
        end
        
        if diagnostics
            #Diagnostics 
            errs = push!(errs, norm(Data - A*Cnew));
            times = push!(times, t);
            print("Iteration number = ",iter,"\n Norm error = ",norm(Data - A*Cnew),"\n Time elapsed =",t)
            println("\n--------------------------------------------------")
        end

    end

    #compute assignments vector
    assigns = zeros(n);
    for i=1:n
    a = argmax(A[i,:]);
    assigns[i] = a;
    end


    return KMz(A,C,assigns,times,errs,a_init, c_init)
end

end