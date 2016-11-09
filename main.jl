using LightGraphs
import LightGraphs: components
using JuMP,Gurobi,Distributions,DataFrames,Ipopt,NPZ

# Base instance
include("instancegen.jl")

# Unique-ranking DP
include("subgraph.jl")
include("dynamicprog.jl")

# Cvx Instance
include("instancegen-cvx.jl")

# Cvx DP & MIP solver
include("dynamicprog-cvx.jl")

function run_computational()
  time_c = zeros(3,4,2,20)
  for (ib,b) in enumerate([0.3,0.5,0.7])
    for (ik,k) in enumerate([1000,1500,2000,2500])
      println(b,"-",k)
      for iter = 1:20
        println(iter)
        I = Instance(b,k,20)
        Initialization!(I)
        R = recursiveGraph(I)
        tic()
        iteration(R)
        t = toq()
        r = solveLP(R)
        time_c[ib,ik,1,iter] = time_c[ib,ik,1,iter] + t + r
        ICvx = InstanceCvx(I)
        r = MIPsolver(ICvx)
        time_c[ib,ik,2,iter] = time_c[ib,ik,2,iter] + r
      end
      npzwrite("FinalRuns/computational-runningtime-cs.npz",time_c)
    end
  end
end


function run_computational_cvx()
  time_c = zeros(4,5,2,50)
  for (iN,N) in enumerate([20,40,50,80])
    for (ik,k) in enumerate([500,1000,1500,2000,2500])
      println(N,"-",k)
      for iter = 1:50
        println(iter)
        I = InstanceCvx(N,k,1)
        Initialization_CVX!(I,0.3)
        tic()
        dynamicprogCvx(I)
        t = toq()
        time_c[iN,ik,1,iter] = time_c[iN,ik,1,iter] + t
        r = MIPsolver(I)
        time_c[iN,ik,2,iter] = time_c[iN,ik,2,iter] + r
      end
      npzwrite("FinalRuns/computational-runningtime-cvx.npz",time_c)
    end
  end
end
