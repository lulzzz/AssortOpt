This directory contains optimization and estimation methods for consider-then-choose choice models discussed in the paper at (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2618823)

## Dependencies

Julia 0.4.3

LightGraphs 0.5.0

Distributions 0.10.1+

DataFrames 0.8.2

JuMP 0.14.0

NPZ 0.1.0

Gurobi 0.2.3

Ipopt 0.2.4

## Types

`Instance` is the fundamental type, describing all parameters of an instance of the unique-ranking model: N, K, prices, consideration sets, probabilities and graph representation. A constructor and random generative model is provided.

`InstanceCvx` combines the "Instance" type with additional fields: ranking functions, ground truth MMNL models, assortment data, learning state. Various constructors (random, from data) are provided

The `Subproblem` type corresponds to one node of the recursion. It describes each as a bit array encoding products and consideration sets.

The `recursiveGraph` composite contains all information relative to the unique-ranking recursion: computational tree, dynamic programming values and decisions.


## Functional description

`subgraph.jl`:  optimized methods for connected subgraphs decomposition (avoiding costly initializations of new instances). Overloads some LightGraphs methods. The computations are conducted on a master graph using oracle calls to the residual subproblem. TO DO: once subgraph is small enough, initialize a new instance.

`dynamicprog.jl`:  methods to run the unique-ranking dynamic program. The second-pass is solved through LP. TO DO: backward recursion

`dynamicprog-cvx.jl`: specialized state space collapse for the quasi-convex model and MIP formulation.

`learning-cvx.jl`: methods for sequential greedy learning of quasi-convex preference lists, L1 and L2 calibration, dual information, hyper-parameter selection. TO DO: batching ideas, bagging

`learning-mmnl.jl`:  methods for exact and approximate maximum likelihood inference of the MMNL model.

Usage examples are provided in `main.jl`.
