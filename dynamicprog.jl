##############################################################################
# Manages the computational tree and runs the two-pass algorithms
# Allocates computational resources between  the subproblems
##############################################################################

type recursiveGraph
  # Our Instanceu
  instance::Instance
  # Subproblem to depth array
  depth::Vector{Int64}
  # Unmarked node list (of int)
  unmarked_nodes::Vector{Subproblem}
  # Decision 3 dimensional Array (subproblem name,decision,size of the list followed by children list)
  decision_graph::Array{Int32,3}
  # Incremental value to each decision
  decision_value::Array{Float64,2}
  # Subproblem value
  DP_value::Vector{Float64}
  # Dictionary from bit name to int name (in preparation of parallelization)
  name_synchronicity::Dict{BitArray{1},Int64}
  # Next name available
  maximal_name::Int64
  # Father nodes
  father_name::Dict{Int64,Vector{Int64}}
  father_decision::Dict{Int64,Vector{Int64}}
  recursiveGraph(I::Instance,
                 d::Vector{Int64},
                 u::Vector{Subproblem},
                 dg::Array{Int32,3},
                 dv::Array{Float64,2},
                 Dv::Vector{Float64},
                 ns::Dict{BitArray{1},Int64},
                 mn::Int64,
                 fn::Dict{Int64,Vector{Int64}},
                 fd::Dict{Int64,Vector{Int64}}
                 ) = new(I,d,u,dg,dv,Dv,ns,mn,fn,fd)
end

function recursiveGraph(I::Instance)
  #=
  Initialization of Data Structures
  =#
  d = zeros(Int64,500000)
  u = Subproblem[]
  dg = zeros(Int32,500000,I.nb_prod,min(I.nb_prod,I.nb_cst_type))
  dv = zeros(Float64,500000,I.nb_prod)
  Dv = zeros(Float64,500000)
  m_n = Int64(2)
  #Now generate the first subproblem
  name = naming(trues(I.nb_prod + I.nb_cst_type))
  S = Subproblem(trues(I.nb_prod + I.nb_cst_type),Int64(1),Int64(1),Int64(1))
  push!(u,S)
  ns = Dict(name => Int64(1))
  fn = Dict{Int64,Vector{Int64}}()
  fd = Dict{Int64,Vector{Int64}}()
  return recursiveGraph(I,d,u,dg,dv,Dv,ns,m_n,fn,fd)
end


function generateSubproblems(S::Subproblem,product::Int64,R::recursiveGraph)
  #=
  Applies one single DP decision,
  Generates children subproblems
  Updates the computational tree
  S <-> Father subproblem
  Product <-> Decision
  R <-> Graph
  =#
  #Indices of the new subgraph
  indices_prod = falses(R.instance.nb_prod)
  indices_prod[product+1:R.instance.nb_prod] = true
  #indices = find( vcat(indices_prod,~R.instance.consideration_sets[:,product])&(S.subgraph)
  #              )
  indices = vcat(indices_prod,~R.instance.consideration_sets[:,product])&(S.subgraph)

  #Indices of consideration sets being allocated
  bool_indices_allocation = (R.instance.consideration_sets[:,product] &
                             (S.subgraph[(R.instance.nb_prod+1):(R.instance.nb_prod+R.instance.nb_cst_type)
                                          ]
                              )
                            )

  #(subgraph,name_dict) = induced_subgraph(R.instance.master_graph,indices)
  #connected_subgraphs = connected_components(subgraph)

  #Computing the connected subgraphs
  connected_subgraphs = connected_components(R.instance.master_graph,indices)

  #Analyzing each new component
  i = 2
  for child =connected_subgraphs
    #if (size(child)[1] > 1) & (maximum(child)>R.instance.nb_prod)
    if (size(child)[1] > 1)
      #child = map!(x->name_dict[x],child)
      child_bool = falses(R.instance.nb_prod + R.instance.nb_cst_type)
      child_bool[child] = true
      name = naming(child_bool)
      if (~haskey(R.name_synchronicity,name))
        R.name_synchronicity[name] = R.maximal_name
        m_n = minimum(child)
        #Next subproblem
        push!(R.unmarked_nodes,Subproblem(child_bool,m_n,R.maximal_name,S.depth+1))
        #Reward-to-go
        R.decision_graph[S.name,product,i] = R.maximal_name
        R.father_name[R.maximal_name] = [S.name]
        R.father_decision[R.maximal_name] = [product]
        R.maximal_name += 1
      else
        #Reward-to-go
        #if r > 1
        aleph = R.name_synchronicity[name]
        R.decision_graph[S.name,product,i] = aleph
        push!(R.father_name[aleph],S.name)
        push!(R.father_decision[aleph],product)
        #end
      end
      i += 1
    end
    #TODO Refinement: Treat the terminating childs
  end
  #Immediate reward
  R.decision_value[S.name,product] = (R.instance.prices[product]*
                                      sum(R.instance.lambdas[bool_indices_allocation]))

  # Update the size of decisions
  R.decision_graph[S.name,product,1] = i - 2
end


function iteration(R::recursiveGraph)
  #=
  Function for allocating computational resources during the first pass - exploration
  =#
  #println(R.unmarked_nodes)
  #Use sizehint! to prepare for the size of the queue
  np = R.instance.nb_prod
  nc = R.instance.nb_cst_type

  while (size(R.unmarked_nodes)[1] > 0)
    ##Regime k >> N
    #for x in find(S.subgraph[1:R.instance.nb_prod])
    #   generateSubproblems(S,x,R)
    #end
    #Regime N >> k
    S = shift!(R.unmarked_nodes)
    #Eliminates redundant products
    find_prod = find(S.subgraph[1:np])
    mat = R.instance.consideration_sets[find(S.subgraph[(np+1):(np+nc)]),find_prod]
    a = []
    for i = 1:(size(mat,2)-1)
      if (mat[:,i] != mat[:,i+1])
        push!(a,i)
      end
    end
    push!(a,size(mat,2))
    for x in find_prod[a]
      generateSubproblems(S,x,R)
    end
  end
end


function solveLP(R::recursiveGraph)
  #=
  Function encoding the second-pass to solve the recursion
  (Alternative to backward recursion)
  =#
  t0 = tic()
  m = Model(solver = GurobiSolver(OutputFlag= 0))
  @defVar(m,Decisions[1:(R.maximal_name-1),1:R.instance.nb_prod]>=0)
  @addConstraint(m,sum{Decisions[1,decision],
                       decision = 1:R.instance.nb_prod ;
                       R.decision_value[1,decision]> 0}
                    == 1.)
  @addConstraints(m, begin
                  sum_to_one[subprob = 2:(R.maximal_name-1)],
                  - sum{
                        Decisions[R.father_name[subprob][u],
                                  R.father_decision[subprob][u]
                                  ],
                        u = 1:size(R.father_decision[subprob],1)
                        } +
                    sum{
                        Decisions[subprob,decision], decision = 1:R.instance.nb_prod ;
                        R.decision_value[subprob,decision]> 0
                        }== 0.
                  end
                  )
  @setObjective(m, Max, vecdot(R.decision_value[1:(R.maximal_name-1),:],Decisions))
  tic()
  status = solve(m)
  return(toq())
end
