##############################################################################
# Defines an Instance type with constructor
##############################################################################

type Instance
  bernouilli_param::Float64 #size of consideration sets
  nb_prod::Int64 #number of products
  nb_cst_type::Int64 #number of customer types
  master_graph::Graph #graph representation
  prices::Vector{Float64} #prices array
  lambdas::Vector{Float64} #probability array
  consideration_sets::BitArray{2} #binary matrix of the consideration sets
  Instance(b,np,nc) = new(Float64(b),Int64(np),Int64(nc),LightGraphs.Graph(),
                        zeros(Float64,np),zeros(Float64,nc),
                        zeros(Bool,nc,np))
end


function Initialization!(I::Instance)
  #=
  Initializes the intance
  Filling (lambdas, prices, consideration_sets) with numerical values using random generators
  =#
  randn!(I.prices)
  I.prices = exp(0.3*sort(I.prices))
  rand!(I.lambdas)
  I.lambdas = I.lambdas/sum(I.lambdas)
  I.consideration_sets = rand(I.nb_cst_type,I.nb_prod) .< I.bernouilli_param

  #Initializes the master graph
  I.master_graph = Graph(Int64(I.nb_prod + I.nb_cst_type))
  for i=1:I.nb_prod
    for j=1:I.nb_cst_type
      if I.consideration_sets[j,i]
        add_edge!(I.master_graph,i,I.nb_prod + j)
      end
    end
  end
end


function naming(subgraph::BitArray{1})
  # Naming hashing (TO DO)
  return subgraph
end
