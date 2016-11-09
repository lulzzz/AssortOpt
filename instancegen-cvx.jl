##############################################################################
# Generates instances formed by quasi-convex list;
# Captures data from an MMNL ground truth model &  Infoscout data format
##############################################################################

type InstanceCvx
  instance::Instance
  rankings::Array{Float64,2} #permutations over the products
  left_endpoints::Vector{Int64} #interval structure
  right_endpoints::Vector{Int64} #interval structure
  k_mixture::Int64 #size of the mixture in the MMNL generative model
  source_model::Array{Float64,2} #Ground truth in synthetic instances from MMNL
  dataI_probability::Array{Float64,2} #data: purchase probability
  dataI_assortments::BitArray{2} #data: assortments observed
  dataO_probability::Array{Float64,2} #data: purchase probability
  dataO_assortments::BitArray{2} #data: assortments observed
  X_purchase::Array{Int64,3} #purchase indicators (nc x n_assort x nb_prod)
  res_obj::Array{Float64,2} #purchase indicators (nc x n_assort x nb_prod) = dual information
  data_frame::DataFrame #Data_base
  InstanceCvx(np,nc,km) = new(Instance(0,np,nc),zeros(Float64,nc,np),
                                 zeros(Int64,nc),zeros(Int64,nc),km,zeros(Float64,km,np),
                                 zeros(Float64,1,1),zeros(Bool,1,1),zeros(Float64,1,1),
                                 zeros(Bool,1,1),zeros(Int64,1,1,1), zeros(Float64,1,1),
                                 DataFrame())
  function InstanceCvx(I::Instance)
    #=
    Initializes the convex-instance and data structures from an Instance
    =#
    np = I.nb_prod
    nc = I.nb_cst_type
    rankings = ones((nc,np))
    for i =1:np
      rankings[:,i] = 10e-5 + np - i
    end
    new(I,rankings,
        zeros(Int64,nc),zeros(Int64,nc),1,zeros(Float64,1,np),
        zeros(Float64,1,1),zeros(Bool,1,1),zeros(Float64,1,1),
        zeros(Bool,1,1),zeros(Int64,1,1,1), zeros(Float64,1,1),
        DataFrame()
        )
  end
end


function truncate_instance!(ICM::InstanceCvx, nb_prod::Int64)
  #=
  Limits the number of products and adjusts the instance accordingly
  =#
  ICM.instance.nb_prod = nb_prod
  ICM.instance.prices = ICM.instance.prices[1:nb_prod]
  ICM.instance.prices = ICM.instance.prices[1:nb_prod]
  ICM.instance.consideration_sets = ICM.instance.consideration_sets[:,1:nb_prod]
  ICM.rankings = ICM.rankings[:,1:nb_prod]
  ICM.source_model = ICM.source_model[:,1:nb_prod]
end

function Initialization_CVX!(ICM::InstanceCvx,P::Float64)
  #=
  Initializes the prices with a convex-instance
  P <-> Scale of log-normal generator
  =#
  #Transforms left-right endpoints into consideration sets
  for k = 1:ICM.instance.nb_cst_type
    #Increasing part
    left_list = sort(sample(1:(ICM.instance.nb_prod-1),floor(Int,rand()*ICM.instance.nb_prod),replace = false))
    #Decreasing part
    right_list = sort(setdiff(1:(ICM.instance.nb_prod-1),left_list), rev = true)
    #left end point & right end point
    a =floor(Int,rand()*(ICM.instance.nb_prod-1)+1)
    b= floor(Int,rand()*(ICM.instance.nb_prod-1)+1)
    ICM.left_endpoints[k] = min(a,b)
    ICM.right_endpoints[k] = max(a,b)
    ICM.rankings[k,:] = vcat(left_list,right_list,[0])
    ICM.instance.consideration_sets[k,(ICM.left_endpoints[k]):(ICM.right_endpoints[k])] = 1
  end
  ICM.instance.prices = exp(P*randn(ICM.instance.nb_prod))
  ICM.instance.lambdas = rand(ICM.instance.nb_cst_type)
  ICM.instance.lambdas = ICM.instance.lambdas/(sum(ICM.instance.lambdas))
end


function Initialization_MMNL!(ICM::InstanceCvx,B::Float64,S::Float64)
  #=
  Initializes the prices and ground truth MMNL model
  B <-> No-purchase weight
  S <-> Scale parameter in log-normal price generator
  =#
  #Initializes the intance, filling with numerical values
  randn!(ICM.instance.prices)
  ICM.instance.prices = exp(0.3*sort(ICM.instance.prices))
  ICM.source_model = exp(-S*randn(ICM.k_mixture,ICM.instance.nb_prod))
  ICM.source_model[:,ICM.instance.nb_prod] = B*ICM.source_model[:,ICM.instance.nb_prod]
end


function transform_data(name::ASCIIString)
  #=
  Converts raw Infoscout category purchase data to a sales & price dataset
  =#
  println("Forming the master database")
  df_1 = readtable(string("Raw/longitudinal-considerationset-",name,"2015.txt"))
  df_1 = df_1[df_1[:Brand].!= "unknown",:]
  df_1 = df_1[~isna(df_1[:Brand]),:]
  #df = df_1[1:999,:]
  # df_2 = readtable(string("Raw/longitudinal-considerationset-",name,"2016.txt"))
  # df_2 = df_2[df_2[:Brand].!= "unknown",:]
  # df_2 = df_2[~isna(df_2[:Brand]),:]
  # df = vcat(df_1,df_2)
  df = df_1
  metrics = unstack(df, :Trip_ID, :Metrics, :Metric)
  df = df[1:3:size(df,1),1:18]
  df = join(df,metrics,on =:Trip_ID,kind= :left)
  #println(head(df))
  println("Converting the data")
  df[symbol("Item Dollars / Unit")] =  map(x-> parse(Float64,x[2:endof(x)]),df[symbol("Item Dollars / Unit")])
  df[symbol("Item Dollars")] =  map(x-> parse(Float64,x[2:endof(x)]),df[symbol("Item Dollars")])
  df[symbol("Item Units")] =  map(x-> parse(Float64,x[1:endof(x)]),df[symbol("Item Units")])

  println("Generating the sales matrix")
  brands = unique(df[:Parent_Brand])
  assortments = unique(df[[:Retailer,:State]])
  for b in brands
    assortments[symbol(b)] = 0.0
  end

  for i = 1:size(assortments)[1]
    #println(i)
    for row in find((df[:Retailer] .== assortments[i,:Retailer]) & (df[:State] .== assortments[i,:State]))
      assortments[i,symbol(df[row,:Parent_Brand])] = assortments[i,symbol(df[row,:Parent_Brand])] + df[row,symbol("Item Units")]
    end
  end

  writetable(string("Filtered/sales-",name,".csv"), assortments)

  println("Extracting the average prices")
  prices = unique(df[[:Retailer,:State]])
  for b in brands
    prices[symbol(b)] = 0.0
  end

  for i = 1:size(assortments)[1]
    for row in find((df[:Retailer] .== assortments[i,:Retailer]) & (df[:State] .== assortments[i,:State]))
      prices[i,symbol(df[row,:Parent_Brand])] = prices[i,symbol(df[row,:Parent_Brand])] + df[row,symbol("Item Units")]*df[row,symbol("Item Dollars / Unit")]
    end
  end

  prices = colwise(sum,prices[:,3:size(prices)[2]])./colwise(sum,assortments[:,3:size(prices)[2]])
  price_data_frame = DataFrame()
  for (i,b) in enumerate(brands)
    price_data_frame[symbol(b)]= prices[i][1]
  end
  writetable(string("Filtered/prices-",name,".csv"), price_data_frame)
end


function filter_data_from_frame(ICM::InstanceCvx,name::ASCIIString)
  #=
  Restricts attention to data with significant samples (TO DO: hard coded parameters)
  =#
  #In sample data
  ICM.data_frame = readtable(string("Filtered/sales-",name,".csv"))
  #println(names(ICM.data_frame)[26])
  # Filter out products with small sales ; hard coded param
  cols = find(map(x -> x[1],colwise(sum,ICM.data_frame[:,3:(size(ICM.data_frame)[2])])) .> 1000)
  ICM.data_frame = ICM.data_frame[:,vcat([1,2],map(x-> x+2,cols))]
  # Filter out too small assortments ; hard coded param
  rows_support = find([sum(Array(ICM.data_frame[i,3:size(ICM.data_frame,2)]).>=1).>10 for i in 1 : size(ICM.data_frame,1)])
  ICM.data_frame = ICM.data_frame[rows_support,:]
  # Filter out assortment without enough samples; hard coded param
  rows_support = find([sum(Array(ICM.data_frame[i,3:size(ICM.data_frame,2)])).>1000 for i in 1 : size(ICM.data_frame,1)])
  ICM.data_frame = ICM.data_frame[rows_support,:]

  #Prices
  price_frame = readtable(string("Filtered/prices-",name,".csv"))
  price_frame = price_frame[:,names(ICM.data_frame[:,3:size(ICM.data_frame,2)])]

  #Central order of rankings
  truncate_instance!(ICM, size(ICM.data_frame,2) -1)
  println(ICM.instance.nb_prod)
  vector_price = reshape(Array(price_frame),(ICM.instance.nb_prod-1,1))[:,1]
  ind_perm = sortperm(vector_price)
  ICM.data_frame = ICM.data_frame[:,vcat([1,2],map(x -> x+2,ind_perm))]
  ICM.instance.prices = vcat(sort(vector_price),[0])
end


function generate_data_from_frame(ICM::InstanceCvx,n_assort::Int64,nO_assort::Int64, r::Array{Int64,1}=zeros(1))
  #=
  Generate the Out-of-sample and In-sample datasets using the data frame
  =#
  #In sample data
  ICM.dataI_assortments = zeros(Bool,n_assort,ICM.instance.nb_prod)
  ICM.dataI_probability = zeros(Float64,n_assort,ICM.instance.nb_prod)

  if sum(r) == 0
    remaining_assortments = sample(Array(1:size(ICM.data_frame,1)),nO_assort, replace = false)
  else
    remaining_assortments = r
  end

  assortment_lists = setdiff(Array(1:size(ICM.data_frame,1)),remaining_assortments)
  ICM.dataI_assortments[:,ICM.instance.nb_prod] = 1

  for i in 1:n_assort
    ICM.dataI_assortments[i,1:(ICM.instance.nb_prod-1)] = Array(ICM.data_frame[assortment_lists[i],3:size(ICM.data_frame,2)]) .> 0
    purchase_proba = sum(ICM.source_model[1,1:(ICM.instance.nb_prod-1)].* ICM.dataI_assortments[i,1:(ICM.instance.nb_prod-1)])
    purchase_proba = purchase_proba/(purchase_proba + ICM.source_model[1,ICM.instance.nb_prod])
    ICM.dataI_probability[i,1:(ICM.instance.nb_prod-1)] = purchase_proba/(sum(Array(ICM.data_frame[assortment_lists[i],3:size(ICM.data_frame,2)])))*
                                                          Array(ICM.data_frame[assortment_lists[i],3:size(ICM.data_frame,2)])
    ICM.dataI_probability[i,ICM.instance.nb_prod] = 1-purchase_proba
    ICM.dataI_probability[i,find(ICM.dataI_probability[i,:])] = max(0.001,ICM.dataI_probability[i,find(ICM.dataI_probability[i,:])])
    ICM.dataI_probability[i,:] = ICM.dataI_probability[i,:]/(sum(ICM.dataI_probability[i,:]))
  end

  assortment_lists = sample(remaining_assortments,nO_assort, replace = false)

  #Out-of-sample data
  ICM.dataO_assortments = zeros(Bool,nO_assort,ICM.instance.nb_prod)
  ICM.dataO_probability = zeros(Float64,nO_assort,ICM.instance.nb_prod)
  ICM.dataO_assortments[:,ICM.instance.nb_prod] = 1

  for i in 1:nO_assort
    ICM.dataO_assortments[i,1:(ICM.instance.nb_prod-1)] = Array(ICM.data_frame[assortment_lists[i],3:size(ICM.data_frame,2)]) .> 0
    purchase_proba = sum(ICM.source_model[1,1:(ICM.instance.nb_prod-1)].* ICM.dataO_assortments[i,1:(ICM.instance.nb_prod-1)])
    purchase_proba = purchase_proba/(purchase_proba + ICM.source_model[1,ICM.instance.nb_prod])
    ICM.dataO_probability[i,1:(ICM.instance.nb_prod-1)] = purchase_proba/(sum(Array(ICM.data_frame[assortment_lists[i],3:size(ICM.data_frame,2)])))*
                                                          Array(ICM.data_frame[assortment_lists[i],3:size(ICM.data_frame,2)])
    ICM.dataO_probability[i,ICM.instance.nb_prod] = 1-purchase_proba
    ICM.dataO_probability[i,find(ICM.dataO_probability[i,:])] = max(0.001,ICM.dataO_probability[i,find(ICM.dataO_probability[i,:])])
    ICM.dataO_probability[i,:] = ICM.dataO_probability[i,:]/(sum(ICM.dataO_probability[i,:]))
  end

end


function generate_data!(ICM::InstanceCvx,n_assort::Int64,nO_assort::Int64, a::Array{Int64,1}=zeros(Int64,1),b::BitArray{2}=falses(1,1))
  #=
  Generates synthetic data based on the MMNL model
  n_assort <-> number of in-sample assortments
  nO_assort <-> number of out-of-sample assortments
  a <-> remaining assortments (or void)
  b <-> pre-generated list of assortments
  =#
  if sum(a) == 0
    remaining_assortments = sample(Array(1:(n_assort+nO_assort)),nO_assort, replace = false)
    pool_assortments = rand((n_assort+nO_assort,ICM.instance.nb_prod)).< 0.5
  else
    remaining_assortments = a
    pool_assortments = b
  end

  assortment_lists = setdiff(Array(1:(n_assort+nO_assort)),remaining_assortments)


  #In sample data
  ICM.dataI_assortments = zeros(Bool,n_assort,ICM.instance.nb_prod)
  ICM.dataI_probability = zeros(Float64,n_assort,ICM.instance.nb_prod)
  for d = 1:n_assort
    ICM.dataI_assortments[d,:] = pool_assortments[assortment_lists[d],:]
    ICM.dataI_assortments[d,ICM.instance.nb_prod] =  1
    for i= 1:ICM.instance.nb_prod
      if ICM.dataI_assortments[d,i]
        for km = 1:ICM.k_mixture
          #Computing MMMNL probabilities
          ICM.dataI_probability[d,i] = ICM.dataI_probability[d,i] +
                                      ICM.source_model[km,i]/
                                      (ICM.k_mixture*
                                      (0.0000001 +sum(
                                          ICM.dataI_assortments[d,:].*
                                          ICM.source_model[km,:]
                                          )
                                      )
                                      )
        end
      end
    end
  end

  #Out of sample data
  ICM.dataO_assortments = zeros(Bool,nO_assort,ICM.instance.nb_prod)
  ICM.dataO_probability = zeros(Float64,nO_assort,ICM.instance.nb_prod)
  for d = 1:nO_assort
    ICM.dataO_assortments[d,:] = pool_assortments[remaining_assortments[d],:]
    ICM.dataO_assortments[d,ICM.instance.nb_prod] =  1
    for i= 1:ICM.instance.nb_prod
      if ICM.dataO_assortments[d,i]
        for km = 1:ICM.k_mixture
          #Computing MMMNL probabilities
          ICM.dataO_probability[d,i] = ICM.dataO_probability[d,i] +
                                        ICM.source_model[km,i]/
                                        (ICM.k_mixture*
                                        (0.0000001 +sum(
                                            ICM.dataO_assortments[d,:].*
                                            ICM.source_model[km,:]
                                            )
                                        )
                                        )
        end
      end
    end
  end
end


function generate_data_intervals!(ICM::InstanceCvx,n_assort::Int64,nO_assort::Int64)
  #=
  Generates synthetic data based on the unique-ranking intervals model
  =#
  remaining_assortments = sample(Array(1:(n_assort+nO_assort)),nO_assort, replace = false)
  pool_assortments = rand((n_assort+nO_assort,ICM.instance.nb_prod)).< 0.5
  assortment_lists = setdiff(Array(1:(n_assort+nO_assort)),remaining_assortments)
  permutation = (Array(1:ICM.instance.nb_prod))
  lambda_probas = exp(randn((ICM.instance.nb_prod-1)*(ICM.instance.nb_prod-1)))
  list_of_indices = []
  for i =1:(ICM.instance.nb_prod-1)
    for j = i:(ICM.instance.nb_prod-1)
      push!(list_of_indices,(i,j))
    end
  end
  lambda_probas = lambda_probas[1:size(list_of_indices,1)]
  lambda_probas = lambda_probas/sum(lambda_probas)

  #In sample data
  ICM.dataI_assortments = zeros(Bool,n_assort,ICM.instance.nb_prod)
  ICM.dataI_probability = zeros(Float64,n_assort,ICM.instance.nb_prod)
  for d = 1:n_assort
    #ICM.dataI_assortments[d,:] = rand(ICM.instance.nb_prod).< 0.5
    ICM.dataI_assortments[d,:] = pool_assortments[assortment_lists[d],:]
    ICM.dataI_assortments[d,ICM.instance.nb_prod] =  1
    for km = 1:size(list_of_indices,1)
      cs = zeros(Int64,1,ICM.instance.nb_prod)
      cs[1,ICM.instance.nb_prod] = 1
      cs[list_of_indices[km][1]:list_of_indices[km][2]] = 1
      i = findmax((ICM.dataI_assortments[d,:] & cs)[1,permutation])[2]
      #println(i)
      ICM.dataI_probability[d,i] = ICM.dataI_probability[d,i] + lambda_probas[km]
    end
  end

  #Out of sample data
  ICM.dataO_assortments = zeros(Bool,nO_assort,ICM.instance.nb_prod)
  ICM.dataO_probability = zeros(Float64,nO_assort,ICM.instance.nb_prod)
  for d = 1:nO_assort
    #ICM.dataO_assortments[d,:] = rand(ICM.instance.nb_prod).< 0.5
    ICM.dataO_assortments[d,:] = pool_assortments[remaining_assortments[d],:]
    ICM.dataO_assortments[d,ICM.instance.nb_prod] =  1
    for km = 1:size(list_of_indices,1)
      cs = zeros(Int64,1,ICM.instance.nb_prod,)
      cs[1,ICM.instance.nb_prod] = 1
      cs[list_of_indices[km][1]:list_of_indices[km][2]] = 1
      i = findmax((ICM.dataO_assortments[d,:] & cs)[1,permutation])[2]
      ICM.dataO_probability[d,i] = ICM.dataO_probability[d,i] + lambda_probas[km]
    end
  end
end


function scoreO(ICM::InstanceCvx)
  #=
  Out-of-sample scores using current cvx-model predictor
  =#
  #Empty predicted values
  nO_assort = size(ICM.dataO_assortments)[1]
  valO_probability = zeros(Float64,nO_assort,ICM.instance.nb_prod)
  for d = 1:nO_assort
    for k = 1:findlast(ICM.instance.lambdas)
      calc_vec = zeros(Bool,ICM.instance.nb_prod)
      if ICM.right_endpoints[k] >= ICM.left_endpoints[k]
        calc_vec[ICM.left_endpoints[k]:ICM.right_endpoints[k]] = 1
      else
        calc_vec[ICM.left_endpoints[k]:(ICM.instance.nb_prod-1)] = 1
        calc_vec[1:ICM.right_endpoints[k]] = 1
      end
      #Intersection of assortment and consideration set
      calc_vec = reshape(calc_vec,size(ICM.dataO_assortments[d,:])) & ICM.dataO_assortments[d,:]
      if sum(calc_vec)> 0
        purchased = indmax( calc_vec.*ICM.rankings[k,:] )
        indices = find(x -> x == ICM.rankings[k,purchased], ICM.rankings[k,:] )
        #println("A",d," ",k," ",ICM.instance.lambdas[k])
        #println("sum1",sum(valO_probability[d,indices])," ", indices)
        valO_probability[d,indices] = valO_probability[d,indices] + (ICM.instance.lambdas[k])/(size(indices)[1])
        #println("sum2",sum(valO_probability[d,indices]))
      else
        valO_probability[d,ICM.instance.nb_prod] = valO_probability[d,ICM.instance.nb_prod] + ICM.instance.lambdas[k]
      end
    end
  end
  valO_probability = valO_probability./(sum(valO_probability,2)*ones((1,ICM.instance.nb_prod)))
  ini_score_local =zeros(4)
  ini_score_local[4] = sum(
                          ICM.dataO_assortments .*
                          ((ICM.dataO_probability-valO_probability).*(ICM.dataO_probability-valO_probability)./(max(0.01,ICM.dataO_probability)))
                          )
  ini_score_local[3] = sum(
                          ICM.dataO_assortments .*
                          (abs(ICM.dataO_probability-valO_probability)./(max(0.01,ICM.dataO_probability)))
                          )
  ini_score_local[1] = sum((ICM.dataO_probability-valO_probability).*(ICM.dataO_probability-valO_probability))/
                         sum(ICM.dataO_probability.*ICM.dataO_probability)
  ini_score_local[2] = sum(abs(ICM.dataO_probability-valO_probability))/
                        sum(abs(ICM.dataO_probability))
  #println("Out-of-sample score (MSE): ",ini_score_local[4])
  #println("Out-of-sample score (MAE): ",ini_score_local[3])
  return(ini_score_local)
end


function separate_demand!(ICM::InstanceCvx)
  #=
  Forces the n first permutations to be singletons (separate demand)
  =#
  n_assort = size(ICM.dataI_probability)[1]

  for i = 1:(ICM.instance.nb_prod-1)
    ICM.left_endpoints[i] = i
    ICM.right_endpoints[i] = i
    ICM.rankings[i,:] = zeros(ICM.instance.nb_prod)
    ICM.rankings[i,i] = 1
    ICM.X_purchase[i,:,:] = zeros((n_assort,ICM.instance.nb_prod))
    ICM.X_purchase[i,:,i] = ICM.dataI_assortments[:,i]
  end
  ICM.left_endpoints[ICM.instance.nb_prod] = 1
  ICM.right_endpoints[ICM.instance.nb_prod] = 0
  ICM.rankings[ICM.instance.nb_prod,:] = zeros(ICM.instance.nb_prod)
  ICM.rankings[ICM.instance.nb_prod,ICM.instance.nb_prod] = 1
  ICM.X_purchase[ICM.instance.nb_prod,:,:] = zeros((n_assort,ICM.instance.nb_prod))
  ICM.X_purchase[ICM.instance.nb_prod,:,ICM.instance.nb_prod] = 1

end
