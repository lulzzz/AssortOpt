
function sampleMNLcvx(ICM::InstanceCvx,k_step::Int64,weights::Array{Float64,1})
  #= Samples quasiconvex rankings using the marginal parwise probabilities of an MNL model

  k_step <-> rank of the customer type in the current model
  weights <-> MNL parameters to use
  =#
  n = ICM.instance.nb_prod
  cum_weights =cumsum(weights)/sum(weights)
  mode = findmax(cum_weights .> rand())[2]
  rank_base = Array(1:ICM.instance.nb_prod)
  cvx_rank = push!([],pop!(rank_base))
  left = mode
  right = mode

  while mode != n
    if left > right
      proba = weights[n]/(sum( weights[(right+1):(left-1)] ) +weights[n])
    else
      #println(left,right)
      proba = weights[n]/(sum(weights[1:(left-1)]) + sum(weights[(right+1):n]) )
    end
    if rand() < proba
      mode = n
      break
    end
    candidate_left = (left - 2 + (n -1)) % (n-1) + 1
    candidate_right = (right)% (n-1) +1

    if rand() < weights[candidate_left]/(weights[candidate_left] + weights[candidate_right])
      #Left extension
      left = candidate_left
      unshift!(cvx_rank,pop!(rank_base))
    else
      #Right extension
      right = candidate_right
      push!(cvx_rank,pop!(rank_base))
    end
  end

  #Enter the convex ranking into basis
  if left > right
    ICM.rankings[k_step,:] = vcat(cvx_rank[(n-left+1):((right+(n-1))-left+1)],ones(left-right-1),cvx_rank[1:(n-left)],[1])
  else
    ICM.rankings[k_step,:] = vcat(ones(left-1),cvx_rank,ones(n-right))
  end
  ICM.left_endpoints[k_step] = left
  ICM.right_endpoints[k_step] = right
  reduced_cost,indices = calculate_shadow_cost(ICM,transpose(round(Int,ICM.rankings[k_step,:]))[:,1],left,right,1)
  #Purchase covariates
  for i in indices
    ICM.X_purchase[k_step,i] = 1
  end

end


function learning!(ICM::InstanceCvx, steps::Int64, k_0::Int64 = 1, switch::Int64 = 0)
  #= Column generation procedure to learn quasi-convex rankings

  steps <-> number of column generation steps
  k_0 <-> starting rank for the column generation procedure
  switch <-> integer parameter that indicates several possible modes
  0: picks only quasi-convex permutations
  1: picks permutations in the neighborhood of the central ordering
  2: returns in-sample score

  TODO hardcoded parameters
  =#

  n_assort = size(ICM.dataI_probability)[1]
  #convex preference list covariate matrix
  ICM.X_purchase =zeros(ICM.instance.nb_cst_type,n_assort,ICM.instance.nb_prod)
  #residual objective (linear objective for the subproblem of list selection)
  ICM.res_obj = ICM.dataI_probability
  # ground = MMNLfitscore(ICM,1,1)[1:(ICM.instance.nb_prod-1)]
  # ground = sortperm(reshape(ground,ICM.instance.nb_prod-1),rev = true)
  # separate_demand!(ICM)
  # fitting!(ICM,ICM.instance.nb_prod)
  # scoreO(I)
  ini_score = 1000000*ones(4)
  k = k_0
  t = 1
  while (k < steps+1) & (t < 2000)
    if (k %400) == 1
      # No-purchase customer type
      ICM.X_purchase[k,:,ICM.instance.nb_prod] = 1
      ICM.left_endpoints[k] = ICM.instance.nb_prod
      ICM.right_endpoints[k] = ICM.instance.nb_prod
      ICM.rankings[k,:] = 1
      #ICM.instance.lambdas[1] = 1.0
    elseif switch == 0
      #best_ranking_bigMIP!(ICM,k,1.)
      if mod(k,2)==0
        best_ranking_heuristicLS!(ICM,k)
      else
        best_ranking_DP!(ICM,k)
      end
    elseif switch == 1
      best_ranking_heuristicNeigh!(ICM,k)
      #best_ranking_heuristicNeigh!(ICM,k,ground)
    end
    fitting!(ICM,k,k_0)
    if (switch > 2) & ((k == 100) | (k == 200) | (k == 400))
      ini_score = min(scoreO(ICM),ini_score)
      fittingL2!(ICM,k)
      ini_score = min(scoreO(ICM),ini_score)
    end
    t += 1
    k += 1
  end
  #Transform left-right endpoints into consideration sets
  for k = k_0:steps
     ICM.instance.consideration_sets[k,(ICM.left_endpoints[k]):(ICM.right_endpoints[k])] = 1
  end
  return(ini_score)
end


function eval_holdout(ICM::InstanceCvx,holdout::Array{Int64,1})
  #= Evaluates performance on in-sample "holdout" for hyper-parameters selection

  holdout <-> indices of the out-of-sample assortments
  =#
  nO_assort = size(holdout,1)
  valO_probability = zeros(Float64,nO_assort,ICM.instance.nb_prod)
  for (i_d,d) in enumerate(holdout)
    for k = 1:findlast(ICM.instance.lambdas)
      calc_vec = zeros(Bool,ICM.instance.nb_prod)
      if ICM.right_endpoints[k] >= ICM.left_endpoints[k]
        calc_vec[ICM.left_endpoints[k]:ICM.right_endpoints[k]] = 1
      else
        calc_vec[ICM.left_endpoints[k]:(ICM.instance.nb_prod-1)] = 1
        calc_vec[1:ICM.right_endpoints[k]] = 1
      end
      #Intersection of assortment and consideration set
      calc_vec = reshape(calc_vec,size(ICM.dataI_assortments[d,:])) & ICM.dataI_assortments[d,:]
      if sum(calc_vec)> 0
        purchased = indmax( calc_vec.*ICM.rankings[k,:] )
        indices = find(x -> x == ICM.rankings[k,purchased], ICM.rankings[k,:] )
        valO_probability[i_d,indices] = valO_probability[i_d,indices] + (ICM.instance.lambdas[k])/(size(indices)[1])
        #println("sum2",sum(valO_probability[d,indices]))
      else
        valO_probability[i_d,ICM.instance.nb_prod] = valO_probability[i_d,ICM.instance.nb_prod] + ICM.instance.lambdas[k]
      end
    end
  end
  valO_probability = valO_probability./(sum(valO_probability,2)*ones((1,ICM.instance.nb_prod)))
  ini_score_local =zeros(4)
  ini_score_local[4] = sum(
                          ICM.dataI_assortments[holdout,:] .*
                          ((ICM.dataI_probability[holdout,:]-valO_probability).*(ICM.dataI_probability[holdout,:]-valO_probability)./(max(0.01,ICM.dataI_probability[holdout,:])))
                          )
  ini_score_local[3] = sum(
                          ICM.dataI_assortments[holdout,:] .*
                          (abs(ICM.dataI_probability[holdout,:]-valO_probability)./(max(0.01,ICM.dataI_probability[holdout,:])))
                          )
  ini_score_local[1] = sum((ICM.dataI_probability[holdout,:]-valO_probability).*(ICM.dataI_probability[holdout,:]-valO_probability))/
                         sum(ICM.dataI_probability[holdout,:].*ICM.dataI_probability[holdout,:])
  ini_score_local[2] = sum(abs(ICM.dataI_probability[holdout,:]-valO_probability))/
                        sum(abs(ICM.dataI_probability[holdout,:]))
  #println("Out-of-sample score (MSE): ",ini_score_local[1])
  #println("Out-of-sample score (MAE): ",ini_score_local[2])
  return(ini_score_local)

end


function crossval!(ICM::InstanceCvx, bootstrap::Int64,switch::Int64 = 0)
  #= Runs a crossvalidation procedure to pick the hyper parameters,
  namely the customer type complexity and the norms

  bootstrap <-> number of iterations over which the learning procedure is averaged
  switch <-> see learning! function

  TODO hardcoded parameters
  =#
  #Selects the right step
  n_assort = size(ICM.dataI_assortments)[1]
  ini_score = 1000000*ones(4)
  stopping_times = [100,200,300,400]
  s_times = size(stopping_times)[1]

  #iterations before averaging (we call it bootstrapping)
  for b = 1:bootstrap
    learning!(ICM,400*b,1 + 400*(b-1),switch)
  end

  score = zeros(s_times*2,4)
  for iter = 1:35
    holdout = sample(Array(1:n_assort),2,replace= false)
    insample = setdiff(Array(1:n_assort),holdout)
    # L1 calibration
    for (i_k,k) in enumerate(stopping_times)
      ICM.instance.lambdas[1:ICM.instance.nb_cst_type] = 0.0
      #for b = 1:bootstrap
      #  fitting!(ICM,400*(b-1) + k,400*(b-1)+1,insample)
      #end
      #ICM.instance.lambdas[1:400*bootstrap] = ICM.instance.lambdas[1:400*bootstrap]/bootstrap
      fitting!(ICM,400*bootstrap,1,insample,
               reduce((x,y)-> union(x,y),
                      [Array(((i-1)*400 + k + 1):(i*400)) for i in 1:bootstrap]
                      )
              )
      score[i_k,:] = score[i_k,:] + reshape(
                                            eval_holdout(ICM,holdout),
                                            size(score[i_k,:])
                                            )
    end
    # L2 calibration
    for (i_k,k) in enumerate(stopping_times)
      ICM.instance.lambdas[1:ICM.instance.nb_cst_type] = 0.0
      for b = 1:bootstrap
        fittingL2!(ICM,400*(b-1) + k,400*(b-1)+1,insample)
      end
      ICM.instance.lambdas[1:400*bootstrap] = ICM.instance.lambdas[1:400*bootstrap]/bootstrap
      score[s_times+i_k,:] = score[s_times+i_k,:] + reshape(
                                                            eval_holdout(ICM,holdout),
                                                            size(score[i_k,:])
                                                            )
    end
  end

  #Now pick the best between customer type complexity and norms
  for j = 1:4
    ind = findmin(score[:,j])[2]
    #println(j," ",ind)
    if div(ind-1,s_times) ==  0
      k = stopping_times[ind]
      ICM.instance.lambdas[1:ICM.instance.nb_cst_type] = 0.0
      #for b = 1:bootstrap
      #  fitting!(ICM,400*(b-1) + k,400*(b-1)+1)
      #end
      #ICM.instance.lambdas[1:400*bootstrap] = ICM.instance.lambdas[1:400*bootstrap]/bootstrap
      fitting!(ICM,400*bootstrap,1,zeros(Int64,1),
               reduce((x,y)-> union(x,y),
                      [Array(((i-1)*400 + k + 1):(i*400)) for i in 1:bootstrap]
                      )
              )
      ini_score[j] = scoreO(ICM)[j]
    else
      k = stopping_times[((ind-1) % s_times)+1]
      ICM.instance.lambdas[1:ICM.instance.nb_cst_type] = 0.0
      for b = 1:bootstrap
        fittingL2!(ICM,400*(b-1) + k,400*(b-1)+1)
      end
      ICM.instance.lambdas[1:400*bootstrap] = ICM.instance.lambdas[1:400*bootstrap]/bootstrap
      ini_score[j] = scoreO(ICM)[j]
    end
  end
  return(ini_score)
end


function fitting!(ICM::InstanceCvx, k_step::Int64, k_0 ::Int64 = 1,
                  insample::Array{Int64,1} = zeros(Int64,1),
                  trim::Array{Int64,1}=zeros(Int64,1))
  #= L1 (master problem): calibrates model using current convex rankings

  k_0 - k_step <-> range of customer types to considera
  insample <-> restricts attention to insample data coordinates
  trim <-> fixing certain lambda probabilities to zero
  =#
  n_assort = size(ICM.dataI_probability)[1]
  #To be coded with nice heuristic
  m = Model(solver = GurobiSolver(OutputFlag=0))
  @variable(m,lambdas[1:(k_step-k_0+1)]>=0)
  #Norm 2 error variables
  @variable(m, -1 <= eps[1:n_assort,1:ICM.instance.nb_prod] <= 1)
  @variable(m, 0 <= eps2[1:n_assort,1:ICM.instance.nb_prod] <= 1)
  @constraint(m, sum(lambdas[i] for i=1:(k_step-k_0+1)) <= 1.0)

  for i in setdiff(trim,[0])
    @constraint(m, lambdas[i] == 0.0)
  end

  #Fitting
  if sum(insample) > 0
    @constraint(m, constr[d = insample,i=1:ICM.instance.nb_prod],
                sum{ICM.X_purchase[k_0+alpha-1,d,i]*lambdas[alpha], alpha = 1:(k_step-k_0+1)} +
                eps[d,i] == ICM.dataI_probability[d,i]
                )
  else
    @constraint(m, constr[d = 1:n_assort,i=1:ICM.instance.nb_prod],
                sum{ICM.X_purchase[k_0+alpha-1,d,i]*lambdas[alpha], alpha = 1:(k_step-k_0+1)} +
                eps[d,i] == ICM.dataI_probability[d,i]
                )
  end

  # Norm 1 on the error terms
  for d = 1:n_assort
    @constraints(m, begin
                      negatives[i = 1:ICM.instance.nb_prod],
                      eps2[d,i] >=eps[d,i]
                      end
                    )
    @constraints(m, begin
                        positives[i = 1:ICM.instance.nb_prod],
                        eps2[d,i] >= - eps[d,i]
                      end
                    )
  end
  #minimize error terms
  @objective(m, Min, sum(eps2))
  status = solve(m)
  # println(k_0,k_step)
  ICM.instance.lambdas[k_0:k_step] = getvalue(lambdas)
  #println("Vector Lambda",ICM.instance.lambdas[k_0:k_step])
  #ICM.instance.lambdas[(k_step+1):ICM.instance.nb_cst_type] = 0
  ICM.instance.lambdas[(k_step+1):(k_0+400-1)] = 0
  if sum(insample) == 0
    #println("update of dual")
    #println(size(ICM.res_obj))
    #println(size(ICM.X_purchase[k_step,:,:]))
    #println(ICM.X_purchase[k_step,2,1],"-",ICM.X_purchase[k_step,2])
    #println(sum(ICM.res_obj != getdual(constr)))
    ICM.res_obj = getdual(constr)
    #println("Should be 0?",sum(ICM.X_purchase[k_step,:,:].*ICM.res_obj))
  end
  #println("Obj ", getobjectivevalue(m))
  #println("Sparsity ",size(find(ICM.instance.lambdas)))
  return()
end


function fittingL2!(ICM::InstanceCvx, k_step::Int64,k_0 ::Int64 = 1, insample::Array{Int64,1} = zeros(Int64,1))
  #= L2 (master problem): calibrates model using current convex rankings

  k_0 - k_step <-> range of customer types to considera
  insample <-> restricts attention to insample data coordinates
  =#
  n_assort = size(ICM.dataI_probability)[1]
  m = Model(solver = GurobiSolver(OutputFlag=0))
  @variable(m,lambdas[1:(k_step-k_0+1)]>=0)
  #Norm 2 error variables
  @variable(m, -1 <= eps[1:n_assort,1:ICM.instance.nb_prod] <= 1)
  #@variable(m, 0 <= eps2[1:n_assort,1:ICM.instance.nb_prod] <= 1)
  @constraint(m, sum{lambdas[i], i=1:(k_step-k_0+1)} <= 1.0)
  #Fitting
  if sum(insample) > 0
    @constraint(m, constr[d = insample,i=1:ICM.instance.nb_prod], sum{ICM.X_purchase[k_0+alpha-1,d,i]*lambdas[alpha], alpha = 1:(k_step-k_0+1)} +
                  eps[d,i] == ICM.dataI_probability[d,i])
  else
    @constraint(m, constr[d = 1:n_assort,i=1:ICM.instance.nb_prod], sum{ICM.X_purchase[k_0+alpha-1,d,i]*lambdas[alpha], alpha = 1:(k_step-k_0+1)} +
                    eps[d,i] == ICM.dataI_probability[d,i])
  end

  #minimize error terms
  @objective(m, Min, sum(eps.*eps))
  status = solve(m)
  ICM.instance.lambdas[k_0:k_step] = getvalue(lambdas)
  #ICM.instance.lambdas[(k_step+1):ICM.instance.nb_cst_type] = 0
  ICM.instance.lambdas[(k_step+1):(k_0+400-1)] = 0.0
  #ICM.res_obj = ICM.res_obj + getdual(constr)
  return()
end


# Calculates the reduced cost associated with one convex ranking, given current state of the master problem
function calculate_shadow_cost(ICM::InstanceCvx,ranking::Array{Int64,1},left::Int64,right::Int64,shift::Int64)
  #= Computes the shadow cost associated with a certain ranking

  ranking <-> permutation describing the preferences
  left <-> left end point of the consideration set
  right <-> right end point of the consideration set
  shift <-> indicates by how much the mode is shifted
  =#
  n_assort = size(ICM.dataI_assortments)[1]
  cs = zeros((1,ICM.instance.nb_prod))
  if right >= left
    cs[1,left:right] = 1
  else
    cs[1,left:(ICM.instance.nb_prod-1)] = 1
    cs[1,1:right] = 1
  end
  rankingshifted = zeros(ICM.instance.nb_prod)
  rankingshifted[shift:ICM.instance.nb_prod-1] = ranking[1:(ICM.instance.nb_prod-shift)]
  rankingshifted[1:(shift-1)] = ranking[(ICM.instance.nb_prod-shift+1):(ICM.instance.nb_prod-1)]
  cs = (ones(n_assort,1)*(cs.*reshape(rankingshifted+0.01,(1,ICM.instance.nb_prod)))).*ICM.dataI_assortments
  #println(cs[2,:])
  cs[:,ICM.instance.nb_prod] = 1
  indices = findmax(cs,2)[2]
  #println(indices)
  #println(cs[2,:])
  #println("I",indices)
  #println("S",sum([ICM.res_obj[indices[i]] for i = 1:n_assort]))
  return(sum([ICM.res_obj[indices[i]] for i = 1:n_assort]), indices)
end

function best_ranking_DP!(ICM::InstanceCvx,k_step::Int64)
  n = ICM.instance.nb_prod -1
  n_assort = size(ICM.res_obj)[1]
  DP_value = zeros(n,n)
  DP_arg = zeros(n,n)
  for DDelta in 0:(n-1)
    for l in 1:n
        Delta = DDelta + 1
        r = mod(Delta+l-2,n)+1
        if l <= r
          old_support = l:r
        else
          old_support = union(1:r,l:n)
        end
        if Delta > 1
          gradient_left = reshape((ICM.dataI_assortments[:,l].>0) &
                                  (sum(ICM.dataI_assortments[:,setdiff(old_support,[l])],2).==0),
                                  (n_assort,))
          #println(sum(gradient_left) - sum(ICM.dataI_assortments[:,l].>0))
          gradient_right = reshape((ICM.dataI_assortments[:,r].>0) &
                                   (sum(ICM.dataI_assortments[:,setdiff(old_support,[r])],2).==0),
                                   (n_assort,))
          #println(sum(gradient_right) - sum(ICM.dataI_assortments[:,r].>0))
        else
          gradient_left = reshape((ICM.dataI_assortments[:,l].>0),(n_assort,))
          gradient_right = reshape((ICM.dataI_assortments[:,r].>0),(n_assort,))
        end
        left_choice = dot(gradient_left,ICM.res_obj[:,l] - ICM.res_obj[:,n+1])
        right_choice = dot(gradient_right,ICM.res_obj[:,r] - ICM.res_obj[:,n+1])
        #println("l",l,"Delta",Delta,"-",left_choice,"-",right_choice)
        # println("l",sum((ICM.dataI_assortments[:,l].>0).*ICM.res_obj[:,l]),"r",sum((ICM.dataI_assortments[:,r].>0).*ICM.res_obj[:,r]))
        # println("n",sum((ICM.dataI_assortments[:,l].>0).*ICM.res_obj[:,n+1]))
        if Delta > 1
          left_choice = left_choice + DP_value[mod(l,n)+1,Delta-1]
          right_choice = right_choice + DP_value[l,Delta-1]
        end
        #println(left_choice,"-",right_choice)
        if left_choice >= right_choice
          DP_value[l,Delta] = left_choice
          DP_arg[l,Delta] = 1
        else
          DP_value[l,Delta] = right_choice
          DP_arg[l,Delta] = 0
        end
    end
  end
  #optimal indices
  #println(DP_value)
  println("Score", sum(ICM.res_obj[:,n+1])+findmax(DP_value)[1])
  println("Min", findmin(DP_value)[1])
  set_of_indices = findmax(DP_value)[2]
  left_end = mod(last(set_of_indices)-1,n)+1
  Delta = div(last(set_of_indices)-1,n)+1
  right_end = mod(left_end + Delta -2,n) +1
  #println("D",Delta)
  # iterations to find the rank
  #push! unshift!
  left_list = Array(Int64,0)
  right_list = Array(Int64,0)
  l = left_end
  #r = right_end
  for d in 0:(Delta-1)
    if DP_arg[l,Delta-d] > 0.5
      push!(left_list,n-Delta+1+d)
      l = mod(left_end,n)+1
    else
      unshift!(right_list,n-Delta+1+d)
      #r = mod(right_end - 2,n) + 1
    end
  end
  append!(right_list,ones(n-size(left_list,1)-size(right_list,1)))
  shift = left_end
  # Update
  score,X_purchase = calculate_shadow_cost(ICM,vcat(left_list,right_list,[0]), left_end,right_end,shift)
  for i in X_purchase
    ICM.X_purchase[k_step,i] = 1
  end
  println("Score DP",score)
  #println(left_end,"-",right_end)
  #println(vcat(left_list,right_list,[0]))

  ICM.left_endpoints[k_step] = left_end
  ICM.right_endpoints[k_step] = right_end
  #ICM.rankings[k_step,:] = vcat(left_list,right_list,[0])
  ranking = vcat(left_list,right_list,[0])
  ICM.rankings[k_step,shift:ICM.instance.nb_prod-1] = ranking[1:(ICM.instance.nb_prod-shift)]
  ICM.rankings[k_step,1:(shift-1)] = ranking[(ICM.instance.nb_prod-shift+1):(ICM.instance.nb_prod-1)]

end

function best_ranking_heuristicLS!(ICM::InstanceCvx,k_step::Int64, target_score::Int64=-100)
  #= Local Search heuristic for the (quasi-convex) column generation step
  by improving the reduced cost from random start

  k_step <-> rank of the customer type in the column generation array
  target_score <-> minimal  shadow cost for admissible ranking
  =#
  #Initialization
  score = - 1000
  #Increasing part
  left_list = sort(sample(1:(ICM.instance.nb_prod-1),floor(Int,rand()*ICM.instance.nb_prod),replace = false))
  #Decreasing part
  right_list = sort(setdiff(1:(ICM.instance.nb_prod-1),left_list), rev = true)
  #left end point
  left_end = floor(Int,rand()*(ICM.instance.nb_prod-1)+1)
  #right end point
  right_end = floor(Int,rand()*(ICM.instance.nb_prod-1)+1)
  #Shift of permutation
  shift = 1
  #Purchase matrix
  X_purchase = zeros((size(ICM.dataI_assortments)[1],ICM.instance.nb_prod))
  step = 0

  while (score <= target_score) & (step < 1000)
    #Resample the candidate
    left_list = sort(sample(1:(ICM.instance.nb_prod-1),floor(Int,rand()*ICM.instance.nb_prod),replace = false))
    right_list = sort(setdiff(1:(ICM.instance.nb_prod-1),left_list), rev = true)
    left_end = floor(Int,rand()*(ICM.instance.nb_prod-1)+1)
    right_end = floor(Int,rand()*(ICM.instance.nb_prod-1)+1)
    shift = 1
    # if right_end < left_end
    #   r = right_end
    #   right_end = left_end
    #   left_end = r
    # end

    #Initial score
    score,X_purchase = calculate_shadow_cost(ICM,vcat(left_list,right_list,[0]), left_end,right_end,shift)
    incr = 1
    while (step < 1000) & (incr > 0)
      #println("A ",step," ",score)
      #Stock the beginning
      res_l,res_r,l,r,sh = left_list,right_list,left_end,right_end,shift
      score_ini = score

      #Exchange i,j
      for i in left_list
        for j in right_list
          res_left_list = setdiff(left_list,[i])
          res_right_list = copy(right_list)
          push!(res_right_list,i)
          res_right_list = setdiff(res_right_list,[j])
          sort!(res_right_list)
          push!(res_left_list,j)
          sort!(res_left_list, rev = true)
          score_new = calculate_shadow_cost(ICM,vcat(res_left_list,res_right_list,[0]), left_end,right_end,shift)
          if score_new[1] > score
            score = score_new[1]
            X_purchase = score_new[2]
            res_l,res_r,l,r,sh = res_left_list,res_right_list,left_end,right_end,shift
          end
        end
      end

      # delete a left element to from left_list
      for i in left_list
        res_left_list = setdiff(left_list,[i])
        sort!(res_left_list)
        res_right_list = copy(right_list)
        push!(res_right_list,i)
        sort!(res_right_list, rev = true)
        score_new = calculate_shadow_cost(ICM,vcat(res_left_list,res_right_list,[0]), left_end,right_end,shift)
        if score_new[1] > score
          score = score_new[1]
          X_purchase = score_new[2]
          res_l,res_r,l,r,sh = res_left_list,res_right_list,left_end,right_end,shift
        end
      end

      # insert a right element to left_list
      for i in right_list
        res_right_list = setdiff(right_list,[i])
        sort!(res_right_list)
        res_left_list = copy(left_list)
        push!(res_left_list,i)
        sort!(res_left_list, rev = true)
        score_new = calculate_shadow_cost(ICM,vcat(res_left_list,res_right_list,[0]), left_end,right_end,shift)
        if score_new[1] > score
          score = score_new[1]
          X_purchase = score_new[2]
          res_l,res_r,l,r,sh = res_left_list,res_right_list,left_end,right_end,shift
        end
      end

      # shift the right end point
      for i in (right_end+1):(ICM.instance.nb_prod -2 + left_end)
        i_new = (i-1) %(ICM.instance.nb_prod-1) + 1
        score_new = calculate_shadow_cost(ICM,vcat(left_list,right_list,[0]), left_end, i_new,shift)
        if score_new[1] > score
          score = score_new[1]
          X_purchase = score_new[2]
          res_l,res_r,l,r,sh = left_list,right_list,left_end,i_new,shift
        end
      end
      # shift the left end point
      for i in (1- right_end+1):(left_end-1)
        i_new = (i-1 + ICM.instance.nb_prod-1) %(ICM.instance.nb_prod-1) + 1
        score_new = calculate_shadow_cost(ICM,vcat(left_list,right_list,[0]), i_new,right_end,shift)
        if score_new[1] > score
          score = score_new[1]
          X_purchase = score_new[2]
          res_l,res_r,l,r,sh = left_list,right_list,i_new,right_end,shift
        end
      end
      # shift
      for s in 1:(ICM.instance.nb_prod-1)
        score_new = calculate_shadow_cost(ICM,vcat(left_list,right_list,[0]), left_end,right_end,s)
        if score_new[1] > score
          score = score_new[1]
          X_purchase = score_new[2]
          res_l,res_r,l,r,sh = left_list,right_list,left_end,right_end,s
        end
      end
      step += 1
      incr = score - score_ini
      if incr > 0
        left_list,right_list,left_end,right_end,shift = res_l,res_r,l,r,sh
      end
    end
  end

  for i in X_purchase
    ICM.X_purchase[k_step,i] = 1
  end

  ICM.left_endpoints[k_step] = left_end
  ICM.right_endpoints[k_step] = right_end
  #ICM.rankings[k_step,:] = vcat(left_list,right_list,[0])
  ranking =  vcat(left_list,right_list,[0])
  ICM.rankings[k_step,shift:ICM.instance.nb_prod-1] = ranking[1:(ICM.instance.nb_prod-shift)]
  ICM.rankings[k_step,1:(shift-1)] = ranking[(ICM.instance.nb_prod-shift+1):(ICM.instance.nb_prod-1)]
  println("Score LS:", score)
  #println("Consideration set: ",ICM.left_endpoints[k_step]," to ", ICM.right_endpoints[k_step])
  #println("Ranking: ",round(Int64,ICM.rankings[k_step,:]))
  return
end


function best_ranking_bigMIP!(ICM::InstanceCvx,k_step::Int64, dual_val::Float64)
  #= MIP-based column generation step to pick the best residual quasi-convex ranking

  k_step <-> rank of the customer type in the column generation array
  dual_val <-> TODO linearization by guessing the probability
  =#
  m = Model(solver = GurobiSolver(OutputFlag=0))
  n_assort = size(ICM.dataI_probability)[1]
  #Distribution variables
  #Convex ranking
  #@variable(m,CVranking[1:I.instance.nb_prod,1:I.instance.nb_prod],Bin)
  @variable(m,0 <= CVranking[1:ICM.instance.nb_prod] <= ICM.instance.nb_prod)
  #Inversion variable
  @variable(m, Inversion[1:ICM.instance.nb_prod],Bin)
  #Left-right inversion to encode the consideration set
  @variable(m, LeftInversion[1:ICM.instance.nb_prod],Bin)
  @variable(m, RightInversion[1:ICM.instance.nb_prod],Bin)
  #Purchase variable of additional convex ranking
  @variable(m, X_purchase[1:n_assort,1:ICM.instance.nb_prod],Bin)

  #Convex constraint
  for i = 2:(ICM.instance.nb_prod-1)
    @constraint(m, Inversion[i] <= Inversion[i-1])
    @constraint(m, LeftInversion[i] <= LeftInversion[i-1])
    @constraint(m, RightInversion[i] <= RightInversion[i-1])
    @constraint(m, CVranking[i] - CVranking[i-1] >= 1-(ICM.instance.nb_prod)*(1-Inversion[i]))
    @constraint(m, CVranking[i] - CVranking[i-1] <= -1 + (ICM.instance.nb_prod)*(Inversion[i]))
  end
  @constraint(m, Inversion[1] == 1 )
  @constraint(m, Inversion[ICM.instance.nb_prod] == 0 )
  @constraint(m, LeftInversion[ICM.instance.nb_prod] == 0)
  @constraint(m, RightInversion[ICM.instance.nb_prod] == 1)
  @constraint(m, CVranking[ICM.instance.nb_prod] == 0.)
  @constraint(m, sum{CVranking[alpha], alpha= 1:(ICM.instance.nb_prod-1)} == sum(1:ICM.instance.nb_prod-1))


  #Consistency constraints on the purchase
  for d = 1:n_assort
    last = findfirst(ICM.dataI_assortments[d,:])
    for i= (last):ICM.instance.nb_prod
      if ICM.dataI_assortments[d,i] >0
        @constraint(m, X_purchase[d,i] <= 1- LeftInversion[i])
        @constraint(m, X_purchase[d,i] <= RightInversion[i])
        if last < i
          #If decayed after i than i is good
          @constraint(m, X_purchase[d,i]  <= (CVranking[i] - CVranking[last]-1)/(ICM.instance.nb_prod) + 1 + LeftInversion[last])
          #If last is purchased means that it has decayed before i
          @constraint(m, X_purchase[d,last]  <= 1- Inversion[i] + (1-RightInversion[i]))
          #If it has indeed decayed by i, then the ranking should be larger
          @constraint(m, X_purchase[d,last]  <= (CVranking[last] - CVranking[i]-1)/(ICM.instance.nb_prod)+ 1 + (1-RightInversion[i]))

        end
        last = i
      else
        @constraint(m,X_purchase[d,i] ==0)
      end
    end
    @constraint(m, sum{X_purchase[d,alpha], alpha = 1:ICM.instance.nb_prod } == 1)
  end

  @objective(m, Max, vecdot(ICM.res_obj,X_purchase))
  status = solve(m)

  #update parameters of the incumbent model
  ICM.X_purchase[k_step,:,:] = getvalue(X_purchase)
  ICM.left_endpoints[k_step] = findlast(getvalue(LeftInversion))+1
  ICM.right_endpoints[k_step] = findlast(getvalue(RightInversion[1:(ICM.instance.nb_prod-1)]))
  ICM.rankings[k_step,:] = getvalue(CVranking)
  #println("Consideration set: ",ICM.left_endpoints[k_step]," to ", ICM.right_endpoints[k_step])
  #println("Ranking: ",round(Int64,ICM.rankings[k_step,:]))

end

function best_ranking_heuristicNeigh!(ICM::InstanceCvx,k_step::Int64,
                                      ground::Vector{Int64}=zeros(Int64,1))
  #= Local Search heuristic for the column generation step in the neighborhood
  of the central permutation

  k_step <-> rank of the customer type in the column generation array
  ground <-> (optional) central permutation defining the neighborhood

  TODO hardcoded parameter: d
  =#
  d = ICM.instance.nb_prod - 1
  #Initialization
  score = - 1000
  n =ICM.instance.nb_prod - 1
  variations = sample((-d):d,n,replace = true)
  if sum(ground) == 0
    ground = Array(1:n)
  end
  ranking = ground + variations + 1  - minimum(ground+variations)
  #left end point
  left_end = floor(Int,rand()*(ICM.instance.nb_prod-1)+1)
  #right end point
  right_end = floor(Int,rand()*(ICM.instance.nb_prod-1)+1)
  if right_end < left_end
    a =left_end
    left_end = right_end
    right_end = a
  end
  #Shift of permutation
  shift = 1
  #Purchase matrix
  X_purchase = zeros((size(ICM.dataI_assortments)[1],ICM.instance.nb_prod))
  step = 0

  while (score <= -100) & (step < 1000)
    #Resample the candidate
    variations = sample((-d):d,n,replace = true)
    ranking = ground + variations + 1  - minimum(ground+variations)
    left_end = floor(Int,rand()*(ICM.instance.nb_prod-1)+1)
    right_end = floor(Int,rand()*(ICM.instance.nb_prod-1)+1)

    if right_end < left_end
      a =left_end
      left_end = right_end
      right_end = a
    end

    shift = 1

    # if right_end < left_end
    #   r = right_end
    #   right_end = left_end
    #   left_end = r
    # end

    #Initial score
    score,X_purchase = calculate_shadow_cost(ICM,vcat(ranking,[0]), left_end,right_end,shift)
    incr = 1
    while (step < 1000) & (incr > 0)
      #println("A ",step," ",score,incr)
      #Stock the beginning
      res,l,r,sh = ranking,left_end,right_end,shift
      score_ini = score

      copy_variations = copy(variations)
      copy_ranking= copy(ranking)
      if left_end <= right_end
        for i = left_end:right_end
          for j = (i):(i)
            j = mod(j - 1,n) + 1
            for exi in (-d):d
              for exj in (0):0
                if (exi != variations[i]) | (exj != variations[j])
                  copy_variations[i] = exi
                  copy_variations[j] = exj
                  copy_ranking = ground + copy_variations + 1 - minimum(ground+copy_variations)
                  score_new = calculate_shadow_cost(ICM,vcat(copy_ranking,[0]), left_end,right_end,shift)
                  if score_new[1] > score
                    score = score_new[1]
                    X_purchase = score_new[2]
                    res,l,r,sh = copy_ranking,left_end,right_end,shift
                  end
                  copy_variations[i] = variations[i]
                  copy_variations[j] = variations[j]
                end
              end
            end
          end
        end
      else
        for i = left_end:n
          for j = (i):(i)
            j = mod(j - 1,n) + 1
            for exi in (-d):d
              for exj in (0):0
                if (exi != variations[i]) | (exj != variations[j])
                  copy_variations[i] = exi
                  copy_variations[j] = exj
                  copy_ranking = ground + copy_variations + 1 - minimum(ground+copy_variations)
                  score_new = calculate_shadow_cost(ICM,vcat(copy_ranking,[0]), left_end,right_end,shift)
                  if score_new[1] > score
                    score = score_new[1]
                    X_purchase = score_new[2]
                    res,l,r,sh = copy_ranking,left_end,right_end,shift
                  end
                  copy_variations[i] = variations[i]
                  copy_variations[j] = variations[j]
                end
              end
            end
          end
        end
        for i = 1:right_end
          for j = (i):(i)
            j = mod(j - 1,n) + 1
            for exi in (-d):d
              for exj in (0):0
                if (exi != variations[i]) | (exj != variations[j])
                  copy_variations[i] = exi
                  copy_variations[j] = exj
                  copy_ranking = ground + copy_variations + 1 - minimum(ground+copy_variations)
                  score_new = calculate_shadow_cost(ICM,vcat(copy_ranking,[0]), left_end,right_end,shift)
                  if score_new[1] > score
                    score = score_new[1]
                    X_purchase = score_new[2]
                    res,l,r,sh = copy_ranking,left_end,right_end,shift
                  end
                  copy_variations[i] = variations[i]
                  copy_variations[j] = variations[j]
                end
              end
            end
          end
        end
      end

      #reverse
      ground = n+1 - ground
      copy_ranking = ground + variations + 1 - minimum(ground+variations)
      score_new = calculate_shadow_cost(ICM,vcat(copy_ranking,[0]), left_end, right_end,shift)
      if score_new[1] < score
        ground = n+1 - ground
      else
        score = score_new[1]
        X_purchase = score_new[2]
      end


      # shift the right end point
      for i in (left_end+1):(ICM.instance.nb_prod -2 + left_end)
        i_new = (i-1) %(ICM.instance.nb_prod-1) + 1
        score_new = calculate_shadow_cost(ICM,vcat(ranking,[0]), left_end, i_new,shift)
        if score_new[1] > score
          score = score_new[1]
          X_purchase = score_new[2]
          res,l,r,sh = ranking,left_end,i_new,shift
        end
      end

      # shift the left end point
      for i in (1- right_end+1):(right_end-1)
        i_new = (i-1 + ICM.instance.nb_prod-1) %(ICM.instance.nb_prod-1) + 1
        score_new = calculate_shadow_cost(ICM,vcat(ranking,[0]), i_new,right_end,shift)
        if score_new[1] > score
          score = score_new[1]
          X_purchase = score_new[2]
          res,l,r,sh = ranking,i_new,right_end,shift
        end
      end
      step += 1
      incr = score - score_ini
      if incr > 0
        ranking,left_end,right_end,shift = res,l,r,sh
      end
    end
  end

  for i in X_purchase
    ICM.X_purchase[k_step,i] = 1
  end

  ICM.left_endpoints[k_step] = left_end
  ICM.right_endpoints[k_step] = right_end
  #ICM.rankings[k_step,:] = vcat(left_list,right_list,[0])
  ranking =  vcat(ranking,[0])
  ICM.rankings[k_step,shift:ICM.instance.nb_prod-1] = ranking[1:(ICM.instance.nb_prod-shift)]
  ICM.rankings[k_step,1:(shift-1)] = ranking[(ICM.instance.nb_prod-shift+1):(ICM.instance.nb_prod-1)]

  #println("Consideration set: ",ICM.left_endpoints[k_step]," to ", ICM.right_endpoints[k_step])
  #println("Ranking: ",round(Int64,ICM.rankings[k_step,:]))
  return
end
