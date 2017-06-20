
function MNLfitscore(ICM::InstanceCvx)
  #= Fitting the MNL model through an approximate ML algorithm (fast approximation)
  =#
  transition_probabilities = zeros((ICM.instance.nb_prod,ICM.instance.nb_prod)) +0.000001
  n_assort = size(ICM.dataI_probability)[1]
  for ass in 1:n_assort
    assort = find(ICM.dataI_assortments[ass,:])
    for p1 in assort
      for p2 in assort
          tot = sum(ICM.dataI_assortments[:,p1] & ICM.dataI_assortments[:,p2])
          if p1 != p2
              #println(sum(transition_probabilities)," ",ICM.dataI_probability[ass,p2]," ",tot)
              transition_probabilities[p2,p1] = transition_probabilities[p2,p1] + ICM.instance.nb_prod/tot*ICM.dataI_probability[ass,p1]/(ICM.dataI_probability[ass,p2] + 0.0000001)
          end
      end
    end
  end
  #println(sum(transition_probabilities))
  #Renormalize the transition probabilities
  transition_probabilities = transition_probabilities ./ (reshape(sum(transition_probabilities,2),(ICM.instance.nb_prod,1))*ones((1,ICM.instance.nb_prod)))

  #initialization
  weights = 1.0/ICM.instance.nb_prod*ones((1,ICM.instance.nb_prod))

  #calculating the weights as steady state distribution
  for j in 1:1000
      weights = weights*transition_probabilities
  end

  weights = reshape(weights,(ICM.instance.nb_prod,))

  #Running the prediction
  nO_assort = size(ICM.dataO_probability)[1]
  predictions = zeros((nO_assort,ICM.instance.nb_prod))
  for ass in 1:nO_assort
      new_assortment = find(ICM.dataO_assortments[ass,:])
      tot = sum([weights[i] for i in new_assortment])+0.00001
      for i in new_assortment
        predictions[ass,i] = weights[i] / tot
      end

      predictions[ass,ICM.instance.nb_prod] = 1 - sum(predictions[ass,1:(ICM.instance.nb_prod-1)])
  end
  ini_score = ones(4)
  ini_score[4] = sum(
                     ICM.dataO_assortments .*
                     ((ICM.dataO_probability-predictions).*
                     (ICM.dataO_probability-predictions)./
                     max(0.01,ICM.dataO_probability)
                     )
                    )
  ini_score[3] = sum(
                    ICM.dataO_assortments .*
                    (abs(ICM.dataO_probability-predictions)./(max(0.01,ICM.dataO_probability)))
                    )
  ini_score[1] = sum((ICM.dataO_probability-predictions).*(ICM.dataO_probability-predictions))/
                 sum(ICM.dataO_probability.*ICM.dataO_probability)
  ini_score[2] = sum(abs(ICM.dataO_probability-predictions))/
                 sum(abs(ICM.dataO_probability));
  #println("W",weights)
  #println("P", predictions)
  #println("MNL prediction (MSE): ", MSE)
  #println("MNL prediction (MAE): ", MAE)
  return(ini_score)
end


function learningMNLbased!(ICM::InstanceCvx)
  #= Using the MNL fit to generate a random convex ranking
  =#
  n_assort = size(ICM.dataI_probability)[1]
  #convex preference list covariate matrix
  ICM.X_purchase =zeros(ICM.instance.nb_cst_type,n_assort,ICM.instance.nb_prod)
  #residual objective (linear objective for the subproblem of list selection)
  ICM.res_obj = ICM.dataI_probability
  #MNL weights
  weights = MNLfitscore(ICM)
  #Random generation
  for k = 1:ICM.instance.nb_cst_type
    sampleMNLcvx(ICM,k,weights)
    if k %1000 == 0
      println("Step ",k)
      fitting!(ICM,k)
      scoreO(ICM)
    end
  end

  #Transform left-right endpoints into consideration sets
  for k = 1:ICM.instance.nb_cst_type
     ICM.instance.consideration_sets[k,(ICM.left_endpoints[k]):(ICM.right_endpoints[k])] = 1
  end

end


function MMNLfitscore(ICM::InstanceCvx, mixture::Int64, ret::Int64 = 0)
  #= MLE estimate of an MMNL model through solving a nonlinear optimization
  problem

  mixture <-> size of the mixture
  ret <-> binary parameter indicating the return mode
  0: returns the score vector
  1: returns the first segment of the mixture
  =#

  #max_iter
  m = Model(solver = IpoptSolver(print_level=0,max_iter = 500,tol = 10e-4,acceptable_tol = 10e-4))
  n = ICM.instance.nb_prod
  @variable(m, w[1:mixture,1:n] >=0.)
  @variable(m, lambda[1:mixture] >= 0.)

  @constraint(m, sum{lambda[i], i=1:mixture} == 1.)
  for i = 1:mixture
    @constraint(m, w[i,1]== 1.)
  end

  #@NLconstraint(m, sum{sum{lambda[i]*w[j,i],j=1:mixture}, i=1:n} <= 1.)
  @NLobjective(m, Max, sum{
                          ICM.dataI_probability[k,i]*
                          log(
                              sum{lambda[j]*w[j,i]/sum{w[j,q]*ICM.dataI_assortments[k,q],q = 1:n}, j =1:mixture}
                              ),
                              k = 1:size(ICM.dataI_probability)[1],
                              i = 1:n
                          }
                )

  solve(m)

  # println("μ = ", getvalue(μ))
  # println("mean(data) = ", mean(data))
  # println("σ^2 = ", getvalue(σ)^2)
  # println("var(data) = ", var(data))
  #println("MLE objective: ", getobjectivevalue(m))
  lambda_MMNL = getvalue(lambda)
  weight_MMNL = getvalue(w)

  #println(lambda_MMNL)
  #println(weight_MMNL)

  #Running the prediction
  nO_assort = size(ICM.dataO_probability)[1]
  predictions = zeros((nO_assort,ICM.instance.nb_prod))
  for ass in 1:nO_assort
    for k in 1:mixture
      new_assortment = find(ICM.dataO_assortments[ass,:])
      tot = sum([weight_MMNL[k,i] for i in new_assortment])+0.00001
      for i in new_assortment
        predictions[ass,i] = predictions[ass,i] + lambda_MMNL[k]* weight_MMNL[k,i]/tot
      end
    end
    predictions[ass,ICM.instance.nb_prod] = 1 - sum(predictions[ass,1:(ICM.instance.nb_prod-1)])
  end

  ini_score = ones(4)
  ini_score[4] = sum(
                      ICM.dataO_assortments .*
                      ((ICM.dataO_probability-predictions).*(ICM.dataO_probability-predictions)./
                      max(0.01,ICM.dataO_probability)
                      )
                    )
  ini_score[3] = sum(
                      ICM.dataO_assortments .*
                      (abs(ICM.dataO_probability-predictions)./(max(0.01,ICM.dataO_probability)))
                    )
  ini_score[1] = sum((ICM.dataO_probability-predictions).*(ICM.dataO_probability-predictions))/
                sum(ICM.dataO_probability.*ICM.dataO_probability)
  ini_score[2] = sum(abs(ICM.dataO_probability-predictions))/
                 sum(abs(ICM.dataO_probability));
  #println("W",weights)
  #println("P", predictions)
  #println("MNL prediction (MSE): ", MSE)
  #println("MNL prediction (MAE): ", MAE)
  if ret > 0.5
    return(weight_MMNL[1,:])
  else
    return(ini_score)
  end

end
