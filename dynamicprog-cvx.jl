
function dynamicprogCvx(I::InstanceCvx)
  #=
  Collapsed DP on the quasi-convex instance
  =#
  # Matrix containing the rankings functions over products
  ranking_mat = zeros((I.instance.nb_cst_type,I.instance.nb_prod+1))
  ranking_mat[:,2:(I.instance.nb_prod+1)] = I.instance.consideration_sets.*I.rankings

  # Value function
  decision_value = zeros((I.instance.nb_prod+1,I.instance.nb_prod))

  # Optimal dynamic programming decisions
  decision_bin = zeros((I.instance.nb_prod+1,I.instance.nb_prod))

  # for k = 1:I.instance.nb_cst_type
  #   ranking_mat[k,(1+I.left_endpoints[k]):(1+I.right_endpoints[k])] = 1
  #   ranking_mat[k,2:] = ranking_mat[k,2:].*I.rankings[k,:]
  # end

  # Backward induction
  for i1 = 3:(I.instance.nb_prod+1)
    for i2 = 2:(i1-1)
      #println(i1,i2)
      d_list = [
                (
                decision_value[i2,i3] +
                sum(I.instance.lambdas.*(ranking_mat[:,i2] .>
                ranking_mat[:,i3]).*(ranking_mat[:,i2] .>= ranking_mat[:,i1]))*
                I.instance.prices[i2-1],
                i3
                )
                for i3= (1):(i2-1)
                ]

      # DP updates
      decision_value[i1,i2] = maximum(d_list)[1]
      decision_bin[i1,i2] = maximum(d_list)[2]

    end
  end
  #println(maximum(decision_value))

end


function MIPsolver(ICM::InstanceCvx)
  #=
  IP solution, using JuMP & Gurobi
  Termination at 1% optimality gap & time limit of 1000 sec.
  =#
  for i = 1:size(ICM.rankings)[1]
    for j = 1:size(ICM.rankings)[2]
      ICM.rankings[i,j] = ICM.rankings[i,j] + 0.01*j
    end
  end
  #println("OK")
  t0 = tic()
  m = Model(solver = GurobiSolver(OutputFlag=0, MIPGap = 0.01, TimeLimit = 1000))
  @variable(m, Assortment[1:ICM.instance.nb_prod],Bin)
  @variable(m, X_purchase[
                          u = 1:ICM.instance.nb_cst_type,
                          v=find(ICM.instance.consideration_sets[u,:])
                          ],
              Bin)

  for k = 1:ICM.instance.nb_cst_type
    #println(size(X_purchase[k,:]))
    #println(size(Assortment))
    for i in 1:ICM.instance.nb_prod
      if ICM.instance.consideration_sets[k,i] == 1
        @constraint(m, X_purchase[k,i] <= Assortment[i])
      end
    end

    for i in find(ICM.instance.consideration_sets[k,:])
      for ipref in  find(ICM.instance.consideration_sets[k,:].*(ICM.rankings[k,:].>ICM.rankings[k,i]))
        @constraint(m, X_purchase[k,i] + Assortment[ipref] <= 1.)
      end
    end

    # Tightening of the LP
    @constraint(m, sum([X_purchase[k,i] for i in find(ICM.instance.consideration_sets[k,:])]) <= 1.)

  end
  @objective(m, Max, sum([
                          ICM.instance.lambdas[k]*
                          sum([X_purchase[k,i]*ICM.instance.prices[i]
                               for i in find(ICM.instance.consideration_sets[k,:])]
                             )
                          for k=1:ICM.instance.nb_cst_type
                          ]
                        )
            )
  tic()
  status = solve(m)
  return(toq())
end
