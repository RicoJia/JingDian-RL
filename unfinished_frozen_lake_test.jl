# README : THIS IS STILL BUGGY, this is why it's called' unfinished
# RUN THIS FIRST TO build necessary packages
ENV["PYTHON"]="/usr/bin/python3"
using Pkg
Pkg.add("PyCall")
Pkg.build("PyCall")
Pkg.add("Plots")
Pkg.add("Distributions")
Pkg.add("StatsPlots")

using PyCall
using Libdl
using Plots
plot()

MAX_STEPS = 1000
# add current directory to search path
pushfirst!(PyVector(pyimport("sys")["path"]), "")
env = pyimport("frozen_lake_env").FrozenLakeEnv
deterministic_env = env(map_name="4x4",is_slippery=false)
deterministic_large_env = env(map_name="8x8",is_slippery=false)
stochastic_env = env(map_name="4x4",is_slippery=true)
pyimport("sys").stdout.flush()


function policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3)
	# policy: np.array[nS]. The policy to evaluate. Maps states to actions.
    value_function = fill(0.0, nS)

	return value_function
end

function policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9)
	# Given the value function from policy improve the policy.
	new_policy = fill(0.0, nS)

	############################
	# YOUR IMPLEMENTATION HERE #


	############################
	return new_policy

end


function policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3)

	value_function = fill(0.0, nS)
	policy = fill(0, ns)

	############################
	# YOUR IMPLEMENTATION HERE #


	############################
	return value_function, policy

end

#   P[s][a] == [(probability, nextstate, reward, done), ...]
function value_iteration(P, nS, nA, gamma=0.9, tol=1e-3)
	value_function = fill(-100.0, nS)
    value_function_next = fill(0.0, nS)
    # julia -> python
	policy = fill(1, nS)
    println("value iteration")
	############################
	# YOUR IMPLEMENTATION HERE #
    value_function = value_function_next
    for state in 1:length(value_function)
        next_action = policy[state]
        print(state, next_action, length(P[state]))
        next_transitions = P[state][next_action]
        for next_transition in next_transitions
            probability, next_state, reward, done = next_transition
        end
        # Python -> Julia
        next_state += 1
    # while true
    #     if abs.(value_function_next .- value_function) .< tol:
    #         # Converged
    #         break
    #     value_function = value_function_next
    #     for state in 1:length(value_function)
    #         next_action = policy[state]
    #         probability, next_state, reward, done = P[state][next_action]
    #         # Python -> Julia
    #         next_state += 1
    #     end
    # end
    end

	############################
	return value_function, policy
end

#####################################################
# Testing - TODO: this is still buggy
#####################################################
function train(env, description, value_iteration)
  episode_reward = 0
  # Julia array starts from 1
  ob = env.reset() + 1
  P, nS, nA = env.P, env.nS, env.nA
  for t in 1:MAX_STEPS
    env.render("frozen_lake_state_output.tmp")
    sleep(0.25)
    _, policy = value_iteration(P, nS, nA)
    println( ob)
    a = policy[ob]
    ob, rew, done, _ = env.step(a)
    episode_reward += rew
    #TODO
    done = true
    if !done
        println("The agent didn't reach a terminal state in $MAX_STEPS steps.")
    else
        println("Episode reward: $episode_reward")
        break
    end
  end

end

train(deterministic_env, "Deterministic-4x4-FrozenLake-v0", value_iteration)
# train(deterministic_large_env, "Deterministic-8x8-FrozenLake-v0", value_iteration)
# train(stochastic_env, "Stochastic-4x4-FrozenLake-v0", value_iteration)

pyimport("sys").stdout.flush()
