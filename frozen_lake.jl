# P: nested dictionary
# 	From gym.core.Environment
# 	For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
# 	tuple of the form (probability, nextstate, reward, terminal) where
# 		- probability: float
# 			the probability of transitioning from "state" to "nextstate" with "action"
# 		- nextstate: int
# 			denotes the state we transition to (in range [0, nS - 1])
# 		- reward: int
# 			either 0 or 1, the reward for transitioning from "state" to
# 			"nextstate" with "action"
# 		- terminal: bool
# 		  True when "nextstate" is a terminal state (hole or goal), False otherwise
# nS: int
# 	number of states in the environment
# nA: int
# 	number of actions in the environment
# gamma: float
# 	Discount factor. Number in range [0, 1)
# Returns: index of action

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

function value_iteration(P, nS, nA, gamma=0.9, tol=1e-3)
	value_function = fill(0.0, nS)
	policy = fill(0, nS)
    println("value iteratio")
	############################
	# YOUR IMPLEMENTATION HERE #


	############################
	return value_function, policy
end

