using StatsBase
mutable struct AgentData
    # Since Julia is not an object oriented language, 
    # [action_1_q ...]
    q::Array{Float64}
    # [action_1_occurance]
    action_nums::Array{Float64}
    # [reward1 ...]
    rewards_history::Array{Float64}
    EPSILON::Float64
    OPTIMISTIC_INITIALIZED::Bool
end

function init_agent_data(k::Int) 
    agent_data = AgentData(fill(0, k), fill(0, k),[], 0.2, false)
    return agent_data
end

function update_agent_data!(agent_data::AgentData, reward::Float64, action::Int)
    # need to subtract by 1 because Julia array indexing starts from 1
    action += 1
    agent_data.action_nums[action] += 1
    # K-Armed Bandit's incremental update already is alpha-step update
    agent_data.q[action] = agent_data.q[action] + (reward - agent_data.q[action]) / agent_data.action_nums[action]
    push!(agent_data.rewards_history, reward)
end

function get_action(agent_data::AgentData)
    EPSILON = agent_data.EPSILON
    pi = rand()
    if pi < EPSILON 
        k = length(agent_data.action_nums)
        # need to subtract by 1 because Julia array indexing starts from 1
        return rand(1: k) - 1
    else
        # need to subtract by 1 because Julia array indexing starts from 1
        return argmax(agent_data.q) - 1
    end
end
    
function get_action_decaying_epsilon(agent_data::AgentData, optimistic::Bool, use_ucb::Bool)
    if optimistic && !agent_data.OPTIMISTIC_INITIALIZED
        agent_data.q = fill(5, length(agent_data.q))
        agent_data.OPTIMISTIC_INITIALIZED = true
    end

    EPISODE_LEARNING_RATE = 0.98
    agent_data.EPSILON *= EPISODE_LEARNING_RATE
    pi = rand()
    if pi < agent_data.EPSILON
        k = length(agent_data.action_nums)
        action = rand(1: k)
    else
        if use_ucb
            # Ucb stands for upper confidence bound. Here we are also evaluating other "potential"
            # Candidates
            num_steps = sum(agent_data.action_nums)
            CONFIDENCE = 0.5
            # must use 0.0 instead of 0, otherwise julia complains about typing
            q_ucb = [0.0 for i in 1:length(agent_data.q)]
            for (i, q) in enumerate(agent_data.q)
                if agent_data.action_nums[i] != 0
                    q_ucb[i] = q + CONFIDENCE * sqrt(log(num_steps)/agent_data.action_nums[i])
                else
                    q_ucb[i] = q
                end
            end
            action=argmax(q_ucb)
        else
            action = argmax(agent_data.q)
        end
    end
    # need to subtract by 1 because Julia array indexing starts from 1
    return action - 1
end

## SGA Method
mutable struct AgentDataSGA
    # Since Julia is not an object oriented language, 
    # [action_1_q ...]
    pi::Array{Float64}
    h::Array{Float64}
    # [reward1 ...]
    rewards_history::Array{Float64}
end

function init_agent_data_SGA(k::Int)
    # because we want to have an initial value
    agent_data_sga = AgentDataSGA(fill(1.0/k, k), fill(0.0, k),[0.0])
    return agent_data_sga
end

function update_agent_data_SGA!(agent_data::AgentDataSGA, reward::Float64, action::Int)
    # need to add 1 because Julia array indexing starts from 1
    action += 1
    ALPHA = 1
    tmp_pi = [exp(h) for h in agent_data.h]
    denominator = sum(tmp_pi)
    agent_data.pi = [tmp_pi[i] / denominator for i in 1:length(tmp_pi)]
    for (i, h) in enumerate(agent_data.h)
        if i == action
            m = 1 - agent_data.pi[i]
        else
            m = -agent_data.pi[i]
        end
        agent_data.h[action] += ALPHA * (reward - mean(agent_data.rewards_history)) * m
    end
    push!(agent_data.rewards_history, reward)
end

function get_action_SGA(agent_data::AgentDataSGA)
    k = length(agent_data.pi)
    # must make it weights
    action = sample(1:k, Weights(agent_data.pi))
    # need to subtract by 1 because Julia array indexing starts from 1
    return action - 1
end