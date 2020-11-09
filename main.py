from policy_iteration_agent import PolicyIterationAgent
from simulation_runner import SimulationRunner

# sub-task 1: computes the optimal policy
pi_agent = PolicyIterationAgent()
pi_agent.learn()

# sub-task 2: performs a single simulation run for the optimal policy
simulation_runner = SimulationRunner()
simulation_runner.run_simulation(pi_agent)

# sub-task 3: for each state in the check_states array, we print the number representing the state and the value function
# of the state.
print('opt_policy:', pi_agent.get_opt_policy())