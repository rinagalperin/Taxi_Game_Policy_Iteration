from policy_iteration_agent import PolicyIterationAgent
from simulation_runner import SimulationRunner

pi_agent = PolicyIterationAgent()
pi_agent.learn()

simulation_runner = SimulationRunner()
simulation_runner.run_simulation(pi_agent)

print('opt_policy:', pi_agent.get_opt_policy())