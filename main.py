from policy_iteration_agent import PolicyIterationAgent
from random_agent import RandomAgent
from simulation_runner import SimulationRunner

pi_agent = PolicyIterationAgent()
pi_agent.learn()

print('opt_policy:', pi_agent.get_opt_policy())

simulation_runner = SimulationRunner()
simulation_runner.run_simulation(pi_agent)

random_agent = RandomAgent()
simulation_runner.run_simulation(random_agent)
