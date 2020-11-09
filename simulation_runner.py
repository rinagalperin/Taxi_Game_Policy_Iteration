import gym
from gym.envs.toy_text import taxi


class SimulationRunner:
    def __init__(self):
        self.problem = gym.make('Taxi-v3')

    def run_simulation(self, agent):
        print('---- start simulation with {} ----'.format(agent.name))
        env = taxi.TaxiEnv()
        t = 0
        s_t = self.problem.reset()
        done = False
        score = 0  # reword sum
        env.render()

        while not done and t < 100:
            a_t = agent.get_action(s_t)
            s_t, r_t, done, _ = env.step(a_t)
            score += r_t
            t += 1
            env.render()

        return score