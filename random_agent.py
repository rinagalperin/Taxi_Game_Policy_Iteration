import gym


class RandomAgent(object):
    def __init__(self):
        self.problem = gym.make('Taxi-v3')
        self.name = 'Random Agent'

    def get_action(self, s_t):
        return self.problem.action_space.sample()



