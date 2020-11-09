import numpy as np
import gym
from gym.envs.toy_text import taxi


class PolicyIterationAgent(object):
    def __init__(self):
        self.problem = gym.make('Taxi-v3')
        self.decay_factor = 0.95
        self.delta = 0.0001
        self.num_states, self.num_actions = self.problem.observation_space.n, self.problem.action_space.n
        self.R, self.T = self.evaluate_rewards_and_transitions(self.problem)

    def evaluate_rewards_and_transitions(self, problem):
        R = np.zeros((self.num_states, self.num_actions, self.num_states))
        T = np.zeros((self.num_states, self.num_actions, self.num_states))

        for s in range(self.num_states):
            for a in range(self.num_actions):
                for transition in problem.env.P[s][a]:
                    probability, next_state, reward, done = transition
                    R[s, a, next_state] = reward
                    T[s, a, next_state] = probability

                T[s, a, :] /= np.sum(T[s, a, :])

        return R, T

    def encode_policy(self, policy, shape):
        encoded_policy = np.zeros(shape)
        encoded_policy[np.arange(shape[0]), policy] = 1
        return encoded_policy

    def get_net_q(self, v_t):
        v_next = np.zeros((self.num_states, self.num_actions))

        for s in range(self.num_states):
            for a in range(self.num_actions):
                v_next[s][a] = np.sum(self.T[s][a] * (self.R[s][a] + self.decay_factor * v_t))

        return v_next

    def policy_estimate(self, policy):
        value_fn = np.zeros(self.num_states)
        while True:
            previous_value_fn = value_fn.copy()
            Q = self.get_net_q(value_fn)
            value_fn = np.sum(self.encode_policy(policy, (self.num_states, self.num_actions)) * Q, 1)
            if np.max(np.abs(previous_value_fn - value_fn)) < self.delta:
                return value_fn

    def policy_update(self, value_fn):
        Q = self.get_net_q(value_fn)
        return np.argmax(Q, axis=1)

    def policy_iteration(self):
        policy = np.array([self.problem.action_space.sample() for _ in range(self.num_states)])

        while True:
            previous_policy = policy.copy()
            value_fn = self.policy_estimate(policy)
            policy = self.policy_update(value_fn)
            if np.array_equal(policy, previous_policy):
                break

        return policy

    def run_simulation(self, pol_policy, render=False):
        env = taxi.TaxiEnv()
        t = 0
        s_t = self.problem.reset()
        done = False
        score = 0  # reward sum

        while not done and t < 100:
            s_t, r_t, done, _ = env.step(pol_policy[s_t])
            score += r_t
            t += 1
            if render:
                env.render()

        return score

    def learn(self):
        opt_policy = self.policy_iteration()
        score = []
        for _ in range(1000):
            score.append(self.run_simulation(opt_policy))

        print('opt_policy:', opt_policy.tolist())
        print('average score:', np.average(score))
        print('---- start simulation ----')
        print(self.run_simulation(opt_policy, render=True))

        return opt_policy
