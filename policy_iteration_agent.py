import numpy as np
import gym


class PolicyIterationAgent(object):
    def __init__(self, decay_factor=0.95):
        self.problem = gym.make('Taxi-v3')
        self.decay_factor = decay_factor
        self.delta = 0.0001
        self.num_states, self.num_actions = self.problem.observation_space.n, self.problem.action_space.n
        self.check_states = list(range(self.num_states))
        self.R, self.T = self.evaluate_rewards_and_transitions(self.problem)
        self.opt_policy = None
        self.name = 'Policy Iteration Agent'

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

    def get_action(self, s_t):
        return self.opt_policy[s_t]

    def get_opt_policy(self):

        return {i: self.opt_policy[i] for i in self.check_states}

    def learn(self):
        opt_policy = self.policy_iteration()
        self.opt_policy = opt_policy
        return opt_policy
