
from abc import abstractmethod
from util import draw_arg
import numpy as np


class Model(object):
    def __init__(self, env):
        """
        Expected attributes in env:
            model_name
            model_spec
            discount
            costs
            values
            states
            actions
            observations
            T
            Z
            R
        """
        for k, v in env.items():
            # print("k: ", k, ",  v: ", v, "\n")
            # print("\t v.type: ", type(v), "\n")
            self.__dict__[k] = v

        self.curr_state = self.init_state or np.random.choice(self.states)
        print(" $$$$$$  self.init_state:  ", self.init_state)
        print(" $$$$$$  self.curr_state:  ", self.curr_state)

    @property
    def num_states(self):
        return len(self.states)

    @property
    def num_actions(self):
        return len(self.actions)

    def gen_particles(self, n, prob=None):
        if prob is None:
            # by default use uniform distribution for particles generation
            prob = [1 / len(self.states)] * len(self.states)

        return [self.states[draw_arg(prob)] for i in range(n)]

    def get_legal_actions(self, state):
        """
        Simplest situation is every action is legal, but the actual model class
        may handle it differently according to the specific knowledge domain
        :param state:
        :return: actions selectable at the given state
        """
        return self.actions

    def observation_function(self, action, state, obs):
        return self.Z.get((action, state, obs), 0.0)  # 0.0

    def transition_function(self, action, si, sj):
        # print("----- self.T type:  ", type(self.T), str((action, si, sj)))
        # f = open("checkKKKingTransi.txt", "w+")
        # f.write(str(self.T)+"\n")
        # f.close()
        return self.T.get((action, si, sj), 0.0)

    def reward_function(self, action='*', si='*', sj='*', obs='*'):
        # print(" &*&*&*   self.R:  ", self.R)
        # f = open("SelfR3.txt", "w+")
        # f.write("len is: "+str(len(self.R)))
        # f.write(str(self.R) + "\n")
        # f.close()
        return self.R.get((action, si, sj, obs), 0.0)

    def immediate_reward_function(self, action, si, m):
        # print(" &*&*&*   self.R:  ", self.R)
        # f = open("SelfR3.txt", "w+")
        # f.write("len is: "+str(len(self.R)))
        # f.write(str(self.R) + "\n")
        # f.close()
        r = 0
        for next_state in m.states:
            for observation in m.observations:
                r += self.T.get((action, si, next_state), 0.0) * self.Z.get((action, next_state, observation), 0.0) * \
                     self.reward_function(action, si, next_state, observation)
        return r

    def cost_function(self, action):
        if not self.costs:
            return 0
        return self.costs[self.actions.index(action)]

    def simulate_action(self, si, ai, debug=True):
        """
        Query the resultant new state, observation and rewards, if action ai is taken from state si

        si: current state
        ai: action taken at the current state
        return: next state, observation and reward
        """
        # get new state
        s_probs = [self.transition_function(ai, si, sj) for sj in self.states]
        print(" @@@@@@ (ai, si):  ", ai, si)
        state = self.states[draw_arg(s_probs)]

        # get new observation
        # print("*********** ai, state: ", ai, state)
        # print(" self.observations:  ", self.observations)
        o_probs = [self.observation_function(ai, state, oj) for oj in self.observations]
        # print(" $#$#$#$#$# o_probs: ", str(o_probs))
        observation = self.observations[draw_arg(o_probs)]

        if debug:
            print('taking action {} at state {}'.format(ai, si))
            print('transition probs: {}'.format(s_probs))
            print('obs probs: {}'.format(o_probs))

        # get new reward
        reward = self.reward_function(ai, si, state, observation) #  --- THIS IS MORE GENERAL!
        # reward = self.immediate_reward_function(ai, si)   # --- THIS IS TMP SOLUTION!
        print(" ###### reward: ", reward)
        cost = self.cost_function(ai)

        return state, observation, reward, cost

    def take_action(self, action):
        """
        Accepts an action and changes the underlying environment state
        
        action: action to take
        return: next state, observation and reward
        """
        # print(" @#$%^&* self.curr_state:  ", self.curr_state)
        # print(" @#$%^&* action:  ", action)
        state, observation, reward, cost = self.simulate_action(self.curr_state, action)
        self.curr_state = state
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print("  state: ", state, ", observation: ", observation, ", reward: ", reward)
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return state, observation, reward, cost

    def print_config(self):
        print("discount:", self.discount)
        print("values:", self.values)
        print("states:", self.states)
        print("actions:", self.actions)
        print("observations:", self.observations)
        print("")
        print("T:", self.T)
        print("")
        print("Z:", self.Z)
        print("")
        print("R:", self.R)
        print("")
