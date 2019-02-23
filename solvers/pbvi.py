import numpy as np

from solvers import Solver
from util.alpha_vector import AlphaVector
from array import array
from util import draw_arg
from numpy import *

MIN = -np.inf
EQUALITY_EPSILON = 0.00000001


class PBVI(Solver):
    def __init__(self, model):
        Solver.__init__(self, model)
        self.belief_points = None
        self.alpha_vecs = None
        self.solved = False
        self.max_size_belief_points = 1000

    def add_configs(self, belief_points):
        Solver.add_configs(self)
        self.alpha_vecs = [AlphaVector(a=-1, v=np.zeros(self.model.num_states))]  # filled with a dummy alpha vector

        self.belief_points = belief_points
        self.compute_gamma_reward()

    def compute_gamma_reward(self):
        """
        :return: Action_a => Reward(s,a) matrix
        """
        # 'halt': array([ 0.,  0.]),
        m = self.model
        self.gamma_reward = {
            a: np.array([self.model.immediate_reward_function(a, s, m) for s in self.model.states])
            for a in self.model.actions
        }

    def compute_gamma_action_obs(self, a, o):
        """
        Computes a set of vectors, one for each previous alpha
        vector that represents the update to that alpha vector
        given an action and observation

        :param a: action index
        :param o: observation index
        """
        m = self.model
        # print(" &&&&&&&&&&&&&& in compute func")
        gamma_action_obs = []
        # print(" &&&&&&&&&&&&&&& self.alpha_vecs:  ", self.alpha_vecs)
        for alpha in self.alpha_vecs:
            v = np.zeros(m.num_states)  # initialize the update vector [0, ... 0]
            for i, si in enumerate(m.states):
                for j, sj in enumerate(m.states):
                    v[i] += m.transition_function(a, si, sj) * \
                            m.observation_function(a, sj, o) * \
                            alpha.v[j]
                v[i] *= m.discount
            gamma_action_obs.append(v)

        # print("********* gamma_action_obs: ", gamma_action_obs)
        return gamma_action_obs

    def solve(self, T):
        # We want it always solve the problem to see the performance during time
        # if self.solved:
        #     return

        # ******   BACKUP( B, Gamma)  for planning horizon T   ******
        m = self.model
        for step in range(T):

            # print(" oooo  Step: ", step, "  oooo")

            # ********   STEP 1   ********
            # First compute a set of updated vectors for every action/observation pair
            # Action(a) => Observation(o) => UpdateOfAlphaVector (a, o)
            gamma_intermediate = {
                a: {
                    o: self.compute_gamma_action_obs(a, o)
                    for o in m.observations
                } for a in m.actions
            }

            # ********   STEP 2   ********
            # Now compute the cross sum
            gamma_action_belief = {}
            for a in m.actions:

                gamma_action_belief[a] = {}
                for bidx, b in enumerate(self.belief_points):

                    gamma_action_belief[a][bidx] = self.gamma_reward[a].copy()

                    for o in m.observations:
                        # only consider the best point
                        best_alpha_idx = np.argmax(np.dot(gamma_intermediate[a][o], b))

                        gamma_action_belief[a][bidx] += gamma_intermediate[a][o][best_alpha_idx]

            # Finally compute the new(best) alpha vector set
            self.alpha_vecs, max_val = [], MIN

            for bidx, b in enumerate(self.belief_points):
                best_av, best_aa = None, None

                for a in m.actions:
                    val = np.dot(gamma_action_belief[a][bidx], b)
                    if best_av is None or val > max_val:
                        max_val = val
                        best_av = gamma_action_belief[a][bidx].copy()
                        best_aa = a

                if len(self.alpha_vecs) == 0:
                    self.alpha_vecs.append(AlphaVector(a=best_aa, v=best_av))
                elif not self.is_duplicate(best_aa, best_av):
                    self.alpha_vecs.append(AlphaVector(a=best_aa, v=best_av))


        # ****** EXPAND(B, Gamma) for planning horizon T ******
        # B_old = []
        # for element in self.belief_points:
        #     B_old.append(element)

        # First Method for belief generation: Fully random
        B_new = self.random_generate_belief_points(m.num_states)

        # Second Method for belief generation: Stochastic Simulation with Random Action (SSRA)
        # B_new = self.stochastic_simulation_random_action()

        # Third Method for belief generation: Stochastic Simulation with Greedy Action (SSGA)
        # B_new = self.stochastic_simulation_greedy_action()

        self.belief_points = self.belief_points + B_new
        if len(self.belief_points) > self.max_size_belief_points:
            temp_belief = []
            belief_indices = random.choice(len(self.belief_points), size=self.max_size_belief_points, replace=False)
            for i in belief_indices:
                temp_belief.append(self.belief_points[i])
            self.belief_points = temp_belief

        print(" ^^^^^^^^^  size belie points: ", len(self.belief_points))
        # f = open("BeliefPoints", "a+")
        # for belief in self.belief_points:
        #     f.write(str(belief) + "\n")
        # f.write("\n")
        # f.close()

        self.solved = True

    def get_action(self, belief):
        max_v = -np.inf
        best = None
        for av in self.alpha_vecs:
            v = np.dot(av.v, belief)
            if v > max_v:
                max_v = v
                best = av

        return best.action

    def get_greedy_action(self, belief):

        max_v = -np.inf
        best = None
        for av in self.alpha_vecs:
            v = np.dot(av.v, belief)
            if v > max_v:
                max_v = v
                best = av

        # If there are more than one best vector, it's better to choose random from them
        equal_vecs = []
        for av in self.alpha_vecs:
            v = np.dot(av.v, belief)
            if abs(v - max_v) < EQUALITY_EPSILON:  # which means (v == best)
                equal_vecs.append(av)

        if len(equal_vecs) > 1:
            best = np.random.choice(equal_vecs)

        return best.action

    def choose_random_act(self, actions):

        random_action_index = np.random.randint(low=1, high=self.model.num_actions)
        random_action = actions[random_action_index]
        return random_action
    #
    # def update_belief(self, belief, action, obs):
    #     m = self.model
    #
    #     b_new = []
    #
    #     for sj in m.states:
    #         p_o_prime = m.observation_function(action, sj, obs)
    #         summation = 0.0
    #         for i, si in enumerate(m.states):
    #             p_s_prime = m.transition_function(action, si, sj)
    #             summation += p_s_prime * float(belief[i])
    #         b_new.append(p_o_prime * summation)
    #
    #     # normalize
    #     total = sum(b_new)
    #     return [x / total for x in b_new]

    def update_belief(self, old_belief, action, obs):
        b_new = []
        # f = open("updateBelief.txt", "w+")
        # f.write(" old_belief: " + str(old_belief) + "\n")
        # f.write(" action, obs: " + str((action, obs)) + "\n")

        for sj in self.model.states:
            p_o_prime = self.model.observation_function(action, sj, obs)
            # f.write("  ********************************************* \n ")
            # f.write(" p_o_prime" + str((action, sj, obs)) + " : " + str(p_o_prime) + "\n")
            summation = 0.0
            for i, si in enumerate(self.model.states):
                p_s_prime = self.model.transition_function(action, si, sj)
                # f.write("T" + str((action, si, sj)) + " : " + str(p_s_prime) + "\n")
                summation += p_s_prime * float(old_belief[i])
            # f.write(" summation: " + str(summation) + "\n")
            # f.write("p_o_prime * summation: " + str(p_o_prime * summation) + "\n")
            b_new.append(p_o_prime * summation)

        # normalize
        # f.write("un-normalized b_new:  " + str(b_new) + "\n")
        total = sum(b_new)
        # f.write(" total:  " + str(total) + "\n")
        normalized_b = [x / total for x in b_new]
        # f.write(" normalized_b: " + str(normalized_b) + "\n \n")
        # f.close()
        return normalized_b

    def random_generate_belief_points(self, num_states):
        belief_points = []
        num_new_random_belief_points = 10
        for i in range(num_new_random_belief_points):
            b_temp = []
            for j in range(num_states):
                b_temp.append(random.uniform(low=0.0, high=1.0))
            b_temp.sort()

            b_new = []
            for kk in range(0, len(b_temp) - 1):
                b_new.append(b_temp[kk + 1] - b_temp[kk])
            b_new.append(1.0 - sum(b_new))

            if b_new not in belief_points:
                belief_points.append(b_new)

        return belief_points

    # Checking whether an alpha_vector object already belongs to Alpha vector list
    def is_duplicate(self, act, vec):

        for element in self.alpha_vecs:
            if element.action == act and self.is_same_vector(element.v, vec):
                return True

        return False

    def is_same_vector(self, vec1, vec2):
        if len(vec1) != len(vec2):
            return False
        else:
            for i in range(len(vec1)):
                if abs(vec1[i] - vec2[i]) > EQUALITY_EPSILON:
                    return False
            return True

    # Stochastic Simulation with Random Action (SSRA)
    def stochastic_simulation_random_action(self):
        B = []
        for belief in self.belief_points:
            si = self.model.states[draw_arg(belief)]
            action = self.model.actions[random.choice(range(self.model.num_actions))]
            s_probs = [self.model.transition_function(action, si, next_state) for next_state in self.model.states]
            sj = self.model.states[draw_arg(s_probs)]
            o_probs = [self.model.observation_function(action, sj, oj) for oj in self.model.observations]
            observation = self.model.observations[draw_arg(o_probs)]
            new_belief = self.update_belief(belief, action, observation)
            if not self.belongs_to(new_belief, self.belief_points):
                B.append(new_belief)

        return B

    # Stochastic Simulation with Greedy Action (SSGA)
    def stochastic_simulation_greedy_action(self):
        B = []
        for belief in self.belief_points:
            si = self.model.states[draw_arg(belief)]

            # Implementing epsilon greedy policy with e = 0.1
            random_num = np.random.randint(10)
            if random_num == 0:
                action = self.choose_random_act(self.model.actions)
            else:
                action = self.get_greedy_action(belief)
            # action = self.model.actions[random.choice(range(self.model.num_actions))]

            s_probs = [self.model.transition_function(action, si, next_state) for next_state in self.model.states]
            sj = self.model.states[draw_arg(s_probs)]
            o_probs = [self.model.observation_function(action, sj, oj) for oj in self.model.observations]
            observation = self.model.observations[draw_arg(o_probs)]
            new_belief = self.update_belief(belief, action, observation)
            if not self.belongs_to(new_belief, self.belief_points):
                B.append(new_belief)

        return B

    def belongs_to(self, b, belief_set):

        for belief in belief_set:
            if self.is_same_vector(belief, b):
                return True

        return False

