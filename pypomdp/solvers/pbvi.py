import numpy as np

from solvers import Solver
from util.alpha_vector import AlphaVector
from array import array
from numpy import *

MIN = -np.inf


class PBVI(Solver):
    def __init__(self, model):
        Solver.__init__(self, model)
        self.belief_points = None
        self.alpha_vecs = None
        self.solved = False

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

        # self.gamma_reward = {
        #     a: np.frombuffer(array('d', [self.model.immediate_reward_function(a, s, m) for s in self.model.states]))
        #     for a in self.model.actions
        # }
        # temp = {}
        # for
        # self.gamma_reward = {
        #     a: (np.array([self.model.reward_function(a, s) for s in self.model.states]))
        #     for a in self.model.actions
        # }

        # f = open("GammaReward9.txt", "w+")
        # # f.write("self.model.reward_function(0, 1): " + str(self.model.reward_function(action="0", sj="1")))
        # f.write(str(self.gamma_reward) + "\n")
        # f.close()

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
        #     print("In if")
        #     return

        # ****** BACKUP( B, Gamma)  for planning horizon T ******
        m = self.model
        for step in range(T):

            # print(" oooo  Step: ", step, "  oooo")
            # print(" ***** size alpha_vec:  ", len(self.alpha_vecs))
            # STEP 1
            # First compute a set of updated vectors for every action/observation pair
            # Action(a) => Observation(o) => UpdateOfAlphaVector (a, o)
            gamma_intermediate = {
                a: {
                    o: self.compute_gamma_action_obs(a, o)
                    for o in m.observations
                } for a in m.actions
            }

            # f = open("gamma_intermediate5.txt", "a+")
            # f.write(str(gamma_intermediate))
            # f.write(" \n ***************   \n")
            # f.close()

            # STEP 2
            # Now compute the cross sum
            gamma_action_belief = {}
            for a in m.actions:

                gamma_action_belief[a] = {}
                for bidx, b in enumerate(self.belief_points):

                    gamma_action_belief[a][bidx] = self.gamma_reward[a].copy()

                    for o in m.observations:
                        # only consider the best point
                        # print(" %%%%%%%% gamma_intermediate[a][o]: ", gamma_intermediate[a][o])
                        # print(" %%%%%%%% b: ", b)
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

                # print(" ****** best_av: ", best_av)
                # print(" ************* type best_av: ", type(best_av))
                if len(self.alpha_vecs) == 0:
                    self.alpha_vecs.append(AlphaVector(a=best_aa, v=best_av))
                elif not self.is_duplicate(best_aa, best_av):
                    self.alpha_vecs.append(AlphaVector(a=best_aa, v=best_av))

            # str_step2 = str(step)
            # file_name = "Alpha_vecstors" + str_step2 + ".txt"
            # f = open(file_name, "w+")
            # f.write(str(best_aa)+"\t")
            # f.write(str(best_av))
            # print(" @@@@@@@@@@@ len alpha vector", len(self.alpha_vecs))
            # print(" ########## type: ", type(self.alpha_vecs))
            # for alph_vector in self.alpha_vecs:
            #     for i in range(len(alph_vector.v)):
            #         f.write(str(alph_vector.v[i]) + "\t")
            #     f.write("\n")
            # f.close()

            # str_step3 = str(step)
            # file_name2 = "ActionsAlpha" + str_step3 + ".txt"
            # f = open(file_name2, "a+")
            # for alph_vector in self.alpha_vecs:
            #     f.write(str(alph_vector.action) + "\n")
            # f.close()

        # ****** EXPAND(B, Gamma) for planning horizon T ******
        B_old = self.belief_points
        num_belief_points = 10
        B_new = self.random_generate_belief_points(num_belief_points, m.num_states)
        self.belief_points = B_old + B_new
        # my_belief = self.remove_duplicate()
        print(" ^^^^^^^^^  size belie points: ", len(self.belief_points))
        f = open("BeliefPoints", "a+")
        for belief in self.belief_points:
            f.write(str(belief) + "\n")
        f.close()

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

        equal_vecs = []
        # f = open("logEqualVecs.txt", "a+")
        for av in self.alpha_vecs:
            v = np.dot(av.v, belief)
            if abs(v - max_v) < 0.00000001:  # which means (v == best)
                equal_vecs.append(av)
                # f.write(" " + str(av.action) + "\t")
                # f.write("  " + str(av.v))
                # f.write("\n")
        if len(equal_vecs) != 0:
            best = np.random.choice(equal_vecs)
            # print(" ******  random choice:   ", best.action)
            # print(" *****  ", len(equal_vecs))
        # f.write(" \n ******* \n")
        # f.close()

        return best.action

    def choose_random_act(self, actions):
        m = self.model
        random_action_index = np.random.randint(low=1, high=m.num_actions)
        # print("**** m.num_actions: ", m.num_actions)
        random_action = actions[random_action_index]
        print("**** random_action: ", random_action)
        # print(" type action chosen: ", type(random_action))
        return random_action

    def update_belief(self, belief, action, obs):
        m = self.model

        b_new = []

        for sj in m.states:
            p_o_prime = m.observation_function(action, sj, obs)
            summation = 0.0
            for i, si in enumerate(m.states):
                p_s_prime = m.transition_function(action, si, sj)
                summation += p_s_prime * float(belief[i])
            b_new.append(p_o_prime * summation)

        # normalize
        total = sum(b_new)
        return [x / total for x in b_new]

    def random_generate_belief_points(self, num_belief_points, num_states):
        belief_points = []

        for i in range(num_belief_points):
            b_temp = []
            for j in range(num_states):
                b_temp.append(random.uniform(low=0.0, high=1.0))
            b_temp.sort()
            # print(" ****  IN random; b_temp: ", str(b_temp))
            # print(" len b_temp", len(b_temp))
            b_new = []
            for k in range(0, len(b_temp) - 1):
                b_new.append(b_temp[k + 1] - b_temp[k])
            b_new.append(1.0 - sum(b_new))
            # print(" ****  IN random; b_new: ", str(b_new))
            # print(" len b_new", len(b_new))
            # print("*** sum :", sum(b_new))
            if b_new not in belief_points:
                belief_points.append(b_new)

        # f = open("BeleifPoints1.txt", "a+")
        # for element in belief_points:
        #     f.write(str(element))
        #     f.write("\n ")
        # f.write("\n ******* \n")
        # f.close()

        return belief_points

    def is_duplicate(self, act, vec):
        # f = open("duplication6.txt", "a+")
        # f.write("  act: " + str(act) + " vec:" + str(vec) + "\n")
        #
        # for element in self.alpha_vecs:
        #     f.write("alpha act: " + element.action)
        #     for counter in range(len(element.v)):
        #         f.write(str(element.v[counter]) + "\t")
        #     f.write("\n ")
        # f.write("\n ******* \n")
        # f.close()

        for element in self.alpha_vecs:
            # print(" %%%%% vec: ", vec)
            # print(" %%%%% element.v: ", element.v)
            if element.action == act and self.is_same_vector(element.v, vec):
                # print(" %%%%%%% True \n ")
                return True

        return False

    def is_same_vector(self, vec1, vec2):
        if len(vec1) != len(vec2):
            return False
        else:
            for i in range(len(vec1)):
                if abs(vec1[i] - vec2[i]) > 0.00000001:
                    return False
            return True
