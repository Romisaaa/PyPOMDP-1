import os

from models import RockSampleModel, Model
from solvers import POMCP, PBVI
from parsers import PomdpParser, GraphViz
from logger import Logger as log
import numpy as np


class PomdpRunner:

    def __init__(self, params):
        self.params = params
        if params.logfile is not None:
            log.new(params.logfile)

    def create_model(self, env_configs):
        """
        Builder method for creating model (i,e, agent's environment) instance
        :param env_configs: the complete encapsulation of environment's dynamics
        :return: concrete model
        """
        MODELS = {
            'RockSample': RockSampleModel,
        }
        # print(" ****** env_configs['model_name']: ", env_configs['model_name'])
        return MODELS.get(env_configs['model_name'], Model)(env_configs)

    def create_solver(self, algo, model):
        """
        Builder method for creating solver instance
        :param algo: algorithm name
        :param model: model instance, e.g, TigerModel or RockSampleModel
        :return: concrete solver
        """
        print(" ***** algo:  ", algo)
        print(" ***** model: ", model)

        SOLVERS = {
            'pbvi': PBVI,
            'pomcp': POMCP,
        }
        # print(" **** SOLVERS.get(algo): ", SOLVERS.get(algo))
        # print(" **** SOLVERS.get(algo)(model) : ", SOLVERS.get(algo)(model))
        return SOLVERS.get(algo)(model)

    def snapshot_tree(self, visualiser, tree, filename):
        visualiser.update(tree.root)
        visualiser.render('./dev/snapshots/{}'.format(filename))  # TODO: parametrise the dev folder path

    def run(self, algo, T, **kwargs):
        visualiser = GraphViz(description='tmp')
        params, pomdp = self.params, None
        total_rewards, budget = 0, params.budget

        log.info('~~~ initialising ~~~')
        with PomdpParser(params.env_config) as ctx:
            # creates model and solver
            model = self.create_model(ctx.copy_env())
            pomdp = self.create_solver(algo, model)

            # supply additional algo params
            belief = ctx.random_beliefs() if params.random_prior else ctx.generate_beliefs()

            # Just for Russel 4x3 problem we changed the belief since all sates are not equiprobable
            #  the agent should not be in terminal states 3 and 6
            # belief = [0.111111, 0.111111, 0.111111, 0.0, 0.111111, 0.111111, 0.0, 0.111112, 0.111111, 0.111111,
            #           0.111111]

            # belief for tiger-grid
            # belief = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            #           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0,
            #           0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

            if algo == 'pbvi':
                print("num_states: ", model.num_states)
                num_belief_points = 5
                belief_points = ctx.random_generate_belief_points(num_belief_points, model.num_states)
                # belief_points = ctx.generate_belief_points(kwargs['stepsize'])
                pomdp.add_configs(belief_points)
            elif algo == 'pomcp':
                pomdp.add_configs(budget, belief, **kwargs)

            # Added by Sara; Current state must be selected on the basis of initial belief
            # At this moment me make it possible
            index = np.random.choice(list(range(len(belief))), p=belief)
            model.curr_state = ctx.states[index]
            print(" ***** New current state: ", model.curr_state)


        # have fun!
        log.info('''
        ++++++++++++++++++++++
            Starting State:  {}
            Starting Budget:  {}
            Init Belief: {}
            Time Horizon: {}
            Max Play: {}
        ++++++++++++++++++++++'''.format(model.curr_state, budget, belief, T, params.max_play))

        for i in range(params.max_play):
            # plan, take action and receive environment feedbacks
            print("*****  play game:  ", i, "  *****")
            pomdp.solve(T)

            file_name = "alpha_vecs" + str(i) + ".txt"
            f = open(file_name, "w+")
            for alph_vector in pomdp.alpha_vecs:
                for j in range(len(alph_vector.v)):
                    f.write(str(alph_vector.v[j]) + "\t")
                f.write("\n")
            f.close()

            file_name2 = "actions" + str(i) + ".txt"
            f = open(file_name2, "a+")
            for alph_vector in pomdp.alpha_vecs:
                f.write(str(alph_vector.action) + "\n")
            f.close()

            # implementing epsilon greedy action selection with two epsilon e1 = 0.3, e2 = 0.1
            random_num = np.random.randint(10)
            # if i < 30:
            #     if random_num <= 4:
            #         action = pomdp.choose_random_act(ctx.actions)
            #     else:
            #         action = pomdp.get_greedy_action(belief)
            # elif i < 50:
            #     if random_num <= 2:
            #         action = pomdp.choose_random_act(ctx.actions)
            #     else:
            #         action = pomdp.get_greedy_action(belief)
            # else:
            if random_num == 0:
                action = pomdp.choose_random_act(ctx.actions)
            else:
                action = pomdp.get_greedy_action(belief)

            print(" @@@@@ action: ", action)

            # action = pomdp.get_action(belief)
            new_state, obs, reward, cost = pomdp.take_action(action)

            if params.snapshot and isinstance(pomdp, POMCP):
                # takes snapshot of belief tree before it gets updated
                self.snapshot_tree(visualiser, pomdp.tree, '{}.gv'.format(i))

            # update states
            belief = pomdp.update_belief(belief, action, obs)
            total_rewards += reward
            budget -= cost

            # writing total rewards in a file
            f = open("Rewards.txt", "a+")
            f.write(str(reward) + "\n")
            f.close()

            # writing total rewards in a file
            f = open("totalRewards.txt", "a+")
            f.write(str(total_rewards) + "\n")
            f.close()

            # print ino
            log.info('\n'.join([
                'Taking action: {}'.format(action),
                'Observation: {}'.format(obs),
                'Reward: {}'.format(reward),
                'Budget: {}'.format(budget),
                'New state: {}'.format(new_state),
                'New Belief: {}'.format(belief),
                '=' * 20
            ]))

            if budget <= 0:
                log.info('Budget spent.')

            # Writing log file
            f = open("logFile.txt", "a+")
            f.write("Step: " + str(i) + "\n")
            f.write("Action: " + str(action) + ", Observation: " + str(obs) + ", New state: " + str(new_state) +
                    ", Reward: " + str(reward) + "\n")
            f.write("New Belief: " + str(belief) + "\n")
            f.write("**************************** \n")
            f.close()

        log.info('{} games played. Toal reward = {}'.format(i + 1, total_rewards))
        return pomdp
