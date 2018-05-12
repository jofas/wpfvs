import numpy as np
import random

from statistics import median, mean
from collections import Counter

# def generate_training_data {{{
def generate_training_data(env, model=False, score_req=50, eps=1000, steps=750):

    # {{{
    # the dataset we want to return. With
    # this data we want to train our model
    # to perform actions to its environment
    # (env)
    # [ ( observation, action ) ]
    # }}}
    training_data = []
    # all scores
    scores = []
    # {{{
    # just the scores from the episodes that
    # were good enough (where the episode's
    # score was higher than the score_requirement)
    # are put to training_data
    # }}}
    accepted_scores = []

    # simulate inital_eps many episodes {{{
    for _ in range(eps):

        score = 0
        # observations and actions done in this episode
        # [ ( observation , action ) ]
        eps_mem = []
        # previous observation (provided by env.step)
        prev_obs = np.random.random((4,))

        # do a maximum of goal_steps many actions {{{
        for _ in range(steps):

            # choose random action (0 or 1) and
            # do it with env.step
            if not model:
                action = random.randrange(0,2)
            else:
                action = np.argmax(
                    model.predict(
                        np.array([prev_obs])
                    )[0]
                )

            observation, reward, done, info = env.step(action)

            # {{{
            # safe previous observation (prev_obs) and
            # the current action. Set prev_obs to the
            # current observation
            # }}}
            if len(prev_obs) > 0 :
                eps_mem.append([prev_obs, action])
            prev_obs = observation

            score+=reward

            # episode failed early
            if done: break
        # }}}

        # if the score is higher than score_requirement,
        # the episode gets added to trainings_data.
        if score >= score_req:
            accepted_scores.append(score)

            # iterate this episodes memory {{{
            for data in eps_mem:

                # {{{
                # parse the action to the output layer
                # format of our model (which action the
                # model should perform)
                # }}}
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]
                # saving our training data
                training_data.append([data[0], output])
            # }}}

        # save score to the overall scores
        scores.append(score)
        # reset env for next episode
        env.reset()
    # }}}

    # stats of the dataset {{{
    print('Average accepted score:',mean(accepted_scores))
    print('Median score for accepted scores:',median(accepted_scores))
    print(Counter(accepted_scores))
    # }}}

    return training_data
# }}}

