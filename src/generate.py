import numpy as np
import random

from statistics import median, mean
from collections import Counter

# def generate_training_data {{{
#
#   (!) IMPORTANT: never provide
#                  model without
#                  providing dim
#
def generate_training_data(
    env,
    score_req,
    eps,
    steps,
    action_space,
    _model=(False,None)
):
    # bind _model
    model = _model[0]
    # dim: {{{
    # if this is the first time this function is called
    # (no model and no dim provided and actions chosen
    # randomly) dim gets set to the input-dimension of
    # our model (size of observation) and should be used
    # further to combine dim with the model as _model as
    # input (we need a random set of dim size, which we
    # provide our model with to generate the first action
    # (basically randomly, because the data is not from
    # an observation), before we can use our first
    # observation in the next iteration.
    # }}}
    dim = _model[1]

    # training_data: {{{
    # the dataset we want to return. With
    # this data we want to train our model
    # to perform actions to its environment
    # (env)
    # [ [ observation, action ] ]
    # }}}
    training_data = []
    # all scores
    scores = []
    # accepted_scores: {{{
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
        if dim == None:
            prev_obs = []
        else:
            prev_obs = np.random.random((dim,))
        # do a maximum of goal_steps many actions {{{
        for _ in range(steps):

            # if model is not provided choose random action
            # (0 or 1) and do it with env.step, else predict
            # next action with the model
            if not model:
                action = random.randrange(0,action_space)
            else:
                action = np.argmax(
                    model.predict(
                        np.array([prev_obs])
                    )[0]
                )

            observation, reward, done, info = env.step(action)

            # safe previous observation (prev_obs) and
            # the current action. Set prev_obs to the
            # current observation
            if len(prev_obs) > 0 :
                eps_mem.append([prev_obs, action])
            prev_obs = observation

            if dim == None:
                dim = len(prev_obs)

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

                # parse the action to the output layer
                # format of our model (which action the
                # model should perform)
                output = [0 for i in range(action_space)]
                for x in range(action_space):
                    if data[1] == x:
                        output[x] = 1

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

    # return training_data and dim which should be collected
    # when running this function the first time.
    return training_data, dim
# }}}

