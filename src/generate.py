import numpy as np
import random
import math

from statistics import median, mean
from collections import Counter

from .bench import Bench

# def generate_data {{{
#
#   (!) IMPORTANT: never provide
#                  model without
#                  providing dim
#
@Bench
def generate_data(

    # reference to env
    env,

    # how many episodes are evaluated
    eps,

    # maximum numper of steps
    steps,

    # how many actions our agent can perform
    action_space,

    # minimum score the generated data for
    # our data set has to have
    score_req,

    # our model for generating the data
    _model=(False,None),

    # our data set we continously improve
    data_set = False
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

    # data_set: {{{
    # the data set we want to return. With
    # this data we want to train our model
    # to perform actions to its environment
    # (env)
    # [ [ observation, action, score ] ]
    # }}}
    if not data_set:
        data_set = []


    # accepted_scores: {{{
    # just the scores from the episodes that
    # were good enough (where the episode's
    # score was higher than the score_requirement)
    # are put to data_set
    # }}}
    accepted_scores = []

    # simulate inital_eps many episodes {{{
    for _ in range(eps):

        score = 0

        # observations and actions done in this episode
        # [ ( observation , action ) ]
        eps_mem = []

        # previous observation (provided by env.step),
        # initialized as empty list when model is not
        # defined, else with a random numpy array with
        # the dimension (dim,)
        if dim == None:
            prev_obs = []
        else:
            prev_obs = np.random.random((dim,))

        # do a maximum of steps many actions {{{
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
                data_set.append([data[0], output, score])
            # }}}

        # reset env for next episode
        env.reset()
    # }}}

    # remove entries that are not
    # good enough for training
    clear = lambda set, req: [x for x in set if x[2] >= req]

    data_set = clear(
        data_set,
        math.floor(
            (score_req + mean(accepted_scores)) / 2
        )
    )

    print('Average data set score:',mean([x[2] for x in data_set]))

    # return data_set and dim which should be collected
    # when running this function the first time.
    return data_set, dim
# }}}
