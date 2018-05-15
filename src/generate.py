import numpy as np
import random
import math

from statistics import median, mean
from collections import Counter
from concurrent import futures

from .protocol import Protocol

# def concurrent_generate_data {{{
def concurrent_generate_data(

    # our data set we continously improve
    data_set = [],

    # how many processes we use for com-
    # puting our data set
    procs = 1

):
    with futures.ProcessPoolExecutor(max_workers=procs) as e:

        fs = {
            e.submit(
                generate_data,
                #data_set = data_set,
            ) : i for i in range(procs)
        }

        for f in futures.as_completed(fs):
            f_ds, = f.result()
            data_set.append[f_ds]

        return data_set
# }}}

# def generate_data {{{
#
#   (!) IMPORTANT: never provide
#                  model without
#                  providing dim
#
@Protocol
def generate_data(

    # data_set: {{{
    #
    # the data set we want to return. With
    # this data we want to train our model
    # to perform actions to its environment
    # (env)
    # [ [ observation, action, score ] ]
    #
    # }}}
    data_set = [],

):

    # import global variables {{{
    from .main import env,          \
                      steps,        \
                      eps,          \
                      score_req,    \
                      action_space, \
                      model
    # }}}

    # accepted_scores: {{{
    #
    # just the scores from the episodes that
    # were good enough (where the episode's
    # score was higher than the score_requirement)
    # are put to data_set
    #
    # }}}
    accepted_scores = []

    # simulate inital_eps many episodes {{{
    for _ in range(eps):

        score = 0

        # observations and actions done in this episode
        # [ ( observation , action ) ]
        eps_mem = []

        # previous observation (provided by env.step),
        # initialized as empty list when the data_set is
        # empty, else with a random numpy array
        if data_set == []:
            prev_obs = []
        else:
            prev_obs = np.random.random((len(data_set[0][0]),))

        # do a maximum of steps many actions {{{
        for _ in range(steps):

            # if model is not provided choose random action
            # (0 or 1) and do it with env.step, else predict
            # next action with the model
            action = random.randrange(0,action_space)
            '''
            if not model:
                action = random.randrange(0,action_space)
            else:
                action = np.argmax(
                    model.predict(
                        np.array([prev_obs])
                    )[0]
                )
            '''

            observation, reward, done, info = env.step(action)

            # safe previous observation (prev_obs) and
            # the current action. Set prev_obs to the
            # current observation
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

    '''
    # remove entries that are not good enough for training {{{
    clear = lambda set, req: [x for x in set if x[2] >= req]

    data_set = clear(
        data_set,
        math.floor(
            (score_req + mean(accepted_scores)) / 2
        )
    )
    # }}}
    '''

    print('Average data set score:',mean([x[2] for x in data_set]))

    return data_set
# }}}
