import numpy as np
import random
import math
import copy
from statistics import median, mean
from collections import Counter
from concurrent import futures

from .protocol import Protocol

generator_models = []

# def _generator {{{
# {{{
#
# id:
#   process number
#
# n:
#   the number of episodes the
#   process should do.
#
# }}}
def _generator(id, n):

    global generator_models

    import tensorflow as tf
    import keras.backend as K

    K.set_session(tf.Session())

    # import global variables {{{
    from .main import env,          \
                      steps,        \
                      score_req,    \
                      action_space, \
                      input_dim,    \
                      model
    # }}}

    # generate local env to avoid race conditions
    _env = copy.deepcopy(env)

    #if id % 2 == 0  and model != None:
    if model != None:
        _model = generator_models[id]
    else:
        _model = None

    data_set = []

    # simulate inital_eps many episodes {{{
    for _ in range(n):

        score = 0

        # observations and actions done in this episode
        # [ ( observation , action ) ]
        eps_mem = []

        # previous observation (provided by env.step),
        # initialized as empty list when the data_set is
        # empty, else with a random numpy array
        #if data_set == []:
        prev_obs = []

        #else:
        #    prev_obs = np.random.random(
        #       (len(data_set[0][0]),))

        # do a maximum of steps many actions {{{
        for _ in range(steps):
            if _model != None and prev_obs != []:
                print('XXX: {} : {}'.format(id,_model))
                action = np.argmax(
                    _model.predict(
                        np.array([prev_obs])
                    )[0]
                )
                print('Halooo 2 aus {} : {}'.format(
                    id,action))
            else:
                action = random.randrange(0,action_space)


            observation, reward, done, info = \
                _env.step(action)

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
        _env.reset()
    # }}}

    if len(data_set) > 0:
        print(
            '{}: avg: {}, len: {}'.format(
                id,
                mean([x[2] for x in data_set]),
                len(data_set)
            )
        )
    else:
        print('{}: NO DATA GENERATED'.format(id))

    return data_set
# }}}

# def generator {{{
# {{{
#
# procs:
#   how many processes should be used
#   for computing our data set
#
# data_set:
#   the data set we continously build
#
# }}}
@Protocol
def generator(procs = 1, data_set = []):

    global generator_models

    from .main import eps, model
    from keras import Sequential

    if model != None:
        generator_models = [
            Sequential.from_config(model.get_config()) \
                for i in range(procs)
        ]
    else:
        generator_models = [None for i in range(procs)]

    with futures.ProcessPoolExecutor(
        max_workers=procs
    ) as e:

        fs = {
            e.submit(_generator,i, int(eps / procs)) \
                : i for i in range(procs)
        }

        for f in futures.as_completed(fs):
            f_ds = f.result()
            data_set += f_ds

    return data_set, mean([x[2] for x in data_set])
# }}}

# def generate_data DEPRECATED {{{
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
