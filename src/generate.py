import numpy as np
import random
import math
import copy
import gym

from statistics import median, mean
from collections import Counter
from concurrent import futures

from .protocol import Protocol

GEN_RND = 0
GEN_MDL = 1

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
def generator(procs = 1, data_set = {}):

    from .main import eps, gen_rand_rat, gen_rand_eps

    # start the processes and await their
    # return values before returning the
    # new data_set
    with futures.ProcessPoolExecutor(
        max_workers=procs
    ) as e:

        # safe every process in fs
        fs = {
            e.submit(
                _generator,
                i,
                int(eps / procs),
                GEN_MDL
            ) : i for i in range(procs)
        }

        # generate random generator processes
        for i in range(procs,procs*gen_rand_rat):
            fs[e.submit(_generator,i,eps*gen_rand_eps)] = i

        # await the return
        for f in futures.as_completed(fs):
            # since passing dics between processes is a
            # pain in the ass we concat the newly generated
            # data with our already existing data_set
            # sequentially, so we don't have the
            # synchronization overhead anymore which made
            # our program way slower.
            for v in f.result():
                k = str(v['obs'])
                data_set[k] = v

                # e
                #if k in data_set and \
                #    v['reward'] > data_set[k]['reward']:
                #        data_set[k] = v
                #else:
                #    data_set[k] = v

    return data_set
# }}}

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
# proc_type:
#  either GEN_RND or GEN_MDL (generate random data or use
#  model to make predictions)
#
# }}}
def _generator(id, n, proc_type=GEN_RND):

    # imports {{{
    import tensorflow as tf
    from keras import Sequential
    import keras.backend as K

    from .main import env_,         \
                      steps,        \
                      action_space, \
                      goal_score,   \
                      input_dim,    \
                      r_take_eps,   \
                      r_clean_eps,  \
                      r_clean_cut,  \
                      model
    # }}}

    # start tf session {{{
    with tf.Session() as s:

        # tell Keras to use newly started
        # tf session
        K.set_session(s)

        # generate local env to avoid race conditions
        _env = gym.make(env_)
        _env.reset()

        if model != None:
            _model = Sequential.from_config(model[0])
            _model.set_weights(model[1])
        else:
            _model = None

        data_set = []

        # simulate n many episodes {{{
        for _ in range(n):

            prev_obs = []
            score    = 0
            eps_mem  = []

            # do a maximum of steps many actions {{{
            for _ in range(steps):

                if _model     != None    and \
                    len(prev_obs) > 0    and \
                    proc_type != GEN_RND:
                        # if it is not the first time we
                        # generate data nor the first move this
                        # episode, our model predicts the next
                        # action
                        action = np.argmax(
                            _model.predict(
                                np.array([prev_obs])
                            )[0]
                        )
                else:
                    # if the model does not exist yet
                    # (first time this function is called)
                    # or we have to do our first move in
                    # this episode, we make a random move
                    action = random.randrange(0,action_space)

                observation, reward, done, info = \
                    _env.step(action)

                if len(prev_obs) > 0:
                    # parse the action to a format our
                    # model will understand (right now
                    # action is an integer saying which
                    # action was performed, for training we
                    # need another representation
                    _action = [0 for i in range(action_space)]
                    _action[action] = 1

                    # safe our observations
                    eps_mem.append({
                        'obs'        : prev_obs,
                        'prev_score' : score,
                        'action'     : _action,
                        'reward'     : reward,
                    })

                prev_obs = observation
                score += reward

                # episode failed early
                if done: break
            # }}}

            # analyse episode
            if score/goal_score >= r_take_eps :
                data_set += eps_mem
            elif score/goal_score >= r_clean_eps:
                data_set += list(filter(
                    lambda x: \
                        x['prev_score']/score < r_clean_cut
                    ,
                    eps_mem
                ))

            # reset env for next episode
            _env.reset()
        # }}}
        print('Generator Process ' + str(id) + ' finished')
        return data_set
    # }}}
# }}}
