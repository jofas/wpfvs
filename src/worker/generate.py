import numpy as np
import random
import math
import gym

from statistics import median, mean
from collections import Counter
from concurrent import futures

GEN_RND = 0
GEN_MDL = 1

# def generator {{{
# {{{
#
# procs:
#   how many processes should be used
#   for computing our data set
#
# }}}
def generator(procs = 1, data_set = []):

    from ..config import eps, gen_rand, rand_eps
    from .main    import generated_data

    # start the processes and await their
    # return values before returning the
    # new data_set
    with futures.ProcessPoolExecutor(
        max_workers=procs + gen_rand
    ) as e:

        # safe every process in fs
        fs = {
            e.submit(
                _generator,
                i,
                eps,
                GEN_MDL
            ) : i for i in range(procs)
        }

        # generate random generator processes
        for i in range(procs, procs + gen_rand):
            fs[e.submit(_generator,i,rand_eps)] = i

        # await the return
        for f in futures.as_completed(fs):

            r = f.result()

            # safe the amount of data generated for the
            # protocol
            generated_data += len(r)

            data_set += r

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
    from keras.models import model_from_json
    import keras.backend as K

    from .main import MDL_CONFIG,   \
                      M_MDL_CONFIG, \
                      MDL_WEIGHTS,  \
                      M_MDL_WEIGHTS

    from ..config import steps,        \
                         goal_score,   \
                         r_take_eps,   \
                         r_clean_eps,  \
                         r_clean_cut,  \
                         env_name
    # }}}

    # start tf session {{{
    with tf.Session() as s:

        # tell Keras to use newly started
        # tf session
        K.set_session(s)

        # generate local env to avoid race conditions
        _env = gym.make(env_name)
        _env.reset()
        action_space = _env.action_space.n

        _model = None

        # instantiate model {{{
        if proc_type == GEN_MDL:
            with M_MDL_CONFIG and M_MDL_WEIGHTS:
                # only if we have data from our executor
                # we can initialize the model
                if len(MDL_CONFIG)  > 0  and len(MDL_WEIGHTS) > 0:

                    _model = model_from_json(
                        MDL_CONFIG['model_json_str']
                    )

                    print('parsing MDL_WEIGHTS...')
                    MDL_WEIGHTS['wieghts_as_list'] = list(
                        map(
                            lambda x: np.array(x),
                            MDL_WEIGHTS['weights_as_list']
                        )
                    )

                    _model.set_weights(
                        MDL_WEIGHTS['weights_as_list']
                    )
        # }}}

        data_set = []

        # simulate n many episodes {{{
        for _i in range(n):

            prev_obs = []
            score    = 0
            min      = 1000000
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
                        'obs'        : prev_obs.tolist(),
                        'prev_score' : score,
                        'action'     : _action,
                        'reward'     : reward,
                    })

                if reward < min: min = reward

                prev_obs = observation
                score += reward

                # episode failed early
                if done: break
            # }}}

            # sanitize episode {{{
            if score/goal_score >= r_take_eps :
                data_set += eps_mem
            elif score/goal_score >= r_clean_eps:
                # normalize eps_mem and take the ones
                # greater equal to r_clean_cut
                data_set += list(filter(
                    lambda x: (x['reward'] - min)/  \
                              (goal_score - min) >= \
                              r_clean_cut
                    ,
                    eps_mem
                ))
            # }}}

            if _i % int(n / 4) == 0:
                print('Generator Process ' + str(id) + ' eps: ' + str(_i))

            # reset env for next episode
            _env.reset()
        # }}}

        print('Generator Process ' + str(id) + ' finished')

        # send only the data which is necessary for
        # training
        return list(map(lambda x: {
            'obs':x['obs'],
            'action': x['action']
        }, data_set))
    # }}}
# }}}
