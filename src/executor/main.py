import pika
import gym
import json
import os
import signal
import numpy as np
from multiprocessing import Manager, Process

from .model import train_model, test_model

from ..config import HOST,          \
                     DATAQUEUE,     \
                     MODELEXCHANGE, \
                     MSG_CONFIG,    \
                     MSG_WEIGHTS,   \
                     MSG_DONE,      \
                     cp,            \
                     ll

# global values {{{
# {{{
#
# env:
#   global reference to our gym environment
#
# goal_score:
#   highscore
#
# steps:
#   is used to loop the actions performed in
#   model.train_model and generate.generate_data.
#
# goal_cons:
#   how often the agent should reach the goal_score
#   consecutively before we can say our agent has
#   solved the environment
#
# action_space:
#   represents the set of actions our agent can
#   perform to this environment. Actually it is just
#   an integer n and the actions the agent can perform
#   are 0, 1, ..., n-1
#
# import_:
#   name of the import module located
#   at $WPFVS_HOME/models.
#   Every module there has a function
#   model(input_size, output_size) that
#   returns a compiled Keras model
#
# m_data_set:
#   mutex for data_set.
#
# data_set:
#   data_set which is used for training and is build by
#   workers.
#
# main_pid:
#   process id of the main process (needed in ttsl process
#   for killing it).
#
# }}}
env          = None
goal_score   = None
steps        = None
goal_cons    = None
action_space = None
import_      = None
m_data_set   = Manager().Lock()
data_set     = Manager().list()
main_pid     = os.getpid()
# }}}

# def main {{{
# {{{
#
# visual:
#   defines if the gym environment should
#   be rendered (output on screen) or not
#   when the agent is tested (model.test_model).
#   if gym does not have to render it can
#   perform the testing much faster.
#
# env_name:
#   defines the gym environment. Right now
#   cp und ll can be used (others may as
#   well, but aren't jet tested)
#
# _model:
#   defines which neural net is imported from
#   $WPFVS_HOME/models. Every module in models
#   has a model-function which returns a compiled
#   keras Sequential. _model is passed to
#   model.train_model as import_ parameter when
#   initializing the model (first call of model.
#   train_model
#
# }}}
def main(visual, env_name, _model):

    global env
    global goal_score
    global steps
    global goal_cons
    global action_space
    global import_
    global data_set
    global m_data_set
    global main_pid

    # set global meta for each environment {{{
    env = gym.make(str(env_name))
    env.reset()
    action_space = env.action_space.n

    import_ = _model

    if env_name == cp:
        goal_score   = 200
        steps        = 1000
        goal_cons    = 10
    elif env_name == ll:
        goal_score   = 200
        steps        = 1000
        goal_cons    = 100
    # }}}


    ttsl = Process(
        target = _training_testing_sending_loop,
        args   = (visual,)
    )
    ttsl.start()

    _recv_data()

# }}}

# def _training_testing_sending_loop {{{
def _training_testing_sending_loop(visual):
    global data_set
    global m_data_set

    connection = _connect_rmq()
    channel = connection.channel()

    channel.exchange_declare(
        exchange      = MODELEXCHANGE,
        exchange_type = 'fanout'
    )

    X         = []
    y         = []
    conf_send = False
    model     = None
    test      = False


    # loop {{{
    while True:
        body = {}

        m_data_set.acquire()
        if len(data_set) > 0:
            print('training...')

            # parse the training data to an
            # input format Keras can use for
            # training
            print('parsing data_set for training...')
            if len(X) > 0:
                X = np.concatenate((
                    X,
                    np.array([i['obs']    for i in data_set])
                ))
            else:
                X = np.array([i['obs']    for i in data_set])

            if len(y) > 0:
                y = np.concatenate((
                    y,
                    np.array([i['action'] for i in data_set])
                ))
            else:
                y = np.array([i['action'] for i in data_set])

            del(data_set[:])

            model = train_model(X, y, model=model)
            test = True

        m_data_set.release()

        if test:
            print('testing...')
            (done, score, cons) = test_model(visual, model)

            print('sending...')

            if done:
                body['type'] = MSG_DONE
            else:
                if conf_send:
                    body['type']  = MSG_WEIGHTS
                    body['model'] = list(map(
                        lambda y: y.tolist(),
                        model.get_weights()
                    ))
                else:
                    body['type']  = MSG_CONFIG
                    body['model'] = model.to_json()

                    conf_send = True

            channel.basic_publish(
                exchange    = MODELEXCHANGE,
                routing_key = '',
                body=json.dumps(body)
            )

            test = False

            if done:
                # TODO: dump protocol here
                global main_pid
                print('done!')
                os.kill(main_pid, signal.SIGKILL)
                os.kill(os.getpid(), signal.SIGKILL)
    # }}}
# }}}

# def _recv_data {{{
def _recv_data():
    connection = _connect_rmq()
    channel = connection.channel()

    channel.queue_declare(queue=DATAQUEUE)

    channel.basic_consume(
        _recv_callback,
        queue=DATAQUEUE,
        no_ack=True
    )

    channel.start_consuming()
# }}}

# def _recv_callback {{{
def _recv_callback(ch, method, properties, body):
    global data_set
    global m_data_set

    m_data_set.acquire()
    print('hallo aus callback')
    data_set += json.loads(body.decode())
    m_data_set.release()

# }}}

# def _connect_rmq {{{
def _connect_rmq():
    return pika.BlockingConnection(
        pika.ConnectionParameters(host=HOST)
    )
# }}}

if __name__ == '__main__':
    print('run $WPFVS_HOME/executor.py instead')
