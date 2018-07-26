import gym
import os
import pika
from multiprocessing import Manager, Process
from time import time

from .rabbitmq import data_callback, meta_callback
from .ttsl     import training_testing_sending_loop

from ..config   import init_conf, DATAQUEUE, METAQUEUE
from ..rabbitmq import start_receiver

# global values {{{
# {{{
#
# env:
#   global reference to our gym environment
#
# action_space:
#   represents the set of actions our agent can
#   perform to this environment. Actually it is just
#   an integer n and the actions the agent can perform
#   are 0, 1, ..., n-1
#
# t_start:
#   time the worker is started.
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
# meta_pid:
#   process id of the meta process (needed in ttsl process
#   for killing it).
#
# }}}
env          = None
action_space = None
t_start      = None

m_data_set   = Manager().Lock()
data_set     = Manager().list()

main_pid     = os.getpid()
meta_pid     = None

protocol     = Manager().dict()
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
# model:
#   defines which neural net is imported from
#   $WPFVS_HOME/models. Every module in models
#   has a model-function which returns a compiled
#   keras Sequential. _model is passed to
#   model.train_model as import_ parameter when
#   initializing the model (first call of model.
#   train_model
#
# }}}
def main(visual, env_name, model):

    global env
    global action_space
    global meta_pid
    global t_start
    t_start = time()

    env = gym.make(str(env_name))
    env.reset()
    action_space = env.action_space.n

    # throw error if $WPFVS_HOME is not set
    if not 'WPFVS_HOME' in os.environ:
        raise Exception(
            'ENVIRONMENT VARIABLE WPFVS_HOME is not set'
        )

    # (!) MUST ALWAYS BEEN CALLED BEFORE WORKING WITH THE
    #     GYM ENVIRONMENT
    init_conf(env_name, model)

    # start process receiving on METAQUEUE and answering on
    # METAEXCHANGE.
    meta = Process(
        target = start_receiver,
        args   = (METAQUEUE, meta_callback)
    )
    meta.start()
    meta_pid = meta.pid

    ttsl = Process(
        target = training_testing_sending_loop,
        args   = (visual,)
    )
    ttsl.start()

    # use main to receive new data which was send by a wor-
    # ker.
    while True:
        try:
            print('connecting to DATAQUEUE')
            start_receiver(DATAQUEUE, data_callback)
        except pika.exceptions.ConnectionClosed:
            print('CONNECTION TO DATAQUEUE CLOSED')
# }}}

def reset_lock():
    global m_data_set
    m_data_set = Manager().Lock()

if __name__ == '__main__':
    print('run $WPFVS_HOME/executor.py instead')
