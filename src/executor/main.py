import gym
import os
from multiprocessing import Manager, Process

from .rabbitmq import data_callback
from .ttsl     import training_testing_sending_loop

from ..config   import init_conf, DATAQUEUE
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
    global action_space
    global import_

    env = gym.make(str(env_name))
    env.reset()
    action_space = env.action_space.n

    import_ = _model

    # (!) MUST ALWAYS BEEN CALLED BEFORE WORKING WITH THE
    #     GYM ENVIRONMENT
    init_conf(env_name)

    ttsl = Process(
        target = training_testing_sending_loop,
        args   = (visual,)
    )
    ttsl.start()

    start_receiver(DATAQUEUE, data_callback)
# }}}

if __name__ == '__main__':
    print('run $WPFVS_HOME/executor.py instead')
