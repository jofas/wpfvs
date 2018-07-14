import gym
import os
from multiprocessing import Manager, Process

from .rabbitmq import recv_data
from .ttsl     import training_testing_sending_loop

from ..config import cp, ll

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
        target = training_testing_sending_loop,
        args   = (visual,)
    )
    ttsl.start()

    recv_data()
# }}}

if __name__ == '__main__':
    print('run $WPFVS_HOME/executor.py instead')
