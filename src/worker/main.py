import gym
import numpy as np
from multiprocessing import Manager, Process

from .gsl      import generating_sending_loop
from .rabbitmq import model_callback

from ..config   import init_conf, MODELEXCHANGE, T_EXCHANGE
from ..rabbitmq import start_receiver

# global shared model (config and weights) + mutexes
MDL_CONFIG    = Manager().dict()
M_MDL_CONFIG  = Manager().Lock()
MDL_WEIGHTS   = Manager().dict()
M_MDL_WEIGHTS = Manager().Lock()

# global values {{{
# {{{
#
# env_:
#   global reference to the name of our gym environment.
#
# gsl:
#  generate-send-loop process.
#
# }}}
env_         = None
gsl          = None
# }}}

# def main {{{
# {{{
#
# env_name:
#   defines the gym environment. Right now
#   cp und ll can be used (others may as
#   well, but aren't jet tested)
#
# procs:
#   CPython's threads only run on a single core,
#   which is not what we need, instead we need
#   to generate our data using multiple cores,
#   which is only possible with Processes in
#   CPython.
#
# }}}
def main(env_name, procs):

    global env_
    global gsl

    env_ = env_name

    # (!) MUST ALWAYS BEEN CALLED BEFORE WORKING WITH THE
    #     GYM ENVIRONMENT
    init_conf(env_name)

    # spawn generating-sending loop
    gsl = Process(
        target = generating_sending_loop,
        args   = (procs,)
    )
    gsl.start()

    start_receiver(MODELEXCHANGE, model_callback, T_EXCHANGE)
# }}}

if __name__ == '__main__':
    print('run $WPFVS_HOME/worker.py instead')
