import pika
import gym
import json
import numpy as np
from multiprocessing import Manager, Process

from .gsl      import generating_sending_loop
from .rabbitmq import recv_model

from ..config import cp, ll

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
# goal_score:
#   highscore.
#
# steps:
#   is used to loop the actions performed in
#   model.train_model and generate.generate_data.
#
# eps:
#   episodes that are performed every time in
#   generate.generate_data (eps * steps actions are
#   performed).
#
# rand_eps:
#   episodes that are performed by every random generator.
#
# gen_rand:
#   to every proc (generator using our model, not random)
#   we add some random generators (adds new aspects to our
#   training data). We add gen_rand many random generators.
#
# r_take_eps:
#   DATA-SANITATION: we take every generated episodes where
#   the episode's score devided by goal_score  is greater
#   equal r_takes_eps.
#
# r_clean_eps:
#   DATA-SANITATION: if an episode's score devided by
#   goal_score is greater equal r_clean_eps we sanitize the
#   episode's data and take it (if the episode generated a
#   score lower than r_clean_eps we throw it away).
#
# r_clean_cut:
#   DATA-SANITATION: if the episode's score devided by
#   goal_score is between r_clean_eps and r_take_eps we
#   normalize the data of the episode and take only the
#   steps that generated a single score greater equal
#   r_clean_cut.
#
# gsl:
#  generate-send-loop process.
#
# }}}
env_         = None
goal_score   = None
steps        = None
eps          = None
rand_eps     = None
gen_rand     = None
r_take_eps   = None
r_clean_eps  = None
r_clean_cut  = None
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
    global goal_score
    global steps
    global eps
    global gen_rand
    global rand_eps
    global r_take_eps
    global r_clean_eps
    global r_clean_cut
    global gsl

    # set global meta for each environment {{{
    env_ = env_name

    if env_name == cp:
        goal_score   = 200
        steps        = 1000
        eps          = 1000
        rand_eps     = 2000
        gen_rand     = 1
        r_take_eps   = 0.95
        r_clean_eps  = 0.2
        r_clean_cut  = -1
    elif env_name == ll:
        goal_score   = 200
        steps        = 1000
        eps          = 40000
        rand_eps     = 40000
        gen_rand     = procs
        r_take_eps   = 0
        r_clean_eps  = -3
        r_clean_cut  = 0.4
    # }}}

    # spawn generating-sending loop
    gsl = Process(
        target = generating_sending_loop,
        args   = (procs,)
    )
    gsl.start()

    recv_model()
# }}}

if __name__ == '__main__':
    print('run $WPFVS_HOME/worker.py instead')
