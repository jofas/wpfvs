import pika
import gym
import json
import os
import signal
import numpy as np
from multiprocessing import Manager, Process

from .generate import generator

from ..config import HOST,          \
                     DATAQUEUE,     \
                     MODELEXCHANGE, \
                     MSG_CONFIG,    \
                     MSG_WEIGHTS,   \
                     MSG_DONE,      \
                     cp,            \
                     ll

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
        target = _generating_sending_loop,
        args   = (procs,)
    )
    gsl.start()

    _recv_model()
# }}}

# def _generating_sending_loop {{{
def _generating_sending_loop(procs):
    connection = _connect_rmq()

    channel = connection.channel()
    channel.queue_declare(queue=DATAQUEUE)

    while True:
        # generate
        data_set = generator(procs=procs)
        # send
        channel.basic_publish(
            exchange    = '',
            routing_key = DATAQUEUE,
            body        = json.dumps(data_set)
        )

    connection.close()
# }}}

# def _recv_model {{{
def _recv_model():
    connection = _connect_rmq()
    channel = connection.channel()
    channel.exchange_declare(
        exchange      = MODELEXCHANGE,
        exchange_type = 'fanout'
    )

    result = channel.queue_declare(exclusive=True)
    qn = result.method.queue

    channel.queue_bind(
        exchange = MODELEXCHANGE,
        queue = qn
    )

    channel.basic_consume(
        _recv_callback,
        queue=qn,
        no_ack=True
    )

    channel.start_consuming()
# }}}

# def _recv_callback {{{
def _recv_callback(ch, method, properties, body):

    # global shared model (config and weights) + mutexes
    global M_MDL_CONFIG
    global MDL_CONFIG
    global M_MDL_WEIGHTS
    global MDL_WEIGHTS


    mdl = json.loads(body.decode())

    if mdl['type'] == MSG_CONFIG:
        with M_MDL_CONFIG:
            MDL_CONFIG['model_json_str'] = mdl['model']

    elif mdl['type'] == MSG_WEIGHTS:
        with M_MDL_WEIGHTS:
            MDL_WEIGHTS['weights_as_list'] = mdl['model']

    elif mdl['type'] == MSG_DONE:

        global gsl

        print('done!')

        # TODO: send protocol to executor

        gsl.terminate()
        os.kill(os.getpid(), signal.SIGKILL)
# }}}

# def _connect_rmq {{{
#
# connect to rabbitmq
#
def _connect_rmq():
    return pika.BlockingConnection(
        pika.ConnectionParameters(host=HOST)
    )
# }}}

if __name__ == '__main__':
    print('run $WPFVS_HOME/worker.py instead')
