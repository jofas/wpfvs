import gym
import numpy as np
from multiprocessing import Manager, Process
from time import time, sleep

from .gsl      import generating_sending_loop
from .rabbitmq import model_callback, rcv_meta_callback

from ..config   import init_conf,     \
                       MODELEXCHANGE, \
                       METAEXCHANGE,  \
                       METAQUEUE,     \
                       T_EXCHANGE

from ..rabbitmq import start_receiver, \
                       channel_init,   \
                       channel_send

# global values {{{
# {{{
#
# snd_meta_pid:
#   pid of the snd_meta process
#
# gsl_pid:
#   pid of the generate-send-loop process.
#
# SND_META_SLEEP:
#   seconds the snd_meta process sleeps before sending
#   request to the METAEXCHANGE.
#
# t_start:
#   time the worker is started.
#
# generated_data:
#   amound of data which was generated before executioner
#   sends the finished message.
#
# MDL_CONFIG:
#   config of our model as json (as shared memory).
#
# M_MDL_CONFIG:
#   mutex for MDL_CONFIG (as shared memory).
#
# MDL_WEIGHTS:
#   weights of our model as list (as shared memory).
#
# M_MDL_WEIGHT:
#   mutex for MDL_WEIGHTS (as shared memory).
#
# }}}
snd_meta_pid   = None
gsl_pid        = None
SND_META_SLEEP = 15
t_start        = None

generated_data = 0

MDL_CONFIG    = Manager().dict()
M_MDL_CONFIG  = Manager().Lock()

MDL_WEIGHTS   = Manager().dict()
M_MDL_WEIGHTS = Manager().Lock()
# }}}

# def main {{{
# {{{
#
# procs:
#   CPython's threads only run on a single core,
#   which is not what we need, instead we need
#   to generate our data using multiple cores,
#   which is only possible with Processes in
#   CPython.
#
# }}}
def main(procs):

    global snd_meta_pid
    global gsl_pid
    global t_start
    t_start = time()

    # start process sending to METAQUEUE that we are
    # here.
    snd_meta = Process(
        target = _snd_meta,
        args   = ()
    )
    snd_meta.start()
    snd_meta_pid = snd_meta.pid

    # start receiving from METAEXCHANGE the gym environ-
    # ment. Blocks until we receive the environement from
    # the executor.
    #
    # (!) AFTER HERE init_conf IS CALLED AND CONFIG IS SET
    #     SO WE CAN CONTINUE EXECUTION.
    start_receiver(
        METAEXCHANGE,
        rcv_meta_callback,
        T_EXCHANGE
    )

    # spawn generating-sending loop
    gsl = Process(
        target = generating_sending_loop,
        args   = (procs,)
    )
    gsl.start()
    gsl_pid = gsl.pid

    start_receiver(MODELEXCHANGE,model_callback,T_EXCHANGE)
# }}}

# def _snd_meta {{{
def _snd_meta():
    channel = channel_init(METAQUEUE)

    while True:
        channel_send(channel, METAQUEUE, {'env':True})
        sleep(SND_META_SLEEP)
# }}}

if __name__ == '__main__':
    print('run $WPFVS_HOME/worker.py instead')
