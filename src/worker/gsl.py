from .generate  import generator

from ..config   import DATAQUEUE
from ..rabbitmq import channel_init, channel_send

# def generating_sending_loop {{{
def generating_sending_loop(procs):

    channel = channel_init(DATAQUEUE)

    while True:
        # generate
        data_set = generator(procs=procs)
        # send
        channel_send(channel, DATAQUEUE, data_set)
# }}}
