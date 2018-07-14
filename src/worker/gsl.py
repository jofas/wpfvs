# def generating_sending_loop {{{
def generating_sending_loop(procs):

    from .rabbitmq import gsl_channel, gsl_send
    from .generate import generator

    channel = gsl_channel()

    while True:
        # generate
        data_set = generator(procs=procs)
        # send
        gsl_send(channel, data_set)
# }}}


