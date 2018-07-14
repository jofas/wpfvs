import pika
import json
import os
import signal

from ..config import HOST,          \
                     DATAQUEUE,     \
                     MODELEXCHANGE, \
                     MSG_CONFIG,    \
                     MSG_WEIGHTS,   \
                     MSG_DONE

# def gsl_channel {{{
def gsl_channel():
    connection = _connect_rmq()

    channel = connection.channel()
    channel.queue_declare(queue=DATAQUEUE)

    return channel
# }}}

# def gsl_send {{{
def gsl_send(channel, body):
    channel.basic_publish(
        exchange    = '',
        routing_key = DATAQUEUE,
        body        = json.dumps(body)
    )
# }}}

# def recv_model {{{
def recv_model():
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

    from .main import M_MDL_CONFIG, MDL_CONFIG,   \
                      M_MDL_WEIGHTS, MDL_WEIGHTS, \
                      gsl

    mdl = json.loads(body.decode())

    if mdl['type'] == MSG_CONFIG:
        with M_MDL_CONFIG:
            MDL_CONFIG['model_json_str'] = mdl['model']

    elif mdl['type'] == MSG_WEIGHTS:
        with M_MDL_WEIGHTS:
            MDL_WEIGHTS['weights_as_list'] = mdl['model']

    elif mdl['type'] == MSG_DONE:

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

