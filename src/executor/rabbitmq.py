import pika
import json

from ..config import HOST,          \
                     DATAQUEUE,     \
                     MODELEXCHANGE

# def ttsl_channel {{{
def ttsl_channel():
    connection = _connect_rmq()
    channel = connection.channel()

    channel.exchange_declare(
        exchange      = MODELEXCHANGE,
        exchange_type = 'fanout'
    )

    return channel
# }}}

# def ttsl_send {{{
def ttsl_send(channel, body):
    channel.basic_publish(
        exchange    = MODELEXCHANGE,
        routing_key = '',
        body=json.dumps(body)
    )
# }}}

# def recv_data {{{
def recv_data():
    connection = _connect_rmq()
    channel = connection.channel()

    channel.queue_declare(queue=DATAQUEUE)

    channel.basic_consume(
        _recv_callback,
        queue=DATAQUEUE,
        no_ack=True
    )

    channel.start_consuming()
# }}}

# def _recv_callback {{{
def _recv_callback(ch, method, properties, body):

    from .main import data_set, m_data_set

    m_data_set.acquire()
    data_set += json.loads(body.decode())
    m_data_set.release()

# }}}

# def _connect_rmq {{{
def _connect_rmq():
    return pika.BlockingConnection(
        pika.ConnectionParameters(host=HOST)
    )
# }}}

