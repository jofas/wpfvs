import pika
import json
from .config import HOST, T_QUEUE

# def start_receiver {{{
def start_receiver(qx, callback, type=T_QUEUE):
    connection = _connect_rmq()
    channel = connection.channel()

    if type == T_QUEUE:
        channel.queue_declare(queue=qx)

        channel.basic_consume(
            callback,
            queue=qx,
            no_ack=True
        )
    else:
        channel.exchange_declare(
            exchange      = qx,
            exchange_type = 'fanout'
        )

        result = channel.queue_declare(exclusive=True)
        qn = result.method.queue

        channel.queue_bind(
            exchange = qx,
            queue = qn
        )

        channel.basic_consume(
            callback,
            queue=qn,
            no_ack=True
        )

    channel.start_consuming()
# }}}

# def channel_init {{{
def channel_init(qx, type=T_QUEUE):

    connection = _connect_rmq()
    channel = connection.channel()

    if type == T_QUEUE:
        channel.queue_declare(queue=qx)
    else:
        channel.exchange_declare(
            exchange      = qx,
            exchange_type = 'fanout'
        )

    return channel
# }}}

# def channel_send {{{
def channel_send(channel, qx, body, type=T_QUEUE):
    if type == T_QUEUE:
        channel.basic_publish(
            exchange    = '',
            routing_key = qx,
            body        = json.dumps(body)
        )
    else:
        channel.basic_publish(
            exchange    = qx,
            routing_key = '',
            body=json.dumps(body)
        )
# }}}

# def _connect_rmq {{{
def _connect_rmq():
    return pika.BlockingConnection(
        pika.ConnectionParameters(host=HOST)
    )
# }}}
