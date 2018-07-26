import json

from ..config   import METAEXCHANGE, T_EXCHANGE
from ..rabbitmq import channel_init, channel_send

meta_channel = channel_init(METAEXCHANGE, T_EXCHANGE)
worker_id    = 0

# def data_callback {{{
def data_callback(ch,method,properties,body):

    from .main import data_set, m_data_set
    print('Hallo aus data_callback')
    m_data_set.acquire()
    data_set += json.loads(body.decode())
    m_data_set.release()
# }}}

# def meta_callback {{{
def meta_callback(ch,method,properties,body):

    from .main    import protocol
    from ..config import env_name

    global meta_channel
    global worker_id

    body = json.loads(body.decode())

    if 'env' in body:
        meta_channel = channel_init(METAEXCHANGE, T_EXCHANGE)
        
        channel_send(
            meta_channel,
            METAEXCHANGE,
            {
                'env': env_name,
            },
            T_EXCHANGE
        )
    elif 'protocol' in body:
        protocol['worker_'+str(worker_id)] =  \
            body['protocol']
        worker_id += 1
    else:
        raise Exception(
            'ON META RECEIVE: received bad entry: ' +
            str(body)
        )
# }}}
