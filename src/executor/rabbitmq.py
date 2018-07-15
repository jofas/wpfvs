import json

# def data_callback {{{
def data_callback(ch, method, properties, body):

    from .main import data_set, m_data_set

    m_data_set.acquire()
    data_set += json.loads(body.decode())
    m_data_set.release()
# }}}
