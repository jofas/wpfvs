import json
import os
import signal
from time import time


from ..rabbitmq import channel_init, channel_send
from ..config import init_conf,     \
                     MSG_CONFIG,    \
                     MSG_WEIGHTS,   \
                     METAQUEUE,     \
                     MSG_DONE

# def model_callback {{{
def model_callback(ch, method, properties, body):

    from .main import M_MDL_CONFIG, MDL_CONFIG,        \
                      M_MDL_WEIGHTS, MDL_WEIGHTS,      \
                      gsl_pid, t_start, generated_data

    mdl = json.loads(body.decode())

    if mdl['type'] == MSG_CONFIG:
        with M_MDL_CONFIG:
            MDL_CONFIG['model_json_str'] = mdl['model']

    elif mdl['type'] == MSG_WEIGHTS:
        with M_MDL_WEIGHTS:
            MDL_WEIGHTS['weights_as_list'] = mdl['model']

    elif mdl['type'] == MSG_DONE:

        print('done!')

        channel = channel_init(METAQUEUE)
        channel_send(channel, METAQUEUE, {
            'protocol' : {
                'time': time() - t_start,
                'data': generated_data
            }
        })

        os.kill(gsl_pid, signal.SIGKILL)
        os.kill(os.getpid(), signal.SIGKILL)
# }}}

# def rcv_meta_callback {{{
def rcv_meta_callback(ch,method,properties,body):

    from .main import snd_meta_pid

    body = json.loads(body.decode())

    if 'env' in body:
        # kill sender
        os.kill(snd_meta_pid, signal.SIGKILL)

        # (!) MUST ALWAYS BEEN CALLED BEFORE WORKING WITH
        #     THE GYM ENVIRONMENT.
        init_conf(body['env'])

        # close channel
        ch.close()
    else:
        raise Exception(
            'ON META RECEIVE: received bad entry: ' +
            str(body)
        )
# }}}
