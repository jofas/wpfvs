import json
import os
import signal

from ..config import MSG_CONFIG,    \
                     MSG_WEIGHTS,   \
                     MSG_DONE

# def model_callback {{{
def model_callback(ch, method, properties, body):

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
