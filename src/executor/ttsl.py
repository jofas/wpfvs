import os
import signal
import numpy as np

from .model    import train_model, test_model

from ..config import MSG_CONFIG,    \
                     MSG_WEIGHTS,   \
                     MSG_DONE,      \
                     MODELEXCHANGE, \
                     T_EXCHANGE

from ..rabbitmq import channel_init, channel_send

# def training_testing_sending_loop {{{
def training_testing_sending_loop(visual):

    from .main import data_set, m_data_set, main_pid

    channel = channel_init(MODELEXCHANGE, T_EXCHANGE)

    X         = []
    y         = []
    conf_send = False
    model     = None
    test      = False


    # loop {{{
    while True:
        body = {}

        m_data_set.acquire()
        if len(data_set) > 0:
            print('training...')

            # parse the training data to an
            # input format Keras can use for
            # training
            print('parsing data_set for training...')
            if len(X) > 0:
                X = np.concatenate((
                    X,
                    np.array([i['obs']    for i in data_set])
                ))
            else:
                X = np.array([i['obs']    for i in data_set])

            if len(y) > 0:
                y = np.concatenate((
                    y,
                    np.array([i['action'] for i in data_set])
                ))
            else:
                y = np.array([i['action'] for i in data_set])

            del(data_set[:])

            model = train_model(X, y, model=model)
            test = True

        m_data_set.release()

        if test:
            print('testing...')
            (done, score, cons) = test_model(visual, model)

            print('sending...')

            if done:
                body['type'] = MSG_DONE
            else:
                if conf_send:
                    body['type']  = MSG_WEIGHTS
                    body['model'] = list(map(
                        lambda y: y.tolist(),
                        model.get_weights()
                    ))
                else:
                    body['type']  = MSG_CONFIG
                    body['model'] = model.to_json()

                    conf_send = True

            channel_send(
                channel,
                MODELEXCHANGE,
                body,
                T_EXCHANGE
            )

            test = False

            if done:
                # TODO: dump protocol here
                print('done!')
                os.kill(main_pid, signal.SIGKILL)
                os.kill(os.getpid(), signal.SIGKILL)
    # }}}
# }}}


