import random
import importlib
import numpy as np

from concurrent import futures

from .protocol import Protocol

# def train_model {{{
# {{{
#
# data:
#   data set we pass over to our
#   worker process executing
#   _train_model
#
# }}}
@Protocol
def train_model(data):

    # start new process (madatory for
    # using Keras concurrently)
    with futures.ProcessPoolExecutor(
        max_workers=1
    ) as e:

        # return model tuple with config
        # and weights
        return e.submit(
            _train_model,
            data=data,
        ).result()
# }}}

# def _train_model {{{
# {{{
#
# data:
#   the data set used for training
#   our model
#
# }}}
def _train_model(data):

    # imports {{{
    import tensorflow as tf
    from keras import Sequential
    import keras.backend as K

    from .main import model, import_
    # }}}

    if import_ == None:
        raise Exception(
            'Model to import is not defined'
        )

    # parse the training data to an
    # input format Keras can use for
    # training
    X = np.array([i['obs']    for _, i in data.items()])
    y = np.array([i['action'] for _, i in data.items()])

    # start tf session {{{
    with tf.Session() as s:

        # tell Keras to use newly started
        # tf session
        K.set_session(s)

        if model == None:
            # initialize model
            _model = importlib.import_module(
                'models.'+import_
            ).model(
                input_size = len(X[0]),
                output_size = len(y[0])
            )
        else:
            # model already initialized
            _model = Sequential.from_config(model[0])
            _model.set_weights(model[1])
            _model = importlib.import_module(
                'models.'+import_
            ).compile(_model)

        # actual training
        _model.fit(X,y,epochs=5,batch_size=1024)

        # after the session is closed our model is use-
        # less, so we have to export it's config and
        # the weights as a tuple to our global variable
        # .main.model
        return (_model.get_config(), _model.get_weights())
    # }}}
# }}}

# def test_model {{{
# {{{
#
# visual:
#   boolean whether gym should
#   render the tests or not
#
# }}}
@Protocol
def test_model(visual):
    with futures.ProcessPoolExecutor(
        max_workers=1
    ) as e:

        # return whether the model has solved the
        # environment or not
        return e.submit(_test_model,visual).result()
# }}}

# def _test_model {{{
# {{{
#
# visual:
#   boolean whether gym should
#   render the environment or
#   not
#
# }}}
def _test_model(visual):

    # imports {{{
    import tensorflow as tf
    from keras import Sequential
    import keras.backend as K

    from .main import action_space, \
                      steps,        \
                      model,        \
                      goal_score,   \
                      goal_cons,    \
                      env
    # }}}

    # start tf session {{{
    with tf.Session() as s:

        # tell Keras to use newly started
        # tf session
        K.set_session(s)

        # initialize local model from .main.model
        _model = Sequential.from_config(model[0])
        _model.set_weights(model[1])

        env.reset()

        cons = 0

        # run tests while the goal_score is reached {{{
        while True:

            score = 0
            prev_obs = []

            # run test {{{
            for _ in range(steps):
                if visual:
                    env.render()

                if prev_obs == []:
                    action = random.randrange(0,action_space)
                else:
                    # predict action
                    action = np.argmax(
                        _model.predict(
                            np.array([prev_obs])
                        )[0]
                    )

                new_observation, reward, done, info = \
                    env.step(action)

                prev_obs = new_observation
                score+=reward

                if done:
                    env.reset()
                    print('Score: ', score)
                    break
            # }}}

            print('Consecutive: ', cons)

            if score == goal_score:
                cons += 1
            else:
                return ( False, score, cons )

            if cons == goal_cons:
                return ( True, score, cons )
        # }}}
    # }}}
# }}}
