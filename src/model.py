import random
import importlib

import numpy as np

from .protocol import Protocol

# def train_model {{{
@Protocol
def train_model(data,imp=False,model=False):

    # parse the training data to an
    # input format keras can use for
    # training
    X = np.array([i[0] for i in data])
    y = np.array([i[1] for i in data])

    if not model:
        if not imp:
            raise Exception('Model to import is not defined')

        # initialize model
        model = importlib.import_module('models.'+imp) \
            .model(
                input_size = len(X[0]),
                output_size = len(y[0])
            )

    # actual training
    model.fit(X,y,epochs=5)

    return model
# }}}

# def test_model {{{
#
# basically the same function like
# generate.generate_test_data. Only
# one game is played and the score
# is returned.
#
def test_model(visual):

    from .main import action_space, \
                      steps,        \
                      model,        \
                      env

    score = 0

    prev_obs = []

    env.reset()

    for _ in range(steps):
        if visual:
            env.render()

        if prev_obs == []:
            action = random.randrange(0,action_space)
        else:
            # predict action
            action = np.argmax(
                model.predict(
                    np.array([prev_obs])
                )[0]
            )

        new_observation, reward, done, info = \
            env.step(action)

        prev_obs = new_observation

        score+=reward

        if done:
            env.reset()
            break

    print('Score: ', score)

    return score
# }}}
