import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

import numpy as np

from .generate import generate_training_data

# def neural_network_model {{{
def neural_network_model(input_size):

    # neural net foundation we use {{{
    model = Sequential([
        Dense(128, activation='relu',input_dim=input_size),
        Dropout(0.5),
        Dense(256,activation='relu'),
        Dropout(0.5),
        Dense(512,activation='relu'),
        Dropout(0.5),
        Dense(256,activation='relu'),
        Dropout(0.5),
        Dense(128,activation='relu'),
        Dropout(0.5),
        Dense(2,activation='softmax'),
    ])
    # }}}

    # smaller net for faster testing {{{
    '''
    model = Sequential([
        Dense(64, activation='relu',input_dim=input_size),
        #Dropout(0.8),
        Dense(64,activation='relu'),
        #Dropout(0.8),
        Dense(2,activation='softmax'),
    ])
    '''
    # }}}

    # meta we need for training
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
# }}}

# def train_model {{{
def train_model(data,model=False):

    # parse the training data to an
    # input format keras can use for
    # training
    X = np.array([i[0] for i in data])
    y = np.array([i[1] for i in data])

    if not model:
        # initialize model
        model = neural_network_model(
            input_size = len(X[0])
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
def test_model(env,model,dim,steps):

    choices = []
    score = 0

    # random input for generating first
    # action
    prev_obs = np.random.random((dim,))

    env.reset()

    for _ in range(steps):
        env.render()

        # predict action
        action = np.argmax(
            model.predict(
                np.array([prev_obs])
            )[0]
        )

        choices.append(action)

        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation

        score+=reward

        if done:
            env.reset()
            break

    # score and statistics for the console {{{
    print('Score: ', score)
    print('choice 1:{}  choice 0:{}'.format(
        choices.count(1)/len(choices),
        choices.count(0)/len(choices)
    ))
    # }}}

    return score
# }}}
