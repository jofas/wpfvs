import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

import numpy as np

from .generate import generate_training_data

# def neural_network_model {{{
def neural_network_model(input_size):

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

    model = Sequential([
        Dense(64, activation='relu',input_dim=input_size),
        #Dropout(0.8),
        Dense(64,activation='relu'),
        #Dropout(0.8),
        Dense(2,activation='softmax'),
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
# }}}

# def train_model {{{
def train_model(env,model=False,best_test_run=None):

    if not model:
        training_data = generate_training_data(env=env)

        X = np.array([i[0] for i in training_data])
        y = np.array([i[1] for i in training_data])

        model = neural_network_model(
            input_size = len(X[0])
        )

    else:
        training_data = generate_training_data(
            env=env,
            model=model,
            score_req=best_test_run
        )

        X = np.array([i[0] for i in training_data])
        y = np.array([i[1] for i in training_data])

    model.fit(X,y,epochs=5)

    return model
# }}}

# def test_model {{{
def test_model(env,model,steps=750):

    choices = []
    score = 0

    game_memory = []

    prev_obs = np.random.random((4,))

    env.reset()

    for _ in range(steps):
        env.render()
        action = np.argmax(
            model.predict(
                np.array([prev_obs])
            )[0]
        )

        choices.append(action)

        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation

        game_memory.append([new_observation, action])

        score+=reward

        if done:
            env.reset()
            break


    print('Score: ', score)
    print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))

    return score
# }}}
