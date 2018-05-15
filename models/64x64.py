import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

def model(input_size, output_size):

    model = Sequential([
        Dense(64, activation='relu',input_dim=input_size),
        Dense(64,activation='relu'),
        Dense(output_size,activation='softmax'),
    ])

    # meta we need for training
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
