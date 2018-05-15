import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

def model(input_size, output_size):
    model = Sequential([
        Dense(128, activation='relu',input_dim=input_size),
        Dropout(0.5),
        Dense(256,activation='relu'),
        Dropout(0.5),
        Dense(512,activation='relu'),
        Dropout(0.5),
        Dense(512,activation='relu'),
        Dropout(0.5),
        Dense(256,activation='relu'),
        Dropout(0.5),
        Dense(128,activation='relu'),
        Dropout(0.5),
        Dense(output_size,activation='softmax'),
    ])

    # meta we need for training
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
