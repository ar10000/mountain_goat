import numpy as np
import pandas as pd
from colorama import Fore, Style
from tensorflow.keras import models, layers, Model
from tensorflow.keras.callbacks import EarlyStopping

def initialize_model() :
    """initialize our model with random weights"""
        #initialize sequential
    model = models.Sequential()
    #adding masking layer
    model.add(layers.Masking(mask_value=-1000))
    #adding rnn layers
    model.add(layers.LSTM(units=20, return_sequences=True, input_shape=(21, 16)))
    model.add(layers.LSTM(units=5, return_sequences=False))
    #adding dense layers
    model.add(layers.Dense(30, activation='relu'))
    model.add(layers.Dense(16, activation='linear'))

    print("\n✅ model initialized")
    return model

def compile_model(model:Model):
    """compile model"""
    model.compile(loss= 'mse',
                  optimizer='adam',
                  metrics=["mae"]
        )
    print("\n✅ model compiled")
    return model

def train_model(model: Model,
                X: np.ndarray,
                y: np.ndarray,
                batch_size=64,
                patience=20,
                validation_split=0.3):
    """
    Fit model and return a the tuple (fitted_model, history)
    """

    print(Fore.BLUE + "\nTrain model..." + Style.RESET_ALL)

    es = EarlyStopping(monitor="val_loss",
                       patience=patience,
                       restore_best_weights=True,
                       )

    history = model.fit(X,
                        y,
                        validation_split=validation_split,
                        epochs=500,
                        batch_size=batch_size,
                        callbacks=[es],
                        verbose=0)

    print(f"\n✅ model trained ({len(X)} rows)")

    return model, history
