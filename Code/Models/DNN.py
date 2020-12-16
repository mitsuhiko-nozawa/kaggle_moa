from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Concatenate, Embedding, Activation, BatchNormalization
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras import regularizers
from tensorflow_addons.layers import WeightNormalization
from tensorflow_addons.optimizers import AdamW

import numpy as np
import pandas as pd


def DNN_model(input_size, output_size):

    """inputs = Input((input_size, ))

    outputs = BatchNormalization()(inputs)
    outputs = Dropout(0.20)(outputs)
    outputs = Dense(1024, activation="relu")(outputs)
    outputs = WeightNormalization()(outputs)


    outputs = BatchNormalization()(outputs)
    outputs = Dropout(0.20)(outputs)
    outputs = Dense(1024, activation="relu")(outputs)
    outputs = WeightNormalization()(outputs)

    outputs = BatchNormalization()(outputs)
    outputs = Dense(output_size, activation="sigmoid")(outputs)
    outputs = WeightNormalization()(outputs)"""

    model = Sequential([Input(input_size)])

    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(WeightNormalization(Dense(1024, activation="relu")))

    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(WeightNormalization(Dense(1024, activation="relu")))

    model.add(BatchNormalization())
    model.add(WeightNormalization(Dense(output_size, activation="sigmoid")))

    #model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=AdamW(lr=1e-3, weight_decay=1e-5, clipvalue=756), loss="binary_crossentropy")

    return model
