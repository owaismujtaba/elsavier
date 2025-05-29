import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, metrics
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, metrics
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping


class OvertCoverRestClassifier(tf.keras.Model):
    def __init__(self, inputShape, numClasses=3):
        super(OvertCoverRestClassifier, self).__init__()

        self.model = models.Sequential([
            layers.Input(shape=inputShape),
            layers.Reshape((inputShape[0], inputShape[1], 1)),

            layers.Conv2D(16, kernel_size=(1, 10), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, kernel_size=(inputShape[0], 1), activation='relu', padding='valid'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(1, 4)),

            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(numClasses, activation='softmax')
        ])

    def compileModel(self, learningRate=0.001):
        self.model.compile(
        optimizer=Adam(learning_rate=learningRate),
        loss=SparseCategoricalCrossentropy(),
        metrics=[metrics.SparseCategoricalAccuracy()]
    )

    def trainWithSplit(self, X, y, validationSplit=0.2, epochs=50, batchSize=32, shuffle=True):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validationSplit, stratify=y, random_state=42, shuffle=shuffle
        )

        earlyStop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        return self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batchSize,
            callbacks=[earlyStop]
        )

    def evaluate(self, testData):
        return self.model.evaluate(testData)

    def predict(self, inputData):
        return self.model.predict(inputData)

    def summary(self):
        return self.model.summary()
