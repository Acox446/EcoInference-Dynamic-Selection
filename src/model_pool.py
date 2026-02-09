import numpy as np
import pickle
import sys
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from .base_model import GreenModel

layers = tf.keras.layers
models = tf.keras.models

class SklearnBase(GreenModel):
    def __init__(self, model, name):
        super().__init__(name)
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_model_size(self):
        # Quick estimation in bytes using pickle
        return len(pickle.dumps(self.model))

class KerasBase(GreenModel):
    def __init__(self, model, name, epochs=5, batch_size=64):
        super().__init__(name)
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def fit(self, X, y):
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

    def predict(self, X):
        probs = self.model(X, training=False).numpy()
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        return self.model(X, training=False).numpy()

    def get_model_size(self):
        # Count parameters * 4 bytes (float32)
        return self.model.count_params() * 4


# 1. TINY: Decision Tree (Simplest possible)
class TinyModel(SklearnBase):
    def __init__(self):
        super().__init__(DecisionTreeClassifier(max_depth=5), "Tiny (DTree)")

# 2. SMALL: Logistic Regression (Limited)
class SmallModel(SklearnBase):
    def __init__(self):
        super().__init__(LogisticRegression(max_iter=1000), "Small (LogReg)")
        

# 3. MEDIUM: Random Forest (Small ensemble)
class MediumModel(SklearnBase):
    def __init__(self):
        super().__init__(RandomForestClassifier(n_estimators=20, max_depth=10), "Medium (RForest)")

# 4. LARGE: MLP (Simple Neural Network)
class LargeModel(KerasBase):
    def __init__(self, input_shape=(28, 28, 1)):
        model = models.Sequential([
            layers.Flatten(input_shape=input_shape),
            layers.Dense(128, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        super().__init__(model, "Large (MLP)", epochs=5)

# 5. EXTRA LARGE: CNN (Convolutional)
class ExtraLargeModel(KerasBase):
    def __init__(self, input_shape=(28, 28, 1)):
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        super().__init__(model, "Extra Large (CNN)", epochs=5)