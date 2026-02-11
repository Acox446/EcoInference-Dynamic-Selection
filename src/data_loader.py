import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

class DataLoader:
    def __init__(self, validation_split=0.2, random_state=42):
        self.val_split = validation_split
        self.seed = random_state
        self._load_and_split()

    def _load_and_split(self):
        print("📥 Downloading Fashion-MNIST...")
        #TODO: Normalitzar les dades (0-1) i convertir a float32

        (X_train_full, y_train_full), (self.X_test, self.y_test) = tf.keras.datasets.fashion_mnist.load_data()

        X_train_full = X_train_full.astype('float32') / 255.0
        self.X_test = self.X_test.astype('float32') / 255.0

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_train_full, 
            y_train_full, 
            test_size=self.val_split, 
            random_state=self.seed,
            stratify=y_train_full
        )
        
        print(f"✅ Data loaded: Train={self.X_train.shape}, Val={self.X_val.shape}, Test={self.X_test.shape}")

    def get_data(self, flatten=False):
        """
        Returns (X_train, y_train), (X_val, y_val), (X_test, y_test).
        
        :param flatten: If True, converts 28x28 images to 784 vectors.
                        (Required for Sklearn models like Trees or Random Forest).
                        (False for CNNs).
        """
        if flatten:
            # Flatten images: (N, 28, 28) -> (N, 784)
            X_train_flat = self.X_train.reshape(self.X_train.shape[0], -1)
            X_val_flat = self.X_val.reshape(self.X_val.shape[0], -1)
            X_test_flat = self.X_test.reshape(self.X_test.shape[0], -1)
            return (X_train_flat, self.y_train), (X_val_flat, self.y_val), (X_test_flat, self.y_test)
        
        # Otherwise, return 28x28 images (for CNNs)
        # Add extra dimension for color channel (N, 28, 28, 1) that Keras needs
        X_train_exp = np.expand_dims(self.X_train, -1)
        X_val_exp = np.expand_dims(self.X_val, -1)
        X_test_exp = np.expand_dims(self.X_test, -1)
        
        return (X_train_exp, self.y_train), (X_val_exp, self.y_val), (X_test_exp, self.y_test)