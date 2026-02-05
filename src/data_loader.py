import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

class DataLoader:
    def __init__(self, validation_split=0.2, random_state=42):
        self.val_split = validation_split
        self.seed = random_state
        self._load_and_split()

    def _load_and_split(self):
        print("📥 Descarregant Fashion-MNIST...")

        (X_train_full, y_train_full), (self.X_test, self.y_test) = tf.keras.datasets.fashion_mnist.load_data()

        X_train_full = X_train_full.astype('float32') / 255.0
        self.X_test = self.X_test.astype('float32') / 255.0

        # 3. Creem el Validation Set a partir del Train (Separació Train/Val)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_train_full, 
            y_train_full, 
            test_size=self.val_split, 
            random_state=self.seed,
            stratify=y_train_full # Assegura que hi hagi la mateixa proporció de roba a train i val
        )
        
        print(f"✅ Dades carregades: Train={self.X_train.shape}, Val={self.X_val.shape}, Test={self.X_test.shape}")

    def get_data(self, flatten=False):
        """
        Retorna (X_train, y_train), (X_val, y_val), (X_test, y_test).
        
        :param flatten: Si és True, converteix les imatges 28x28 a vectors de 784.
                        (Necessari per a Sklearn models com Arbres o Random Forest).
                        (Fals per a CNNs).
        """
        if flatten:
            # Aplanem les imatges: (N, 28, 28) -> (N, 784)
            X_train_flat = self.X_train.reshape(self.X_train.shape[0], -1)
            X_val_flat = self.X_val.reshape(self.X_val.shape[0], -1)
            X_test_flat = self.X_test.reshape(self.X_test.shape[0], -1)
            return (X_train_flat, self.y_train), (X_val_flat, self.y_val), (X_test_flat, self.y_test)
        
        # Si no, retornem imatges 28x28 (per a CNNs)
        # Afegim una dimensió extra pel canal de color (N, 28, 28, 1) que Keras necessita
        X_train_exp = np.expand_dims(self.X_train, -1)
        X_val_exp = np.expand_dims(self.X_val, -1)
        X_test_exp = np.expand_dims(self.X_test, -1)
        
        return (X_train_exp, self.y_train), (X_val_exp, self.y_val), (X_test_exp, self.y_test)