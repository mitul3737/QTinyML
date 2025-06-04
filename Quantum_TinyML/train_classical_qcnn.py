# train_classical_qcnn.py
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# 1. Define a classical replacement for the QCNN
def create_classical_qcnn(input_shape=(10, 1)):
    model = models.Sequential([
        layers.Conv1D(4, kernel_size=2, activation='relu', input_shape=input_shape),
        layers.Flatten(),
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary output
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 2. Generate synthetic data (replace with your dataset)
X_train = np.random.rand(1000, 10, 1)  # 1000 samples, 10 timesteps, 1 feature
y_train = (np.sum(X_train, axis=1) > 5).astype(np.float32)  # Dummy labels

# 3. Train and save the model
model = create_classical_qcnn()
model.fit(X_train, y_train, epochs=10, batch_size=32)
model.save('classical_qcnn.h5')
print("Model saved as classical_qcnn.h5")
