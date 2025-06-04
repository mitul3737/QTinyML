import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models

# Generate synthetic data for a simple binary classification problem
def generate_data(num_samples=1000):
    np.random.seed(42)
    x = np.random.rand(num_samples, 10).astype(np.float32)  # 10 features
    y = (np.sum(x, axis=1) > 5).astype(np.int32)  # Simple threshold classification
    return x, y

# Create a simple model
def create_model():
    model = models.Sequential([
        layers.Dense(16, activation='relu', input_shape=(10,)),
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

# Main training script
def main():
    # Generate data
    x_train, y_train = generate_data(1000)
    x_test, y_test = generate_data(200)
    
    # Create and train model
    model = create_model()
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
    
    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Save the model
    model.save('simple_model.h5')
    print("Model saved as simple_model.h5")

if __name__ == "__main__":
    main()
