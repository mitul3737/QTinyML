import tensorflow as tf

# 1. Load the trained model
model = tf.keras.models.load_model('classical_qcnn.h5')

# 2. Convert to quantized TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Quantization
tflite_model = converter.convert()

# 3. Save the TFLite model
with open('classical_qcnn.tflite', 'wb') as f:
    f.write(tflite_model)
print("TFLite model saved as classical_qcnn.tflite")
