import tensorflow as tf
import numpy as np
import os

# Load the trained model
model = tf.keras.models.load_model('simple_model.h5')

# Get the original model size by saving it and checking file size
model.save('temp_model.keras')  # Save in native Keras format
original_size = os.path.getsize('temp_model.keras')
# Clean up
os.remove('temp_model.keras')

# Convert to TensorFlow Lite - new method for Keras 3+
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Apply quantization for smaller size and faster inference
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Representative dataset for full integer quantization
def representative_data_gen():
    for _ in range(100):
        x = np.random.rand(1, 10).astype(np.float32)
        yield [x]

converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8

# Convert the model
tflite_quant_model = converter.convert()

# Save the quantized model
with open('quantized_model.tflite', 'wb') as f:
    f.write(tflite_quant_model)

print("Quantized model saved as quantized_model.tflite")
print(f"Original model size: {original_size/1024:.2f} KB")
print(f"Quantized TFLite size: {len(tflite_quant_model)/1024:.2f} KB")
print(f"Reduction: {(original_size - len(tflite_quant_model))/original_size*100:.1f}% smaller")
