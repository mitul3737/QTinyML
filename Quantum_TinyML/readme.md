1. Run the train_classical_qcnn.py which acts equivalent as [QCNN](https://github.com/mitul3737/Quantum-CNN) &  convert_to_tflite.py, 

```
python train_classical_qcnn.py
python convert_to_tflite.py
python convert_to_c_array.py
```

2. Deploy on Arduino:

- Copy **model.h** and **arduino_inference.ino** to your Arduino sketch folder.

- Install the **Arduino_TensorFlowLite** library via Arduino IDE.

- Upload to a compatible board (e.g., Nano 33 BLE).


# How to run it in your arduino?
We need an image to detect (from dataset)

This is what "detecting an image" usually implies.

## Choose a Camera: You'll need a camera module compatible with your Arduino.
Common Choices:
- OV7670 (and similar): Inexpensive, but can be complex to interface due to varying quality and require careful timing and data handling.
- Arducam Mini series: Often easier to use with provided libraries, but might be slightly more expensive.
- Arduino Nano 33 BLE Sense / Portenta H7: These boards have built-in cameras and native TensorFlow Lite Micro support, making integration much easier if you switch boards.
### Considerations:
- Resolution: Ensure the camera can output a resolution from which you can derive 28x28 (e.g., 32x32, 64x64, or larger that you then downsample).
- Color Format: Your model expects grayscale (1 channel). Most cameras output RGB. You'll need to convert RGB to grayscale (e.g., grayscale = (R + G + B) / 3 or weighted average 0.299*R + 0.587*G + 0.114*B).
-  Interface: SPI, I2C, DCMI (Digital Camera Interface), etc. This dictates how you'll connect and communicate with the camera.

## Then Integrate Camera Code:

Find a library for your chosen camera module.
Write code to initialize the camera, capture a frame, and read the pixel data.
### Pre-process the image:
- Downsample to 28x28: If the camera captures a larger image, you'll need an algorithm to scale it down. This can be complex, often involving averaging or picking pixels.
- Convert to Grayscale (if needed): Iterate through RGB pixels and calculate grayscale.
- Normalize to [0, 1]: Divide each pixel value (0-255) by 255.0.
- Store in float_input: Create a float float_input[784]; array and populate it with the preprocessed 28x28 grayscale pixel values.

Finally in the arduino_inference.ino file's loop

```
#include <TensorFlowLite.h>
#include "model.h"
// #include <YourCameraLibrary.h> // Include your camera library here

// ... (rest of your setup variables)

// Global array to hold the preprocessed image data
float camera_image_buffer[28 * 28 * 1]; // 784 pixels

void setup() {
  // ... (your existing setup code)
}

void loop() {
  // 1. Capture and Preprocess Image from Camera
  // (This is placeholder code - replace with actual camera library calls)
  // Example: YourCameraLibrary.captureFrame(camera_image_buffer, 28, 28, GRAYSCALE);
  // For demonstration, let's assume you have a function that populates camera_image_buffer
  // with a 28x28 normalized grayscale image.
  // For now, let's just fill with some dummy data to simulate a 28x28 image
  // (YOU MUST REPLACE THIS WITH ACTUAL CAMERA DATA)
  for (int i = 0; i < 784; i++) {
      camera_image_buffer[i] = (float)random(0, 256) / 255.0; // Random pixel values between 0 and 1
      // Or, for a simple test: camera_image_buffer[i] = 0.5; // All pixels grey
  }


  // 2. Quantize Input Data
  for (int i = 0; i < (28 * 28 * 1); i++) { // Loop 784 times
    input->data.int8[i] = camera_image_buffer[i] / input->params.scale + input->params.zero_point;
  }

  // 3. Run Inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed!");
    return;
  }

  // 4. Dequantize Output
  float prediction = (output->data.int8[0] - output->params.zero_point) * output->params.scale;

  // 5. Interpret Prediction
  Serial.print("Prediction: ");
  Serial.println(prediction);

  if (prediction > 0.5) { // Your model outputs 1 for class 0 (T-shirt/top), 0 for others
    Serial.println("Detected: T-shirt/Top (Class 0)");
  } else {
    Serial.println("Detected: Other Fashion Item");
  }

  delay(2000); // Wait for 2 seconds before next inference
}
```


