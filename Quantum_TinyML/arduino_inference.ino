#include <TensorFlowLite.h>
#include "model.h"  // Generated header

// TensorFlow Lite setup
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 4 * 1024;  // Adjust if needed
uint8_t tensor_arena[kTensorArenaSize];
}

void setup() {
  Serial.begin(9600);
  while (!Serial);

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load the TFLite model
  model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model version mismatch!");
    while(1);
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed!");
    while(1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  // Print quantization params (if quantized)
  Serial.print("Input scale: "); Serial.println(input->params.scale);
  Serial.print("Input zero point: "); Serial.println(input->params.zero_point);
}

void loop() {
  // Simulate input data (replace with real sensor data)
  float float_input[10] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
  
  // Quantize input (if model is quantized)
  for (int i = 0; i < 10; i++) {
    input->data.int8[i] = float_input[i] / input->params.scale + input->params.zero_point;
  }

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed!");
    return;
  }

  // Dequantize output
  float prediction = (output->data.int8[0] - output->params.zero_point) * output->params.scale;
  
  Serial.print("Prediction: "); Serial.println(prediction);
  delay(2000);
}
