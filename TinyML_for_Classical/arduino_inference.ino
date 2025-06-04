#include <TensorFlowLite.h>
#include "model.h"

namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 4 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

void setup() {
  Serial.begin(9600);
  while (!Serial);

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

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

  // Print quantization parameters
  Serial.print("Input scale: "); Serial.println(input->params.scale);
  Serial.print("Input zero point: "); Serial.println(input->params.zero_point);
  Serial.print("Output scale: "); Serial.println(output->params.scale);
  Serial.print("Output zero point: "); Serial.println(output->params.zero_point);
}

void loop() {
  // Generate random input (simulate sensor data)
  float float_input[10];
  for (int i = 0; i < 10; i++) {
    float_input[i] = random(100) / 100.0;
  }

  // Quantize input
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

  // Print results
  Serial.print("Input: ");
  for (int i = 0; i < 10; i++) {
    Serial.print(float_input[i]); Serial.print(" ");
  }
  Serial.print("\nPrediction: "); Serial.println(prediction);
  Serial.print("Class: "); Serial.println(prediction > 0.5 ? "1" : "0");
  
  delay(2000);
}
