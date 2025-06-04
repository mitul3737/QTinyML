import tensorflow as tf
import numpy as np
import time

def load_and_test_model(model_path, num_tests=5, verbose=True):
    """Comprehensive model verification function"""
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if verbose:
        print("\n" + "="*50)
        print("Model Verification Report")
        print("="*50)
        print("\nModel Input Details:")
        for detail in input_details:
            print(f"- Name: {detail['name']}")
            print(f"  Shape: {detail['shape']}")
            print(f"  Type: {detail['dtype']}")
            print(f"  Quantization: {detail['quantization']}")

        print("\nModel Output Details:")
        for detail in output_details:
            print(f"- Name: {detail['name']}")
            print(f"  Shape: {detail['shape']}")
            print(f"  Type: {detail['dtype']}")
            print(f"  Quantization: {detail['quantization']}")

    # Get quantization parameters
    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']

    # Performance metrics
    inference_times = []
    test_cases = []

    # Test cases including edge cases
    test_inputs = [
        np.random.rand(1, 10).astype(np.float32),  # Random normal case
        np.zeros((1, 10)),  # All zeros
        np.ones((1, 10)),  # All ones
        np.full((1, 10), 0.5),  # Mid-range values
        np.random.uniform(low=-1, high=2, size=(1, 10)).astype(np.float32)  # Out-of-range (will be clipped)
    ]

    for i, float_input in enumerate(test_inputs[:num_tests]):
        # Quantize input
        quantized_input = np.clip(
            (float_input / input_scale) + input_zero_point,
            np.iinfo(np.int8).min,
            np.iinfo(np.int8).max
        ).astype(np.int8)

        # Run inference
        start_time = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], quantized_input)
        interpreter.invoke()
        inference_time = (time.perf_counter() - start_time) * 1000  # in ms
        inference_times.append(inference_time)

        # Get output
        quantized_output = interpreter.get_tensor(output_details[0]['index'])
        float_output = (quantized_output - output_zero_point) * output_scale

        # Store results
        test_cases.append({
            "test_num": i+1,
            "input": float_input,
            "quantized_input": quantized_input,
            "output": quantized_output,
            "dequantized_output": float_output,
            "inference_time_ms": inference_time
        })

        if verbose:
            print("\n" + "-"*40)
            print(f"TEST CASE {i+1}")
            print("-"*40)
            print("Input (float32):\n", float_input)
            print("Quantized Input (int8):\n", quantized_input)
            print("Quantized Output (int8):", quantized_output)
            print("Dequantized Output (float32):", float_output)
            print(f"Inference Time: {inference_time:.2f} ms")

    # Summary statistics
    if verbose:
        avg_time = np.mean(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        
        print("\n" + "="*50)
        print("Verification Summary")
        print("="*50)
        print(f"Total Test Cases: {num_tests}")
        print(f"Average Inference Time: {avg_time:.2f} ms")
        print(f"Minimum Inference Time: {min_time:.2f} ms")
        print(f"Maximum Inference Time: {max_time:.2f} ms")
        print("\nAll test cases completed successfully!")

    return test_cases

if __name__ == "__main__":
    # Run comprehensive tests
    test_results = load_and_test_model("quantized_model.tflite", num_tests=5)
    
    # You can access individual test results like:
    # test_results[0]['input']  # First test case input
    # test_results[0]['dequantized_output']  # First test case output
