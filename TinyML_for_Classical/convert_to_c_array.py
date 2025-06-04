def convert_to_c_array(bytes_model):
    hex_array = [f"0x{byte:02x}" for byte in bytes_model]
    return "{\n  " + ",\n  ".join(", ".join(hex_array[i:i+12]) for i in range(0, len(hex_array), 12)) + "\n}"

with open('quantized_model.tflite', 'rb') as f:
    tflite_model = f.read()

header_content = f"""
#ifndef MODEL_H
#define MODEL_H

const unsigned char model_data[] = {convert_to_c_array(tflite_model)};
const int model_data_len = {len(tflite_model)};

#endif
"""

with open('model.h', 'w') as f:
    f.write(header_content)
