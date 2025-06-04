1. Run **train_model.py, convert_model.py, test_quantized_model.py, convert_to_c_array.py** sequentially.
2. Copy the generated **model.h** to your Arduino sketch folder
3. Install the** Tensorflow Lite Micro** (via Arduino Library Manager)
4. **Upload the sketch** (arduino_inference.ino) to your compatible Arduino board

**Note:** 
- Ensure the Arduino board is compatible.
- The model.h file is in the same folder as the sketch.
- The library is installed (no missing #include errors).
