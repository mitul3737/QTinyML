1. Run the 

train_classical_qcnn.py which acts equivalent as [QCNN](https://github.com/mitul3737/Quantum-CNN) &  convert_to_tflite.py, 

```
python train_classical_qcnn.py
python convert_to_tflite.py
python convert_to_c_array.py
```

2. Deploy on Arduino:

- Copy **model.h** and **arduino_inference.ino** to your Arduino sketch folder.

- Install the **Arduino_TensorFlowLite** library via Arduino IDE.

- Upload to a compatible board (e.g., Nano 33 BLE).
