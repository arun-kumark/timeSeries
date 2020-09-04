import numpy as np
import platform
import tflite_runtime.interpreter as tflite
import time

# Read the binary data for time series data (ts_data)
ts_data= np.fromfile("data/data.bin", dtype=float)
ts_data = ts_data.reshape(60,2)
ts_data = np.expand_dims(ts_data, axis=0)

# loading EdgeTPU
EDGETPU_SHARED_LIB = {'Linux': 'libedgetpu.so.1'}[platform.system()]

def make_interpreter(model_file):
	model_file, *device = model_file.split('@')
	return tflite.Interpreter(
	model_path=model_file,
	experimental_delegates=[
	tflite.load_delegate(EDGETPU_SHARED_LIB,{'device': device[0]} if device else {})])

def main():
	for model in ['ts_ptq.tflite', 'ts_ptq_edgetpu.tflite']:
		print('Evaluating: ', model)
		interpreter = make_interpreter(model)
		interpreter.allocate_tensors()
		input_details = interpreter.get_input_details()
		output_details = interpreter.get_output_details()
		casted_array = ts_data.astype(np.float32)
		interpreter.set_tensor(input_details[0]['index'], casted_array)
		for _ in range(10):
			start = time.perf_counter()
			interpreter.invoke()
			inference_time = time.perf_counter() - start
			print('%.2f us' % (inference_time * 1e6))

if __name__ == '__main__':
	main()
