import tensorflow as tf
import numpy as np
import os

model = tf.keras.models.load_model('bestModel.h5')
model.summary()
converter = tf.lite.TFLiteConverter.from_keras_model(model)
#converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file('Keras_Model.h5')
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Read the binary data for time series data (ts_data)
ts_data= np.fromfile("data/data.bin", dtype=float)
ts_data = ts_data.reshape(60,2)
ts_data = np.expand_dims(ts_data, axis=0)
casted_data = ts_data.astype(np.float32) 

def representative_dataset_gen():
  for _ in range(50):
    # Get sample input data as a numpy array in a method of your choosing.
    input = casted_data
    yield [input]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.float32 #int8 or uint8 for quantized input
converter.inference_output_type = tf.float32
converter.experimental_new_converter = False
tflite_quant_model = converter.convert()
with open('ts_ptq.tflite', 'wb') as f:
      f.write(tflite_quant_model)

os.system('edgetpu_compiler ./ts_ptq.tflite')
