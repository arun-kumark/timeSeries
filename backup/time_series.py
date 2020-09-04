"""
 _____ _                  ____            _
|_   _(_)_ __ ___   ___  / ___|  ___ _ __(_) ___  ___
  | | | | '_ ` _ \ / _ \ \___ \ / _ \ '__| |/ _ \/ __|
  | | | | | | | | |  __/  ___) |  __/ |  | |  __/\__ \
  |_| |_|_| |_| |_|\___| |____/ \___|_|  |_|\___||___/ Model test on EdgeTPU

"""

# import the required packages
import numpy as np

# for reading binary data file
import struct

# for the Auto Encoder
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp

from tensorflow.keras.constraints import UnitNorm, Constraint
from tensorflow.keras.layers import Input, InputLayer, Dense, Lambda, Reshape, Conv1D, BatchNormalization, Layer, InputSpec, Flatten, UpSampling1D
from tensorflow.keras.layers import GlobalAvgPool1D, RepeatVector, TimeDistributed, Dropout
from tensorflow.keras.models import save_model
from tensorflow.keras import metrics, Sequential, Model, regularizers, activations, initializers, constraints
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import UnitNorm, Constraint
tfd = tfp.distributions
tfpl = tfp.layers

# for EdgeTPU
from tflite_runtime.interpreter import load_delegate
import platform

# Read the binary data for time series data (ts_data)
ts_data= np.fromfile("data/data.bin", dtype=float)
ts_data = ts_data.reshape(60,2)
ts_data = np.expand_dims(ts_data, axis=0)


# Model constraints
input_shape = (60, 2)
BATCH_SIZE = 128
latent_dim = 4


# The Encoder part of the model
inp = Input(shape=input_shape, name='input')
encoder_conv1 = Conv1D(filters=16, kernel_size=3, padding='same', dilation_rate=1, activation='relu', 
                       kernel_regularizer=regularizers.l2(3e-5)
                      )(inp)
encoder_conv2 = Conv1D(filters=16, kernel_size=3, padding='same', dilation_rate=2, activation='relu', 
                       kernel_regularizer=regularizers.l2(3e-5)
                      )(encoder_conv1)
encoder_conv3 = Conv1D(filters=16, kernel_size=3, padding='same', dilation_rate=4, activation='relu', 
                       kernel_regularizer=regularizers.l2(3e-5)
                      )(encoder_conv2)
#encoder_conv4 = Conv1D(filters=16, kernel_size=3, padding='same', dilation_rate=8, activation=tf.nn.relu)(encoder_conv3)
encoder_flat = Flatten()(encoder_conv3)
#encoder_dense1 = Dense(32, activation=tf.nn.relu, kernel_constraint=UnitNorm(axis=0))(encoder_flat)
drop1 = Dropout(0.2)(encoder_flat)
encoded = Dense(latent_dim, activation='linear', name='bottleneck', kernel_constraint=UnitNorm(axis=0))(drop1)
encoder = Model(inputs=inp, outputs=encoded, name='encoder')
encoder.summary()

# the decoder part of the model
dec_inp = Input(shape=(latent_dim,), name='dec_input')
#dec_dense1 = Dense(32, activation=tf.nn.relu, kernel_constraint=UnitNorm(axis=1))(dec_inp)
dec_dense2 = Dense(60*16, activation='relu', kernel_constraint=UnitNorm(axis=1))(dec_inp)
dec_drop = Dropout(0.2)(dec_dense2)
dec_reshape = Reshape((60, 16))(dec_drop)
#decoder_conv = Conv1D(filters=16, kernel_size=3, padding='same', dilation_rate=8, activation=tf.nn.relu)(dec_reshape)
decoder_conv1 = Conv1D(filters=16, kernel_size=3, padding='same', dilation_rate=4, activation='relu', 
                       kernel_regularizer=regularizers.l2(3e-5)
                      )(dec_reshape)
decoder_conv2 = Conv1D(filters=16, kernel_size=3, padding='same', dilation_rate=2, activation='relu', 
                       kernel_regularizer=regularizers.l2(3e-5)
                      )(decoder_conv1)
decoder_conv3 = Conv1D(filters=16, kernel_size=3, padding='same', dilation_rate=1, activation='relu', 
                       kernel_regularizer=regularizers.l2(3e-5)
                      )(decoder_conv2)
decoded = Conv1D(filters=2, kernel_size=1, padding='same', activation=None, name='reconstructed')(decoder_conv3)
decoder = Model(inputs=dec_inp, outputs=decoded, name='decoder')
decoder.summary()

#autoencoder = Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs[0]), name='vae')
autoencoder = Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs), name='ae')
autoencoder.summary()

# loading EdgeTPU
EDGETPU_SHARED_LIB = {'Linux': 'libedgetpu.so.1'}[platform.system()]

# compiling the model
optimizer = keras.optimizers.Adam(1e-4)
autoencoder.compile(loss=keras.losses.logcosh,
                    optimizer=optimizer, 
                    metrics=['mean_absolute_error', 'mean_squared_error'])

# load best model
autoencoder.load_weights('Model/best.h5')

test_predictions = autoencoder.predict(ts_data)
print("Predictions on the x86 PC are ... ")
print(test_predictions)

def make_interpreter(model_file):
	model_file, *device = model_file.split('@')
	return tf.lite.Interpreter(
	model_path=model_file,
	experimental_delegates=[
	load_delegate(EDGETPU_SHARED_LIB,{'device': device[0]} if device else {})])

def main():
	interpreter = make_interpreter('Model/google_q.tflite')
	interpreter.allocate_tensors()
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
	casted_array = ts_data.astype(np.float32)
	interpreter.set_tensor(input_details[0]['index'], casted_array)
	interpreter.invoke()

	print("Predictions on the EdgeTPU are ... ")
	test_predictions = interpreter.get_tensor(output_details[0]['index'])
	print(test_predictions)


if __name__ == '__main__':
	main()
