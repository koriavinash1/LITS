import numpy as np
import tensorflow as tf
import os
from cell import ConvGRUCell
from cell import ConvLSTMCell

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# # Create a placeholder for videos.
# inputs = tf.placeholder(tf.float32, [batch_size, timesteps] + shape + [channels])


# # # Add the ConvLSTM step.

# cell = ConvLSTMCell(shape, filters, kernel)
# outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=inputs.dtype)

# # There's also a ConvGRUCell that is more memory efficient.

# cell = ConvGRUCell(shape, filters, kernel)
# outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=inputs.dtype)

# # It's also possible to enter 2D input or 4D input instead of 3D.
# shape = [100]
# kernel = [3]
# inputs = tf.placeholder(tf.float32, [batch_size, timesteps] + shape + [channels])
# cell = ConvLSTMCell(shape, filters, kernel)
# outputs, state = tf.nn.bidirectional_dynamic_rnn(cell, cell, inputs, dtype=inputs.dtype)

# shape = [50, 50, 50]
# kernel = [1, 3, 5]
# inputs = tf.placeholder(tf.float32, [batch_size, timesteps] + shape + [channels])
# cell = ConvGRUCell(shape, filters, kernel)
# outputs, state= tf.nn.bidirectional_dynamic_rnn(cell, cell, inputs, dtype=inputs.dtype)

def count_variables():    
    total_parameters = 0
    #iterating over all variables
    for variable in tf.trainable_variables():  
        local_parameters=1
        shape = variable.get_shape()  #getting shape of a variable
        for i in shape:
            local_parameters*=i.value  #mutiplying dimension values
        total_parameters+=local_parameters
    print('Total Number of Trainable Parameters:', total_parameters) 

# Patientwise batches
iters = 2
batch_size = 1
slices = 10
shape = [240, 240]
kernel = [3, 3]
channels = 3
filters = 12
logdir = './logdir'
# Cell type is LSTM or GRU
cell_type = 'LSTM'
# Prefer using GRU cell typoe because of reduced number of parameters and simplicity
# ****** Dont use peephole connection: it will blow up number of parameters almost 1000 times

if not os.path.exists(logdir):
    os.makedirs(logdir)

# Create a placeholder for videos.
inputs_pl = tf.placeholder(tf.float32, [slices, batch_size] + shape + [channels])



# # Add the ConvLSTM step.
# Graph
if cell_type=='LSTM':
	print '@@@@@@@@@'
	cell = ConvLSTMCell(shape, filters, kernel,  peephole=False)
	# Approach to combine LSTM states:
	# from: https://github.com/tensorflow/tensorflow/issues/2838
	c_state_pl = tf.placeholder(tf.float32, [batch_size] + shape + [filters])
	h_state_pl = tf.placeholder(tf.float32, [batch_size] + shape + [filters])
	init_state = tf.nn.rnn_cell.LSTMStateTuple(c_state_pl, h_state_pl)
	lstm_outputs, lstm_states = tf.nn.dynamic_rnn(cell, inputs_pl, sequence_length=None,
	                                                initial_state=init_state,
	                                                dtype=inputs_pl.dtype,
	                                                parallel_iterations=None,
	                                                swap_memory=False,
	                                                time_major=True,
	                                                scope='conv_lstm')
	c_state, h_state = lstm_states
else:
	print '**********'
	cell = ConvGRUCell(shape, filters, kernel)
	state_pl = tf.placeholder(tf.float32, [batch_size] + shape + [filters])
	gru_outputs, gru_states = tf.nn.dynamic_rnn(cell, inputs_pl, sequence_length=None,
	                                                initial_state=state_pl,
	                                                dtype=inputs_pl.dtype,
	                                                parallel_iterations=None,
	                                                swap_memory=False,
	                                                time_major=True,
	                                                scope='conv_gru')

# Print the number of trainable variables
count_variables()

# Evaluation
with tf.Session() as sess:
	# Summarise the graph
	sess.run(tf.global_variables_initializer())
	tf.summary.FileWriter(logdir, sess.graph)
	# Evaluate
	if cell_type == 'LSTM':
		zero_state = cell.zero_state(batch_size, tf.float32)
		c_state_val, h_state_val = zero_state
		c_state_val = c_state_val.eval()
		h_state_val = h_state_val.eval()	
		for i in range(iters):
			test_input = np.random.randn(slices, batch_size , shape[0], shape[1], channels)
			feed_dict = {inputs_pl:test_input, c_state_pl: c_state_val, h_state_pl: h_state_val}
			outputs, c_state_val, h_state_val = sess.run([lstm_outputs, c_state, h_state], feed_dict=feed_dict)
			print '#'*10
			print i
			print len(outputs)
			print c_state_val.shape 
			print h_state_val.shape
	else:
		# For Initial state of the GRU memory.
		state_val = cell.zero_state(batch_size, tf.float32).eval()
		# state_val = np.ones_like(state_val)
		for i in range(iters):
			test_input = np.random.randn(slices, batch_size , shape[0], shape[1], channels)
			feed_dict = {inputs_pl:test_input, state_pl: state_val}
			outputs, state_val = sess.run([gru_outputs, gru_states], feed_dict=feed_dict)
			print '#'*10
			print i
			print len(outputs)
			print state_val.shape
			print outputs.shape 