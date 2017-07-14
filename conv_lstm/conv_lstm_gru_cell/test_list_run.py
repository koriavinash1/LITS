import numpy as np
import tensorflow as tf
import os
from cell import ConvGRUCell
from cell import ConvLSTMCell
np.random.seed(0)
tf.set_random_seed(0)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
iters = 3
batch_size = 1
slices = 10
shape = [64, 64]
kernel = [3, 3]
channels = 80
filters = 80
logdir = './logdir'
# Cell type is LSTM or GRU
cell_type = 'GRU'
# Prefer using GRU cell typoe because of reduced number of parameters and simplicity
# ****** Dont use peephole connection: it will blow up number of parameters almost 1000 times

if not os.path.exists(logdir):
    os.makedirs(logdir)

graph = tf.get_default_graph()
# Create a placeholder for videos.
inputs_pl = tf.placeholder(tf.float32, [slices, batch_size] + shape + [channels])
# # Add the ConvLSTM step.
# Graph
if cell_type=='LSTM':
	print '@@@@@@@@@'
	c_state_var = tf.get_variable('c_state_var', shape=[batch_size]+shape+[filters], trainable=False, initializer=tf.zeros_initializer())
	h_state_var = tf.get_variable('h_state_var', shape=[batch_size]+shape+[filters], trainable=False, initializer=tf.zeros_initializer())
	init_state = tf.nn.rnn_cell.LSTMStateTuple(c_state_var, h_state_var)
	cell = ConvLSTMCell(shape, filters, kernel, peephole=False)
	lstm_outputs, lstm_states = tf.nn.dynamic_rnn(cell, inputs_pl, sequence_length=None,
	                                                initial_state=init_state,
	                                                dtype=inputs_pl.dtype,
	                                                parallel_iterations=None,
	                                                swap_memory=False,
	                                                time_major=True,
	                                                scope='conv_lstm')
	c_state_var_ops = tf.assign(c_state_var, lstm_states[0])
	h_state_var_ops = tf.assign(h_state_var, lstm_states[1])
	state_var_ops = tf.group(c_state_var_ops, h_state_var_ops)
	reset_ops = tf.variables_initializer([c_state_var, h_state_var], name="reset_states")	

else:
	print '**********'
	state_var = tf.get_variable('state_var', shape=[batch_size]+shape+[filters], trainable=False, initializer=tf.zeros_initializer())
	print 'STATE shape'
	print state_var.shape
	cell = ConvGRUCell(shape, filters, kernel)
	gru_outputs, gru_states = tf.nn.dynamic_rnn(cell, inputs_pl, sequence_length=None,
	                                                initial_state=state_var,
	                                                dtype=inputs_pl.dtype,
	                                                parallel_iterations=None,
	                                                swap_memory=False,
	                                                time_major=True,
	                                                scope='conv_gru')

	state_var_ops = tf.assign(state_var, gru_states)
	reset_ops = tf.variables_initializer([state_var], name="reset_states")

# Print the number of trainable variables
count_variables()

# Evaluation
with tf.Session() as sess:
	# Summarise the graph
	sess.run(tf.global_variables_initializer())
	tf.summary.FileWriter(logdir, sess.graph)
	# Evaluate
	if cell_type == 'LSTM':
		test_input = np.random.randn(slices, batch_size , shape[0], shape[1], channels)
		for epoch in range(2):
			sess.run(reset_ops)
			for i in range(iters):
				feed_dict = {inputs_pl:test_input}
				outputs, state_val, _ = sess.run([lstm_outputs, lstm_states, state_var_ops], feed_dict=feed_dict)
				print '#'*10
				print i
				print len(outputs)
				print state_val
				# print outputs 
	else:
		test_input = np.random.randn(slices, batch_size , shape[0], shape[1], channels)
		print 'INPUT shape'
		print test_input.shape
		# print test_input
		for epoch in range(2):
			sess.run(reset_ops)
			for i in range(iters):
				feed_dict = {inputs_pl:test_input}
				outputs, state_val, _ = sess.run([gru_outputs, gru_states, state_var_ops], feed_dict=feed_dict)
				print '#'*10
				print i
				print len(outputs)
				print state_val.shape
				print outputs.shape

# reset_state_pl = tf.placeholder(tf.bool, shape=[], name="reset_state")
# for i in xrange(1):
#     with tf.name_scope('myscope_%d' % i):
#         state_var_ops = state_var.assign(gru_states)

# all_operations = [graph.get_operation_by_name('myscope_%d' % i) for i in range(i)]
# all_op = tf.group(*all_operations)
	# state_var = tf.cond(tf.equal(reset_state_pl, True), lambda: cell.zero_state(batch_size, tf.float32),
	# 								lambda: state_var)