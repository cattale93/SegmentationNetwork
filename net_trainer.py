
# -*- coding: utf-8 -*-
############################ ~ IMPORT LIBRARIES ~ ##############################
import tensorflow as tf
import time
import datetime

from time import sleep

import numpy as np
import os
import gc

import os
import sys

sys.path.insert(0,'/home/alessandrocattoi/Seg_net/library/DataGenerator_ImageSeg')
os.chdir('/home/alessandrocattoi/Seg_net/library/DataGenerator_ImageSeg')

from DataGenerator_ImageSeg import DataGenerator_ImageSeg

###############################################################################

########################## ~ TRAINING PARAMETERS ~ ############################

LABELS = ["PASCOLO<5","PASCOLO5-20", "PASCOLO20-50", "BOSCO", "NON COLTIVABILE"]

N_of_classes = len(LABELS) # Number of classes

batch_size = 32

patch_size = 128

TRININING_ITERS = 50  #Number of training iterations -> MAX 50

learning_rate = 1e-5

start_date = time.asctime(time.localtime(time.time()))

print('Please insert job name:')
name = input()
print('Restore Model? y for yes n for no')
temp = input()

if 'y' in temp:
	restore_model = True
else:
	restore_model = False

model_data = '12_prc10_ps64_bs32_new__Ep50_Bs32_Ps64_gr1e-05_CACHE_Sun_Mar__8_11_09_34_2020/my_model-11'

net_output = ('/home/alessandrocattoi/Seg_net/model_output/' + str(name) + '_Ep' + str(TRININING_ITERS) + '_Bs' +
str(batch_size) + '_Ps' + str(patch_size) + '_gr' + str(learning_rate))+ '_OUTPUT_' + start_date

net_output = net_output.replace(' ','_')
net_output = net_output.replace(':','_')

net_parameter_set = ('/home/alessandrocattoi/Seg_net/model_cache/' + str(name) + '_Ep' + str(TRININING_ITERS) + '_Bs' +
str(batch_size) + '_Ps' + str(patch_size) + '_gr' + str(learning_rate))  + '_CACHE_' + start_date

net_parameter_set = net_parameter_set.replace(' ','_')
net_parameter_set = net_parameter_set.replace(':','_')

os.mkdir(net_output)
os.mkdir(net_parameter_set)

###############################################################################
def correct_map(mappa_orig,tara_gt):
	tara_flat = np.zeros((tara_gt.shape[0:3]), dtype = np.float32)
	tara_temp_gt = np.zeros((tara_gt.shape), dtype = np.float32)
	mappa_temp_orig = np.zeros((mappa_orig.shape), dtype = np.float32)

	tara_temp_gt[:,:,:,0] = tara_gt[:,:,:,0]*1
	tara_temp_gt[:,:,:,1] = tara_gt[:,:,:,1]*2
	tara_temp_gt[:,:,:,2] = tara_gt[:,:,:,2]*3
	tara_temp_gt[:,:,:,3] = tara_gt[:,:,:,3]*4
	tara_temp_gt[:,:,:,4] = tara_gt[:,:,:,4]*5
	N_of_pix = 0

	for i in range(0,tara_gt.shape[0]):
		temp = np.where(tara_temp_gt[i,:,:,0] == 1, 1, 0)
		tara_flat[i,:,:] = np.add(tara_flat[i,:,:],temp)
		temp = np.where(tara_temp_gt[i,:,:,1] == 2, 2, 0)
		tara_flat[i,:,:]  = np.add(tara_flat[i,:,:],temp)
		temp = np.where(tara_temp_gt[i,:,:,2] == 3, 3, 0)
		tara_flat[i,:,:]  = np.add(tara_flat[i,:,:],temp)
		temp = np.where(tara_temp_gt[i,:,:,3] == 4, 4, 0)
		tara_flat[i,:,:]  = np.add(tara_flat[i,:,:],temp)
		temp = np.where(tara_temp_gt[i,:,:,4] == 5, 5, 0)
		tara_flat[i,:,:]  = np.add(tara_flat[i,:,:],temp)

		mask  = np.where(tara_flat[i,:,:]  == 0, 0, 1)

		mappa_temp_orig[i,:,:,0] = np.multiply(mappa_orig[i,:,:,0],mask)
		mappa_temp_orig[i,:,:,1] = np.multiply(mappa_orig[i,:,:,1],mask)
		mappa_temp_orig[i,:,:,2] = np.multiply(mappa_orig[i,:,:,2],mask)
		mappa_temp_orig[i,:,:,3] = np.multiply(mappa_orig[i,:,:,3],mask)
		mappa_temp_orig[i,:,:,4] = np.multiply(mappa_orig[i,:,:,4],mask)

		mask  = np.where(tara_flat[i,:,:]  == 0, 1, 0)

		mappa_temp_orig[i,:,:,4] = np.add(mappa_temp_orig[i,:,:,4],mask)

		ignore, temp_N = np.unique(mask,return_counts=True)

		N_of_pix = N_of_pix + temp_N[0]

	return mappa_temp_orig, N_of_pix

##############################################################################
def fake_tara(mappa_tot,tara_orig):
	tara_flat_orig = np.zeros((tara_orig.shape[0:3]), dtype = np.float32)
	tara_fake = np.zeros((tara_orig.shape), dtype = np.float32)

	for i in range(0,tara_orig.shape[0]):
		temp = np.where(tara_orig[i,:,:,0] == 1, 1, 0)
		tara_flat_orig[i,:,:] = np.add(tara_flat_orig[i,:,:],temp)
		temp = np.where(tara_orig[i,:,:,1] == 1, 1, 0)
		tara_flat_orig[i,:,:]  = np.add(tara_flat_orig[i,:,:],temp)
		temp = np.where(tara_orig[i,:,:,2] == 1, 1, 0)
		tara_flat_orig[i,:,:]  = np.add(tara_flat_orig[i,:,:],temp)
		temp = np.where(tara_orig[i,:,:,3] == 1, 1, 0)
		tara_flat_orig[i,:,:]  = np.add(tara_flat_orig[i,:,:],temp)

		inv_mask  = np.where(tara_flat_orig[i,:,:]  == 0, 1, 0)

		tara_fake[i,:,:,0] = np.add(tara_orig[i,:,:,0],np.multiply(inv_mask,mappa_tot[i,:,:,0]))
		tara_fake[i,:,:,1] = np.add(tara_orig[i,:,:,1],np.multiply(inv_mask,mappa_tot[i,:,:,1]))
		tara_fake[i,:,:,2] = np.add(tara_orig[i,:,:,2],np.multiply(inv_mask,mappa_tot[i,:,:,2]))
		tara_fake[i,:,:,3] = np.add(tara_orig[i,:,:,3],np.multiply(inv_mask,mappa_tot[i,:,:,3]))

	return tara_fake
########################### ~ NETWORK PARAMETERS ~ #############################

stride_value = 3 # segnet_value = 3

conv_stride = 1 #stride of convolutional layer during image compression
conv_stride_d = 1 #stride of convolutional layer during image decompression
deconv_stride = 2 #stride of deconvolutional layer during image compression
pool_stride = 2 #stride of max pooling layer apllied during compression

pool_kernel = 2 #kernel dim for deconv and maxpool
conv_kernel = 3 #kernel dim for convolutional layer during image compression
conv_kernel_d = 1 #kernel dim for convolutional layer during image decompression

N_of_band = 5 #image band 3 for RGB + 1 for NIR

def maxpool2d(x, _name, hw_kernel_dim = pool_kernel, nc_kernel_dim = 1, nc_strides = 1, hw_strides = pool_stride):
	return tf.nn.max_pool(x,
		          ksize = [nc_kernel_dim, hw_kernel_dim, hw_kernel_dim, nc_kernel_dim],
		          strides = [nc_strides, hw_strides, hw_strides, nc_strides],
		          padding = 'SAME',
		          data_format = 'NHWC',
		          name = _name)

def conv2d(x, W, b, _name, hw_strides, nc_strides = 1):
	# Conv2D wrapper, with bias and relu activation
	# W contains the filter shape
	'''
	La conv2d prende in input il dato entrante x poi W rappresenta il filtro, che sono poi i pesi
	questo deve avere la forma corretta: [filter_height, filter_width, in_channels, out_channels]
	La stride, non mi è chiarissimo perche ma deve avere sempre 1 1, poi rispetto a pytorch anche gli
	altri due valori devono essere uguali
	'''
	x = tf.nn.conv2d(x,
		     W,
		     strides=[nc_strides, hw_strides, hw_strides, nc_strides],
		     padding='SAME',
		     data_format='NHWC',
		     name = _name)

	x = tf.nn.bias_add(x, b, data_format='NHWC')
	return tf.nn.relu(x)

def deconv2d(x, n_of_filters, _kernel, _name, strides = deconv_stride):
	x = tf.layers.conv2d_transpose(x,
		                   n_of_filters,
		                   _kernel,
		                   strides=(strides, strides),
		                   padding = 'SAME',
		                   data_format='channels_last',
		                   name = _name)
	return x


def net(x, weights, biases):
	'''La conv2d prende in input '''
	x = conv2d(x, weights['wc1_1'], biases['bc1_1'], _name ='conv_1_1', hw_strides = conv_stride)
	x = tf.layers.batch_normalization(x, name='batch_norm_1_1')
	x = conv2d(x, weights['wc1_2'], biases['bc1_2'], _name ='conv_1_2', hw_strides = conv_stride)
	x = tf.layers.batch_normalization(x, name='batch_norm_1_2')
	x = maxpool2d(x, _name ='maxpool_1')
	print(x)

	x = conv2d(x, weights['wc2_1'], biases['bc2_1'], _name ='conv_2_1', hw_strides = conv_stride)
	x = tf.layers.batch_normalization(x, name='batch_norm_2_1')
	x = conv2d(x, weights['wc2_2'], biases['bc2_2'], _name ='conv_2_2', hw_strides = conv_stride)
	x = tf.layers.batch_normalization(x, name='batch_norm_221')
	x = maxpool2d(x, _name ='maxpool2')
	print(x)
	x = conv2d(x, weights['wc3_1'], biases['bc3_1'], _name ='conv_3_1', hw_strides = conv_stride)
	x = tf.layers.batch_normalization(x, name='batch_norm_3_1')
	x = conv2d(x, weights['wc3_2'], biases['bc3_2'], _name ='conv_3_2', hw_strides = conv_stride)
	x = tf.layers.batch_normalization(x, name='batch_norm_3_2')
	x = conv2d(x, weights['wc3_3'], biases['bc3_3'], _name ='conv_3_3', hw_strides = conv_stride)
	x = tf.layers.batch_normalization(x, name='batch_norm_3_3')
	x = maxpool2d(x, _name ='maxpool3')
	print(x)
	x = conv2d(x, weights['wc4_1'], biases['bc4_1'], _name ='conv_4_1', hw_strides = conv_stride)
	x = tf.layers.batch_normalization(x, name='batch_norm_4_1')
	x = conv2d(x, weights['wc4_2'], biases['bc4_2'], _name ='conv_4_2', hw_strides = conv_stride)
	x = tf.layers.batch_normalization(x, name='batch_norm_4_2')
	x = conv2d(x, weights['wc4_3'], biases['bc4_3'], _name ='conv_4_3', hw_strides = conv_stride)
	x = tf.layers.batch_normalization(x, name='batch_norm_4_3')
	x = maxpool2d(x, _name ='maxpool4')
	print(x)
	x = conv2d(x, weights['wc5_1'], biases['bc5_1'], _name ='conv_5_1', hw_strides = conv_stride)
	x = tf.layers.batch_normalization(x, name='batch_norm_5_1')
	x = conv2d(x, weights['wc5_2'], biases['bc5_2'], _name ='conv_5_2', hw_strides = conv_stride)
	x = tf.layers.batch_normalization(x, name='batch_norm_5_2')
	x = conv2d(x, weights['wc5_3'], biases['bc5_3'], _name ='conv_5_3', hw_strides = conv_stride)
	x = tf.layers.batch_normalization(x, name='batch_norm_5_3')
	x = maxpool2d(x, _name ='maxpool5')
	print(x)
	x = deconv2d(x, 512, conv_kernel, _name ='deconv1')
	x = conv2d(x, weights['wc_d_5_1'], biases['bc_d_5_1'], _name ='conv_d_5_1', hw_strides = conv_stride_d)
	x = tf.layers.batch_normalization(x, name='batch_norm_d_5_1')
	x = conv2d(x, weights['wc_d_5_2'], biases['bc_d_5_2'], _name ='conv_d_5_2', hw_strides = conv_stride_d)
	x = tf.layers.batch_normalization(x, name='batch_norm_d_5_2')
	x = conv2d(x, weights['wc_d_5_3'], biases['bc_d_5_3'], _name ='conv_d_5_3', hw_strides = conv_stride_d)
	x = tf.layers.batch_normalization(x, name='batch_norm_d_5_3')
	print(x)
	x = deconv2d(x, 512, pool_kernel, _name ='deconv2')
	x = conv2d(x, weights['wc_d_4_1'], biases['bc_d_4_1'], _name ='conv_d_4_1', hw_strides = conv_stride_d)
	x = tf.layers.batch_normalization(x, name='batch_norm_d_4_1')
	x = conv2d(x, weights['wc_d_4_2'], biases['bc_d_4_2'], _name ='conv_d_4_2', hw_strides = conv_stride_d)
	x = tf.layers.batch_normalization(x, name='batch_norm_d_4_2')
	x = conv2d(x, weights['wc_d_4_3'], biases['bc_d_4_3'], _name ='conv_d_4_3', hw_strides = conv_stride_d)
	x = tf.layers.batch_normalization(x, name='batch_norm_d_4_3')
	print(x)
	x = deconv2d(x, 256, pool_kernel, _name ='deconv3')
	x = conv2d(x, weights['wc_d_3_1'], biases['bc_d_3_1'], _name ='conv_d_3_1', hw_strides = conv_stride_d)
	x = tf.layers.batch_normalization(x, name='batch_norm_d_3_1')
	x = conv2d(x, weights['wc_d_3_2'], biases['bc_d_3_2'], _name ='conv_d_3_2', hw_strides = conv_stride_d)
	x = tf.layers.batch_normalization(x, name='batch_norm_d_3_2')
	x = conv2d(x, weights['wc_d_3_3'], biases['bc_d_3_3'], _name ='conv_d_3_3', hw_strides = conv_stride_d)
	x = tf.layers.batch_normalization(x, name='batch_norm_d_3_3')
	print(x)
	x = deconv2d(x, 128, pool_kernel, _name ='deconv4')
	x = conv2d(x, weights['wc_d_2_1'], biases['bc_d_2_1'], _name ='conv_d_2_1', hw_strides = conv_stride_d)
	x = tf.layers.batch_normalization(x, name='batch_norm_d_2_1')
	x = conv2d(x, weights['wc_d_2_2'], biases['bc_d_2_2'], _name ='conv_d_2_2', hw_strides = conv_stride_d)
	x = tf.layers.batch_normalization(x, name='batch_norm_d_2_2')
	print(x)
	x = deconv2d(x, 64, pool_kernel, _name ='deconv5')
	x = conv2d(x, weights['wc_d_1_1'], biases['bc_d_1_1'], _name ='conv_d_1_1', hw_strides = conv_stride_d)
	x = tf.layers.batch_normalization(x, name='batch_norm_d_1_1')
	x = conv2d(x, weights['wc_d_1_2'], biases['bc_d_1_2'], _name ='conv_d_1_2', hw_strides = conv_stride_d)

	x = tf.nn.softmax(x, axis = 3,name ='softmax')
	print(x)
	return x



biases = {
	'bc1_1': tf.get_variable('B0', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
	'bc1_2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
	'bc2_1': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
	'bc2_2': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
	'bc3_1': tf.get_variable('B4', shape=(256), initializer=tf.contrib.layers.xavier_initializer()),
	'bc3_2': tf.get_variable('B5', shape=(256), initializer=tf.contrib.layers.xavier_initializer()),
	'bc3_3': tf.get_variable('B6', shape=(256), initializer=tf.contrib.layers.xavier_initializer()),
	'bc4_1': tf.get_variable('B7', shape=(512), initializer=tf.contrib.layers.xavier_initializer()),
	'bc4_2': tf.get_variable('B8', shape=(512), initializer=tf.contrib.layers.xavier_initializer()),
	'bc4_3': tf.get_variable('B9', shape=(512), initializer=tf.contrib.layers.xavier_initializer()),
	'bc5_1': tf.get_variable('B10', shape=(512), initializer=tf.contrib.layers.xavier_initializer()),
	'bc5_2': tf.get_variable('B11', shape=(512), initializer=tf.contrib.layers.xavier_initializer()),
	'bc5_3': tf.get_variable('B12', shape=(512), initializer=tf.contrib.layers.xavier_initializer()),
	'bc_d_5_1': tf.get_variable('B13', shape=(512), initializer=tf.contrib.layers.xavier_initializer()),
	'bc_d_5_2': tf.get_variable('B14', shape=(512), initializer=tf.contrib.layers.xavier_initializer()),
	'bc_d_5_3': tf.get_variable('B15', shape=(512), initializer=tf.contrib.layers.xavier_initializer()),
	'bc_d_4_1': tf.get_variable('B16', shape=(512), initializer=tf.contrib.layers.xavier_initializer()),
	'bc_d_4_2': tf.get_variable('B17', shape=(512), initializer=tf.contrib.layers.xavier_initializer()),
	'bc_d_4_3': tf.get_variable('B18', shape=(256), initializer=tf.contrib.layers.xavier_initializer()),
	'bc_d_3_1': tf.get_variable('B19', shape=(256), initializer=tf.contrib.layers.xavier_initializer()),
	'bc_d_3_2': tf.get_variable('B20', shape=(256), initializer=tf.contrib.layers.xavier_initializer()),
	'bc_d_3_3': tf.get_variable('B21', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
	'bc_d_2_1': tf.get_variable('B22', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
	'bc_d_2_2': tf.get_variable('B23', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
	'bc_d_1_1': tf.get_variable('B24', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
	'bc_d_1_2': tf.get_variable('B25', shape=(N_of_classes), initializer=tf.contrib.layers.xavier_initializer()),
}

weights = {
	'wc1_1': tf.get_variable('W0', shape=(conv_kernel,conv_kernel,N_of_band,64), initializer=tf.contrib.layers.xavier_initializer()),
	'wc1_2': tf.get_variable('W1', shape=(conv_kernel,conv_kernel,64,64), initializer=tf.contrib.layers.xavier_initializer()),
	'wc2_1': tf.get_variable('W2', shape=(conv_kernel,conv_kernel,64,128), initializer=tf.contrib.layers.xavier_initializer()),
	'wc2_2': tf.get_variable('W3', shape=(conv_kernel,conv_kernel,128,128), initializer=tf.contrib.layers.xavier_initializer()),
	'wc3_1': tf.get_variable('W4', shape=(conv_kernel,conv_kernel,128,256), initializer=tf.contrib.layers.xavier_initializer()),
	'wc3_2': tf.get_variable('W5', shape=(conv_kernel,conv_kernel,256,256), initializer=tf.contrib.layers.xavier_initializer()),
	'wc3_3': tf.get_variable('W6', shape=(conv_kernel,conv_kernel,256,256), initializer=tf.contrib.layers.xavier_initializer()),
	'wc4_1': tf.get_variable('W7', shape=(conv_kernel,conv_kernel,256,512), initializer=tf.contrib.layers.xavier_initializer()),
	'wc4_2': tf.get_variable('W8', shape=(conv_kernel,conv_kernel,512,512), initializer=tf.contrib.layers.xavier_initializer()),
	'wc4_3': tf.get_variable('W9', shape=(conv_kernel,conv_kernel,512,512), initializer=tf.contrib.layers.xavier_initializer()),
	'wc5_1': tf.get_variable('W10', shape=(conv_kernel,conv_kernel,512,512), initializer=tf.contrib.layers.xavier_initializer()),
	'wc5_2': tf.get_variable('W11', shape=(conv_kernel,conv_kernel,512,512), initializer=tf.contrib.layers.xavier_initializer()),
	'wc5_3': tf.get_variable('W12', shape=(conv_kernel,conv_kernel,512,512), initializer=tf.contrib.layers.xavier_initializer()),
	'wc_d_5_1': tf.get_variable('W13', shape=(conv_kernel_d,conv_kernel_d,512,512), initializer=tf.contrib.layers.xavier_initializer()),
	'wc_d_5_2': tf.get_variable('W14', shape=(conv_kernel_d,conv_kernel_d,512,512), initializer=tf.contrib.layers.xavier_initializer()),
	'wc_d_5_3': tf.get_variable('W15', shape=(conv_kernel_d,conv_kernel_d,512,512), initializer=tf.contrib.layers.xavier_initializer()),
	'wc_d_4_1': tf.get_variable('W16', shape=(conv_kernel_d,conv_kernel_d,512,512), initializer=tf.contrib.layers.xavier_initializer()),
	'wc_d_4_2': tf.get_variable('W17', shape=(conv_kernel_d,conv_kernel_d,512,512), initializer=tf.contrib.layers.xavier_initializer()),
	'wc_d_4_3': tf.get_variable('W18', shape=(conv_kernel_d,conv_kernel_d,512,256), initializer=tf.contrib.layers.xavier_initializer()),
	'wc_d_3_1': tf.get_variable('W19', shape=(conv_kernel_d,conv_kernel_d,256,256), initializer=tf.contrib.layers.xavier_initializer()),
	'wc_d_3_2': tf.get_variable('W20', shape=(conv_kernel_d,conv_kernel_d,256,256), initializer=tf.contrib.layers.xavier_initializer()),
	'wc_d_3_3': tf.get_variable('W21', shape=(conv_kernel_d,conv_kernel_d,256,128), initializer=tf.contrib.layers.xavier_initializer()),
	'wc_d_2_1': tf.get_variable('W22', shape=(conv_kernel_d,conv_kernel_d,128,128), initializer=tf.contrib.layers.xavier_initializer()),
	'wc_d_2_2': tf.get_variable('W23', shape=(conv_kernel_d,conv_kernel_d,128,64), initializer=tf.contrib.layers.xavier_initializer()),
	'wc_d_1_1': tf.get_variable('W24', shape=(conv_kernel_d,conv_kernel_d,64,64), initializer=tf.contrib.layers.xavier_initializer()),
	'wc_d_1_2': tf.get_variable('W25', shape=(conv_kernel_d,conv_kernel_d,64,N_of_classes), initializer=tf.contrib.layers.xavier_initializer()),
    }

###############################################################################

#Network prediction
x = tf.placeholder("float32", shape=[batch_size, patch_size, patch_size, N_of_band])
y = tf.placeholder("float32", shape=[batch_size, patch_size, patch_size, N_of_classes])

pred_ph = tf.placeholder("float32", shape=[batch_size, patch_size, patch_size, N_of_classes])

predictions = net(x,weights,biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#Accuracuy
accuracy_train = tf.count_nonzero(tf.keras.metrics.categorical_accuracy(pred_ph,y))
#accuracy_test = tf.count_nonzero(tf.keras.metrics.categorical_accuracy(predictions,y)),(patch_size*patch_size*batch_size))
cost_train = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_ph, labels=y))

##############################################################################

#Initilizer
init = tf.global_variables_initializer()

#Parameter saving function
saver = tf.train.Saver()

################################# SET UP #######################################
#datapath preparation
start_time_set_up = time.time()
root = '/home/alessandrocattoi/Seg_net/dataset/' + '28_Patch_size128_prc20__50_overlapping_Sun_Mar_24_8_07_10_2020/'


path_data_train = root + 'train' + str(batch_size) + '/data/'
path_data_test = root + 'test' + str(batch_size) + '/data/'
path_data_valid = root + 'valid' + str(batch_size) + '/data/'

path_tare_train = root + 'train' + str(batch_size) + '/tare/'
path_tare_test = root + 'test' + str(batch_size) + '/tare/'
path_tare_valid = root + 'valid' + str(batch_size) + '/tare/'

#Data class inizialization
data_train = DataGenerator_ImageSeg(path_data_train, path_tare_train, batch_size, shuffle = True)
data_test = DataGenerator_ImageSeg(path_data_test, path_tare_test, batch_size, shuffle = True)
data_valid = DataGenerator_ImageSeg(path_data_valid, path_tare_valid, batch_size, shuffle = True)

N_of_train_batches = data_train.get_available_batches()
N_of_test_batches = data_test.get_available_batches()
N_of_valid_batches = data_valid.get_available_batches()

#Vector measurments
training_time = []
training_acc = np.zeros(shape = [TRININING_ITERS, N_of_train_batches], dtype = np.float32)
train_cost = np.zeros(shape = [TRININING_ITERS, N_of_train_batches], dtype = np.float32)
test_acc = np.zeros(shape = [TRININING_ITERS, N_of_test_batches], dtype = np.float32)
test_cost = np.zeros(shape = [TRININING_ITERS, N_of_test_batches], dtype = np.float32)
validation_acc = []

#Printig support varaibles
str_time_epoch1 = []
str_time_epoch2 = []
str_epoch = []
str_resume = []
################################################################################
elapsed_time_set_up = time.time() - start_time_set_up
str_set_up = '\nSet up took {}'.format( time.strftime('%H h %M m %S s', time.gmtime(elapsed_time_set_up)))
print(str_set_up)

########################### TRAINING OPERATIONS ################################

with tf.Session() as sess:
	sess.run(init)
	if restore_model:
		saver.restore(sess, '/home/alessandrocattoi/Seg_net/model_cache/' + model_data)

	for iterations in range(TRININING_ITERS):

		print('\nEpoch {} of {}'.format(iterations,TRININING_ITERS))

		start_time_epoch = time.time()

		for i in range(0, N_of_train_batches):

			start_time_batch = time.time()


			#NETWORK TRAINING
			#Extract batch
			batch_RGB_NIR, batch_tare = data_train.patch_extractor()
			#Optimization
			#batch_tare_fake = fake_tara(batch_RGB_NIR,batch_tare)
			sess.run(optimizer, feed_dict={x:batch_RGB_NIR, y:batch_tare})

			#BATCH ACCURACY
			#Run network
			pred = sess.run(predictions, feed_dict={x:batch_RGB_NIR})
			predizioni, N = correct_map(pred, batch_tare)
			acc_temp = sess.run(accuracy_train, feed_dict={pred_ph:predizioni, y: batch_tare})
			acc_temp = acc_temp / N
			#calculate cost
			cost_temp = sess.run(cost_train, feed_dict={pred_ph:pred, y: batch_tare})


			#Store result
			training_acc[iterations, i] = acc_temp
			train_cost[iterations, i] = cost_temp

			#Print Info of this batch
			print('\nActual Epoch {} of {}'.format(iterations,TRININING_ITERS))
			elapsed_time_batch = time.time() - start_time_batch
			print('Batch {} -> {} accurate. Cost func -> {}. Time {}'.format(i, acc_temp, cost_temp, time.strftime('%M m %S s', time.gmtime(elapsed_time_batch))))
			print(str_resume)
			print(str_time_epoch2)

		#TEST ACCURACY
		for j in range(0,N_of_test_batches):
			#Extract a batch of test image
			batch_RGB_NIR, batch_tare = data_test.patch_extractor()
			pred = sess.run(predictions, feed_dict={x:batch_RGB_NIR})
			#Calculate accuracy
			predizioni, N = correct_map(pred, batch_tare)
			acc_temp_test = sess.run(accuracy_train, feed_dict={pred_ph:pred, y: batch_tare}) / N
			cost_temp_test = sess.run(cost_train, feed_dict={pred_ph:pred, y: batch_tare})

			test_acc[iterations, j] = acc_temp_test
			test_cost[iterations, j] = cost_temp_test

		#Print training status
		str_epoch ='\nEpoch {} of {}'.format(iterations+1,TRININING_ITERS)
		str_resume = 'TRAIN accuracy is: {} - TEST accuracy is: {} - COST is: {}'.format(np.mean(training_acc[iterations,:]),np.mean(test_acc[iterations,:]),np.mean(train_cost[iterations,:]))
		print(str_epoch)
		print(str_resume)

		#EPOCH TIME DURATION PRINTER
		elapsed_time = time.time() - start_time_epoch
		str_time_epoch1 = 'Last epoch took {}'.format(time.strftime('%H h %M m %S s', time.gmtime(elapsed_time)))
		training_time.append(elapsed_time)
		epoch_duration_mean = np.mean(training_time)
		str_time_epoch2 = 'Mean epoch took {}.\nTraining process running from {} of approximately {}'.format(
					                time.strftime('%H h %M m %S s', time.gmtime(epoch_duration_mean)),
					                time.strftime('%d d %H h %M m %S s', time.gmtime((epoch_duration_mean * (iterations + 1)-(3600*24)))),
					                time.strftime('%d d %H h %M m %S s', time.gmtime((epoch_duration_mean * TRININING_ITERS)-(3600*24))))

		print(str_time_epoch1)
		print(str_time_epoch2)
		data_train.on_epoch_end()
		data_test.on_epoch_end()

		#SAVE ACCURACY RESULTS
		np.save(net_output + '/test_acc.npy', test_acc)
                #SAVE ACCURACY RESULTS
		np.save(net_output + '/train_acc.npy', training_acc)
		#SAVE COST TRAIN
		np.save(net_output + '/cost_train.npy', train_cost)
		#SAVE COST TEST
		np.save(net_output + '/cost_test.npy', test_cost)
		#SAVE TIME
		np.save(net_output + '/time.npy', (epoch_duration_mean * (iterations + 1)-(3600*24)))
		#Save model
		save_path = saver.save(sess, net_parameter_set + '/my_model',iterations)
		print('Model saved in path: %s',save_path)



	#VALIDATION ACCURACY
	for j in range(0,N_of_valid_batches):
		#Extract a batch of validation image
		batch_RGB_NIR, batch_tare = data_valid.patch_extractor()
		pred = sess.run(predictions, feed_dict={x:batch_RGB_NIR})
		#Calculate accuracy
		predizioni, N = correct_map(pred, batch_tare)
		acc_valid_temp = sess.run(accuracy_train, feed_dict={pred_ph:pred, y: batch_tare})
		acc_valid_temp = acc_valid_temp / N
		validation_acc.append(acc_valid_temp)

	print('VALID accuracy is: {}'.format(np.mean(validation_acc)))
	np.save(net_output + '/valid_acc.npy', validation_acc)
	print('END --> Execution time = {}'.format(np.sum(training_time)))
