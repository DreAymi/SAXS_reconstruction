import numpy as np
import tensorflow as tf
import random
import os
import exceptions

BATCH_SIZE = 100
SEED=56297
z_dim=200


def variable_on_cpu(name,shape,stddev,trainable=False):
	with tf.device('/cpu:0'):
		var=tf.get_variable(name,shape,initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32),trainable=trainable)
	return var


def encode(input,batchsize):
	with tf.variable_scope('conv1') as scope:
		weight1_1=variable_on_cpu('weight1',[3,3,3,1,64],np.sqrt(2./(3*3*3)))
		bias1_1=var=variable_on_cpu('bias1',[64],0)
		conv=tf.nn.conv3d(input,weight1_1,strides=[1,1,1,1,1],padding='SAME') + bias1_1
		weight1_2=variable_on_cpu('weight2',[3,3,3,64,64],np.sqrt(2./(3*3*3*64)))
		bias1_2=var=variable_on_cpu('bias2',[64],0)
		conv1=tf.nn.conv3d(conv,weight1_2,strides=[1,1,1,1,1],padding='SAME')
		relu=tf.nn.relu(conv1 + bias1_2)
	pool1=tf.nn.max_pool3d(relu,ksize=[1,2,2,2,1],strides=[1,2,2,2,1],padding='SAME')

	with tf.variable_scope('conv2') as scope:
		weight2_1=variable_on_cpu('weight1',[3,3,3,64,128],np.sqrt(2./(3*3*3*64)))
		bias2_1=var=variable_on_cpu('bias1',[128],0)
		conv=tf.nn.conv3d(pool1,weight2_1,strides=[1,1,1,1,1],padding='SAME') + bias2_1
		weight2_2=variable_on_cpu('weight2',[3,3,3,128,128],np.sqrt(2./(3*3*3*128)))
		bias2_2=var=variable_on_cpu('bias2',[128],0)
		conv2=tf.nn.conv3d(conv,weight2_2,strides=[1,1,1,1,1],padding='SAME')
		relu=tf.nn.relu(conv2 + bias2_2)
	pool2=tf.nn.max_pool3d(relu,ksize=[1,2,2,2,1],strides=[1,2,2,2,1],padding='SAME')

	with tf.variable_scope('conv3') as scope:
		weight3_1=variable_on_cpu('weight1',[3,3,3,128,128],np.sqrt(2./(3*3*3*128)))
		bias3_1=var=variable_on_cpu('bias1',[128],0)
		conv=tf.nn.conv3d(pool2,weight3_1,strides=[1,1,1,1,1],padding='SAME') + bias3_1
		weight3_2=variable_on_cpu('weight2',[3,3,3,128,128],np.sqrt(2./(3*3*3*128)))
		bias3_2=var=variable_on_cpu('bias2',[128],0)
		conv=tf.nn.conv3d(conv,weight3_2,strides=[1,1,1,1,1],padding='SAME') + bias3_2
		weight3_3=variable_on_cpu('weight3',[3,3,3,128,128],np.sqrt(2./(3*3*3*128)))
		bias3_3=var=variable_on_cpu('bias3',[128],0)
		conv3=tf.nn.conv3d(conv,weight3_3,strides=[1,1,1,1,1],padding='SAME')
		relu=tf.nn.relu(conv3 + bias3_3)

	reshape_=tf.reshape(relu,[batchsize,-1])
	dim=reshape_.get_shape()[-1].value
	with tf.variable_scope('fc1') as scope:
		weight4=variable_on_cpu('weight',[dim,z_dim],np.sqrt(2./dim),trainable=True)
		bias4=variable_on_cpu('bias',[z_dim],0,trainable=True)
		z=tf.nn.relu(tf.matmul(reshape_,weight4) + bias4)
		#z=tf.nn.sigmoid(tf.matmul(reshape_,weight4) + bias4)

	return z
  

	
def decode(z,batchsize):
	
	with tf.variable_scope('fc2',reuse=tf.AUTO_REUSE) as scope:
		weight5=variable_on_cpu('weight',[z_dim,8*8*8*32],np.sqrt(2./z_dim),trainable=True)
		bias5=variable_on_cpu('bias',[8*8*8*32],0,trainable=True)
		h=tf.nn.relu(tf.matmul(z,weight5) + bias5)
		h=tf.reshape(h,[-1,8,8,8,32])
	with tf.variable_scope('deconv1',reuse=tf.AUTO_REUSE) as scope:
		weight6=variable_on_cpu('weight',[5,5,5,64,32],np.sqrt(2./(5*5*5*32)))
		bias6=variable_on_cpu('bias',[64],0)
		deconv=tf.nn.conv3d_transpose(h,weight6,[batchsize,16,16,16,64],[1,2,2,2,1],padding='SAME')
		deconv1=tf.nn.relu(deconv+bias6)
	with tf.variable_scope('deconv2',reuse=tf.AUTO_REUSE) as scope:
		weight7=variable_on_cpu('weight',[5,5,5,128,64],np.sqrt(2./(5*5*5*64)))
		bias7=variable_on_cpu('bias',[128],0)
		deconv=tf.nn.conv3d_transpose(deconv1,weight7,[batchsize,32,32,32,128],[1,2,2,2,1],padding='SAME')
		deconv2=tf.nn.relu(deconv+bias7) 
	with tf.variable_scope('conv4',reuse=tf.AUTO_REUSE) as scope:
		weight8=variable_on_cpu('weight',[3,3,3,128,1],np.sqrt(2./(3*3*3*128)))
		bias8=variable_on_cpu('bias',[1],0)
		conv=tf.nn.conv3d(deconv2,weight8,strides=[1,1,1,1,1],padding='SAME')
		logits=conv+bias8

	return logits

'''
def generate_session_decode(gpu_num):
	in_=[]
	out=[]
	for ii in rnage(gpu_num):
		with tf.device('/gpu:%d'%ii):
			train_in=tf.placeholder(shape=[BATCH_SIZE,200],dtype=tf.float32)
			train_logits =decode(train_in,BATCH_SIZE)
			train_out=tf.nn.sigmoid(train_logits)
			in_.append(train_in)
			out.append(train_out)
			tf.get_variable_scope().reuse_variables()
	saver=tf.train.Saver()
	sess=tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False))
	model_path=tf.train.latest_checkpoint(saved_model_path)
	saver.restore(sess,model_path)
	return sess,in_,out

def generate_session(gpu_num):
	in_=[]
	z=[]
	out=[]
	for ii in range(gpu_num):
		with tf.device('/gpu:%d'%ii):
			train_in=tf.placeholder(shape=[1,32,32,32,1],dtype=tf.float32)
			train_z=encode(train_in,1)
			train_logits =decode(train_z,1)
			train_out=tf.nn.sigmoid(train_logits)
			tf.get_variable_scope().reuse_variables()
			in_.append(train_in)
			z.append(train_z)
			out.append(train_out)

	saver=tf.train.Saver()
	sess=tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False))
	sess.run(tf.global_variables_initializer())
	model_path=tf.train.latest_checkpoint(saved_model_path)
	saver.restore(sess,model_path)
	return sess,in_,z,out
'''
def generate_session_decode(gpu_num):
	print 'BATCH_SIZE:::',BATCH_SIZE
	in_=[]
	out=[]
	for ii in range(gpu_num):
		with tf.device('/gpu:%d'%ii):
			train_in=tf.placeholder(shape=[BATCH_SIZE,z_dim],dtype=tf.float32)
			train_logits =decode(train_in,BATCH_SIZE)
			train_out=tf.nn.sigmoid(train_logits)
			in_.append(train_in)
			out.append(train_out)
			tf.get_variable_scope().reuse_variables()
	return in_,out

def generate_session(gpu_num):
	in_=[]
	z=[]
	out=[]
	for ii in range(gpu_num):
		with tf.device('/gpu:%d'%ii):
			train_in=tf.placeholder(shape=[1,32,32,32,1],dtype=tf.float32)
			train_z=encode(train_in,1)
			train_logits =decode(train_z,1)
			train_out=tf.nn.sigmoid(train_logits)
			tf.get_variable_scope().reuse_variables()
			in_.append(train_in)
			z.append(train_z)
			out.append(train_out)
	return in_,z,out













