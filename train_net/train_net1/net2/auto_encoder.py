import numpy as np
import tensorflow as tf
import random
import os
import sys

BATCH_SIZE = 64
NUM_EPOCHS = 15
SEED=56297
z_dim=200

cur_path=sys.path[0]
folder_of_well_trained_model='%s/../net1/model'%cur_path
folder_to_save_model='%s/model'%cur_path
folder_to_save_log='%s/log'%cur_path

def read_tfrecords(filename):
	filename_quene=tf.train.string_input_producer([filename])
	reader=tf.TFRecordReader()
	_,serialized_example=reader.read(filename_quene)
	features=tf.parse_single_example(serialized_example,features={
								      'data' : tf.FixedLenFeature([32768], tf.int64)
								      })
	data=tf.reshape(features['data'],(32,32,32,1))
	train_data=tf.cast(data,tf.float32)
	label_data=tf.cast(data,tf.float32)
	return train_data , label_data

def variable_on_cpu(name,shape,stddev,trainable=False):
	with tf.device('/cpu:0'):
		var=tf.get_variable(name,shape,initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32),trainable=trainable)
	return var


def encode(input):
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

	reshape_=tf.reshape(relu,[BATCH_SIZE,-1])
	dim=reshape_.get_shape()[-1].value
	with tf.variable_scope('fc1') as scope:
		weight4=variable_on_cpu('weight',[dim,z_dim],np.sqrt(2./dim),trainable=True)
		bias4=variable_on_cpu('bias',[z_dim],0,trainable=True)
		z=tf.nn.relu(tf.matmul(reshape_,weight4) + bias4)

	var_dict1 = {'conv1/weight1':weight1_1,'conv1/bias1':bias1_1,
			'conv1/weight2':weight1_2,'conv1/bias2':bias1_2,
			'conv2/weight1':weight2_1,'conv2/bias1':bias2_1,
			'conv2/weight2':weight2_2,'conv2/bias2':bias2_2,
			'conv3/weight1':weight3_1,'conv3/bias1':bias3_1,
			'conv3/weight2':weight3_2,'conv3/bias2':bias3_2,
			'conv3/weight3':weight3_3,'conv3/bias3':bias3_3,
			}
	var_list1=[weight4,bias4]
	return z , var_dict1 
	
def decode(z):
	
	with tf.variable_scope('fc2') as scope:
		weight5=variable_on_cpu('weight',[z_dim,8*8*8*32],np.sqrt(2./z_dim),trainable=True)
		bias5=variable_on_cpu('bias',[8*8*8*32],0,trainable=True)
		h=tf.nn.relu(tf.matmul(z,weight5) + bias5)
	h=tf.reshape(h,[-1,8,8,8,32])
	with tf.variable_scope('deconv1') as scope:
		weight6=variable_on_cpu('weight',[5,5,5,64,32],np.sqrt(2./(5*5*5*32)))
		bias6=variable_on_cpu('bias',[64],0)
		deconv=tf.nn.conv3d_transpose(h,weight6,[BATCH_SIZE,16,16,16,64],[1,2,2,2,1],padding='SAME')
		deconv1=tf.nn.relu(deconv+bias6)
	with tf.variable_scope('deconv2') as scope:
		weight7=variable_on_cpu('weight',[5,5,5,128,64],np.sqrt(2./(5*5*5*64)))
		bias7=variable_on_cpu('bias',[128],0)
		deconv=tf.nn.conv3d_transpose(deconv1,weight7,[BATCH_SIZE,32,32,32,128],[1,2,2,2,1],padding='SAME')
		deconv2=tf.nn.relu(deconv+bias7) 
	with tf.variable_scope('conv4') as scope:
		weight8=variable_on_cpu('weight',[3,3,3,128,1],np.sqrt(2./(3*3*3*128)))
		bias8=variable_on_cpu('bias',[1],0)
		conv=tf.nn.conv3d(deconv2,weight8,strides=[1,1,1,1,1],padding='SAME')
		logits=conv+bias8

	var_dict2 = {
			'deconv1/weight':weight6,'deconv1/bias':bias6,
			'deconv2/weight':weight7,'deconv2/bias':bias7,
			'conv4/weight':weight8,'conv4/bias':bias8}
	var_list2=[weight5,bias5]

	return logits , var_dict2 

with tf.device('/cpu:0'):
	train_set='../../database/train.tfrecords'
	test_set='../../database/test.tfrecords'
	train_data,train_label=read_tfrecords(train_set)
	train_data,train_label=tf.train.shuffle_batch([train_data,train_label],batch_size=BATCH_SIZE,capacity=6400,min_after_dequeue=3200)
	test_data,test_label=read_tfrecords(test_set)
	test_data,test_label=tf.train.shuffle_batch([test_data,test_label],batch_size=BATCH_SIZE,capacity=3000,min_after_dequeue=1500)

train_z , var_dict1=encode(train_data)
train_logits , var_dict2=decode(train_z)
train_out=tf.nn.sigmoid(train_logits)
train_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=train_label,logits=train_logits))
optimizer=tf.train.AdamOptimizer(0.001).minimize(train_loss)

tf.get_variable_scope().reuse_variables()
test_z,_=encode(test_data)
test_logits,_=decode(test_z)
test_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=test_label,logits=test_logits))
	
var_dict=dict(var_dict1.items() + var_dict2.items())

saver1=tf.train.Saver(var_list=var_dict)
saver2=tf.train.Saver(max_to_keep=2)
sess=tf.Session(config=tf.ConfigProto(log_device_placement=True))

sess.run(tf.global_variables_initializer())

model_path=tf.train.latest_checkpoint(folder_of_well_trained_model)
saver1.restore(sess,model_path)
threads=tf.train.start_queue_runners(sess=sess)

log_path=os.path.join(folder_to_save_log,'log.txt')
step=0
for e in range(NUM_EPOCHS):
	for ii in range(50000//BATCH_SIZE):
		step=step+1
		loss_,_=sess.run([train_loss,optimizer])
		print 'iteration:{}/{},{}batchs,Training loss: {:.4f}'.format(e+1,NUM_EPOCHS,step,loss_)
		if step%10==0:
			logfile=open(log_path,'a')
			logfile.write('iteration:{}/{},{}batchs,Training loss: {:.4f}\n'.format(e+1,NUM_EPOCHS,step,loss_))
			logfile.close()
	
	model_path=os.path.join(folder_to_save_model,'model.ckpt')	
	saver_path=saver2.save(sess,model_path,global_step=step)

	test_loss_list=[]
	for jj in range(10000//BATCH_SIZE):
		test_loss_=sess.run(test_loss)
		test_loss_list.append(test_loss_)
		print 'epoch {},{}batches,test loss :{:.4f}'.format(e,jj,test_loss_)
	print 'epoch {}, average value for testing loss is {:.4f}'.format(e,np.mean(test_loss_list))
	logfile=open(log_path,'a')
	logfile.write('epoch {}, average value for testing loss is {:.4f}'.format(e,np.mean(test_loss_list)))
	logfile.close()
sess.close()









