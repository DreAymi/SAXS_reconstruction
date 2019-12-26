import numpy as np
import tensorflow as tf
import random
import os
import exceptions


BATCH_SIZE = 64
NUM_EPOCHS = 30
SEED=56297
z_dim=3000
z_dim2=200


cur_path=sys.path[0]
print 'cur_path',cur_path
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

def variable_on_cpu(name,shape,stddev):
	with tf.device('/cpu:0'):
		var=tf.get_variable(name,shape,initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
	return var


def encode(input):
	with tf.variable_scope('conv1') as scope:
		weight1=variable_on_cpu('weight1',[3,3,3,1,64],np.sqrt(2./(3*3*3)))
		bias1=var=variable_on_cpu('bias1',[64],0)
		conv=tf.nn.conv3d(input,weight1,strides=[1,1,1,1,1],padding='SAME') + bias1
		weight2=variable_on_cpu('weight2',[3,3,3,64,64],np.sqrt(2./(3*3*3*64)))
		bias2=var=variable_on_cpu('bias2',[64],0)
		conv1=tf.nn.conv3d(conv,weight2,strides=[1,1,1,1,1],padding='SAME')
		relu=tf.nn.relu(conv1 + bias2)
	pool1=tf.nn.max_pool3d(relu,ksize=[1,2,2,2,1],strides=[1,2,2,2,1],padding='SAME')

	with tf.variable_scope('conv2') as scope:
		weight1=variable_on_cpu('weight1',[3,3,3,64,128],np.sqrt(2./(3*3*3*64)))
		bias1=var=variable_on_cpu('bias1',[128],0)
		conv=tf.nn.conv3d(pool1,weight1,strides=[1,1,1,1,1],padding='SAME') + bias1
		weight2=variable_on_cpu('weight2',[3,3,3,128,128],np.sqrt(2./(3*3*3*128)))
		bias2=var=variable_on_cpu('bias2',[128],0)
		conv2=tf.nn.conv3d(conv,weight2,strides=[1,1,1,1,1],padding='SAME')
		relu=tf.nn.relu(conv2 + bias2)
	pool2=tf.nn.max_pool3d(relu,ksize=[1,2,2,2,1],strides=[1,2,2,2,1],padding='SAME')

	with tf.variable_scope('conv3') as scope:
		weight1=variable_on_cpu('weight1',[3,3,3,128,128],np.sqrt(2./(3*3*3*128)))
		bias1=var=variable_on_cpu('bias1',[128],0)
		conv=tf.nn.conv3d(pool2,weight1,strides=[1,1,1,1,1],padding='SAME') + bias1
		weight2=variable_on_cpu('weight2',[3,3,3,128,128],np.sqrt(2./(3*3*3*128)))
		bias2=var=variable_on_cpu('bias2',[128],0)
		conv=tf.nn.conv3d(conv,weight2,strides=[1,1,1,1,1],padding='SAME') + bias2
		weight3=variable_on_cpu('weight3',[3,3,3,128,128],np.sqrt(2./(3*3*3*128)))
		bias3=var=variable_on_cpu('bias3',[128],0)
		conv3=tf.nn.conv3d(conv,weight3,strides=[1,1,1,1,1],padding='SAME')
		relu=tf.nn.relu(conv3 + bias3)

	reshape_=tf.reshape(relu,[BATCH_SIZE,-1])
	dim=reshape_.get_shape()[-1].value
	with tf.variable_scope('fc1') as scope:
		weight=variable_on_cpu('weight',[dim,z_dim],np.sqrt(2./dim))
		bias=variable_on_cpu('bias',[z_dim],0)
		z1=tf.nn.relu(tf.matmul(reshape_,weight) + bias)
	with tf.variable_scope('fc1_2') as scope:
		weight=variable_on_cpu('weight',[z_dim,z_dim2],np.sqrt(2./z_dim))
		bias=variable_on_cpu('bias',[z_dim2],0)
		z2=tf.nn.relu(tf.matmul(z1,weight) + bias)

	return z2
	
def decode(z):
	with tf.variable_scope('fc2') as scope:
		weight=variable_on_cpu('weight',[z_dim2,8*8*8*32],np.sqrt(2./z_dim))
		bias=variable_on_cpu('bias',[8*8*8*32],0)
		h=tf.nn.relu(tf.matmul(z,weight) + bias)
	h=tf.reshape(h,[-1,8,8,8,32])
	with tf.variable_scope('deconv1') as scope:
		weight=variable_on_cpu('weight',[5,5,5,64,32],np.sqrt(2./(5*5*5*32)))
		bias=variable_on_cpu('bias',[64],0)
		deconv=tf.nn.conv3d_transpose(h,weight,[BATCH_SIZE,16,16,16,64],[1,2,2,2,1],padding='SAME')
		deconv1=tf.nn.relu(deconv+bias)
	with tf.variable_scope('deconv2') as scope:
		weight=variable_on_cpu('weight',[5,5,5,128,64],np.sqrt(2./(5*5*5*64)))
		bias=variable_on_cpu('bias',[128],0)
		deconv=tf.nn.conv3d_transpose(deconv1,weight,[BATCH_SIZE,32,32,32,128],[1,2,2,2,1],padding='SAME')
		deconv2=tf.nn.relu(deconv+bias) 
	with tf.variable_scope('conv4') as scope:
		weight=variable_on_cpu('weight',[3,3,3,128,1],np.sqrt(2./(3*3*3*128)))
		bias=variable_on_cpu('bias',[1],0)
		conv=tf.nn.conv3d(deconv2,weight,strides=[1,1,1,1,1],padding='SAME')
		logits=conv+bias
	return logits

with tf.device('/cpu:0'):
	train_data,train_label=read_tfrecords('../database/train.tfrecords')
	train_data,train_label=tf.train.shuffle_batch([train_data,train_label],batch_size=BATCH_SIZE,capacity=6400,min_after_dequeue=3200)
	test_data,test_label=read_tfrecords('../database/test.tfrecords')
	test_data,test_label=tf.train.shuffle_batch([test_data,test_label],batch_size=BATCH_SIZE,capacity=3000,min_after_dequeue=1500)

train_z=encode(train_data)
train_logits=decode(train_z)
train_out=tf.nn.sigmoid(train_logits)
train_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=train_label,logits=train_logits))
optimizer=tf.train.AdamOptimizer(0.0001).minimize(train_loss)

tf.get_variable_scope().reuse_variables()
test_z=encode(test_data)
test_logits=decode(test_z)
test_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=test_label,logits=test_logits))


saver=tf.train.Saver(max_to_keep=2)
sess=tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

threads=tf.train.start_queue_runners(sess=sess)

sess.run(tf.global_variables_initializer())

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
	saver_path=saver.save(sess,model_path,global_step=step)

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









