import tensorflow as tf 
import numpy as np
import pdb2voxel

def create_tfrecords():
	file=open('pdbname.txt')
	errorlog=open('wrong_pdb.txt','a')
	writer=tf.python_io.TFRecordWriter('train.tfrecords')
	writer2=tf.python_io.TFRecordWriter('test.tfrecords')
	lines=file.readlines()
	count=0
	for line in lines:
		count+=1
		print count,line.split('\n')[0]
		arg=['pdbfile='+line.split('\n')[0]]
		try:
			cube=pdb2voxel.run(arg)
			cube=cube.astype(int)
			cube_list=cube.flatten()
						
			example = tf.train.Example(features=tf.train.Features(feature={
		                "data": tf.train.Feature(int64_list=tf.train.Int64List(value=cube_list)),       #require a list of int 
		            }))
			if count%5==0:
				writer2.write(example.SerializeToString())
			else:
				writer1.write(example.SerializeToString())
		except:
			errorlog.write(line)
	writer1.close()
	writer2.close()
	errorlog.close()
	file.close()

if __name__=='__main__':
	create_tfrecords()
