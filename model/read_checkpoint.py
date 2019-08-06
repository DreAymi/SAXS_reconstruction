import os
import numpy as np
from tensorflow.python import pywrap_tensorflow
import tensorflow as tf  
file=open('tensorfile.txt','a')
checkpoint_path = tf.train.latest_checkpoint('./') #os.path.join('/mnt/data2/liucan/auto_encoder/model/3D', "model.ckpt")  
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)  
var_to_shape_map = reader.get_variable_to_shape_map()  
for key in var_to_shape_map:  
	print >> file,"tensor_name: %s"%key
	print>>file,np.shape(reader.get_tensor(key))  
	#print>>file,reader.get_tensor(key) # Remove this is you want to print only variable names  
