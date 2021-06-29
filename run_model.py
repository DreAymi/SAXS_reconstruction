import os
import sys
import numpy as np
import tensorflow as tf
import exceptions
import auto_encoder_t
from sastbx.zernike_model import pdb2zernike
from scitbx.array_family import flex


def get_encode(voxel_value):
    out = sess.run(z_out[0], feed_dict={in_tensor[0]: voxel_value.reshape(1,32,32,32,1)})
    return out

def get_decode(interpolation_value):
    out = sess.run(out_tensor[0], feed_dict={z_in[0]: interpolation_value.reshape(1, 200)})
    return out



if __name__ == '__main__':
    saved_model_path = "run_process/model"
    output_folder = "results"
    in_tensor, z_out = auto_encoder_t.generate_encode_session(1)
    z_in, out_tensor = auto_encoder_t.generate_decode_session(1)
    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        model_path = tf.train.latest_checkpoint(saved_model_path)
        saver.restore(sess, model_path)

        #input voxel -> z
        voxel_value = np.ones(shape=(1,32,32,32,1))
        z_vector = get_encode(voxel_value)
        #input z -> voxel
        voxel_out = get_decode(z_vector)
        ccp4data_data=flex.double(np.greater(voxel_out[0,:31,:31,:31,0].reshape((31,31,31)),0.1).astype(float))
        #save ccp4
        pdb2zernike.ccp4_map_type(ccp4data_data, 15, rmax/0.9,file_name='%s/voxel_out.ccp4'%(output_folder))
