import os
import numpy as np
np.random.seed(1234)

# Tensorflow set up
os.environ['KERAS_BACKEND']='tensorflow'
import keras.backend as K
from keras.backend.tensorflow_backend import set_session
import tensorflow

# CONFIG: Set to True for fergusoci
if False:
	config = tensorflow.ConfigProto(allow_soft_placement=True, log_device_placement=True)
	config.gpu_options.allow_growth = True
	config.gpu_options.per_process_gpu_memory_fraction = 0.7
	config.gpu_options.visible_device_list = '1'
	K.tensorflow_backend.set_session(tensorflow.Session(config=config))

K.set_image_dim_ordering('tf')
K.set_image_data_format('channels_last')

# Directory set up
base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')

data_dir = os.path.join(base_dir, 'data')
submissions_dir = os.path.join(base_dir, 'submissions')
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
supplementary_dir = [os.path.join(base_dir, 'train_external', 'ISBI'),
                     os.path.join(base_dir, 'train_external', 'nsb'),
                     os.path.join(base_dir, 'train_external', 'nsb_crop')]


group_id_file = os.path.join(data_dir, 'data_ids.csv')
if not os.path.exists(group_id_file):
    from clustering_functions import run
    run(group_id_file)