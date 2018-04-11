import os
import numpy as np
np.random.seed(1234)

# Tensorflow set up
os.environ['KERAS_BACKEND']='tensorflow'
import keras.backend as K
from keras.backend.tensorflow_backend import set_session
import tensorflow

from tensorflow.python.client import device_lib

local_device_protos = device_lib.list_local_devices()

if os.name == 'nt' and len(local_device_protos) > 2:
    config = tensorflow.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    config.gpu_options.visible_device_list = '1'
    K.tensorflow_backend.set_session(tensorflow.Session(config=config))

K.set_image_dim_ordering('tf')
K.set_image_data_format('channels_last')

# Directory set up
base_dir = 'D:/Kaggle/Data_Science_Bowl_2018' if os.name == 'nt' else os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')

data_dir = os.path.join(base_dir, 'data')
submissions_dir = os.path.join(base_dir, 'submissions')
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
train_mosaics_dir = os.path.join(base_dir, "train_mosaics")
test_mosaics_dir  = os.path.join(base_dir, "test_mosaics")

supplementary_dir = [os.path.join(base_dir, 'train_external', 'ISBI'),
                     os.path.join(base_dir, 'train_external', 'nsb')]

gan_dir = os.path.join(base_dir, 'train_external', 'GAN')

# Run group_id_files if they don't exist

train_group_id_file = os.path.join(data_dir, 'train_data_ids.csv')
if not os.path.exists(train_group_id_file):
    from clustering_functions import run
    print("Generating {}".format(train_group_id_file))
    run(train_group_id_file, [train_dir], supplementary_dir + [gan_dir])

test_group_id_file = os.path.join(data_dir, 'test_data_ids.csv')
if not os.path.exists(test_group_id_file):
    from clustering_functions import run
    print("Generating {}".format(test_group_id_file))
    run(test_group_id_file, [test_dir])

supplementary_group_id_file = os.path.join(data_dir, 'supplementary_data_ids.csv')
if not os.path.exists(supplementary_group_id_file):
    from clustering_functions import run
    print("Generating {}".format(supplementary_group_id_file))
    run(supplementary_group_id_file, supplementary_dir, [train_dir]+ [gan_dir])


gan_group_id_file = os.path.join(data_dir, 'gan_data_ids.csv')
if False and not os.path.exists(gan_group_id_file):
    from clustering_functions import run
    print("Generating {}".format(gan_group_id_file))
    run(gan_group_id_file, gan_dir, [train_dir] + supplementary_dir)