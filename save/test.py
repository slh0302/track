import tensorflow as tf
import os
model_dir = "/home/slh/tf-project/track/"
# reader = tf.train.NewCheckpointReader(file_name)
# print(reader.debug_string().decode("utf-8"))
from tensorflow.python import pywrap_tensorflow
# save/model_1/inception_resnet_v2_2016_08_30.ckpt  save/model_2/model.ckpt save/model_3/model
checkpoint_path = os.path.join(model_dir, "save/model_1/inception_resnet_v2_2016_08_30.ckpt")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
    if key == 'boxnet/Box/classMaps/biases':
        print(var_to_shape_map[key])
    # print(reader.get_tensor(key)) # Remove this is you want to print only variable names