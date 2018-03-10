from utils.Resnetv2 import *
from model.RPN import RPN
import tensorflow.contrib.slim as slim
import tensorflow as tf


class RFCN:
    def __init__(self, inputs, nCategories, resnet_size, name="R_FCN", weightDecay=0.00004,
                 reuse=False, isTraining=True, trainFrom=None, hardMining=True):
        self.boxThreshold = 0.5
        # print("Training network from " + (trainFrom if trainFrom is not None else "end"))
        with tf.variable_scope(name, reuse=reuse) as scope:
            model = self.rfcn(resnet_size, nCategories)
            self.net, self.frames = model(inputs, isTraining)
            self.scope = scope

            with tf.variable_scope("RPN_PART"):
                # Pepeat_1 - last 1/16 layer, Mixed_6a - first 1/16 layer
                scale_16 = self.frames['block_layer3'][-1]
                scale_32 = self.frames['block_layer4']
                # bn + relu
                print(scale_16.shape, scale_32.shape)
                with slim.arg_scope([slim.conv2d],
                                    weights_regularizer=slim.l2_regularizer(weightDecay),
                                    biases_regularizer=slim.l2_regularizer(weightDecay),
                                    padding='SAME',
                                    activation_fn=tf.nn.relu):
                    net = tf.concat([tf.image.resize_bilinear(scale_32, tf.shape(scale_16)[1:3]), scale_16], 3)
                    rpnInput = slim.conv2d(net, 1024, 1)
                    featureInput = slim.conv2d(net, 1536, 1)
                    self.Rpn = RPN(nCategories, rpnInput, 16, [32, 32], featureInput, 16, [32, 32],
                                        weightDecay=weightDecay, hardMining=hardMining)


    def getLoss(self,refBoxes, refClasses):
        return self.Rpn.getLoss(refBoxes, refClasses)

    # def getVariables(self, includeFeatures=False):
    #     if includeFeatures:
    #         return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope.name)
    #     else:
    #         vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope.name + "/Box/")
    #         vars += self.googleNet.getTrainableVars()
    #
    #         print("Training variables: ", [v.op.name for v in vars])
    #         return vars
    #
    # def importWeights(self, sess, filename):
    #     self.googleNet.importWeights(sess, filename, includeTraining=True)

    def rfcn(self, resnet_size, num_classes, data_format=None):
        """Returns the rfcn model for a given size and number of output classes."""
        model_params = {
            18: {'block': building_block, 'layers': [2, 2, 2, 2]},
            34: {'block': building_block, 'layers': [3, 4, 6, 3]},
            50: {'block': bottleneck_block, 'layers': [3, 4, 6, 3]},
            101: {'block': bottleneck_block, 'layers': [3, 4, 23, 3]},
            152: {'block': bottleneck_block, 'layers': [3, 8, 36, 3]},
            200: {'block': bottleneck_block, 'layers': [3, 24, 36, 3]}
        }

        if resnet_size not in model_params:
            raise ValueError('Not a valid resnet_size:', resnet_size)

        params = model_params[resnet_size]
        return resnet_v2_RFCN_generator(
            params['block'], params['layers'], num_classes, data_format)

    def importWeights(self, sess, filename):

        ignores = []
        # if includeTraining or (self.trainFrom is None) else self.getScopes(fromLayer=self.trainFrom,
        #                                                                                inclusive=True)
        
        print("Ignoring blocks:")
        print(ignores)
        # CheckpointLoader.importIntoScope(sess, filename, fromScope="InceptionResnetV2", toScope=self.scope.name,
        #                                      ignore=ignores)
        self.googleNet.importWeights(sess, filename, includeTraining=True)

# Example resnetV2
def imagenet_resnet_v2(resnet_size, num_classes, data_format=None):
  """Returns the ResNet model for a given size and number of output classes."""
  model_params = {
      18: {'block': building_block, 'layers': [2, 2, 2, 2]},
      34: {'block': building_block, 'layers': [3, 4, 6, 3]},
      50: {'block': bottleneck_block, 'layers': [3, 4, 6, 3]},
      101: {'block': bottleneck_block, 'layers': [3, 4, 23, 3]},
      152: {'block': bottleneck_block, 'layers': [3, 8, 36, 3]},
      200: {'block': bottleneck_block, 'layers': [3, 24, 36, 3]}
  }

  if resnet_size not in model_params:
    raise ValueError('Not a valid resnet_size:', resnet_size)

  params = model_params[resnet_size]
  return imagenet_resnet_v2_generator(
      params['block'], params['layers'], num_classes, data_format)


def block_layer_with_framework(inputs, filters, block_fn, blocks, strides, is_training, name,
                data_format, net_framework=None):
    """Creates one layer of blocks for the ResNet model.

    Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    is_training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').

    Returns:
    The output tensor of the block layer.
    """
    # Bottleneck blocks end with 4x the number of filters as they start with
    filters_out = 4 * filters if block_fn is bottleneck_block else filters

    def projection_shortcut(inputs):
        return conv2d_fixed_padding(
            inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
            data_format=data_format)

    # Only the first block per block_layer uses projection_shortcut and strides
    inputs = block_fn(inputs, filters, is_training, projection_shortcut, strides,
                    data_format)
    net_frame = []
    net_frame.append(inputs)
    for i in range(1, blocks):
        inputs = block_fn(inputs, filters, is_training, None, 1, data_format)
        net_frame.append(inputs)

    if net_framework:
        return tf.identity(inputs, name), net_frame
    else:
        return tf.identity(inputs, name)


def resnet_v2_RFCN_generator(block_fn, layers, num_classes,
                                 data_format=None):
    """Generator for ImageNet ResNet v2 models.

      Args:
        block_fn: The block to use within the model, either `building_block` or
          `bottleneck_block`.
        layers: A length-4 array denoting the number of blocks to include in each
          layer. Each layer consists of blocks that take inputs of the same size.
        num_classes: The number of possible classes for image classification.
        data_format: The input format ('channels_last', 'channels_first', or None).
          If set to None, the format is dependent on whether a GPU is available.

      Returns:
        The model function that takes in `inputs` and `is_training` and
        returns the output tensor of the ResNet model.

      """
    if data_format is None:
        data_format = (
            'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    def model(inputs, is_training):
        """Constructs the ResNet model given the inputs."""
        # ALL layers added in framework
        #   has not been passed through relu function
        net_frame = {}

        def AddFrameWork(net, name, form=False):
            net_frame[name] = net

        if data_format == 'channels_first':
            # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
            # This provides a large performance boost on GPU. See
            # https://www.tensorflow.org/performance/performance_guide#data_formats
            inputs = tf.transpose(inputs, [0, 3, 1, 2])

        # First
        inputs = conv2d_fixed_padding(
            inputs=inputs, filters=64, kernel_size=7, strides=2,
            data_format=data_format)
        # First with batch norm
        # inputs = batch_norm_relu(inputs, is_training, data_format)
        inputs = tf.identity(inputs, 'initial_conv')
        AddFrameWork(inputs, 'initial_conv')

        # max-Pool
        inputs = tf.layers.max_pooling2d(
            inputs=inputs, pool_size=3, strides=2, padding='SAME',
            data_format=data_format)
        inputs = tf.identity(inputs, 'initial_max_pool')
        AddFrameWork(inputs, 'initial_max_pool')

        # First Block
        inputs = block_layer(
            inputs=inputs, filters=64, block_fn=block_fn, blocks=layers[0],
            strides=1, is_training=is_training, name='block_layer1',
            data_format=data_format)
        AddFrameWork(inputs, 'block_layer1')

        # 2nd Block
        inputs = block_layer(
            inputs=inputs, filters=128, block_fn=block_fn, blocks=layers[1],
            strides=2, is_training=is_training, name='block_layer2',
            data_format=data_format)
        AddFrameWork(inputs, 'block_layer2')

        # 3rd Block
        inputs, frames = block_layer_with_framework(
            inputs=inputs, filters=256, block_fn=block_fn, blocks=layers[2],
            strides=2, is_training=is_training, name='block_layer3',
            data_format=data_format, net_framework=True)
        AddFrameWork(frames, 'block_layer3')

        # 4th Block
        inputs = block_layer(
            inputs=inputs, filters=512, block_fn=block_fn, blocks=layers[3],
            strides=2, is_training=is_training, name='block_layer4',
            data_format=data_format)
        AddFrameWork(inputs, 'block_layer4')

        inputs = batch_norm_relu(inputs, is_training, data_format)

        # Original Pooling
        inputs = tf.layers.average_pooling2d(
            inputs=inputs, pool_size=7, strides=1, padding='VALID',
            data_format=data_format)
        inputs = tf.identity(inputs, 'final_avg_pool')
        inputs = tf.reshape(inputs,
                            [-1, 512 if block_fn is building_block else 2048])
        inputs = tf.layers.dense(inputs=inputs, units=num_classes)
        inputs = tf.identity(inputs, 'final_dense')
        return inputs, net_frame

    return model
