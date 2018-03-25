import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "12"
def test111():
    test = tf.constant([
        [2, 3, 4],
        [14, 7, 9],
        [1, 1, 12]
    ])
    box = tf.constant([1,1,2])
    res = tf.tuple([test,
                    tf.shape(test)]),box
    return res



if __name__ == "__main__":
    import tensorflow as tf
    from Roi.ROIPoolingWrapper import *

    config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=8)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        tmp = tf.constant([[[2,8],[1,3]],[[1,4],[1,5]], [[2,8],[1,3]],[[2,8],[1,3]],[[2,8],[1,3]]], dtype=tf.float32)
        tmp3 = tf.gather(tmp, tf.range(5,delta=2))
        refOnehot = tf.one_hot([1,0,1,0,1,1], 2, on_value=1.0 - 1 * 0.0002,
                               off_value=0.0002)

        print(sess.run(refOnehot) )
        print("1231")
