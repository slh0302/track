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

    with tf.Session() as sess:
        tmp = tf.constant([[[2,8],[1,3]],[[1,4],[1,5]], [[2,8],[1,3]]], dtype=tf.float32)
        tmp1 = tf.constant([[[1, 4],[2, 8]],[[2, 5],[3, 9]], [[2,8],[1,3]]], dtype=tf.float32)
        tm = tf.constant([1,2,3,1,1])
        yu = [tmp, tmp1]
        # x0, y0, x1, y1 = tf.unstack(tmp1, axis=2)
        # # print(tmp.get_shape().as_list())
        # # boxMatches = tf.cast(tmp > 2, tf.float32)
        # # value = tf.minimum(boxMatches, 2.0)
        # # ma = tf.reduce_max(tmp, axis=1)
        # # to = tf.one_hot(tf.cast(tmp > 2, tf.uint8), 2, on_value=0.999, off_value=0.001)
        # # value = tf.reduce_mean(tmp, axis=[0,1])
        # res = tf.logical_and((x1 - x0) >= (5 - 1), (y1 - y0) >= (5 - 1))
        test = tf.constant([
            [2,3,4],
            [14,7,9],
            [1,1,12]
        ])
        a,b = test111()
        c = a[1]
        total = []
        # for i,j  in zip(tmp, tmp1):
        #     total.append(i+j)
        tmp3 = tf.gather(tmp, tf.range(tm[1]))
        print(sess.run(tmp3) )

