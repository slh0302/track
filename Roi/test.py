import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "11"
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
        tmp1 = tf.constant([[[1, 4],[2, 8],[1,7]],[[2, 5],[3, 9],[3,4]], [[2,8],[1,3],[2,6]]], dtype=tf.float32)
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
        print(sess.run([a,b,c]) )

        img = np.zeros((1, 8, 8, 9), np.float32)
        boxes = tf.constant([[0, 0, 2 * 16, 5 * 16]], dtype=tf.float32)
        print(boxes.get_shape().as_list())

        yOffset = 0
        xOffset = 0
        chOffset = 0
        img[0, yOffset + 0:yOffset + 1, xOffset + 0:xOffset + 1, chOffset + 0:chOffset + 1] = 1
        # img[:,:,:,:]=1
        p = tf.placeholder(tf.float32, shape=img.shape)

        np.set_printoptions(threshold=5000, linewidth=150)

        pooled = positionSensitiveRoiPooling(p, boxes)
        pooled = tf.Print(pooled, [tf.shape(pooled)], "pooled shape", summarize=100)
        print(sess.run(pooled, feed_dict={p: img}))

        loss = tf.reduce_sum(pooled)

        g = tf.gradients(loss, p)

        print(img)
        print(sess.run(g, feed_dict={p: img})[0])
        print(sess.run(g, feed_dict={p: img})[0][:, :, :, 1])
