import tensorflow as tf

def gather(tensors, indices):
    with tf.name_scope("multiGather"):
        res = []
        for a in tensors:
            res.append(tf.gather_nd(a, indices))
        return res

def gatherList(tensors, indices):
    with tf.name_scope("multiGatherList"):
        res = []
        for a in tensors:
            tmp = tf.unstack(a, axis=0)
            tmpTensor = []
            for item in tmp :
                tmpTensor.append(tf.gather_nd(item, indices))
            tmpTensor = tf.stack(tmpTensor, axis=0)
            res.append(tmpTensor)
        return res

def gatherTopK(t, k, others=[], sorted=False):
    res=[]
    with tf.name_scope("gather_top_k"):
        isMoreThanK = tf.shape(t)[-1]>k
        values, indices = tf.cond(isMoreThanK, lambda: tuple(tf.nn.top_k(t, k=k, sorted=sorted)), lambda: (t, tf.zeros((0,1), tf.int32)))
        indices = tf.reshape(indices, [-1,1])
        res.append(values)

        for o in others:
            res.append(tf.cond(isMoreThanK, lambda: tf.gather_nd(o, indices), lambda: o))

    return res