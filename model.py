import tensorflow as tf
from utils import tf_resize_image

def conv2d(x, num_output, kernel_size=3, stride=1, act=tf.nn.relu, name=None):
    return tf.layers.conv2d(x, num_output, kernel_size, stride, 'same',
                            activation=act, name=name,
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1e-5))


def deconv2d(x, num_output, kernel_size, stride, act=None, name=None):
    return tf.layers.conv2d_transpose(x, num_output, kernel_size, stride, 'same',
                                      activation=act, name=name,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1e-5))


class VDSR():
    def __init__(self, scale=4):
        self.scale = scale

    def __call__(self, lr, bic=None):
        with tf.variable_scope('VDSR', reuse=tf.AUTO_REUSE):
            b, h, w, c = tf.unstack(tf.shape(lr))
            scale = self.scale
            layer_num = 20
            if bic == None:
                bic = tf_resize_image(lr, scale)
                bic = tf.reshape(bic, [b, h * scale, w * scale, 1])
            self.bic = bic
            x = bic
            for i in range(layer_num - 1):
                x = conv2d(x, 64)
            x = conv2d(x, 1, act=tf.identity) + bic
            return x


class EDSR():
    def __init__(self, scale=4):
        self.scale = scale

    def res2d(self, x, num_output, kernel_size, name, scale=0.1):
        with tf.variable_scope(name):
            x0 = x
            x = conv2d(x, num_output, kernel_size)
            x = conv2d(x, num_output, kernel_size, act=tf.identity)
            x = x0 + x * scale
            return x

    def __call__(self, lr, bic=None):
        with tf.variable_scope('EDSR', reuse=tf.AUTO_REUSE):
            b, h, w, c = tf.unstack(tf.shape(lr))
            scale = self.scale
            if bic == None:
                bic = tf_resize_image(lr, scale)
                bic = tf.reshape(bic, [b, h * scale, w * scale, 1])
            self.bic = bic
            x = conv2d(lr, 256)
            x0 = x
            for i in range(16):
                x = self.res2d(x, 256, 3, 'res2d_%d' % i)
            x = conv2d(x, 256)
            x = x + x0
            x = deconv2d(x, 1, kernel_size=2 * scale, stride=scale)
            return x + bic

class BICUBIC():
    def __init__(self, scale=4):
        self.scale = scale

    def __call__(self, lr):
        with tf.variable_scope('BICUBIC'):
            b, h, w, c = tf.unstack(tf.shape(lr))
            scale = self.scale
            bic = tf_resize_image(lr, scale)
            bic = tf.reshape(bic, [b, h * scale, w * scale, c])
            return bic


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    lr = tf.zeros([1, 24, 24, 1])
    # model = VDSR()
    model = EDSR()
    res = model(lr)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    out = sess.run(res)
    print(out.shape)
