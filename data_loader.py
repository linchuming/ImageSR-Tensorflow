import tensorflow as tf
import os
import glob
from scipy import misc
import numpy as np
import utils


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class DataLoader():
    def __init__(self, data_dir='DIV2K/DIV2K_train_HR',
                 patch_size=96, scale=4, batch_size=64,
                 shuffle_num=2000, prefetch_num=1000,
                 map_parallel_num=8, one_img_patch_num=128):
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.scale = scale
        self.batch_size = batch_size
        self.shuffle_num = shuffle_num
        self.prefetch_num = prefetch_num
        self.map_parallel_num = map_parallel_num
        self.one_img_patch_num = one_img_patch_num

    def gen_tfrecords(self, save_dir='DIV2K/tfrecords', tfrecord_num=10):
        file_num = tfrecord_num
        sample_num = 0
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fs = []
        for i in range(file_num):
            fs.append(tf.python_io.TFRecordWriter(os.path.join(save_dir, 'data%d.tfrecords' % i)))

        img_paths = sorted(glob.glob(os.path.join(self.data_dir, '*.png')))
        for img_path in img_paths:
            print('processing %s' % img_path)
            img = misc.imread(img_path)
            y = utils.img_to_uint8(utils.rgb2ycbcr(img)[:, :, 0])
            height, width = y.shape
            p = self.patch_size
            step = p // 3 * 2
            for h in range(0, height - p + 1, step):
                for w in range(0, width - p + 1, step):
                    gt = y[h: h + p, w: w + p]
                    assert gt.shape[:] == (p, p)
                    assert gt.dtype == np.uint8
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'gt': _bytes_feature(gt.tostring())
                    }))
                    fs[sample_num % file_num].write(example.SerializeToString())
                    sample_num += 1
        print('example number: %d' % sample_num)
        np.savetxt(os.path.join(save_dir, 'sample_num.txt'), np.asarray([sample_num]), '%d')
        for f in fs:
            f.close()

    def _parse_one_example(self, example):
        features = tf.parse_single_example(
            example,
            features={
                'gt': tf.FixedLenFeature([], tf.string)
            })
        p = self.patch_size
        gt = features['gt']
        gt = tf.decode_raw(gt, tf.uint8)
        gt = tf.reshape(gt, [p, p])
        gt = tf.cast(gt, tf.float32)

        c1 = tf.random_uniform([], 0, 1)
        c2 = tf.random_uniform([], 0, 1)
        gt = tf.cond(c1 < 0.5, lambda: gt[::-1, :], lambda: gt)
        gt = tf.cond(c2 < 0.5, lambda: gt[:, ::-1], lambda: gt)

        lr = tf.py_func(lambda x: misc.imresize(x, 1.0 / self.scale, 'bicubic', 'F'), [gt], tf.float32)
        bic = tf.py_func(lambda x: misc.imresize(x, self.scale / 1.0, 'bicubic', 'F'), [lr], tf.float32)

        gt = tf.reshape(gt, [p, p, 1]) / 255.0
        bic = tf.reshape(bic, [p, p, 1]) / 255.0
        lr = tf.reshape(lr, [p // self.scale, p // self.scale, 1]) / 255.0

        return lr, bic, gt

    def read_tfrecords(self, save_dir='DIV2K/tfrecords'):
        fs_paths = sorted(glob.glob(os.path.join(save_dir, '*.tfrecords')))
        if len(fs_paths) == 0:
            print('No tfrecords. Should run gen_tfrecords() firstly.')
            exit()
        dataset = tf.data.TFRecordDataset(fs_paths)
        dataset = dataset.map(self._parse_one_example, self.map_parallel_num).shuffle(self.shuffle_num) \
            .prefetch(self.prefetch_num).batch(self.batch_size).repeat()
        lrs, bics, gts = dataset.make_one_shot_iterator().get_next()
        return lrs, bics, gts

    def get_generator(self):
        img_paths = sorted(glob.glob(os.path.join(self.data_dir, '*.png')))
        one_img_patch_num = self.one_img_patch_num
        p = self.patch_size
        scale = self.scale
        for img_path in img_paths:
            img = misc.imread(img_path)
            height, width, _ = img.shape
            for i in range(one_img_patch_num):
                h = np.random.randint(height - p + 1)
                w = np.random.randint(width - p + 1)
                patch = img[h: h + p, w: w + p]
                gt = utils.rgb2ycbcr(patch)[:, :, 0]
                gt = np.float32(gt) / 255.0
                c1 = np.random.rand()
                c2 = np.random.rand()
                if c1 < 0.5:
                    gt = gt[::-1, :]
                if c2 < 0.5:
                    gt = gt[:, ::-1]
                lr = misc.imresize(gt, 1.0 / scale, 'bicubic', 'F')
                bic = misc.imresize(lr, scale / 1.0, 'bicubic', 'F')
                yield lr, bic, gt

    def read_pngs(self):
        dataset = tf.data.Dataset.from_generator(self.get_generator, (tf.float32, tf.float32, tf.float32))
        dataset = dataset.shuffle(self.shuffle_num).prefetch(self.prefetch_num).batch(self.batch_size).repeat()
        lrs, bics, gts = dataset.make_one_shot_iterator().get_next()
        p = self.patch_size
        lrs = tf.reshape(lrs, [-1, p // self.scale, p // self.scale, 1])
        gts = tf.reshape(gts, [-1, p, p, 1])
        bics = tf.reshape(bics, [-1, p, p, 1])
        return lrs, bics, gts


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    data_loader = DataLoader()
    # data_loader.gen_tfrecords()
    # lrs, bics, gts = data_loader.read_tfrecords()
    lrs, bics, gts = data_loader.read_pngs()
    sess = tf.Session()
    import matplotlib.pyplot as plt
    # while True:
    #     im1, im2 = sess.run([lrs, gts])
    #     plt.imshow(utils.img_to_uint8(im1[0, :, :, 0]))
    #     plt.show()
    im1, im2 = sess.run([lrs, gts])
    print(im1.shape, im2.shape)
