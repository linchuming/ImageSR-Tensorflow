import tensorflow as tf
from model import VDSR, EDSR, BICUBIC
import glob
import os
import utils
from scipy import misc
import numpy as np

TEST_DIR = 'test_data/Set5'
MODEL = 'VDSR'  # 'VDSR' or 'EDSR' or 'BICUBIC'
MODEL_CKPT_PATH = 'model/{}/checkpoint.ckpt-50000'.format(MODEL)
OUTPUT_DIR = 'result/Set5/{}'.format(MODEL)
DEVICE_MODE = 'GPU'  # 'CPU' or 'GPU'
DEVICE_GPU_ID = '0'
SCALE = 4

if DEVICE_MODE == 'CPU':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_GPU_ID

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

if MODEL == 'VDSR':
    model = VDSR()
elif MODEL == 'EDSR':
    model = EDSR()
else:
    model = BICUBIC()

lr = tf.placeholder(tf.float32, [None, None, None, 1])
res = model(lr)

if not MODEL == 'BICUBIC':
    saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
if not MODEL == 'BICUBIC':
    saver.restore(sess, MODEL_CKPT_PATH)

fs = glob.glob(os.path.join(TEST_DIR, '*.bmp'))
psnrs = []
for f in fs:
    img = misc.imread(f)
    lr_img = misc.imresize(img, 1.0 / SCALE, 'bicubic')
    lr_y = utils.rgb2ycbcr(lr_img)[:, :, :1]
    lr_y = np.expand_dims(lr_y, 0).astype(np.float32) / 255.0
    res_y = sess.run(res, feed_dict={lr: lr_y})
    res_y = np.clip(res_y, 0, 1)[0] * 255.0
    bic_img = misc.imresize(lr_img, SCALE / 1.0, 'bicubic')

    bic_ycbcr = utils.rgb2ycbcr(bic_img)
    bic_ycbcr[:, :, :1] = res_y
    res_img = utils.img_to_uint8(utils.ycbcr2rgb(bic_ycbcr))
    img_name = f.split(os.sep)[-1]
    misc.imsave(os.path.join(OUTPUT_DIR, img_name), res_img)

    gt_y = utils.rgb2ycbcr(img)[:, :, :1]
    psnr = utils.psnr(res_y[SCALE:-SCALE, SCALE:-SCALE], gt_y[SCALE:-SCALE, SCALE:-SCALE])
    psnrs.append(psnr)
    print(img_name, 'PSNR:', psnr)

print('AVG PSNR:', np.mean(psnrs))
