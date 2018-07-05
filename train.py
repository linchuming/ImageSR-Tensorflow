import tensorflow as tf
from model import VDSR, EDSR
import datetime
from data_loader import DataLoader
import os

MODEL = 'VDSR'                  # 'VDSR' or 'EDSR'
TRAIN_DIR = 'output/{}/model'.format(MODEL)
LOG_DIR = 'output/{}/log'.format(MODEL)
BATCH_SIZE = 64
SHUFFLE_NUM = 10000
PREFETCH_NUM = 5000
MAX_TRAIN_STEP = 100000
LR_BOUNDS = [90000]
LR_VALS = [1e-4, 1e-5]
SAVE_PER_STEP = 2000
TRAIN_PNG_PATH = 'DIV2K/DIV2K_train_HR'
TRAIN_TFRECORD_PATH = 'DIV2K/tfrecords'
DATA_LOADER_MODE = 'RAW'        # 'TFRECORD' or 'RAW'
DEVICE_MODE = 'GPU'             # 'CPU' or 'GPU'
DEVICE_GPU_ID = '0'

if not os.path.exists(TRAIN_DIR):
    os.makedirs(TRAIN_DIR)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

if DEVICE_MODE == 'CPU':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_GPU_ID

def restore_session_from_checkpoint(sess, saver):
    checkpoint = tf.train.latest_checkpoint(TRAIN_DIR)
    if checkpoint:
        saver.restore(sess, checkpoint)
        return True
    else:
        return False

if MODEL == 'VDSR':
    model = VDSR()
else:
    model = EDSR()

data_loader = DataLoader(data_dir=TRAIN_PNG_PATH,
                         batch_size=BATCH_SIZE,
                         shuffle_num=SHUFFLE_NUM,
                         prefetch_num=PREFETCH_NUM)

if DATA_LOADER_MODE == 'TFRECORD':
    if len(os.listdir(TRAIN_TFRECORD_PATH)) == 0:
        data_loader.gen_tfrecords(TRAIN_TFRECORD_PATH)
    lrs, gts = data_loader.read_tfrecords(TRAIN_TFRECORD_PATH)
else:
    lrs, gts = data_loader.read_pngs()

res = model(lrs)
with tf.name_scope('train'):
    global_step = tf.Variable(0, trainable=False, name='global_step')
    mse_loss = tf.reduce_mean(tf.square(res - gts)) * 1e3
    reg_loss = tf.losses.get_regularization_loss()
    loss = mse_loss + reg_loss

    learning_rate = tf.train.piecewise_constant(global_step, LR_BOUNDS, LR_VALS)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step)

with tf.name_scope('summaries'):
    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('mse_loss', mse_loss)
    tf.summary.scalar('reg_loss', reg_loss)
    tf.summary.scalar('loss', loss)

    tf.summary.image('lr', lrs, 1)
    tf.summary.image('out', res, 1)
    tf.summary.image('gt', gts, 1)

    summary_op = tf.summary.merge_all()

saver = tf.train.Saver(max_to_keep=500)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

restore_session_from_checkpoint(sess, saver)

start_time = datetime.datetime.now()
writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

while True:
    _, loss_value, step = sess.run([train_op, loss, global_step])
    if step % 20 == 0:
        end_time = datetime.datetime.now()
        print('[{}] Step:{}, loss:{}'.format(
            end_time - start_time, step, loss_value
        ))
        summary_value = sess.run(summary_op)
        writer.add_summary(summary_value, step)
        start_time = end_time
    if step % SAVE_PER_STEP == 0:
        saver.save(sess, os.path.join(TRAIN_DIR, 'checkpoint.ckpt'), global_step=step)
    if step >= MAX_TRAIN_STEP:
        print('Done train.')
        break







