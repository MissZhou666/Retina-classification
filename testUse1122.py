import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.client import graph_util
from tensorflow.python.platform import gfile

BOTTLENECK_TENSOR_SIZE = 2048
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
MODEL_DIR = '/home/zyj/testSave/model'
MODEL_FILE = 'classify_image_graph_def.pb'
CACHE_DIR = '/home/zyj/testSave/tmp/bottleneck'
INPUT_DATA = '/home/zyj/testSave/pic'
VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10
LEARNING_RATE = 0.01
STEPS = 6
BATCH = 400


def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


def get_or_create_bottleneck(sess, image_path, jpeg_data_tensor, bottleneck_tensor):
    image_data = gfile.FastGFile(image_path, 'rb').read()
    bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
    # bottleneck_string=','.join(str(x) for x in bottleneck_values)
    return bottleneck_values


def get_random_cached_bottlenecks(sess, image_path, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    for _ in range(1):
        bottleneck = get_or_create_bottleneck(sess, image_path, jpeg_data_tensor, bottleneck_tensor)
        bottlenecks.append(bottleneck)
    return bottlenecks


def main(_):
    # init=tf.initialize_all_variables()

    with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def, return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])
    bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name='BottleneckInputPlaceholder')
    with tf.name_scope('final_training_ops'):
        weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, 4], stddev=0.001))
        biases = tf.Variable(tf.zeros([4]))
        logits = tf.matmul(bottleneck_input, weights) + biases
        final_tensor = tf.nn.softmax(logits)
    prediction_labels = tf.argmax(final_tensor, 1, name="output")
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # sess.run(init)
        saver.restore(sess, "./model1122/net1124.ckpt")
        test_bottlenecks = get_random_cached_bottlenecks(sess, "/home/zyj/testSave/test/light1.jpg", jpeg_data_tensor, bottleneck_tensor)
        print(sess.run(prediction_labels, feed_dict={bottleneck_input: test_bottlenecks}))




if __name__ == '__main__':
    tf.app.run()