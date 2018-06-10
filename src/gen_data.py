import os
import tensorflow as tf
import random


def floatlist_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[float(v) for v in value]))


def intlist_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytelist_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def rand_bytes(length):
    val = ''
    for _ in range(length):
        val += chr(random.randrange(26) + ord('a'))
    return val.encode('ascii')


def gen_tfrecord_data(num_examples=10000, num_columns=200, output='data/data.tfrecord'):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    writer = tf.python_io.TFRecordWriter(output)

    for n in range(num_examples):
        feature = {}
        for c in range(num_columns):
            colname = 'C{}'.format(c)
            val_length = random.randrange(1, 5)
            value = [rand_bytes(random.randrange(8, 24)) for _ in range(val_length)]
            feature[colname] = bytelist_feature(value)
        feature['label'] = floatlist_feature([random.randrange(2)])
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        serialized = example.SerializeToString()
        # print('example length:', len(serialized))
        writer.write(serialized)


if __name__ == '__main__':
    gen_tfrecord_data()
