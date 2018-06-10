import os
import tensorflow as tf


def build_model_columns(num_columns=200, bucket_size=10000):
    """Builds a set of wide and deep feature columns."""
    feature_columns = [
        tf.feature_column.categorical_column_with_hash_bucket('C{}'.format(i), bucket_size)
        for i in range(num_columns)
    ]

    return feature_columns, []


def input_fn(data_file, num_epochs=None, shuffle=True, batch_size=1024):
    assert tf.gfile.Exists(data_file), ('%s not found.' % data_file)

    feature_columns, _ = build_model_columns()

    def parse_tfrecord(example):
        label_column = tf.feature_column.numeric_column('label', dtype=tf.float32, default_value=0)
        parsed = tf.parse_single_example(example, features=tf.feature_column.make_parse_example_spec(feature_columns + [label_column]))
        label = parsed.pop('label')
        return parsed, label

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TFRecordDataset(data_file)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 10)

    dataset = dataset.prefetch(buffer_size=batch_size*10)

    dataset = dataset.map(parse_tfrecord, num_parallel_calls=5)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    return dataset


def input_fn2(data_file, num_epochs=None, shuffle=True, batch_size=1024):
    assert tf.gfile.Exists(data_file), ('%s not found.' % data_file)

    feature_columns, _ = build_model_columns()

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TFRecordDataset(data_file)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 10)

    dataset = dataset.prefetch(buffer_size=batch_size*10)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    return dataset


def gen_test_data(num_columns, output):
    if not os.path.exists(output):
        print('gen data:', output)
        from gen_data import gen_tfrecord_data
        gen_tfrecord_data(num_columns=num_columns, output=output)
        print('generated data:', output)
    else:
        print('data already exists:', output)
