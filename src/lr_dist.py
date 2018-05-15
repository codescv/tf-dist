import os
import json
import tensorflow as tf
from lr_single import build_model_columns, input_fn


def build_model(filename):
    wide_columns, deep_columns = build_model_columns()
    features, labels = input_fn(filename).make_one_shot_iterator().get_next()
    cols_to_vars = {}
    linear_logits = tf.feature_column.linear_model(features=features, feature_columns=wide_columns, cols_to_vars=cols_to_vars)
    predictions = tf.reshape(tf.nn.sigmoid(linear_logits), (-1,))
    loss = tf.losses.log_loss(labels=labels, predictions=predictions)
    optimizer = tf.train.FtrlOptimizer(learning_rate=0.1, l1_regularization_strength=0.1, l2_regularization_strength=0.1)
    train_op = optimizer.minimize(loss)

    return {
        'train': {
            'train_op': train_op,
            'loss': loss
        },
        'init': {
            'global': [tf.global_variables_initializer()],
            'local': [tf.local_variables_initializer(), tf.tables_initializer()]
        },
        'cols_to_vars': cols_to_vars
    }


def main(tf_config):
    task_type = tf_config['task']['type']
    task_index = tf_config['task']['index']

    # build graph
    with tf.device(tf.train.replica_device_setter(
            worker_device=f"/job:{task_type}/task:{task_index}",
            ps_device="/job:ps",
            cluster=tf_config['cluster'])):
        model = build_model(filename='census_data/adult.data')

    # start server
    cluster = tf.train.ClusterSpec(tf_config['cluster'])
    server = tf.train.Server(
        cluster, job_name=task_type, task_index=task_index)

    # create session
    with tf.Session(target=server.target) as sess:
        sess.run(model['init'])

        for step in range(1, 1000):
            result = sess.run(model['train'])
            print('step =', step, 'loss =', result['loss'])


if __name__ == '__main__':
    main(json.loads(os.environ.get("TF_CONFIG", "{}")))
