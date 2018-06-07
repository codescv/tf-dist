import os
import time
import json
import tensorflow as tf
import logging
from data import build_model_columns, input_fn


def build_model(filename):
    wide_columns, deep_columns = build_model_columns()
    features, labels = input_fn(filename).make_one_shot_iterator().get_next()
    cols_to_vars = {}
    linear_logits = tf.feature_column.linear_model(features=features, feature_columns=wide_columns, cols_to_vars=cols_to_vars)
    predictions = tf.reshape(tf.nn.sigmoid(linear_logits), (-1,))
    loss = tf.reduce_mean(tf.losses.log_loss(labels=labels, predictions=predictions))
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.FtrlOptimizer(learning_rate=0.1,
                                       l1_regularization_strength=0.1,
                                       l2_regularization_strength=0.1)
    train_op = optimizer.minimize(loss, global_step=global_step)

    tf.summary.scalar('prediction/mean', tf.reduce_mean(predictions))
    tf.summary.histogram('prediction', predictions)
    tf.summary.scalar('metrics/loss', loss)
    summary = tf.summary.merge_all()

    global_vars = tf.global_variables()
    uninitialized = tf.report_uninitialized_variables(tf.global_variables())

    global_init = tf.global_variables_initializer()
    local_init = [tf.local_variables_initializer(), tf.tables_initializer()]

    return {
        'train': {
            'train_op': train_op,
            'loss': loss,
        },
        'init': {
            'global': global_init,
            'local': local_init
        },
        'global_variables': global_vars,
        'uninitialized': uninitialized,
        'cols_to_vars': cols_to_vars,
        'summary': summary,
        'global_step': global_step,
    }


def main(tf_config):
    task_type = tf_config['task']['type']
    task_index = tf_config['task']['index']

    # start server
    cluster = tf.train.ClusterSpec(tf_config['cluster'])
    server = tf.train.Server(
        cluster, job_name=task_type, task_index=task_index)

    if task_type == 'ps':
        server.join()
        return

    is_chief = task_type == 'master'

    # build graph
    with tf.device(tf.train.replica_device_setter(
            worker_device=f"/job:{task_type}/task:{task_index}",
            ps_device="/job:ps",
            cluster=tf_config['cluster'])):
        model = build_model(filename='census_data/adult.data')

    writer = None
    if is_chief:
        writer = tf.summary.FileWriter(logdir='tmp/model/lr-dist', graph=tf.get_default_graph())

    # create session
    config = tf.ConfigProto(log_device_placement=True)
    with tf.Session(target=server.target, config=config) as sess:
        tf.get_default_graph().finalize()
        if is_chief:
            sess.run(model['init']['global'])
            sess.run(model['init']['local'])
            step = 0
            while step < 2000:
                step = sess.run(model['global_step'])
                logging.info('global step = %s', step)
                writer.add_summary(sess.run(model['summary']), global_step=step)
                time.sleep(1)
        else:
            ready = False
            while not ready:
                uninitialized = sess.run(model['uninitialized'])
                ready = len(uninitialized) == 0
                if not ready:
                    logging.info('still waiting for variables to initialize: %s', uninitialized)
                    time.sleep(5)

            sess.run(model['init']['local'])
            for step in range(1000):
                result = sess.run(model['train'])
                logging.info('step = %s, loss= %s, global step = %s', step, result['loss'], sess.run(model['global_step']))

    time.sleep(10)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)-15s [%(levelname)s] [%(filename)s:%(lineno)s] %(message)s', level=logging.INFO)
    main(json.loads(os.environ.get("TF_CONFIG", "{}")))
