import os
import time
import tensorflow as tf

from tensorflow.python.client import timeline

from data_tf import build_model_columns, input_fn, gen_test_data


def build_model(filename):
    wide_columns, deep_columns = build_model_columns()
    global_step = tf.train.get_or_create_global_step()
    features, labels = input_fn(filename).make_one_shot_iterator().get_next()
    cols_to_vars = {}
    logits = tf.feature_column.linear_model(features=features, feature_columns=wide_columns, cols_to_vars=cols_to_vars)
    # predictions = tf.reshape(tf.nn.sigmoid(logits), (-1,))
    predictions = tf.nn.sigmoid(logits)
    # print('labels:', labels.shape, 'predictions:', predictions.shape, 'logits:', logits.shape)
    loss = tf.losses.log_loss(labels=labels, predictions=predictions)
    optimizer = tf.train.FtrlOptimizer(learning_rate=0.1, l1_regularization_strength=0.1, l2_regularization_strength=0.1)
    train_op = optimizer.minimize(loss, global_step=global_step)

    return {
        'train': {
            'train_op': train_op,
            'loss': loss,
            'global_step': global_step
        },
        'init': {
            'global': [tf.global_variables_initializer()],
            'local': [tf.local_variables_initializer(), tf.tables_initializer()]
        },
        'cols_to_vars': cols_to_vars
    }


def main():
    import logging
    logging.basicConfig(level=logging.INFO)

    num_columns = 200
    data_path = 'data/data.tfrecord'

    gen_test_data(num_columns, data_path)

    # build graph
    model = build_model(filename=data_path)

    # inspect graph variables
    for col, var in model['cols_to_vars'].items():
        print('Column:  ', col)
        print('Variable:', var)
        print('-' * 50)

    model_dir = 'tmp/lr_single'
    profile_dir = os.path.join(model_dir, 'profile')
    os.makedirs(profile_dir, exist_ok=True)

    # create session
    with tf.Session() as sess:
        sess.run(model['init'])
        step = 0
        time_start = time.time()
        last_step = step
        while step < 10000:
            step += 1
            should_write_profile = 100 < step < 110

            opts = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) if should_write_profile else None
            run_metadata = tf.RunMetadata() if should_write_profile else None

            sess.run(model['train'], options=opts, run_metadata=run_metadata)

            if should_write_profile:
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                trace_file = os.path.join(profile_dir, 'profile-{}.json'.format(step))
                with open(trace_file, 'w') as f:
                    f.write(chrome_trace)
                    print('written trace file:', trace_file)

            if step % 60 == 0:
                print('step/sec:', (step-last_step) / (time.time() - time_start))
                time_start = time.time()
                last_step = step


if __name__ == '__main__':
    main()
