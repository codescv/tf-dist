import os
import tensorflow as tf
from data import build_model_columns, input_fn


def build_model(filename):
    wide_columns, deep_columns = build_model_columns()
    global_step = tf.train.get_or_create_global_step()
    features, labels = input_fn(filename).make_one_shot_iterator().get_next()
    cols_to_vars = {}
    logits = tf.feature_column.linear_model(features=features, feature_columns=wide_columns, cols_to_vars=cols_to_vars)
    predictions = tf.reshape(tf.nn.sigmoid(logits), (-1,))
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

    # build graph
    model = build_model(filename='census_data/adult.data')

    # inspect graph variables
    for col, var in model['cols_to_vars'].items():
        print('Column:  ', col)
        print('Variable:', var)
        print('-' * 50)

    model_dir = 'tmp/lr_single'
    profile_dir = os.path.join(model_dir, 'eval')
    checkpoint_dir = os.path.join(model_dir, 'checkpoint')
    os.makedirs(profile_dir, exist_ok=True)
    # create session
    hooks = [
        tf.train.ProfilerHook(save_secs=30, output_dir=profile_dir),
        tf.train.StopAtStepHook(num_steps=10000)
    ]
    with tf.train.MonitoredTrainingSession(hooks=hooks,
                                           checkpoint_dir=checkpoint_dir,
                                           log_step_count_steps=60) as sess:
        # sess.run(model['init'])
        step = 0
        while not sess.should_stop():
            step += 1
            sess.run(model['train'])


if __name__ == '__main__':
    main()
