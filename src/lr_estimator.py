import os
import tensorflow as tf
from data import build_model_columns, input_fn


def main():
    import logging
    logging.basicConfig(level=logging.INFO)

    wide_columns, deep_columns = build_model_columns()

    model_dir = 'tmp/lr_estimator'
    estimator = tf.estimator.LinearClassifier(
        feature_columns=wide_columns,
        model_dir=model_dir,
        optimizer=tf.train.FtrlOptimizer(learning_rate=0.1, l1_regularization_strength=0.1, l2_regularization_strength=0.1)
    )

    profile_dir = os.path.join(model_dir, 'eval')
    os.makedirs(profile_dir, exist_ok=True)
    hooks = [
        tf.train.ProfilerHook(save_secs=10, output_dir=profile_dir),
        tf.train.StopAtStepHook(num_steps=10000)
    ]

    estimator.train(input_fn=lambda: input_fn('census_data/adult.data'), hooks=hooks, steps=100000)


if __name__ == '__main__':
    main()
