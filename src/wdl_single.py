import tensorflow as tf
import six

from data import build_model_columns, input_fn


def _dnn_logit_fn_builder(units, hidden_units, feature_columns, activation_fn,
                          dropout, input_layer_partitioner):
    """Function builder for a dnn logit_fn.

    Args:
      units: An int indicating the dimension of the logit layer.  In the
        MultiHead case, this should be the sum of all component Heads' logit
        dimensions.
      hidden_units: Iterable of integer number of hidden units per layer.
      feature_columns: Iterable of `feature_column._FeatureColumn` model inputs.
      activation_fn: Activation function applied to each layer.
      dropout: When not `None`, the probability we will drop out a given
        coordinate.
      input_layer_partitioner: Partitioner for input layer.

    Returns:
      A logit_fn (see below).

    Raises:
      ValueError: If units is not an int.
    """
    if not isinstance(units, int):
        raise ValueError('units must be an int.  Given type: {}'.format(
            type(units)))

    def dnn_logit_fn(features, mode):
        """Deep Neural Network logit_fn.

        Args:
          features: This is the first item returned from the `input_fn`
                    passed to `train`, `evaluate`, and `predict`. This should be a
                    single `Tensor` or `dict` of same.
          mode: Optional. Specifies if this training, evaluation or prediction. See
                `ModeKeys`.

        Returns:
          A `Tensor` representing the logits, or a list of `Tensor`'s representing
          multiple logits in the MultiHead case.
        """
        with tf.variable_scope(
                'input_from_feature_columns',
                values=tuple(six.itervalues(features)),
                partitioner=input_layer_partitioner):
            net = tf.feature_column.input_layer(
                features=features, feature_columns=feature_columns)
        for layer_id, num_hidden_units in enumerate(hidden_units):
            with tf.variable_scope(
                    'hiddenlayer_%d' % layer_id, values=(net,)) as hidden_layer_scope:
                net = tf.layers.dense(
                    net,
                    units=num_hidden_units,
                    activation=activation_fn,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    name=hidden_layer_scope)
                if dropout is not None and mode == 'train':
                    net = tf.layers.dropout(net, rate=dropout, training=True)
            # _add_hidden_layer_summary(net, hidden_layer_scope.name)

        with tf.variable_scope('logits', values=(net,)) as logits_scope:
            logits = tf.layers.dense(
                net,
                units=units,
                activation=None,
                kernel_initializer=tf.glorot_uniform_initializer(),
                name=logits_scope)
        # _add_hidden_layer_summary(logits, logits_scope.name)

        return logits

    return dnn_logit_fn


def build_model(features, labels=None, mode='train'):
    wide_columns, deep_columns = build_model_columns()

    global_step = tf.train.get_or_create_global_step()

    cols_to_vars = {}
    with tf.variable_scope(
        'linear',
        values=features.values(),
        reuse=tf.AUTO_REUSE
    ):
        linear_logits = tf.feature_column.linear_model(features=features, feature_columns=wide_columns, cols_to_vars=cols_to_vars)

    with tf.variable_scope(
        'dnn',
        values=features.values(),
        reuse=tf.AUTO_REUSE
    ):
        dnn_logit_fn = _dnn_logit_fn_builder(
            units=1,
            hidden_units=[8, 4],
            feature_columns=deep_columns,
            activation_fn=tf.nn.relu,
            dropout=None,
            input_layer_partitioner=None)
        dnn_logits = dnn_logit_fn(features=features, mode='train')

    logits = linear_logits + dnn_logits

    predictions = tf.reshape(tf.nn.sigmoid(logits), (-1,))

    if mode == 'train':
        loss = tf.losses.log_loss(labels=labels, predictions=predictions)
        linear_optimizer = tf.train.FtrlOptimizer(
            learning_rate=0.1,
            l1_regularization_strength=0.1,
            l2_regularization_strength=0.1,
        )
        linear_train_op = linear_optimizer.minimize(loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='linear'))
        dnn_optimizer = tf.train.AdamOptimizer(
            learning_rate=0.01
        )
        dnn_train_op = dnn_optimizer.minimize(loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dnn'))

        train_ops = tf.group(linear_train_op, dnn_train_op)
        with tf.control_dependencies([train_ops]):
            with tf.colocate_with(global_step):
                train_op = tf.assign_add(global_step, 1)

        return {
            'train': {
                'train_op': train_op,
                'loss': loss
            },
            'init': {
                'global': [tf.global_variables_initializer()],
                'local': [tf.local_variables_initializer(), tf.tables_initializer()]
            },
            'cols_to_vars': cols_to_vars,
        }
    elif mode == 'predict':
        return {
            'predictions': predictions
        }


def input_receiver():
    wide_columns, deep_columns = build_model_columns()
    columns = wide_columns + deep_columns
    feature_spec = tf.feature_column.make_parse_example_spec(columns)
    serialized_tf_example = tf.placeholder(dtype=tf.string,
                                           shape=[None],
                                           name='input')
    receiver_tensors = {'examples': serialized_tf_example}
    features = tf.parse_example(serialized_tf_example, feature_spec)
    rec = tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
    # print('rec:', rec)
    return rec


def main():
    # build graph
    features, labels = input_fn(data_file='census_data/adult.data').make_one_shot_iterator().get_next()
    model = build_model(features=features, labels=labels, mode='train')

    # inspect graph variables
    for col, var in model['cols_to_vars'].items():
        print('Column:  ', col)
        print('Variable:', var)
        print('-' * 50)

    # create session
    with tf.Session() as sess:
        sess.run(model['init'])

        for step in range(1, 1000):
            result = sess.run(model['train'])
            print('step =', step, 'loss =', result['loss'])

        signature_def_map = {}

        receiver = input_receiver()
        receiver_tensors = receiver.receiver_tensors
        pred_model = build_model(features=receiver.features, mode='predict')
        output = tf.estimator.export.PredictOutput(outputs=pred_model['predictions'])
        signature_def_map['serving_default'] = output.as_signature_def(receiver_tensors)
        # print('sig def map:', signature_def_map)
        local_init_op = tf.group(
            tf.local_variables_initializer(),
            tf.tables_initializer(),
            )
        builder = tf.saved_model.builder.SavedModelBuilder('export/wdl_single')
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map=signature_def_map,
            assets_collection=tf.get_collection(
                tf.GraphKeys.ASSET_FILEPATHS),
            legacy_init_op=local_init_op,
            strip_default_attrs=False)
        builder.save(as_text=False)


if __name__ == '__main__':
    main()
