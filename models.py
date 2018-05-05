import numpy as np
import tensorflow as tf


def iterated_model_unrolled(features, labels, mode, params):
    """Model function for iterated network."""
    iterations = params['iterations']
    channels = params['channels']

    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    adapted_to_channels = tf.layers.conv2d(
        inputs=input_layer,
        filters=channels,
        kernel_size=[1, 1],
        padding="same",
        activation=tf.nn.relu
    )

    accumulator = adapted_to_channels

    for i in range(iterations):
        with tf.variable_scope('iteration', reuse=tf.AUTO_REUSE):
            accumulator = tf.layers.conv2d(
                inputs=accumulator,
                filters=channels,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu
            )
            accumulator += adapted_to_channels
            tf.layers.dropout(
                inputs=accumulator, rate=0.4,
                training=mode == tf.estimator.ModeKeys.TRAIN)

    flat = tf.layers.flatten(accumulator)

    dense = tf.layers.dense(inputs=flat, units=1024,
                            activation=tf.nn.relu)

    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def iterated_model_while_op(features, labels, mode, params):
    """Model function for iterated network."""
    iterations = tf.Variable(params['iterations'], tf.int32)
    channels = params['channels']

    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    adapted_to_channels = tf.layers.conv2d(
        inputs=input_layer,
        filters=channels,
        kernel_size=[1, 1],
        padding="same")

    accumulator = adapted_to_channels

    def condition(i, n, _):
        return tf.less(i, n)

    def loop(i, n, input_):
        with tf.variable_scope('iteration', reuse=tf.AUTO_REUSE):
            input_ = tf.layers.conv2d(
                inputs=input_,
                filters=channels,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu
            )
            input_ += adapted_to_channels
            tf.layers.dropout(
                inputs=input_, rate=0.4,
                training=mode == tf.estimator.ModeKeys.TRAIN)
        return [tf.add(i, 1), n, input_]

    _, _, accumulator = tf.while_loop(
        condition,
        loop,
        [tf.constant(0), iterations, accumulator],
    )

    flat = tf.layers.flatten(accumulator)

    dense = tf.layers.dense(inputs=flat, units=1024,
                            activation=tf.nn.relu)

    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def cnn_model(features, labels, mode, params):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024,
                            activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def train(eval_data, eval_labels, train_data, train_labels, steps, model_fn,
          name=None, params=None):
    if name is None:
        name = model_fn.__name__
    if params is None:
        params = {}

    # Create the Estimator
    model = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=f"./data/{name}",
        params=params
    )

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=1000,
        num_epochs=None,
        shuffle=True)
    model.train(
        input_fn=train_input_fn,
        steps=steps,
    )

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = model.evaluate(input_fn=eval_input_fn)
    return model, eval_results
