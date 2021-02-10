import tensorflow as tf


# Jaccard loss
def jaccard_loss(y_true, y_pred, smooth=1.0):
    # Copied from https://gist.github.com/LaurenzReitsam/05e3bb42024ff76955adbf92356d79f2
    intersection = tf.reduce_sum(y_true * y_pred, axis=-1)
    union = tf.reduce_sum(y_true + y_pred, axis=-1) - intersection
    jac = (intersection + smooth)/(union + smooth)
    return (1 - jac) * smooth


# F1 loss
def f1_loss(y_true, y_pred, smooth=1.0):
    # Copied from https://gist.github.com/LaurenzReitsam/05e3bb42024ff76955adbf92356d79f2
    intersection = tf.reduce_sum(y_true * y_pred, axis=-1)
    denominator = tf.reduce_sum(y_true + y_pred, axis=-1)
    f1 = (2 * intersection + smooth)/(denominator + smooth)
    return (1 - f1) * smooth
