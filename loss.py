import tensorflow as tf
import numpy as np

opveclen = 2
n = 2

def loss_fn(y_true, y_pred):
    fq = []
    fp = []
    fq = tf.reshape(y_pred[:,0:opveclen], (-1, 1, opveclen))
    fp = tf.reshape(y_pred[:, opveclen:], (-1, n, opveclen))
    s1 = tf.keras.backend.sum(fp[:,:,:]*fq, axis=-1)
    s2 = s1/tf.keras.backend.sqrt(tf.keras.backend.sum(fq*fq, axis=-1))
    s = s2/tf.keras.backend.sqrt(tf.keras.backend.sum(fp*fp, axis = -1))
    delta = 0.01
    labels = tf.reshape(y_true, (-1, (n+1), opveclen))[:, :-1, 0]
    diff = list()
    for i in range(s.shape[1]):
        d = s[:, i:i+1] - s + labels[:, i:i+1]*delta
        if not diff:
            diff = [d]
        else:
            diff.append(d)
    difference = tf.stack(diff, axis=1)
    loss = []
    for j in range(s.shape[1]):
        l = tf.multiply(difference[:,j,:], tf.cast(labels==1, difference[:,j,:].dtype))
        if not loss:
            loss = [l]
        else:
            loss.append(l)
    losst = tf.stack(loss, axis=1)
    ls = tf.keras.backend.max(tf.keras.backend.max(losst, axis=-1), axis=-1)
    print('inside loss function, prediction = ' + str(y_pred))
    print('inside loss function, loss vector = ' + str(losst))
    print('inside loss function: '+ str(ls))
    return ls

y_true = tf.convert_to_tensor(np.array([[1, 1, 0, 0, 0, 0],[1, 1, 0, 0, 0, 0]]), dtype=tf.float32)
y_pred = tf.convert_to_tensor(np.array([[0.5, 0.1, 0.4, 0.2, -0.8, -0.9],[0.5, 0.1, 0.4, 0.2, -0.8, -0.9]]), dtype=tf.float32)
print(loss_fn(y_true, y_pred))
