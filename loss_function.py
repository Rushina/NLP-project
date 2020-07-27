import tensorflow as tf

def similarity(y_pred, dims):
    n, N, wlen, opveclen = dims
    fq = []
    fp = []
    fq = tf.reshape(y_pred[:,0:opveclen], (-1, 1, opveclen))
    fp = tf.reshape(y_pred[:, opveclen:], (-1, n, opveclen))
    s1 = tf.keras.backend.sum(fp[:,:,:]*fq, axis=-1)
    s2 = s1/tf.keras.backend.sqrt(tf.keras.backend.sum(fq*fq, axis=-1))
    s = s2/tf.keras.backend.sqrt(tf.keras.backend.sum(fp*fp, axis = -1))
    return s

def loss_fn_wrap2(dims):
    def loss_fn(y_true, y_pred):
        n, N, wlen, opveclen = dims
        s = similarity(y_pred, dims)
        delta = 1.0
        l1 = s - s[:, 0:1]
        l2 = l1[:, 1:] + delta
        loss_ = tf.keras.backend.max(l2, axis=-1)
        return tf.math.maximum(tf.convert_to_tensor(0.0), loss_) 
    return loss_fn

def loss_fn_wrap(dims):
    def loss_fn(y_true, y_pred):
        n, N, wlen, opveclen = dims
        s = similarity(y_pred, dims)

        delta = 1.0
        labels = tf.reshape(y_true, (-1, (n+1), opveclen))[:, :-1, 0]
        diff = list()
        for i in range(s.shape[1]):
            d = s[:, i:i+1] - s + (1 - labels[:, i:i+1])*delta
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
        lstemp = tf.keras.backend.max(losst, axis=-1)
        ls = tf.keras.backend.max(lstemp, axis=-1)
        return ls
    return loss_fn
