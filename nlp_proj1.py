import random
import numpy as np
import tensorflow as tf
import os
import pickle

class Logger(object):
    def __init__(self, mode='log'):
        self.mode = 0
        if (mode == 'error'):
            self.mode = 2
        elif (mode == 'warn'):
            self.mode = 1
        else:
            self.mode = 0
    def warn(self, str):
        if (self.mode < 2):
            print('WARN: ', str)
    
    def log(self, str):
        if (self.mode < 1):
            print('Logging: ', str)

    def error(self, str):
        print('ERROR: ', str)

    
def read_question_data(filename, num_fields = 3):
    f = open(filename, "r")
    raw_data = f.readlines()
    f.close()
    q = list()
    pos = list()
    neg = list()
    for td in raw_data:
        qi, posi, negi = [], [], []
        if (num_fields == 3):
            qi, posi, negi = td.split('\t')
        elif (num_fields == 4):
            qi, posi, negi, _ = td.split('\t')
        q.append(qi)
        pos.append(posi)
        neg.append(negi)
    return (q, pos, neg)

def word2vec(word, word_embed):
    if not word in word_embed.keys():
        return np.zeros_like(word_embed['the'])
    return np.array(word_embed[word])

def sentence2vec_avg(sentence, word_embed):
    words = sentence.split()
    n = len(words)
    vec = np.zeros((n, len(word_embed['the'])))
    for (i,w) in enumerate(words):
        vec[i] = word2vec(w, word_embed)
    return vec 

def question2vec(question, N, word_embed):
    title, body = question
    title_vec = sentence2vec_avg(title, word_embed)
    body_vec = sentence2vec_avg(body, word_embed)
    vec = np.vstack((title_vec, body_vec))
    n = vec.shape[0]
    if n > N:
        vec = vec[:N, :]
    elif n < N:
        pad = np.zeros((N-n, vec.shape[1]))
        vec = np.vstack((vec, pad))
    return vec



def average(x):
    return tf.keras.backend.mean(x, axis=-2)

def create_model(dims):
    n, N, wlen, opveclen = dims
    ip = tf.keras.layers.Input(shape=((n+1), N,wlen)) # query question + sample questions
    avg = tf.keras.layers.Lambda(average)
    l1 = tf.keras.layers.Dense(128, activation='relu')
    l2 = tf.keras.layers.Dense(opveclen)
    out = tf.keras.layers.Flatten()

    op = out(l2(l1(avg(ip))))

    model = tf.keras.models.Model(inputs=ip, outputs=op)
    return model


def generate_samples(qset, pos_set, neg_set, batch_inds, dims, question_id, word_embed):
    n, N, wlen, opveclen = dims
    # n : number of sample questions per query
    # N : number of words per sentence
    # wlen : length of word embedding vector
    # opveclen : length of output vector
    batch_size = len(batch_inds) 
    samples = np.zeros((batch_size, n, N, wlen))
    labels = np.zeros((batch_size, n+1, opveclen))
    ques = np.zeros((batch_size, N, wlen))
    r = 0
    not_batch_inds = [] 
    for i in batch_inds:
        q = qset[i] 
        j = 0
        ques[r, :, :] = question2vec(question_id[int(q)], N, word_embed)
        for pos in pos_set[i].split():
            p = int(pos)
            samples[r, j, :, :] = question2vec(question_id[int(p)], N, word_embed)
            labels[r, j, :] = 1
            j += 1
            if (j >= n):
                break
        if(j == 0):
            # logger.warn('No positive examples detected for example ' + str(i))
            r -= 1
            if not_batch_inds:
                not_batch_inds.append(i)
            else:
                not_batch_inds = [i]
            continue
        for neg in neg_set[i].split():
            if (j >= n):
                break
            ns = int(neg)
            samples[r, j, :, :] = question2vec(question_id[int(ns)], N, word_embed)
            labels[r, j, :] = 0
            j += 1
        while (j < n):
            samples[r, j, :, :] = question2vec(question_id[int(ns)], N, word_embed)
            labels[r, j, :] = 0
            j += 1
        r += 1
    batch_size = r
    batch_inds_new = [x for x in batch_inds if x not in not_batch_inds]
    labels = labels[:batch_size, :, :].reshape((batch_size, (n+1)*opveclen))
    ques = tf.convert_to_tensor(ques[:batch_size, :, :].reshape(batch_size, 1, N, wlen), dtype=tf.float32) # batch_size X N X wlen
    samples = tf.convert_to_tensor(samples[:batch_size, :,:,:], dtype=tf.float32) 
    labels = tf.convert_to_tensor(labels, dtype=tf.float32)
    data = tf.concat([ques, samples], axis=1)
    return (data, labels, batch_size, batch_inds_new)

def loss_fn_wrap(dims):
    def loss_fn(y_true, y_pred):
        n, N, wlen, opveclen = dims
        fq = []
        fp = []
        fq = tf.reshape(y_pred[:,0:opveclen], (-1, 1, opveclen))
        fp = tf.reshape(y_pred[:, opveclen:], (-1, n, opveclen))
        s1 = tf.keras.backend.sum(fp[:,:,:]*fq, axis=-1)
        s2 = s1/tf.keras.backend.sqrt(tf.keras.backend.sum(fq*fq, axis=-1))
        s = s2/tf.keras.backend.sqrt(tf.keras.backend.sum(fp*fp, axis = -1))
        delta = 0.25
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
        return ls
    return loss_fn

def fit_model(model, train_q, train_pos, train_neg,\
 batch_size, epochs, dims, question_id, word_embed, callbacks=[]):
    inds = range(len(train_q))
    num_iter = int(len(train_q)*epochs/batch_size)
    for k in range(num_iter):
        batch_inds = random.sample(inds, batch_size)
        inds = [x for x in inds if x not in batch_inds]
        data, labels, batch_size_curr, _ = generate_samples(train_q, train_pos,\
          train_neg, batch_inds, dims, question_id, word_embed)
        model.fit(data, labels, batch_size=batch_size_curr, \
         epochs=1, callbacks=callbacks)
        op = model.predict(data)
    return model
    
def post_process(test_file, model):
    test_q, test_pos, test_neg = read_question_data(test_file, num_fields=4)
    batch_inds = range(len(test_q))
    test_ip, test_labels, _, _ = generate_samples(test_q, test_pos, test_neg, batch_inds)
    test_op = model.predict(test_ip)
    print(test_op.shape)
    print(len(test_q))



def main():
    logger = Logger('log')

    logger.log('Reading in word: to word embedding -- mapping words to vectors...')
    data_folder = "data_folder/created_data"
    f = open(os.path.join(data_folder, "word_embed.pkl"), "rb")
    word_embed = pickle.load(f)
    f.close()
    
    logger.log('Reading in raw text (tokenized) -- question ID maps to question (title + body)...')

    f = open(os.path.join(data_folder, "question_id.pkl"), "rb")
    question_id = pickle.load(f)
    f.close()

    logger.log('Reading in training data -- query question ID, similar questions ID (pos), random questions ID (neg)...')

    train_q, train_pos, train_neg = read_question_data('data_folder/data/train_random.txt')
    logger.log('Creating Model ...')

    n = 120 # number of sample questions per query question
    N = 100 # number of words per question
    opveclen = 30
    wlen = len(word_embed['the'])
    dims = n, N, wlen, opveclen 
    model = create_model(dims)

    logger.log('Model inputs and outputs')
    loss_fn = loss_fn_wrap(dims)
    model.compile(optimizer='adam', loss=loss_fn)
    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    load_from_checkpoint = False
    if (load_from_checkpoint):
        model.load_weights(checkpoint_path)
    else:
        model = fit_model(model, train_q, train_pos, train_neg, \
         batch_size=10, epochs=3, dims=dims, \
         question_id=question_id, word_embed=word_embed, \
         callbacks=[cp_callback])

    # !mkdir -p saved_model
    model.save('saved_model/simple_nn1')
    post_process('data_folder/data/test.txt', model)

if __name__ == "__main__":
    main()
