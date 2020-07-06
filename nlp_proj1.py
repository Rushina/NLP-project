import random
import numpy as np
import tensorflow as tf
from playsound import playsound

class Logger(object):
    def warn(self, str):
        print('WARN: ', str)
    
    def log(self, str):
        print('Logging: ', str)

    def error(self, str):
        print('ERROR: ', str)

logger = Logger()

logger.log('Reading in word: to word embedding -- mapping words to vectors...')
f = open('all_corpora_vectors.txt', "r")
word_embed_raw = f.readlines()
f.close()

word_embed = dict()
i = 0
for w in word_embed_raw:
    data = w.split()
    if (data[1] == '1/2'):
        data[0] = '1 1/2'
        data[1:-1] = data[2:]
        data[1:].pop()
    word_embed[data[0]] = [float(x) for x in data[1:]]

logger.log('Reading in raw text (tokenized) -- question ID maps to question (title + body)...')

f = open('data/texts_raw_fixed.txt', "r")
raw_text_tokenized = f.readlines()
f.close()

question_id = dict()
for q in raw_text_tokenized:
    data = q.split('\t')
    question_id[int(data[0])] = data[1:]

logger.log('Reading in training data -- query question ID, similar questions ID (pos), random questions ID (neg)...')

f = open('data/train_random.txt', "r")
raw_train_data = f.readlines()
f.close()

train_q = list()
train_pos = list()
train_neg = list()

for td in raw_train_data:
    q, pos, neg = td.split('\t')
    train_q.append(q)
    train_pos.append(pos)
    train_neg.append(neg)

logger.log('Processing sentences into a single vector using word embedding...')


def word2vec(word):
    if not word in word_embed.keys():
        return np.zeros_like(word_embed['the'])
    return np.array(word_embed[word])

def sentence2vec_avg(sentence):
    words = sentence.split()
    n = len(words)
    vec = np.zeros((n, len(word_embed['the'])))
    for (i,w) in enumerate(words):
        vec[i] = word2vec(w)
    return vec 

def question2vec(question, N):
    title, body = question
    title_vec = sentence2vec_avg(title)
    body_vec = sentence2vec_avg(body)
    vec = np.vstack((title_vec, body_vec))
    n = vec.shape[0]
    if n > N:
        vec = vec[:N, :]
    elif n < N:
        pad = np.zeros((N-n, vec.shape[1]))
        vec = np.vstack((vec, pad))
    return vec


logger.log('Creating Model ...')


n = 120 # number of sample questions per query question
N = 100 # number of words per question
opveclen = 30
def average(x):
    return tf.keras.backend.mean(x, axis=-2)

wlen = len(word_embed['the'])
ip = tf.keras.layers.Input(shape=((n+1), N,wlen)) # query question + sample questions
avg = tf.keras.layers.Lambda(average)
l1 = tf.keras.layers.Dense(128, activation='relu')
l2 = tf.keras.layers.Dense(opveclen)
out = tf.keras.layers.Flatten()

op = out(l2(l1(avg(ip))))

model = tf.keras.models.Model(inputs=ip, outputs=op)

logger.log('Model inputs and outputs')

batch_size = 10
samples = np.zeros((batch_size, n, N, wlen))
labels = np.zeros((batch_size, n+1, opveclen))
ques = np.zeros((batch_size, N, wlen))

for i, q in enumerate(train_q):
    if (i >= batch_size):
        break
    j = 0
    ques[i, :, :] = question2vec(question_id[int(q)], N)
    for pos in train_pos[i].split():
        p = int(pos)
        samples[i, j, :, :] = question2vec(question_id[int(p)], N)
        labels[i, j, :] = 1
        j += 1
        if (j >= n):
            break
    for neg in train_neg[i].split():
        if (j >= n):
            break
        ns = int(neg)
        samples[i, j, :, :] = question2vec(question_id[int(ns)], N)
        labels[i, j, :] = 0
        j += 1
    while (j < n):
        samples[i, j, :, :] = question2vec(question_id[int(ns)], N)
        labels[i, j, :] = 0
        j += 1

labels = labels.reshape((batch_size, (n+1)*opveclen))
ques = tf.convert_to_tensor(ques.reshape(batch_size, 1, N, wlen), dtype=tf.float32) # batch_size X N X wlen
samples = tf.convert_to_tensor(samples, dtype=tf.float32) 
labels = tf.convert_to_tensor(labels, dtype=tf.float32)
data = tf.concat([ques, samples], axis=1)

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
    # logger.log(str(labels))
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
    # logger.log('inside loss function, true labels = '+ str(y_true))
    # logger.log('inside loss function, prediction = ' + str(y_pred))
    # logger.log('inside loss function, loss vector = ' + str(losst))
    # logger.log('inside loss function: '+ str(ls))
    return ls


model.compile(optimizer='adam',
              loss=loss_fn)

op_old = model.predict(data)
logger.log(str(op_old[0]))
# logger.log(str(loss_fn(labels, op_old)))

model.fit(data, labels, epochs=2)



def fit_model(model, batch_size, epochs):
    inds = range(len(train_q))
    num_iter = int(len(train_q)*epochs/batch_size)
    # logger.log('Number of iterations = '+ str(num_iter))
    for k in range(num_iter):
        batch_inds = random.sample(inds, batch_size)
        inds = [x for x in inds if x not in batch_inds]
        samples = np.zeros((batch_size, n, N, wlen))
        labels = np.zeros((batch_size, n+1, opveclen))
        ques = np.zeros((batch_size, N, wlen))
        r = 0
        for i in batch_inds:
            q = train_q[i] 
            j = 0
            ques[r, :, :] = question2vec(question_id[int(q)], N)
            for pos in train_pos[i].split():
                p = int(pos)
                samples[r, j, :, :] = question2vec(question_id[int(p)], N)
                labels[r, j, :] = 1
                j += 1
                if (j >= n):
                    break
            if(j == 0):
                logger.warn('No positive examples detected for example ' + str(i))
            for neg in train_neg[i].split():
                if (j >= n):
                    break
                ns = int(neg)
                samples[r, j, :, :] = question2vec(question_id[int(ns)], N)
                labels[r, j, :] = 0
                j += 1
            while (j < n):
                samples[r, j, :, :] = question2vec(question_id[int(ns)], N)
                labels[r, j, :] = 0
                j += 1
            # logger.log('Inside fit. r= ' + str(r) + 'labels = '+str(labels[r,:,:]))
            r += 1
        # logger.log('Inside model fit. Labels: '+ str(labels))
        labels = labels.reshape((batch_size, (n+1)*opveclen))
        ques = tf.convert_to_tensor(ques.reshape(batch_size, 1, N, wlen), dtype=tf.float32) # batch_size X N X wlen
        samples = tf.convert_to_tensor(samples, dtype=tf.float32) 
        labels = tf.convert_to_tensor(labels, dtype=tf.float32)
        data = tf.concat([ques, samples], axis=1)
        model.fit(data, labels, batch_size=batch_size, epochs=1)
        op = model.predict(data)
    return model
    
model = fit_model(model, 50, 1)
# model.fit(data, labels, batch_size=32, epochs=5)

op_new = model.predict(data)
logger.log('New output: '+ str(op_new[0]))

playsound('doorbell-1.wav')
