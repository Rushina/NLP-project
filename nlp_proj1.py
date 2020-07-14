import random
import numpy as np
import tensorflow as tf
import os
import pickle
from sample_generator import SampleGenerator
import argparse

class Logger(object):
    def __init__(self, mode='log'):
        self.mode = 0
        if (mode == 'error'):
            self.mode = 2
        elif (mode == 'warn'):
            self.mode = 1
        else:
            self.mode = 0
    def warn(self, string):
        if (self.mode < 2):
            print('WARN: ', string)
    
    def log(self, string):
        if (self.mode < 1):
            print('Logging: ', string)

    def error(self, string):
        print('ERROR: ', string)

    
def read_question_data(filename):
    f = open(filename, "r")
    raw_data = f.readlines()
    f.close()
    q = list()
    pos = list()
    neg = list()
    for td in raw_data:
        qi, posi, negi = [], [], []
        if (len(td.split('\t')) == 3):
            qi, posi, negi = td.split('\t')
        elif (len(td.split('\t')) == 4):
            qi, posi, negi, _ = td.split('\t')
        posi_list = posi.split()
        for pi in posi_list:
            q.append(qi)
            pos.append(pi)
            neg.append(negi)
    return (q, pos, neg)

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

def loss_fn_wrap(dims):
    def loss_fn(y_true, y_pred):
        n, N, wlen, opveclen = dims
        s = similarity(y_pred, dims)
        delta = 0.25
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
        ls = tf.keras.backend.max(tf.keras.backend.max(losst, axis=-1), axis=-1)
        return ls
    return loss_fn

def fit_model_single_data_point(model, sample_gen, batch_ind, epochs, dims, logger):
    print("Training with a single data point ", batch_ind)
    loss_fn = loss_fn_wrap(dims)
    loss = 0.25
    k = 0
    
    batch_inds = [batch_ind]
    data, labels, batch_size_curr, _ = sample_gen.generate_samples(batch_inds)
    print("Labels = ", tf.reshape(labels, (batch_size_curr, 121, -1))[0, :, 0])
    print("Current batch size = ", batch_size_curr)
    if (batch_size_curr == 0):
        logger.error("Data point provided has no positive examples. Please run again with valid data point.")
        return
    while (k < epochs):
        print(k)
        model.fit(data, labels, batch_size=batch_size_curr, epochs=1)
        op = model.predict(data)
        loss = loss_fn(labels, op)
        k += 1

    print("Loss of output: ", loss_fn(labels, op))
    sim = np.array(similarity(op, dims))
    print(sim.shape)
    sim_inds = np.argsort(-sim, axis=1)
    pos_i = sample_gen.pos_set[batch_ind].split()
    print(len(pos_i))
    neg_i = sample_gen.neg_set[batch_ind].split()
    for i, ind in enumerate(sim_inds[0]):
        if (ind < len(pos_i)):
            print("Rank: ", i+1, " positive. Similarity: ", sim[0, ind])
            # print(question_id[int(pos_i[ind])])
        elif (ind < len(pos_i) + len(neg_i)):
            print("Rank: ", i+1, "negative. Similarity: ", sim[0, ind])
            # print(question_id[int(neg_i[ind-len(pos_i)])])
        else:
            print("Rank: ", i+1, "negative. Similarity: ", sim[0, ind])
            # print(question_id[int(neg_i[len(neg_i)-1])])
    return model 

def fit_model(model, sample_gen, batch_size, epochs, dims, callbacks=[], num_data=[]):
    loss_fn = loss_fn_wrap(dims)
    if (num_data == []):
        num_data = len(sample_gen.qset)
    num_iter = int(num_data/batch_size)
    for e in range(epochs):
        print("Actual epoch = ", (e+1),"/", (epochs))
        inds = range(num_data)
        for k in range(num_iter):
            batch_inds = [] 
            if (len(inds) >= batch_size):
                batch_inds = random.sample(inds, batch_size)
            elif (len(inds) == 0):
                break
            else:
                batch_inds = inds
            inds = [x for x in inds if x not in batch_inds]
            data, labels, batch_size_curr, _ = sample_gen.generate_samples(batch_inds)
            model.fit(data, labels, batch_size=batch_size_curr, \
             epochs=1, callbacks=callbacks, verbose=0)
        op = model.predict(data)
        sim = similarity(op, dims)
        curr_loss = tf.keras.backend.mean(loss_fn(labels, op))
        print("Current loss = ", curr_loss, " at the end of epoch ", e+1, ".")
        # if ((curr_loss) < 0.05):
        #    return model 
    return model
    

def main():

    # Arguments to main
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_type", type=str, required=True, \
                        help="Training data set type, options = \
                        single batch_ind, short num_data, full, load_from_ckp ckp_path")
    
    parser.add_argument("-e", "--epochs", type=int, default=20, \
                        help="-e or --epochs number of epochs (default 20)")

    parser.add_argument("-bs", "--batch_size", type=int, default=10, \
     help="-bs or --batch_size training batch size (default 10)")

    parser.add_argument("-s", "--save_model", type=str, \
                        help="-s save_path or --save_model save_path (save_path is relative from current dir)")
    
    parser.add_argument("-cp", "--checkpoint", type=str, \
                        help="-cp or --checkpoint check_point_path to save checkpoints -- relative path")

    args = parser.parse_args()
    
    train_type, train_opts = args.train_type.split()


    logger = Logger('log')

    # Reading in data
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

    # Creating model    
    logger.log('Creating Model ...')

    n = 120 # number of sample questions per query question
    N = 100 # number of words per question
    opveclen = 30
    wlen = len(word_embed['the'])
    dims = n, N, wlen, opveclen 
    model = create_model(dims)
    train_sample_generator = SampleGenerator(question_id, train_q, train_pos, train_neg, dims, word_embed)

    logger.log('Model inputs and outputs')
    loss_fn = loss_fn_wrap(dims)
    model.compile(optimizer='adam', loss=loss_fn)


    # Training
    cp_callback = []
    if (args.checkpoint):
        cp_callback = tf.keras.callbacks.ModelCheckpoint(\
            filepath=args.checkpoint,verbose=1,save_weights_only=True)

    
    if (train_type == "load_from_ckp"):
        model.load_weights(train_opts)
    elif train_type=="single":
        batch_ind = int(train_opts)
        model = fit_model_single_data_point(model, train_sample_generator, epochs=args.epochs, batch_ind = batch_ind, dims=dims, logger=logger)
    elif train_type=="short":
        num_data = int(train_opts) 
        model = fit_model(model, train_sample_generator, \
         batch_size=args.batch_size, epochs=args.epochs, dims=dims, \
         callbacks=cp_callback, num_data = num_data)
    elif train_type == "full":    
        model = fit_model(model, train_sample_generator, \
         batch_size=args.batch_size, epochs=args.epochs, dims=dims, \
         callbacks=[], num_data = num_data)
    else:
        print("Invalid train type entered.")

    # !mkdir -p saved_model
    if (args.save_model):
        model.save(args.save_model)
    # post_process('data_folder/data/test.txt', model)

if __name__ == "__main__":
    main()
