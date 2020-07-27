import random
import numpy as np
import tensorflow as tf
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from post_process import get_metrics
from sample_generator import SampleGenerator
from loss_function import similarity, loss_fn_wrap, loss_fn_wrap2
from read_question_data import read_question_data

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

def average(x):
    return tf.keras.backend.mean(x, axis=-2)

def print_model_weights(model, layers = [4]):
    for i in layers:
        layer = model.layers[i]
        print("Layer number ", i, " type : ", layer.__class__.__name__)
        w, b = layer.get_weights()
        print("Weights: ", w)
        # print("Biases: ", b)
 
def create_model(dims):
    n, N, wlen, opveclen = dims
    ip = tf.keras.layers.Input(shape=((n+1), N,wlen)) # query question + sample questions

    l1 = tf.keras.layers.Dense(150, activation='tanh')
    l2 = tf.keras.layers.Dense(300, activation='tanh') 
    avg = tf.keras.layers.Lambda(average)
    l3 = tf.keras.layers.Dense(opveclen, activation='tanh')
    out = tf.keras.layers.Flatten()

    op = out(l3(avg(l2(l1(ip)))))

    model = tf.keras.models.Model(inputs=ip, outputs=op)
    return model

def fit_model_single_data_point(model, sample_gen, batch_ind, epochs, dims, logger):
    print("Training with a single data point ", batch_ind)
    loss_fn = loss_fn_wrap2(dims)
    loss = 0.25
    k = 0
    
    loss_over_epochs = []
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
        if (loss_over_epochs == []):
            loss_over_epochs = [loss]
        else:
            loss_over_epochs.append(loss)
        k += 1
        print_model_weights(model) 

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
    return (model, loss_over_epochs)

def fit_model(model, sample_gen, dev_sample_gen, batch_size, epochs, dims, checkpoint_path, num_data=[]):
    loss_fn = loss_fn_wrap2(dims)
    loss_fn_orig = loss_fn_wrap(dims)
    if (num_data == []):
        num_data = len(sample_gen.qset)
    num_iter = int(num_data/batch_size)
    loss_over_epochs = []
    if not (checkpoint_path == 0):
        if not (os.path.exists(checkpoint_path)):
            os.makedirs(checkpoint_path)
        elabel = 0
        while (os.path.exists(os.path.join(checkpoint_path, "e"+str(elabel)))):
            elabel += 1
    for e in range(epochs):
        print("Actual epoch = ", (e+1),"/", (epochs))
        inds = range(num_data)
        epoch_loss = 0.0
        if not checkpoint_path == 0:
            cp_path = os.path.join(checkpoint_path, "e"+str(elabel))
            cp_callback = tf.keras.callbacks.ModelCheckpoint(\
                filepath=cp_path,verbose=0,save_weights_only=True)
        else:
            cp_callback = []
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
             epochs=1, callbacks=cp_callback, verbose=0)
            op = model.predict(data)
            sim = similarity(op, dims)
            curr_loss = np.asscalar(np.array(tf.keras.backend.mean(loss_fn(labels, op))))
            epoch_loss += curr_loss
        epoch_loss /= (k+1)
        print("Current loss = ", epoch_loss, "at the end of epoch ", e+1, ".")
        print("Dev metrics = ", get_metrics(model, dev_sample_gen, dims, verbose=False), \
            " at the end of epoch ", e+1, ".")
        elabel += 1
        if (loss_over_epochs == []):
            loss_over_epochs = [curr_loss]
        else:
            loss_over_epochs.append(curr_loss)
        # if ((curr_loss) < 0.05):
        #    return model 
    return (model, loss_over_epochs)
    

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

    parser.add_argument("-pl", "--print_loss", type=str, \
            help="-pl or --print_loss plot_save_path (str) to save loss vs epochs")

    parser.add_argument("-load", type=str, \
            help="-load ckp_path (str) to load model from checkpoint before training")

    args = parser.parse_args()
    train_type, train_opts = " ", " "
    
    if (len(args.train_type.split()) == 2):
        train_type, train_opts = args.train_type.split()
    else:
        train_type = args.train_type.split()[0]


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
    dev_q, dev_pos, dev_neg = read_question_data('data_folder/data/dev.txt')

    # Creating model    
    logger.log('Creating Model ...')

    n = 120 # number of sample questions per query question
    N = 100 # number of words per question
    opveclen = 100
    wlen = len(word_embed['the'])
    dims = n, N, wlen, opveclen 
    model = create_model(dims)
    train_sample_generator = SampleGenerator(question_id, train_q, train_pos, train_neg, dims, word_embed)
    dev_sample_generator = SampleGenerator(question_id, dev_q, dev_pos, dev_neg, dims, word_embed)

    logger.log('Model inputs and outputs')
    loss_fn = loss_fn_wrap2(dims)
    model.compile(optimizer='adam', loss=loss_fn)
    if (args.load):
        model.load_weights(args.load)


    # Training
    cp_path = 0
    if (args.checkpoint):
        cp_path = args.checkpoint
    
    if (train_type == "load_from_ckp"):
        model.load_weights(train_opts)
        print("Model loaded")
    elif train_type=="single":
        batch_ind = int(train_opts)
        model, loss_over_epochs = fit_model_single_data_point(model, train_sample_generator, epochs=args.epochs, batch_ind = batch_ind, dims=dims, logger=logger)
    elif train_type=="short":
        num_data = int(train_opts) 
        model, loss_over_epochs = fit_model(model, train_sample_generator, dev_sample_generator,\
         batch_size=args.batch_size, epochs=args.epochs, dims=dims, \
         checkpoint_path=cp_path, num_data = num_data)
    elif train_type == "full":    
        model, loss_over_epochs = fit_model(model, train_sample_generator, dev_sample_generator,\
         batch_size=args.batch_size, epochs=args.epochs, dims=dims, \
         checkpoint_path=cp_path)
    else:
        print("Invalid train type entered.")

    # !mkdir -p saved_model
    if (args.save_model):
        model.save(args.save_model)

    if (args.print_loss):
        plt.plot(loss_over_epochs)
        plt.show()
        plt.savefig(args.print_loss, format='png')
    # post_process('data_folder/data/test.txt', model)

if __name__ == "__main__":
    main()
