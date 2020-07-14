import tensorflow as tf
import numpy as np
import os
from nlp_proj1 import read_question_data, loss_fn_wrap, similarity
from sample_generator import SampleGenerator
import pickle
import random
import argparse

def rank(y_pred, dims):
    # Takes in an output of size (batch, (n+1)*opvec)
    # Reshaped to (batch, n+1, opvec)
    # The first row represents the vector representing the query question
    # The sencond - last rows represent the sample questions 
    # We rank the sample questions based on their similarity to the query question
    # Most similar sample ranked first
    
    n, N, wlen, opveclen = dims
    s = np.array(similarity(y_pred, dims))
    x = np.argsort(-s, axis=1)
    r = np.argsort(x, axis=1) + 1 # indexing by 1
    return (x,r)

def mrr(y_pred, labels, dims):
    n, N, wlen, opveclen = dims
    _,r = rank(y_pred, dims)
    rtemp = (2*n)*np.ones_like(r)
    rtemp[labels.astype(bool)] = r[labels.astype(bool)]
    lowest_relevant_rank = rtemp.min(axis=1)
    mrr = np.sum(1/lowest_relevant_rank)/len(lowest_relevant_rank)
    return mrr 

def precision_at_k(y_pred, labels, dims, k):
    n, N, wlen, opveclen = dims
    x, _ = rank(y_pred, dims)
    ordered_labels = np.zeros_like(labels)
    for (i, lab) in enumerate(labels):
        ordered_labels[i, :] = labels[i, x[i,:]]
    precision = np.sum(ordered_labels[:, :k])/(k*labels.shape[0])
    return precision 

def map(y_pred, labels, dims):
    sorted_inds, _ = rank(y_pred, dims)
    map_data = 0.0 
    num_data = y_pred.shape[0]
    for i, a in enumerate(sorted_inds):
        cnt = 0
        ap = 0.0
        for j, aj in enumerate(a):
            if (labels[i, j]):
                ap += np.sum(labels[i,a[0]:aj+1])/(j+1)
                cnt += 1
        map_data += ap/cnt
    return map_data/num_data 

def get_metrics(model, sample_gen, dims, metrics = ['mrr', 'pan1', 'pan5', 'map'], num_data=[]):
    n, N, wlen, opveclen = dims
    print("Number of data points: ", len(sample_gen.qset))
    if (num_data == []):
        num_data = len(q)
    res = dict()
    batch_size = min(10, num_data-1)
    i = 0
    mrr_data = 0.0
    prec1_data = 0.0
    prec5_data = 0.0
    total_pos = 0
    map_data = 0.0
    cnt_data_points = 0
    while (i < num_data-batch_size):
        print('Analyzing data points ', i, ' through ', i+batch_size, '...')
        data, labels, batch_size_curr, _ = sample_gen.generate_samples( \
         range(i, i + batch_size))
        labels = np.reshape(labels, (-1, (n+1), opveclen))[:, :-1, 0]
        if not batch_size_curr == 0:
            op = np.array(model.predict(data))
            data = np.array(data)
            labels = np.array(labels)
            total_pos += np.sum(labels)
            if ('mrr' in metrics):
                mrr_data += mrr(op, labels, dims)*batch_size_curr
            if ('pan1' in metrics):
                prec1_data += precision_at_k(op, labels, dims, 1)*batch_size_curr
            if ('pan5' in metrics):
                prec5_data += precision_at_k(op, labels, dims, 5)*batch_size_curr
            if ('map' in metrics):
                map_data += map(op, labels, dims)*batch_size_curr
        i += batch_size
        cnt_data_points += batch_size_curr
    res['mrr'] = mrr_data/cnt_data_points
    res['pan1'] = prec1_data/cnt_data_points
    res['pan5'] = prec5_data/cnt_data_points
    res['map'] = map_data/cnt_data_points
    print('This data set has ', total_pos, ' number of positive examples for ', num_data, ' query questions.')
    return res

def verify_samples(sample_gen, model, dims, batch_size=10, num_pos=5, randomly=True):
    n, N, wlen, opveclen = dims
    loss_fn = loss_fn_wrap(dims)
    batch_inds = random.sample(range(len(sample_gen.qset)), batch_size)
    if not randomly:
        batch_inds = range(batch_size)
    data, labels, batch_size_curr, batch_inds_curr = \
            sample_gen.generate_samples(batch_inds)
    op = model.predict(data)
    print("Loss = ", loss_fn(labels, op))
    labels = tf.reshape(labels, (batch_size_curr, n+1, -1))[:, :, 0] 
    sim = similarity(op, dims)
    sim_inds, _ = rank(op, dims)

    for bi, b in enumerate(batch_inds_curr):
        print("--------------- Data point ", bi, " ---------------")
        pos_i = sample_gen.pos_set[b].split()
        neg_i = sample_gen.neg_set[b].split()
        for i, ind in enumerate(sim_inds[bi]):
            if (i >= num_pos):
                break
            if (ind < len(pos_i)):
                print("Rank: ", i+1, " POSITIVE. Similarity: ", sim[bi, ind])
                # print(question_id[int(pos_i[ind])])
            elif (ind < len(pos_i) + len(neg_i)):
                print("Rank: ", i+1, "negative. Similarity: ", sim[bi, ind])
                # print(question_id[int(neg_i[ind-len(pos_i)])])
            else:
                print("Rank: ", i+1, "negative. Similarity: ", sim[bi, ind])
                # print(question_id[int(neg_i[len(neg_i)-1])])
    
def main():

    # Arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-d", "--data_set", type=str, required=True, \
                        choices = ["train", "dev", "test"], \
                        help="Specify if evaluation to be done on train/dev/test")

    parser.add_argument("-m", "--model", type=str, required=True, \
                        help="Model path")

    parser.add_argument("-n", "--num_data", type=int, \
                        help="Number of data points to evaluate")

    parser.add_argument("-po", "--print_outputs", type=str, default="1 1", \
                        help="-po batch_size num_pos outputs")

    parser.add_argument("-met", "--metrics", type=bool, default=True, \
                        help="-met (True/False) to print metrics")

    args = parser.parse_args()    
    model_path = args.model
    num_data = []
    if (args.num_data):
        num_data = args.num_data
    data_set = args.data_set
    batch_size, num_pos = args.print_outputs.split()
    batch_size = int(batch_size)
    num_pos = int(num_pos)

    f = open("data_folder/created_data/word_embed.pkl", "rb") 
    word_embed = pickle.load(f)
    f.close()

    f = open("data_folder/created_data/question_id.pkl", "rb") 
    question_id = pickle.load(f)
    f.close()

    n = 120
    N = 100
    opveclen = 30
    wlen = len(word_embed['the'])
    dims = n, N, wlen, opveclen
    loss_fn = loss_fn_wrap(dims)

    model = tf.keras.models.load_model(model_path, \
     custom_objects={'loss': loss_fn_wrap(dims)}, compile=False)

    model.compile(optimizer='adam', loss=loss_fn)

    loaded_data = (word_embed, question_id, model)



    data_folder = 'data_folder/data'
    train_file = 'train_random.txt'
    dev_file = 'dev.txt'
    test_file = 'heldout.txt'

    dev_path = os.path.join(data_folder, dev_file)
    test_path = os.path.join(data_folder, test_file)
    train_path = os.path.join(data_folder, train_file)

    sample_gen = dict()

    train_q, train_pos, train_neg = read_question_data(train_path)
    sample_gen["train"] = SampleGenerator(question_id, train_q, train_pos, \
            train_neg, dims, word_embed)

    dev_q, dev_pos, dev_neg = read_question_data(dev_path)
    sample_gen["dev"] = SampleGenerator(question_id, dev_q, dev_pos, \
            dev_neg, dims, word_embed)

    test_q, test_pos, test_neg = read_question_data(test_path)
    sample_gen["test"] = SampleGenerator(question_id, test_q, test_pos, \
            test_neg, dims, word_embed)

    if (args.metrics):
        print(get_metrics(model, sample_gen[data_set], dims, num_data=num_data)) 

    verify_samples(sample_gen[data_set], model, dims, batch_size=batch_size, num_pos=num_pos, randomly=False) 


if __name__ == "__main__":
    main()
