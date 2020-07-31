import tensorflow as tf
import numpy as np
import os
from loss_function import loss_fn_wrap, similarity
from sample_generator import SampleGenerator
from read_question_data import read_question_data
import pickle
import random
import argparse
from tqdm import tqdm

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
            if (labels[i, aj]):
                ap += float(np.sum(labels[i,a[0:j+1]]))/(j+1)
                cnt += 1
        map_data += float(ap)/cnt
    return map_data/num_data 

def get_metrics(model, sample_gen, dims, metrics = ['mrr', 'pan1', 'pan5', 'map'], num_data=[], verbose=True):
    n, N, wlen, opveclen = dims
    if (num_data == []):
        num_data = len(sample_gen.qset)
    res = dict()
    batch_size = min(10, num_data)
    i = 0
    mrr_data = 0.0
    prec1_data = 0.0
    prec5_data = 0.0
    total_pos = 0
    map_data = 0.0
    cnt_data_points = 0
    pbar = tqdm(total = num_data - batch_size + 2)
    while (i < num_data-batch_size+1):
        if (verbose):
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
        pbar.update(batch_size)
        i += batch_size
        cnt_data_points += batch_size_curr
    pbar.close()
    res['mrr'] = mrr_data/cnt_data_points
    res['pan1'] = prec1_data/cnt_data_points
    res['pan5'] = prec5_data/cnt_data_points
    res['map'] = map_data/cnt_data_points
    # print('This data set has ', total_pos, ' number of positive examples for ',cnt_data_points , ' query questions.')
    return res

def verify_inputs(sample_gen, dims, batch_ind):
    n, N, wlen, opveclen = dims
    qset = sample_gen.qset[batch_ind]
    pos_set = sample_gen.pos_set[batch_ind]
    neg_set = sample_gen.neg_set[batch_ind]

    # print(pos_set)
    # print(neg_set)
    for i, pos in enumerate(pos_set.split()):
        if (pos in neg_set.split()):
            print(i, pos) 
    data, labels, batch_size_curr, batch_inds_curr = \
            sample_gen.generate_samples([batch_ind])
    labels = tf.reshape(labels, (batch_size_curr, (n+1), -1))[:,:,0]

    q = int(qset)
    direct = sample_gen.question2vec(sample_gen.question_id[q])
    from_data = data[0, 0, :, :]
    print("Question data correct = ", tf.keras.backend.all(direct == from_data))
    i = 0
    for pos in pos_set.split():
        if (i >= n):
            break
        direct = sample_gen.question2vec(sample_gen.question_id[int(pos)])
        from_data = data[0, i+1, :, :]
        print("Data ", i, " correct = ", tf.keras.backend.all(direct == from_data))
        print("Label ", i, " correct = ", labels[0, i] == 1.0)
        i += 1
    for neg in neg_set.split():
        if (i >= n):
            break
        direct = sample_gen.question2vec(sample_gen.question_id[int(neg)])
        from_data = data[0, i+1, :, :]
        print("Data ", i, " correct = ", tf.keras.backend.all(direct == from_data))
        print("Label ", i, " correct = ", labels[0, i] == 0.0)
        i += 1
        
    return
 
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
    sim = similarity(op, dims)
    sim_inds, _ = rank(op, dims)

    op = tf.reshape(op, (batch_size_curr, n+1, -1))
    labels = tf.reshape(labels, (batch_size_curr, n+1, -1))[:, :, 0] 

    for bi, b in enumerate(batch_inds_curr):
        print("--------------- Data point ", bi, " ---------------")
        pos_i = sample_gen.pos_set[b].split()
        neg_i = sample_gen.neg_set[b].split()
        op_i = op[bi, :, :]

        print("Question : ")
        print(sample_gen.question_id[int(sample_gen.qset[b])])
        for i, ind in enumerate(sim_inds[bi]):
            if (i >= num_pos):
                break
            if (labels[bi, ind] == 1):
                print("Rank: ", i+1, " POSITIVE. Similarity: ", sim[bi, ind])
                print("Positive Question : ")
                if (ind < len(pos_i)):
                    print(sample_gen.question_id[int(pos_i[ind])])
                else:
                    print(sample_gen.question_id[int(pos_i[-1])])
                    
            else:
                print("Rank: ", i+1, "negative. Similarity: ", sim[bi, ind])
                print("Negative Question : ")
                if (ind < len(pos_i) + len(neg_i)):
                    print(sample_gen.question_id[int(neg_i[ind-len(pos_i)])])
                else:
                    print(sample_gen.question_id[int(neg_i[-1])])
    
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

    parser.add_argument("-met", "--metrics", action='store_false', \
                        help="-met (True/False) to print metrics")

    parser.add_argument("-ver_s", "--verify_samples", action='store_false', \
                        help="-met (True/False) to print metrics")

    parser.add_argument("-ver_i", "--verify_input", action='store_false', \
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
    opveclen = 100 
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

    if (args.verify_samples):
        verify_samples(sample_gen[data_set], model, dims, batch_size=batch_size, num_pos=num_pos, randomly=False) 

    if (args.verify_input):
        verify_inputs(sample_gen[data_set], dims, batch_ind=1)

if __name__ == "__main__":
    main()
