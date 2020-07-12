import tensorflow as tf
import numpy as np
import os
from nlp_proj1 import read_question_data, generate_samples, loss_fn_wrap, similarity
import pickle
import random

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
    precision = np.sum(labels[:, :k])/(k*labels.shape[0])
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

def get_metrics(data_path, dims, loaded_data, metrics = ['mrr', 'pan1', 'pan5', 'map'], num_data=[]):
    n, N, wlen, opveclen = dims
    word_embed, question_id, model = loaded_data
    q, pos, neg = read_question_data(data_path)
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
        data, labels, batch_size_curr, _ = generate_samples(q, pos, neg,\
         range(i, i + batch_size), dims, question_id, word_embed)
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
    print('This data set has ', total_pos, ' number of positive examples for ', len(q), ' query questions.')
    return res

def verify_samples(data_path, dims, loaded_data, batch_size=10, num_pos=5, metrics = ['mrr', 'pan1', 'pan5', 'map'], randomly=True):
    n, N, wlen, opveclen = dims
    word_embed, question_id, model = loaded_data
    q, pos, neg = read_question_data(data_path)
    batch_inds = random.sample(range(len(q)), batch_size)
    if not randomly:
        batch_inds = range(batch_size)
    data, labels, batch_size_curr, batch_inds_curr = generate_samples(q, pos, neg,\
     batch_inds, dims, question_id, word_embed)
    print(batch_size_curr)
    print(data.shape)
    op = model.predict(data)
    x, _ = rank(op, dims) 
    print(x.shape, len(batch_inds_curr))
    for i, b in enumerate(batch_inds_curr):
        top_inds = x[i, :num_pos] 
        print("================== Query Question =======================")
        print(question_id[int(q[i])])
        pos_i = pos[i].split()
        neg_i = neg[i].split()
        for j, ind in enumerate(top_inds):
            print("======= Similar question number ", j+1, " ======")
            if (ind < len(pos_i)):
                print("positive")
                print(question_id[int(pos_i[ind])])
            elif (ind < len(pos_i) + len(neg_i)):
                print("negative")
                print(question_id[int(neg_i[ind-len(pos_i)])])
            else:
                print("negative")
                print(question_id[int(neg_i[len(neg_i)-1])])
        
    
def main():
    model_name = 'simple_nn3'
    model_dir = 'saved_model'
    model_path = os.path.join(model_dir, model_name)

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
    test_file = 'test.txt'

    dev_path = os.path.join(data_folder, dev_file)
    test_path = os.path.join(data_folder, test_file)
    train_path = os.path.join(data_folder, train_file)

    # print(get_metrics(dev_path, dims, loaded_data))
    print(get_metrics(train_path, dims, loaded_data, num_data=100))
    verify_samples(train_path, dims, loaded_data, batch_size=10, num_pos=1, randomly=False) 

    # print(get_mrr(test_path, dims, loaded_data))

if __name__ == "__main__":
    main()
