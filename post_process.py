import tensorflow as tf
import numpy as np
import os
from nlp_proj1 import read_question_data, generate_samples, loss_fn_wrap
import pickle

def rank(y_pred, dims):
    # Takes in an output of size (batch, (n+1)*opvec)
    # Reshaped to (batch, n+1, opvec)
    # The first row represents the vector representing the query question
    # The sencond - last rows represent the sample questions 
    # We rank the sample questions based on their similarity to the query question
    # Most similar sample ranked first
    
    n, N, wlen, opveclen = dims
    fq = []
    fp = []
    fq = np.reshape(y_pred[:,0:opveclen], (-1, 1, opveclen))
    fp = np.reshape(y_pred[:, opveclen:], (-1, n, opveclen))
    s1 = np.sum(fp[:,:,:]*fq, axis=-1)
    s2 = s1/np.sqrt(np.sum(fq*fq, axis=-1))
    s = s2/np.sqrt(np.sum(fp*fp, axis = -1))
    r = np.argsort(-s, axis=1)
    r = np.argsort(r, axis=1) + 1 # indexing by 1
    return r

def mrr(y_pred, labels, dims):
    n, N, wlen, opveclen = dims
    labels = np.reshape(labels, (-1, (n+1), opveclen))[:, :-1, 0]
    r = rank(y_pred, dims)
    rtemp = (2*n)*np.ones_like(r)
    rtemp[labels.astype(bool)] = r[labels.astype(bool)]
    lowest_relevant_rank = rtemp.min(axis=1)
    mrr = np.sum(1/lowest_relevant_rank)/len(lowest_relevant_rank)
    return mrr 

def get_mrr(data_path, dims, loaded_data):
    word_embed, question_id, model = loaded_data
    q, pos, neg = read_question_data(data_path, 4)

    batch_size = 10
    i = 0
    mrr_data = 0.0
    while (i < len(q)):
        print('Analyzing data points ', i, ' through ', i+batch_size, '...')
        data, labels = generate_samples(q, pos, neg,\
         range(i, i + batch_size), dims, question_id, word_embed)
        op = np.array(model.predict(data))
        data = np.array(data)
        labels = np.array(labels)
        mrr_data += mrr(op, labels, dims)*batch_size
        i += batch_size
    return mrr_data/float(len(q))

def main():
    model_name = 'simple_nn1'
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
    dev_file = 'dev.txt'
    test_file = 'test.txt'

    dev_path = os.path.join(data_folder, dev_file)
    test_path = os.path.join(data_folder, test_file)

    print(get_mrr(dev_path, dims, loaded_data))

    # print(get_mrr(test_path, dims, loaded_data))

if __name__ == "__main__":
    main()
