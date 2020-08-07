from transfer_learning_samples import AndroidSampleGenerator
from sample_generator import DataStore, SampleGenerator
import pickle
import os
import tensorflow as tf
import argparse
from nlp_proj1 import create_model
from post_process import get_metrics
from read_question_data import read_question_data

def create_data_objs():

    android_data_folder = 'Android-master'
    f = open(os.path.join(android_data_folder, "corpus.pkl"), "rb")
    android_question_id = pickle.load(f)
    f.close()

    data_folder = "data_folder/created_data"
    f = open(os.path.join(data_folder, "word_embed.pkl"), "rb")
    word_embed = pickle.load(f)
    f.close()

    f = open(os.path.join(data_folder, "question_id.pkl"), "rb")
    ubuntu_question_id = pickle.load(f)
    f.close()

    n = 120
    N = 100
    opveclen = 100
    wlen = len(word_embed['the'])
    dims = n, N, wlen, opveclen

    f = open(os.path.join(android_data_folder, "dev_pos.pkl"), "rb")
    android_dev_pos = pickle.load(f)
    f.close()

    f = open(os.path.join(android_data_folder, "dev_neg.pkl"), "rb")
    android_dev_neg = pickle.load(f)
    f.close()

    f = open(os.path.join(android_data_folder, "test_pos.pkl"), "rb")
    android_test_pos = pickle.load(f)
    f.close()

    f = open(os.path.join(android_data_folder, "test_neg.pkl"), "rb")
    android_test_neg = pickle.load(f)
    f.close()

    train_q, train_pos, train_neg = read_question_data('data_folder/data/train_random.txt')
    dev_q, dev_pos, dev_neg = read_question_data('data_folder/data/dev.txt')
    test_q, test_pos, test_neg = read_question_data('data_folder/data/test.txt')

    android_data_obj = DataStore(android_question_id, word_embed)
    ubuntu_data_obj = DataStore(ubuntu_question_id, word_embed)

    android_dev_generator = AndroidSampleGenerator(android_dev_pos,\
     android_dev_neg, dims, android_data_obj)
    android_test_generator = AndroidSampleGenerator(android_test_pos,\
     android_test_neg, dims, android_data_obj)

    
    ubuntu_train_generator = SampleGenerator(train_q, train_pos, train_neg, dims, ubuntu_data_obj)
    ubuntu_dev_generator = SampleGenerator(dev_q, dev_pos, dev_neg, dims, ubuntu_data_obj)
    ubuntu_test_generator = SampleGenerator(test_q, test_pos, test_neg, dims, ubuntu_data_obj)
    return (dims, android_data_obj, android_dev_generator, android_test_generator, ubuntu_data_obj, ubuntu_train_generator, ubuntu_dev_generator, ubuntu_test_generator)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-load", type=str, required=True, \
            help='load ckp_path (str) to load model from checkpoint')
 

    args = parser.parse_args()

    
    dims, android_data_obj, android_dev_generator, android_test_generator, \
        ubuntu_data_obj, ubuntu_train_generator, ubuntu_dev_generator,\
        ubuntu_test_generator = create_data_objs()

    model = create_model(dims, android_data_obj.embedding_matrix)
    model.load_weights(args.load)

    
    print("------------- UBUNTU DEV METRICS ----------------")
    print(get_metrics(model, ubuntu_dev_generator, dims, verbose=False)) 

    print("------------- UBUNTU TEST METRICS ----------------")
    print(get_metrics(model, ubuntu_test_generator, dims, verbose=False)) 
    
    print("------------- ANDROID DEV METRICS ----------------")
    print(get_metrics(model, android_dev_generator, dims, verbose=False)) 

    print("------------- ANDROID TEST METRICS ----------------")
    print(get_metrics(model, android_test_generator, dims, verbose=False)) 

if __name__ == "__main__":
    main()
