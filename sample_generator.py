import tensorflow as tf
import numpy as np
import random

def list_append(ls, elem):
    if (ls == []):
        ls = [elem]
    else:
        ls.append(elem)
    return ls

class DataStore(object):
    def __init__(self, question_id, word_embed):
        self.question_id = question_id
        self.word_embed = word_embed
        self.word_inds = dict()
        num_words = len(self.word_embed.keys())
        wlen = len(word_embed[list(self.word_embed.keys())[0]])
        self.embedding_matrix = np.zeros((num_words, wlen))
        self.create_embedding_matrix()

    def create_embedding_matrix(self):
        i = 0
        for word in self.word_embed.keys():
            self.word_inds[word] = i
            self.embedding_matrix[i, :] = self.word_embed[word]
            i += 1

class SampleGenerator(object):
    def __init__(self, qset, pos_set, neg_set, dims, data_obj):
        self.n, self.N, self.wlen, self.opveclen = dims
        self.word_embed = data_obj.word_embed
        self.question_id = data_obj.question_id
        self.word_inds = data_obj.word_inds
        self.qset, self.pos_set, self.neg_set = qset, pos_set, neg_set

    def word2vec(self, word):
        if not word in self.word_embed.keys():
            return np.zeros_like(self.word_embed['the'])
        return np.array(self.word_embed[word])

    def sentence2vec(self, sentence):
        words = sentence.split()
        n = len(words)
        vec = np.zeros((n, ))
        for (i,w) in enumerate(words):
            if w in self.word_inds.keys():
                vec[i] = self.word_inds[w] 
        return vec 

    def question2vec(self, question):
        title, body = question
        title_vec = self.sentence2vec(title)
        body_vec = self.sentence2vec(body)
        vec = np.hstack((title_vec, body_vec))
        n = vec.shape[0]
        if n > self.N:
            vec = vec[:self.N]
        elif n < self.N:
            pad = np.zeros((self.N-n))
            vec = np.hstack((vec, pad))
        return vec

    def generate_samples(self, batch_inds):
        n, N = self.n, self.N 
        # n : number of sample questions per query
        # N : number of words per sentence
        # self.wlen : length of word embedding vector
        # self.opveclen : length of output vector
        batch_size = len(batch_inds) 
        samples = np.zeros((batch_size, n, N))
        labels = np.zeros((batch_size, n+1, self.opveclen))
        ques = np.zeros((batch_size, N))
        r = 0
        not_batch_inds = [] 
        for i in batch_inds:
            q = self.qset[i] 
            j = 0
            ques[r, :] = self.question2vec(self.question_id[int(q)])
            pos_list = self.pos_set[i].split()
            neg_list = self.neg_set[i].split()

            if(len(pos_list) == 0):
                # logger.warn('No positive or no negative examples detected for example ' + str(i))
                if not_batch_inds:
                    not_batch_inds.append(i)
                else:
                    not_batch_inds = [i]
                continue

            for pos in pos_list:
                p = int(pos)
                samples[r, j, :] = self.question2vec(self.question_id[int(p)])
                labels[r, j, :] = 1
                j += 1
                if (j >= n):
                    break

            for neg in neg_list:
                if (j >= n):
                    break
                ns = int(neg)
                samples[r, j, :] = self.question2vec(self.question_id[int(ns)])
                labels[r, j, :] = 0
                j += 1
            if (len(neg_list)==0):
                ns = p
                extra_label = 1.0
            else:
                extra_label = 0
            while (j < n):
                samples[r, j, :] = self.question2vec(self.question_id[int(ns)])
                labels[r, j, :] = extra_label 
                j += 1
            r += 1
        batch_size = r
        batch_inds_new = [x for x in batch_inds if x not in not_batch_inds]
        labels = labels[:batch_size, :, :].reshape((batch_size, (n+1)*self.opveclen))
        ques = tf.convert_to_tensor(ques[:batch_size, :].reshape(batch_size, 1, N), dtype=tf.float32) # batch_size X N X self.wlen
        samples = tf.convert_to_tensor(samples[:batch_size, :,:], dtype=tf.float32) 
        labels = tf.convert_to_tensor(labels, dtype=tf.float32)
        data = tf.concat([ques, samples], axis=1)
        return (data, labels, batch_size, batch_inds_new)
