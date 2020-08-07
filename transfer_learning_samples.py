import pickle
import numpy as np
import tensorflow as tf
from sample_generator import SampleGenerator

class AndroidSampleGenerator(SampleGenerator):

    def __init__(self, pos_data, neg_data, dims, data_obj):
        qset, pos_set, neg_set = self.format_data(pos_data, neg_data) 
        SampleGenerator.__init__(self, qset, pos_set, neg_set, dims, data_obj)  
   
    def format_data(self, pos_data, neg_data):
        qset = list()
        pos_set = list() 
        neg_set = list()
        for qid in pos_data.keys():
            pos_set_ = ' '.join(map(str, pos_data[qid]))
            neg_set_ = ' '.join(map(str, neg_data[qid]))
            qset_ = qid
            if qset == []:
                qset = [qset_]
                pos_set = [pos_set_]
                neg_set = [neg_set_]
            else:
                qset.append(qset_)
                pos_set.append(pos_set_)
                neg_set.append(neg_set_)
        return (qset, pos_set, neg_set)
