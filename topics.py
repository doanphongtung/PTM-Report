#! /usr/bin/python

# usage: python topics.py [beta-file] [vocab-file] [num words] [result-file]
#
# [beta-file] is output from the dln-c code
# [vocab-file] is a list of words, one per line
# [num words] is the number of words to print from each topic
# [result-file] is file to write top [num words] words 

import sys
import numpy as np

def print_topics(beta_file, vocab_file, nwords, result_file):
    # get the vocabulary
    vocab = open(vocab_file, 'r').readlines()
    # vocab = map(lambda x: x.split()[0], vocab)
    vocab = list(map(lambda x: x.strip(), vocab))    
    # open file to write    
    fp_result = open(result_file, 'w')
    fp_topics = open(beta_file, 'r')
    topic_no = 0
    for topic in fp_topics:
        fp_result.write('topic %03d\n' % (topic_no))
        topic = np.array(list(map(float, topic.split())))    
        indices = np.argsort(-topic)
        for i in range(nwords):
            fp_result.write ('   %s \t\t %f\n' % (vocab[indices[i]], topic[indices[i]]))
        topic_no = topic_no + 1
        fp_result.write( '\n')
    fp_result.close()
    fp_topics.close()

if (__name__ == '__main__'):

    if (len(sys.argv) != 5):
       print('usage: python topics.py [beta-file] [vocab-file] [num words] [result-file]')
       sys.exit(1)

    beta_file = sys.argv[1]
    vocab_file = sys.argv[2]
    nwords = int(sys.argv[3])
    result_file = sys.argv[4]
    print_topics(beta_file, vocab_file, nwords, result_file)
