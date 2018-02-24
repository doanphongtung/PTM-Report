#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 22:51:22 2018

@author: tung doan
"""

import sys, os, shutil
import DualOnlineLDA
import numpy as np
import per_vb

def read_data(filename):
    wordids = list()
    wordcts = list()
    fp = open(filename, 'r')
    while True:
        line = fp.readline()
        # check end of file
        if len(line) < 1:
            break
        terms = line.split(' ')
        doc_length = int(terms[0])
        ids = np.zeros(doc_length, dtype = np.int32)
        cts = np.zeros(doc_length, dtype = np.int32)
        for j in range(1, doc_length + 1):
            term_count = terms[j].split(':')
            ids[j - 1] = int(term_count[0])
            cts[j - 1] = int(term_count[1])
        wordids.append(ids)
        wordcts.append(cts)
    fp.close()
    return(wordids, wordcts)
   
def read_setting(file_name):
    f = open(file_name, 'r')
    settings = f.readlines()
    f.close()
    sets = list()
    vals = list()
    for i in range(len(settings)):
        #print'%s\n'%(settings[i])
        if settings[i][0] == '#':
            continue
        set_val = settings[i].split(':')
        sets.append(set_val[0])
        vals.append(float(set_val[1]))
    ddict = dict(zip(sets, vals))
    ddict['num_terms'] = int(ddict['num_terms'])
    ddict['num_topics'] = int(ddict['num_topics'])
    ddict['iter_train'] = int(ddict['iter_train'])
    ddict['iter_infer'] = int(ddict['iter_infer'])
    ddict['batch_size'] = int(ddict['batch_size'])
    return(ddict)
    
def read_minibatch(fp, batch_size):
    wordtks = list()
    lengths = list()
    for i in range(batch_size):
        line = fp.readline()
        # check end of file
        if len(line) < 5:
            break
        tks = list()
        tokens = line.split(' ')
        counts = int(tokens[0]) + 1
        for j in range(1, counts):
            token_count = tokens[j].split(':')
            token_count = list(map(int, token_count))
            for k in range(token_count[1]):
                tks.append(token_count[0])
        wordtks.append(tks)
        lengths.append(len(tks))
    return(wordtks, lengths) 
    
def read_data_for_perpl(test_data_folder):
    filename_part1 = '%s/data_test_part_1.txt'%(test_data_folder)
    filename_part2 = '%s/data_test_part_2.txt'%(test_data_folder)
    (wordids_1, wordcts_1) = read_data(filename_part1)
    (wordids_2, wordcts_2) = read_data(filename_part2)
    return(wordids_1, wordcts_1, wordids_2, wordcts_2)    
    
def compute_sparsity(doc_tp, batch_size, num_topics, _type):
    sparsity = np.zeros(batch_size, dtype = np.float)
    if _type == 'z':
        for d in range(batch_size):
            N_z = np.zeros(num_topics, dtype = np.int)
            N = len(doc_tp[d])
            for i in range(N):
                N_z[doc_tp[d][i]] += 1.
            sparsity[d] = len(np.where(N_z != 0)[0])
    else:
        for d in range(batch_size):
            sparsity[d] = len(np.where(doc_tp[d] > 1e-10)[0])
    sparsity /= num_topics
    return(np.mean(sparsity))
    
def compute_perplexities_vb(beta, alpha, eta, max_iter, wordids_1, wordcts_1, wordids_2, wordcts_2):
    vb = per_vb.VB(beta, alpha, eta, max_iter)
    LD2 = vb.compute_perplexity(wordids_1, wordcts_1, wordids_2, wordcts_2)
    return(LD2)
    
def write_topics(beta, file_name):
    num_terms = beta.shape[1]
    num_topics = beta.shape[0]
    f = open(file_name, 'w')
    for k in range(num_topics):
        for i in range(num_terms - 1):
            f.write('%.10f '%(beta[k][i]))
        f.write('%.10f\n'%(beta[k][num_terms - 1]))
    f.close()

def write_perplexities(LD2, file_name):
    f = open(file_name, 'a')
    f.writelines('%f,'%(LD2))
    f.close()    
    
def write_time(i, j, time_e, time_m, file_name):
    f = open(file_name, 'a')
    f.write('tloop_%d_iloop_%d, %f, %f, %f,\n'%(i, j, time_e, time_m, time_e + time_m))
    f.close()
    
def write_loop(i, j, file_name):
    f = open(file_name, 'w')
    f.write('%d, %d'%(i,j))
    f.close()    

def write_file(i, j, time_e, time_m, sparsity, LD2, model_folder):
    per_file_name = '%s/perplexities_%d.csv'%(model_folder, i)
    time_file_name = '%s/time_%d.csv'%(model_folder, i)
    loop_file_name = '%s/loops.csv'%(model_folder)
    # write perplexities
    write_perplexities(LD2, per_file_name)
    # write time
    write_time(i, j, time_e, time_m, time_file_name)
    # write loop
    write_loop(i, j, loop_file_name)   
    
def main():
    # Check input
    if len(sys.argv) != 5:
        print("usage: python run.py [train file] [setting file] [model folder] [test data folder]")
        exit()
    # Get environment variables
    train_file = sys.argv[1]
    setting_file = sys.argv[2]
    model_folder = sys.argv[3]
    test_data_folder = sys.argv[4]    
    # Create model folder if it doesn't exist
    if os.path.exists(model_folder):
        shutil.rmtree(model_folder)
    os.makedirs(model_folder)
    # Read settings
    print('reading setting ...')
    ddict = read_setting(setting_file)
    # Read data for computing perplexities
    print('read data for computing perplexities ...')
    (wordids_1, wordcts_1, wordids_2, wordcts_2) = \
    read_data_for_perpl(test_data_folder)
    # Initialize the algorithm
    print('initialize the algorithm ...')
    dualonline_lda = DualOnlineLDA.DualOnlineLDA(ddict['num_terms'], ddict['num_topics'], ddict['alpha'], 
                                                      ddict['tau0'], ddict['kappa'], ddict['iter_infer'])
    # Start
    print('start!!!')
    i = 0
    while i < ddict['iter_train']:
        i += 1
        print('\n***iter_train:%d***\n'%(i))
        datafp = open(train_file, 'r')
        j = 0
        while True:
            j += 1
            (wordids, wordcts) = read_minibatch(datafp, ddict['batch_size'])
            # Stop condition
            if len(wordids) == 0:
                break
            # 
            print('---num_minibatch:%d---'%(j))
            (time_e, time_m, theta) = dualonline_lda.static_online(ddict['batch_size'], wordids, wordcts)
            # Compute sparsity
            sparsity = compute_sparsity(theta, theta.shape[0], theta.shape[1], 't')
            # Compute perplexities
            LD2 = compute_perplexities_vb(dualonline_lda.beta, ddict['alpha'], ddict['eta'], ddict['iter_infer'],\
                                                                       wordids_1, wordcts_1, wordids_2, wordcts_2)
            # Write files
            write_file(i, j, time_e, time_m, sparsity, LD2, model_folder)
        datafp.close()
    # Write final model to file
    file_name = '%s/beta_final.dat'%(model_folder)
    write_topics(dualonline_lda.beta, file_name)
    # Finish
    print('done!!!')        
if __name__ == '__main__':
    main()