#Coding: UTF-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from transformers import BertTokenizer as BertTokenizer
import os
import torch
import numpy as np
import random
import absa_data_utils as data_utils
from absa_data_utils import ABSATokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import math
datasets = [
    'TaskDrop/dat/dsc_json/Amazon_Instant_Video',
    'TaskDrop/dat/dsc_json/Apps_for_Android',
    'TaskDrop/dat/dsc_json/Automotive',
    'TaskDrop/dat/dsc_json/Baby',
    'TaskDrop/dat/dsc_json/Beauty',
    'TaskDrop/dat/dsc_json/Books',
    'TaskDrop/dat/dsc_json/CDs_and_Vinyl',
    'TaskDrop/dat/dsc_json/Cell_Phones_and_Accessories',
    'TaskDrop/dat/dsc_json/Clothing_Shoes_and_Jewelry',
    'TaskDrop/dat/dsc_json/Digital_Music',
    'TaskDrop/dat/dsc_json/Electronics',
    'TaskDrop/dat/dsc_json/Grocery_and_Gourmet_Food',
    'TaskDrop/dat/dsc_json/Health_and_Personal_Care',
    'TaskDrop/dat/dsc_json/Home_and_Kitchen',
    'TaskDrop/dat/dsc_json/Kindle_Store',
    'TaskDrop/dat/dsc_json/Movies_and_TV',
    'TaskDrop/dat/dsc_json/Musical_Instruments',
    'TaskDrop/dat/dsc_json/Office_Products',
    'TaskDrop/dat/dsc_json/Patio_Lawn_and_Garden',
    'TaskDrop/dat/dsc_json/Pet_Supplies',
    'TaskDrop/dat/dsc_json/Sports_and_Outdoors',
    'TaskDrop/dat/dsc_json/Tools_and_Home_Improvement',
    'TaskDrop/dat/dsc_json/Toys_and_Games',
    'TaskDrop/dat/dsc_json/Video_Games',
            ]


domains = [
    'Amazon_Instant_Video',
    'Apps_for_Android',
    'Automotive',
    'Baby',
    'Beauty',
    'Books',
    'CDs_and_Vinyl',
    'Cell_Phones_and_Accessories',
    'Clothing_Shoes_and_Jewelry',
    'Digital_Music',
    'Electronics',
    'Grocery_and_Gourmet_Food',
    'Health_and_Personal_Care',
    'Home_and_Kitchen',
    'Kindle_Store',
    'Movies_and_TV',
    'Musical_Instruments',
    'Office_Products',
    'Patio_Lawn_and_Garden',
    'Pet_Supplies',
    'Sports_and_Outdoors',
    'Tools_and_Home_Improvement',
    'Toys_and_Games',
    'Video_Games',
        ]

def get(logger=None,args=None):
    if os.path.exists('TaskDrop/dat/bin/data_dis6'+'_'+str(args.idrandom)+'.pt') and os.path.exists('TaskDrop/dat/bin/taskcla_dis6'+'_'+str(args.idrandom)+'.pt'):
        data = torch.load('TaskDrop/dat/bin/data_dis6'+'_'+str(args.idrandom)+'.pt')
        taskcla = torch.load('TaskDrop/dat/bin/taskcla_dis6'+'_'+str(args.idrandom)+'.pt')
        return data,taskcla
    data={}
    taskcla=[]

    # Others
    f_name = 'TaskDrop/asc_random_dis'

    with open(f_name,'r') as f_random_seq:
        fseq=f_random_seq.readlines()
        random_sep = fseq[args.idrandom].split()

    print('random_sep: ',random_sep)
    print('domains: ',domains)

    print('random_sep: ',len(random_sep))
    print('domains: ',len(domains))

    for t in range(args.ntasks):
        dataset = datasets[domains.index(random_sep[t])]

        data[t]={}
        data[t]['name']=dataset
        data[t]['ncla']=2
        # if 'Bing' in dataset:
        #     data[t]['name']=dataset
        #     data[t]['ncla']=2
        # elif 'XuSemEval' in dataset:
        #     data[t]['name']=dataset
        #     data[t]['ncla']=3

        processor = data_utils.AscProcessor()
        label_list = processor.get_labels()
        tokenizer = ABSATokenizer.from_pretrained(args.bert_model)
        train_examples = processor.get_train_examples(dataset)
        num_train_steps = int(math.ceil(len(train_examples) / args.train_batch_size)) * args.num_train_epochs
        # num_train_steps = int(len(train_examples) / args.train_batch_size) * args.num_train_epochs

        train_features = data_utils.convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, "asc")
        logger.info("Loading Task"+str(t)+": "+str(random_sep[t]))
        # logger.info("***** Running training *****")
        # logger.info("  Num examples = %d", len(train_examples))
        # logger.info("  Batch size = %d", args.train_batch_size)
        # logger.info("  Num steps = %d", num_train_steps)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_tasks = torch.tensor([t for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids, all_tasks)

        # Store tokens for analyzing attributions
        tokens = []
        for (ex_index, example) in enumerate(train_examples):
            example_tokens = tokenizer.tokenize(example.text_a)
            tokens.append(example_tokens[0:(args.max_seq_length - 2)])
        data[t]['train_tokens'] = tokens
        test_examples = processor.get_test_examples(dataset)
        tokens = []
        for (ex_index, example) in enumerate(test_examples):
            example_tokens = tokenizer.tokenize(example.text_a)
            tokens.append(example_tokens[0:(args.max_seq_length - 2)])
        data[t]['test_tokens'] = tokens

        data[t]['train'] = train_data
        data[t]['num_train_steps']=num_train_steps

        valid_examples = processor.get_dev_examples(dataset)
        valid_features=data_utils.convert_examples_to_features(
            valid_examples, label_list, args.max_seq_length, tokenizer, "asc")
        valid_all_input_ids = torch.tensor([f.input_ids for f in valid_features], dtype=torch.long)
        valid_all_segment_ids = torch.tensor([f.segment_ids for f in valid_features], dtype=torch.long)
        valid_all_input_mask = torch.tensor([f.input_mask for f in valid_features], dtype=torch.long)
        valid_all_label_ids = torch.tensor([f.label_id for f in valid_features], dtype=torch.long)
        valid_all_tasks = torch.tensor([t for f in valid_features], dtype=torch.long)

        valid_data = TensorDataset(valid_all_input_ids, valid_all_segment_ids, valid_all_input_mask, valid_all_label_ids, valid_all_tasks)

        # logger.info("***** Running validations *****")
        # logger.info("  Num orig examples = %d", len(valid_examples))
        # logger.info("  Num split examples = %d", len(valid_features))
        # logger.info("  Batch size = %d", args.train_batch_size)

        data[t]['valid']=valid_data


        processor = data_utils.AscProcessor()
        label_list = processor.get_labels()
        tokenizer = BertTokenizer.from_pretrained(args.bert_model)
        eval_examples = processor.get_test_examples(dataset)
        eval_features = data_utils.convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer, "asc")

        # logger.info("***** Running evaluation *****")
        # logger.info("  Num examples = %d", len(eval_examples))
        # logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_tasks = torch.tensor([t for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids, all_tasks)
        # Run prediction for full data

        data[t]['test']=eval_data

        taskcla.append((t,int(data[t]['ncla'])))



    # Others
    n=0
    for t in data.keys():
        n+=data[t]['ncla']
    data['ncla']=n

    data2={}
    taskcla2=[]
    for t in range(args.ntasks):
        data2[t]=data[args.ntasks-1-t]
        taskcla2.append(taskcla[args.ntasks-1-t])
    torch.save(data,'TaskDrop/dat/bin/data_dis6'+'_'+str(args.idrandom)+'.pt')
    torch.save(taskcla,'TaskDrop/dat/bin/taskcla_dis6'+'_'+str(args.idrandom)+'.pt')
    return data,taskcla


