import numpy as np
import torch

def aggregate_local_to_global(attributions,predictions,targets,tokens,all_tokens):
    mxtx_tp = np.where((predictions==0) & (predictions==targets))[0]
    mxtx_fp = np.where((predictions==0) & (predictions!=targets))[0]
    mxtx_tn = np.where((predictions==1) & (predictions==targets))[0]
    mxtx_fn = np.where((predictions==1) & (predictions!=targets))[0]
    
    attr_pos_mxtx = np.clip(attributions, a_min=0, a_max=None)
    attr_neg_mxtx = np.clip(attributions, a_min=None, a_max=0)
    
    # Collect all local token attributions towards each class separately by correct and incorrect predictions
    # true positives
    attr_mxtx_tp = attr_pos_mxtx[mxtx_tp]
    mxtx_tp_token_global_attr = {}
    for train_idx,attr in zip(mxtx_tp,attr_mxtx_tp):
        train_tokens = tokens[train_idx]
        for token_id in range(len(attr)):
            if token_id>0 and token_id<=len(train_tokens):
                token = train_tokens[token_id-1]
                if token in mxtx_tp_token_global_attr:
                    mxtx_tp_token_global_attr[token].append(attr[token_id])
                else:
                    mxtx_tp_token_global_attr[token] = [attr[token_id]]
            elif token_id==0:
                if '<CLS>' in mxtx_tp_token_global_attr:
                    mxtx_tp_token_global_attr['<CLS>'].append(attr[token_id])
                else:
                    mxtx_tp_token_global_attr['<CLS>'] = [attr[token_id]]
            elif token_id==len(train_tokens)+1:
                if '<SEP>' in mxtx_tp_token_global_attr:
                    mxtx_tp_token_global_attr['<SEP>'].append(attr[token_id])
                else:
                    mxtx_tp_token_global_attr['<SEP>'] = [attr[token_id]]
            else:
                if '<PAD>' in mxtx_tp_token_global_attr:
                    mxtx_tp_token_global_attr['<PAD>'].append(attr[token_id])
                else:
                    mxtx_tp_token_global_attr['<PAD>'] = [attr[token_id]]
    attr_mxtx_tp = attr_neg_mxtx[mxtx_tp]
    mxtx_tn_token_global_attr = {}
    for train_idx,attr in zip(mxtx_tp,attr_mxtx_tp):
        train_tokens = tokens[train_idx]
        for token_id in range(len(attr)):
            if token_id>0 and token_id<=len(train_tokens):
                token = train_tokens[token_id-1]
                if token in mxtx_tn_token_global_attr:
                    mxtx_tn_token_global_attr[token].append(attr[token_id]*-1)
                else:
                    mxtx_tn_token_global_attr[token] = [attr[token_id]*-1]
            elif token_id==0:
                if '<CLS>' in mxtx_tn_token_global_attr:
                    mxtx_tn_token_global_attr['<CLS>'].append(attr[token_id]*-1)
                else:
                    mxtx_tn_token_global_attr['<CLS>'] = [attr[token_id]*-1]
            elif token_id==len(train_tokens)+1:
                if '<SEP>' in mxtx_tn_token_global_attr:
                    mxtx_tn_token_global_attr['<SEP>'].append(attr[token_id]*-1)
                else:
                    mxtx_tn_token_global_attr['<SEP>'] = [attr[token_id]*-1]
            else:
                if '<PAD>' in mxtx_tn_token_global_attr:
                    mxtx_tn_token_global_attr['<PAD>'].append(attr[token_id]*-1)
                else:
                    mxtx_tn_token_global_attr['<PAD>'] = [attr[token_id]*-1]
    
    # false positives
    attr_mxtx_fp = attr_pos_mxtx[mxtx_fp]
    mxtx_fp_token_global_attr = {}
    for train_idx,attr in zip(mxtx_fp,attr_mxtx_fp):
        train_tokens = tokens[train_idx]
        for token_id in range(len(attr)):
            if token_id>0 and token_id<=len(train_tokens):
                token = train_tokens[token_id-1]
                if token in mxtx_fp_token_global_attr:
                    mxtx_fp_token_global_attr[token].append(attr[token_id])
                else:
                    mxtx_fp_token_global_attr[token] = [attr[token_id]]
            elif token_id==0:
                if '<CLS>' in mxtx_fp_token_global_attr:
                    mxtx_fp_token_global_attr['<CLS>'].append(attr[token_id])
                else:
                    mxtx_fp_token_global_attr['<CLS>'] = [attr[token_id]]
            elif token_id==len(train_tokens)+1:
                if '<SEP>' in mxtx_fp_token_global_attr:
                    mxtx_fp_token_global_attr['<SEP>'].append(attr[token_id])
                else:
                    mxtx_fp_token_global_attr['<SEP>'] = [attr[token_id]]
            else:
                if '<PAD>' in mxtx_fp_token_global_attr:
                    mxtx_fp_token_global_attr['<PAD>'].append(attr[token_id])
                else:
                    mxtx_fp_token_global_attr['<PAD>'] = [attr[token_id]]
    attr_mxtx_fp = attr_neg_mxtx[mxtx_fp]
    mxtx_fn_token_global_attr = {}
    for train_idx,attr in zip(mxtx_fp,attr_mxtx_fp):
        train_tokens = tokens[train_idx]
        for token_id in range(len(attr)):
            if token_id>0 and token_id<=len(train_tokens):
                token = train_tokens[token_id-1]
                if token in mxtx_fn_token_global_attr:
                    mxtx_fn_token_global_attr[token].append(attr[token_id]*-1)
                else:
                    mxtx_fn_token_global_attr[token] = [attr[token_id]*-1]
            elif token_id==0:
                if '<CLS>' in mxtx_fn_token_global_attr:
                    mxtx_fn_token_global_attr['<CLS>'].append(attr[token_id]*-1)
                else:
                    mxtx_fn_token_global_attr['<CLS>'] = [attr[token_id]*-1]
            elif token_id==len(train_tokens)+1:
                if '<SEP>' in mxtx_fn_token_global_attr:
                    mxtx_fn_token_global_attr['<SEP>'].append(attr[token_id]*-1)
                else:
                    mxtx_fn_token_global_attr['<SEP>'] = [attr[token_id]*-1]
            else:
                if '<PAD>' in mxtx_fn_token_global_attr:
                    mxtx_fn_token_global_attr['<PAD>'].append(attr[token_id]*-1)
                else:
                    mxtx_fn_token_global_attr['<PAD>'] = [attr[token_id]*-1]

    # true negatives
    attr_mxtx_tn = attr_neg_mxtx[mxtx_tn]
    for train_idx,attr in zip(mxtx_tn,attr_mxtx_tn):
        train_tokens = tokens[train_idx]
        for token_id in range(len(attr)):
            if token_id>0 and token_id<=len(train_tokens):
                token = train_tokens[token_id-1]
                if token in mxtx_tp_token_global_attr:
                    mxtx_tp_token_global_attr[token].append(attr[token_id]*-1)
                else:
                    mxtx_tp_token_global_attr[token] = [attr[token_id]*-1]
            elif token_id==0:
                if '<CLS>' in mxtx_tp_token_global_attr:
                    mxtx_tp_token_global_attr['<CLS>'].append(attr[token_id]*-1)
                else:
                    mxtx_tp_token_global_attr['<CLS>'] = [attr[token_id]*-1]
            elif token_id==len(train_tokens)+1:
                if '<SEP>' in mxtx_tp_token_global_attr:
                    mxtx_tp_token_global_attr['<SEP>'].append(attr[token_id]*-1)
                else:
                    mxtx_tp_token_global_attr['<SEP>'] = [attr[token_id]*-1]
            else:
                if '<PAD>' in mxtx_tp_token_global_attr:
                    mxtx_tp_token_global_attr['<PAD>'].append(attr[token_id]*-1)
                else:
                    mxtx_tp_token_global_attr['<PAD>'] = [attr[token_id]*-1]
    attr_mxtx_tn = attr_pos_mxtx[mxtx_tn]
    for train_idx,attr in zip(mxtx_tn,attr_mxtx_tn):
        train_tokens = tokens[train_idx]
        for token_id in range(len(attr)):
            if token_id>0 and token_id<=len(train_tokens):
                token = train_tokens[token_id-1]
                if token in mxtx_tn_token_global_attr:
                    mxtx_tn_token_global_attr[token].append(attr[token_id])
                else:
                    mxtx_tn_token_global_attr[token] = [attr[token_id]]
            elif token_id==0:
                if '<CLS>' in mxtx_tn_token_global_attr:
                    mxtx_tn_token_global_attr['<CLS>'].append(attr[token_id])
                else:
                    mxtx_tn_token_global_attr['<CLS>'] = [attr[token_id]]
            elif token_id==len(train_tokens)+1:
                if '<SEP>' in mxtx_tn_token_global_attr:
                    mxtx_tn_token_global_attr['<SEP>'].append(attr[token_id])
                else:
                    mxtx_tn_token_global_attr['<SEP>'] = [attr[token_id]]
            else:
                if '<PAD>' in mxtx_tn_token_global_attr:
                    mxtx_tn_token_global_attr['<PAD>'].append(attr[token_id])
                else:
                    mxtx_tn_token_global_attr['<PAD>'] = [attr[token_id]]

    # false negatives
    attr_mxtx_fn = attr_neg_mxtx[mxtx_fn]
    for train_idx,attr in zip(mxtx_fn,attr_mxtx_fn):
        train_tokens = tokens[train_idx]
        for token_id in range(len(attr)):
            if token_id>0 and token_id<=len(train_tokens):
                token = train_tokens[token_id-1]
                if token in mxtx_fp_token_global_attr:
                    mxtx_fp_token_global_attr[token].append(attr[token_id]*-1)
                else:
                    mxtx_fp_token_global_attr[token] = [attr[token_id]*-1]
            elif token_id==0:
                if '<CLS>' in mxtx_fp_token_global_attr:
                    mxtx_fp_token_global_attr['<CLS>'].append(attr[token_id]*-1)
                else:
                    mxtx_fp_token_global_attr['<CLS>'] = [attr[token_id]*-1]
            elif token_id==len(train_tokens)+1:
                if '<SEP>' in mxtx_fp_token_global_attr:
                    mxtx_fp_token_global_attr['<SEP>'].append(attr[token_id]*-1)
                else:
                    mxtx_fp_token_global_attr['<SEP>'] = [attr[token_id]*-1]
            else:
                if '<PAD>' in mxtx_fp_token_global_attr:
                    mxtx_fp_token_global_attr['<PAD>'].append(attr[token_id]*-1)
                else:
                    mxtx_fp_token_global_attr['<PAD>'] = [attr[token_id]*-1]
    attr_mxtx_fn = attr_pos_mxtx[mxtx_fn]
    for train_idx,attr in zip(mxtx_fn,attr_mxtx_fn):
        train_tokens = tokens[train_idx]
        for token_id in range(len(attr)):
            if token_id>0 and token_id<=len(train_tokens):
                token = train_tokens[token_id-1]
                if token in mxtx_fn_token_global_attr:
                    mxtx_fn_token_global_attr[token].append(attr[token_id])
                else:
                    mxtx_fn_token_global_attr[token] = [attr[token_id]]
            elif token_id==0:
                if '<CLS>' in mxtx_fn_token_global_attr:
                    mxtx_fn_token_global_attr['<CLS>'].append(attr[token_id])
                else:
                    mxtx_fn_token_global_attr['<CLS>'] = [attr[token_id]]
            elif token_id==len(train_tokens)+1:
                if '<SEP>' in mxtx_fn_token_global_attr:
                    mxtx_fn_token_global_attr['<SEP>'].append(attr[token_id])
                else:
                    mxtx_fn_token_global_attr['<SEP>'] = [attr[token_id]]
            else:
                if '<PAD>' in mxtx_fn_token_global_attr:
                    mxtx_fn_token_global_attr['<PAD>'].append(attr[token_id])
                else:
                    mxtx_fn_token_global_attr['<PAD>'] = [attr[token_id]]

    # Aggregate to global token attributions
    mxtx_tp_mean_global_attr_dict = {}
    mxtx_tn_mean_global_attr_dict = {}
    mxtx_fp_mean_global_attr_dict = {}
    mxtx_fn_mean_global_attr_dict = {}
    for token in all_tokens:
        try:
            token_mean = statistics.mean(mxtx_tp_token_global_attr[token])
        except:
            token_mean = 0
        mxtx_tp_mean_global_attr_dict[token] = token_mean
        
        try:
            token_mean = statistics.mean(mxtx_tn_token_global_attr[token])
        except:
            token_mean = 0
        mxtx_tn_mean_global_attr_dict[token] = token_mean
        
        try:
            token_mean = statistics.mean(mxtx_fp_token_global_attr[token])
        except:
            token_mean = 0
        mxtx_fp_mean_global_attr_dict[token] = token_mean
        
        try:
            token_mean = statistics.mean(mxtx_fn_token_global_attr[token])
        except:
            token_mean = 0
        mxtx_fn_mean_global_attr_dict[token] = token_mean
    
    global_attr = {}
    global_attr['tp'] = mxtx_tp_mean_global_attr_dict
    global_attr['fp'] = mxtx_fp_mean_global_attr_dict
    global_attr['tn'] = mxtx_tn_mean_global_attr_dict
    global_attr['fn'] = mxtx_fn_mean_global_attr_dict
    
    return global_attr

def aggregate_local_to_global_batch(attributions,predictions,targets,tokens):
    mxtx_pos = np.where((predictions==0))[0]
    mxtx_neg = np.where((predictions==1))[0]
    
    attr_pos_mxtx = np.clip(attributions, a_min=0, a_max=None)
    attr_neg_mxtx = np.clip(attributions, a_min=None, a_max=0)
    
    # Collect all local token attributions towards each class
    # predicted class = positive
    attr_mxtx_pos = attr_pos_mxtx[mxtx_pos]
    mxtx_pos_token_global_attr = {}
    for train_idx,attr in zip(mxtx_pos,attr_mxtx_pos):
        train_tokens = tokens[train_idx]
        for token_id in range(len(attr)):
            if token_id>0 and token_id<=len(train_tokens):
                token = train_tokens[token_id-1]
                if token in mxtx_pos_token_global_attr:
                    mxtx_pos_token_global_attr[token].append(attr[token_id])
                else:
                    mxtx_pos_token_global_attr[token] = [attr[token_id]]
            elif token_id==0:
                if '<CLS>' in mxtx_pos_token_global_attr:
                    mxtx_pos_token_global_attr['<CLS>'].append(attr[token_id])
                else:
                    mxtx_pos_token_global_attr['<CLS>'] = [attr[token_id]]
            elif token_id==len(train_tokens)+1:
                if '<SEP>' in mxtx_pos_token_global_attr:
                    mxtx_pos_token_global_attr['<SEP>'].append(attr[token_id])
                else:
                    mxtx_pos_token_global_attr['<SEP>'] = [attr[token_id]]
            else:
                if '<PAD>' in mxtx_pos_token_global_attr:
                    mxtx_pos_token_global_attr['<PAD>'].append(attr[token_id])
                else:
                    mxtx_pos_token_global_attr['<PAD>'] = [attr[token_id]]
    attr_mxtx_pos = attr_neg_mxtx[mxtx_pos]
    mxtx_neg_token_global_attr = {}
    for train_idx,attr in zip(mxtx_pos,attr_mxtx_pos):
        train_tokens = tokens[train_idx]
        for token_id in range(len(attr)):
            if token_id>0 and token_id<=len(train_tokens):
                token = train_tokens[token_id-1]
                if token in mxtx_neg_token_global_attr:
                    mxtx_neg_token_global_attr[token].append(attr[token_id]*-1)
                else:
                    mxtx_neg_token_global_attr[token] = [attr[token_id]*-1]
            elif token_id==0:
                if '<CLS>' in mxtx_neg_token_global_attr:
                    mxtx_neg_token_global_attr['<CLS>'].append(attr[token_id]*-1)
                else:
                    mxtx_neg_token_global_attr['<CLS>'] = [attr[token_id]*-1]
            elif token_id==len(train_tokens)+1:
                if '<SEP>' in mxtx_neg_token_global_attr:
                    mxtx_neg_token_global_attr['<SEP>'].append(attr[token_id]*-1)
                else:
                    mxtx_neg_token_global_attr['<SEP>'] = [attr[token_id]*-1]
            else:
                if '<PAD>' in mxtx_neg_token_global_attr:
                    mxtx_neg_token_global_attr['<PAD>'].append(attr[token_id]*-1)
                else:
                    mxtx_neg_token_global_attr['<PAD>'] = [attr[token_id]*-1]
 

    # predicted class = negative
    attr_mxtx_neg = attr_neg_mxtx[mxtx_neg]
    for train_idx,attr in zip(mxtx_neg,attr_mxtx_neg):
        train_tokens = tokens[train_idx]
        for token_id in range(len(attr)):
            if token_id>0 and token_id<=len(train_tokens):
                token = train_tokens[token_id-1]
                if token in mxtx_pos_token_global_attr:
                    mxtx_pos_token_global_attr[token].append(attr[token_id]*-1)
                else:
                    mxtx_pos_token_global_attr[token] = [attr[token_id]*-1]
            elif token_id==0:
                if '<CLS>' in mxtx_pos_token_global_attr:
                    mxtx_pos_token_global_attr['<CLS>'].append(attr[token_id]*-1)
                else:
                    mxtx_pos_token_global_attr['<CLS>'] = [attr[token_id]*-1]
            elif token_id==len(train_tokens)+1:
                if '<SEP>' in mxtx_pos_token_global_attr:
                    mxtx_pos_token_global_attr['<SEP>'].append(attr[token_id]*-1)
                else:
                    mxtx_pos_token_global_attr['<SEP>'] = [attr[token_id]*-1]
            else:
                if '<PAD>' in mxtx_pos_token_global_attr:
                    mxtx_pos_token_global_attr['<PAD>'].append(attr[token_id]*-1)
                else:
                    mxtx_pos_token_global_attr['<PAD>'] = [attr[token_id]*-1]
    attr_mxtx_neg = attr_pos_mxtx[mxtx_neg]
    for train_idx,attr in zip(mxtx_neg,attr_mxtx_neg):
        train_tokens = tokens[train_idx]
        for token_id in range(len(attr)):
            if token_id>0 and token_id<=len(train_tokens):
                token = train_tokens[token_id-1]
                if token in mxtx_neg_token_global_attr:
                    mxtx_neg_token_global_attr[token].append(attr[token_id])
                else:
                    mxtx_neg_token_global_attr[token] = [attr[token_id]]
            elif token_id==0:
                if '<CLS>' in mxtx_neg_token_global_attr:
                    mxtx_neg_token_global_attr['<CLS>'].append(attr[token_id])
                else:
                    mxtx_neg_token_global_attr['<CLS>'] = [attr[token_id]]
            elif token_id==len(train_tokens)+1:
                if '<SEP>' in mxtx_neg_token_global_attr:
                    mxtx_neg_token_global_attr['<SEP>'].append(attr[token_id])
                else:
                    mxtx_neg_token_global_attr['<SEP>'] = [attr[token_id]]
            else:
                if '<PAD>' in mxtx_neg_token_global_attr:
                    mxtx_neg_token_global_attr['<PAD>'].append(attr[token_id])
                else:
                    mxtx_neg_token_global_attr['<PAD>'] = [attr[token_id]]

    # Get unique tokens
    all_tokens=[]
    for example in tokens:
        all_tokens += example
    all_tokens = list(set(all_tokens))
    
    # Aggregate to global token attributions
    mxtx_pos_mean_global_attr_dict = {}
    mxtx_neg_mean_global_attr_dict = {}
    for token in all_tokens:
        try:
            token_mean = statistics.mean(mxtx_pos_token_global_attr[token])
        except:
            token_mean = 0
        mxtx_pos_mean_global_attr_dict[token] = token_mean
        
        try:
            token_mean = statistics.mean(mxtx_neg_token_global_attr[token])
        except:
            token_mean = 0
        mxtx_neg_mean_global_attr_dict[token] = token_mean
    
    global_attr = {}
    global_attr['pos'] = mxtx_pos_mean_global_attr_dict
    global_attr['neg'] = mxtx_neg_mean_global_attr_dict
    
    return global_attr
    
def get_common_features(batch_global_attr,global_attr):
    batch_global_attr_common = {}
    global_attr_common = {}
    
    common_features = set(batch_global_attr['pos'].keys()).intersection(list(global_attr['tp'].keys()))
    batch_global_attr_common['tp'] = []
    global_attr_common['tp'] = []
    for feature in common_features:
        batch_global_attr_common['tp'].append(batch_global_attr['pos'][feature])
        global_attr_common['tp'].append(global_attr['tp'][feature])
    
    common_features = set(batch_global_attr['pos'].keys()).intersection(list(global_attr['fp'].keys()))
    batch_global_attr_common['fp'] = []
    global_attr_common['fp'] = []
    for feature in common_features:
        batch_global_attr_common['fp'].append(batch_global_attr['pos'][feature])
        global_attr_common['fp'].append(global_attr['fp'][feature])
    
    common_features = set(batch_global_attr['neg'].keys()).intersection(list(global_attr['tn'].keys()))
    batch_global_attr_common['tn'] = []
    global_attr_common['tn'] = []
    for feature in common_features:
        batch_global_attr_common['tn'].append(batch_global_attr['neg'][feature])
        global_attr_common['tn'].append(global_attr['tn'][feature])
    
    common_features = set(batch_global_attr['neg'].keys()).intersection(list(global_attr['fn'].keys()))
    batch_global_attr_common['fn'] = []
    global_attr_common['fn'] = []
    for feature in common_features:
        batch_global_attr_common['fn'].append(batch_global_attr['neg'][feature])
        global_attr_common['fn'].append(global_attr['fn'][feature])
    
    batch_global_attr_common['tp'] = torch.Tensor(batch_global_attr_common['tp'])
    batch_global_attr_common['fp'] = torch.Tensor(batch_global_attr_common['fp'])
    batch_global_attr_common['tn'] = torch.Tensor(batch_global_attr_common['tn'])
    batch_global_attr_common['fn'] = torch.Tensor(batch_global_attr_common['fn'])
    global_attr_common['tp'] = torch.Tensor(global_attr_common['tp'])
    global_attr_common['fp'] = torch.Tensor(global_attr_common['fp'])
    global_attr_common['tn'] = torch.Tensor(global_attr_common['tn'])
    global_attr_common['fn'] = torch.Tensor(global_attr_common['fn'])

    return batch_global_attr_common, global_attr_common