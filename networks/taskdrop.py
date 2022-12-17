import sys
import torch
from transformers import BertModel, BertConfig
import utils
from torch import nn
import torch.nn.functional as F

class Net(torch.nn.Module):

    def __init__(self,taskcla,args):

        super(Net,self).__init__()

        self.taskcla=taskcla
        self.args=args

        config = BertConfig.from_pretrained(args.bert_model)
        self.bert = BertModel.from_pretrained(args.bert_model,config=config)

        #BERT fixed, i.e. BERT as feature extractor===========
        for param in self.bert.parameters():
            param.requires_grad = False

        self.relu=torch.nn.ReLU()
        self.mcl = MCL(args,taskcla)
        self.ac = AC(args,taskcla)

        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(args.bert_hidden_size,n)) # Multiply by 2 for bidirectional gru with concat

        
        print('BERT (Fixed) + GRU + KAN')


        return

    def forward(self,t, input_ids, segment_ids, input_mask, which_type, s, mask_id=-1, my_debug=0, get_emb_ip=False):
        if my_debug==1:
            # When using captum integrated gradients, the assumption is that the first argument of forward() is the input
            # Since TaskDrop deviates from this argument structure, we need to fix the argument assignment manually
            # Additionally, gradients cannot be calculated for the embedding layer so we take the embeddings outside the forward call
            temp_sequence_output = t
            t = input_ids
            sequence_output = temp_sequence_output
            # print(t, input_ids, segment_ids, input_mask, which_type, s)
        else:
            res = self.bert(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
            sequence_output = res['last_hidden_state']

        gfc=self.ac.mask(t=t,s=s,mask_id=mask_id)

        if which_type == 'train':
            mcl_output,mcl_hidden = self.mcl.gru(sequence_output)
            # Commented out the masking operation
            if t == 0: mcl_hidden = mcl_hidden*torch.ones_like(gfc.expand_as(mcl_hidden)) # everyone open
            else: mcl_hidden=mcl_hidden*gfc.expand_as(mcl_hidden)
            h=self.relu(mcl_hidden)

        elif which_type == 'test':
            mcl_output,mcl_hidden = self.mcl.gru(sequence_output)
            h=self.relu(mcl_hidden)

        #loss ==============
        if my_debug==1:
        # When using captum for attributions, return only the output of the head for the current task
        # TODO: Check that mcl_hidden is not needed
            # print(y)
            current_task_id = t[0]
            # print('current_task_id':, t, t[0])
            return self.last[current_task_id](h)

        y=[]
        for t,i in self.taskcla:
            y.append(self.last[t](h))
        if get_emb_ip:
            return y,mcl_hidden,sequence_output
        return y,mcl_hidden
        # return y

    def get_view_for(self,n,mask):
        if n=='mcl.gru.rnn.weight_ih_l0':
            # print('not none')
            return mask.data.view(1,-1).expand_as(self.mcl.gru.rnn.weight_ih_l0) # Change this for bidirectional gru with summing
        elif n=='mcl.gru.rnn.weight_hh_l0':
            return mask.data.view(1,-1).expand_as(self.mcl.gru.rnn.weight_hh_l0)
        elif n=='mcl.gru.rnn.bias_ih_l0':
            return mask.data.view(-1).repeat(3)
        elif n=='mcl.gru.rnn.bias_hh_l0':
            return mask.data.view(-1).repeat(3)
        return None



class AC(nn.Module):
    def __init__(self,args,taskcla):
        super().__init__()

        self.gru = GRU(
                    embedding_dim = args.bert_hidden_size,
                    hidden_dim = args.bert_hidden_size,
                    n_layers=1,
                    bidirectional=False,
                    # bidirectional=True,
                    dropout=0.5,
                    args=args)

        self.efc=torch.nn.Embedding(args.num_task,args.bert_hidden_size)
        self.gate=torch.nn.Sigmoid()
        self.random_mask=torch.zeros([args.ntasks,args.bert_hidden_size]).cuda() # Multiply by 2 for bidirectional gru with concat
        if args.multi_mask>1:
            self.mask_pool=torch.zeros([args.multi_mask,args.bert_hidden_size]).cuda()
    def mask(self,t,s=1,mask_id=-1):
        # gfc=self.gate(s*self.efc(torch.LongTensor([t]).cuda()))
        if mask_id==-1:
            gfc=self.random_mask[t]
            return gfc
        else:
            gfc=self.mask_pool[mask_id].cuda()
            return gfc


class MCL(nn.Module):
    def __init__(self,args,taskcla):
        super().__init__()

        self.gru = GRU(
                    embedding_dim = args.bert_hidden_size,
                    hidden_dim = args.bert_hidden_size,
                    n_layers=1,
                    bidirectional=False,
                    # bidirectional=True,
                    dropout=0.5,
                    args=args)

class GRU(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_layers,
                 bidirectional, dropout, args):
        super().__init__()

        self.rnn = nn.GRU(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout,
                           batch_first=True)
        self.args = args
        self.bidirectional = bidirectional

    def forward(self, x):
        output, hidden = self.rnn(x)
        if self.bidirectional:
            # print('hidden dimension:',hidden.shape)
            hidden_1 = hidden.view(-1,2,self.args.bert_hidden_size)[:,0,:]
            hidden_2 = hidden.view(-1,2,self.args.bert_hidden_size)[:,1,:]
            # print('hidden dimension after view:',hidden_1.shape,hidden_2.shape)
            hidden = hidden_1 + hidden_2
            # print('summed hidden dimension:',hidden.shape)
            # hidden = torch.cat((hidden_1, hidden_2), dim=1)
            # print('concated hidden dimension:',hidden.shape)
            # print('output dimension:',output.shape)
            output_1 = output.view(-1,self.args.max_seq_length,2,self.args.bert_hidden_size)[:,:,0,:]
            output_2 = output.view(-1,self.args.max_seq_length,2,self.args.bert_hidden_size)[:,:,1,:]
            # print('output dimension after view:',output_1.shape,output_2.shape)
            output = output_1 + output_2
            # print('summed output dimension:',output.shape)
            # output = output.view(-1,self.args.max_seq_length,2*self.args.bert_hidden_size)
            # print('concated output dimension:',output.shape)
        else:
            hidden = hidden.view(-1,self.args.bert_hidden_size)
            # print('output dimension:',output.shape)
            output = output.view(-1,self.args.max_seq_length,self.args.bert_hidden_size)
            # print('output dimension after view:',output.shape)

        return output,hidden