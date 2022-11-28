import sys,time
import numpy as np
from sklearn.metrics import f1_score
import torch
# from copy import deepcopy

import utils
from tqdm import tqdm, trange

import captum
from captum.attr import IntegratedGradients, LRP, Occlusion, configure_interpretable_embedding_layer, remove_interpretable_embedding_layer


rnn_weights = [
    'mcl.lstm.rnn.weight_ih_l0',
    'mcl.lstm.rnn.weight_hh_l0',
    'mcl.lstm.rnn.bias_ih_l0',
    'mcl.lstm.rnn.bias_hh_l0',
    'mcl.gru.rnn.weight_ih_l0',
    'mcl.gru.rnn.weight_hh_l0',
    'mcl.gru.rnn.bias_ih_l0',
    'mcl.gru.rnn.bias_hh_l0']

class Appr(object):
    def __init__(self,model,nepochs=100,sbatch=64,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=3,clipgrad=10000,args=None,logger=None):
    # def __init__(self,model,nepochs=100,sbatch=64,lr=0.001,lr_min=1e-5,lr_factor=2,lr_patience=3,clipgrad=10000,args=None,logger=None):
        self.model=model
        # self.initial_model=deepcopy(model)

        self.nepochs=nepochs
        self.sbatch=sbatch
        self.lr=lr
        self.lr_min=lr_min
        self.lr_factor=lr_factor
        self.lr_patience=lr_patience
        self.clipgrad=clipgrad

        self.criterion=torch.nn.CrossEntropyLoss()
        self.optimizer=self._get_optimizer()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()

        self.smax = 400
        self.thres_cosh=50
        self.thres_emb=6
        self.lamb=0.75

        print('CONTEXTUAL + RNN NCL')

        return

    def _get_optimizer(self,lr=None,which_type=None):
        if lr is None: lr=self.lr
        return torch.optim.SGD(
            [p for p in self.model.mcl.parameters()]+[p for p in self.model.last.parameters()],lr=lr)

    def train(self,t,train,valid,args):
        # self.model=deepcopy(self.initial_model) # Restart model: isolate


        # if t == 0: which_types = ['mcl']
        # else: which_types = ['ac','mcl']

        which_types = ['mcl']
        self.model.ac.random_mask[t]=torch.randint(0,2,(1,2*args.bert_hidden_size)) # Multiply by 2 for bidirectional gru

        for which_type in which_types:

            best_loss=np.inf
            best_model=utils.get_model(self.model)
            lr=self.lr
            patience=self.lr_patience
            self.optimizer=self._get_optimizer(lr,which_type)

            # Loop epochs
            self.nepochs=2
            for e in range(self.nepochs):
                # Train
                clock0=time.time()
                iter_bar = tqdm(train, desc='Train Iter (loss=X.XXX)')
                self.train_epoch(t,train,iter_bar,'train')
                clock1=time.time()
                train_loss,train_acc=self.eval(t,train,'test')
                clock2=time.time()
                print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
                    1000*self.sbatch*(clock1-clock0)/len(train),1000*self.sbatch*(clock2-clock1)/len(train),train_loss,100*train_acc),end='')
                # Valid
                valid_loss,valid_acc=self.eval(t,valid,'test')
                print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
                # Adapt lr
                if valid_loss<best_loss:
                    best_loss=valid_loss
                    best_model=utils.get_model(self.model)
                    patience=self.lr_patience
                    print(' *',end='')
                else:
                    patience-=1
                    if patience<=0:
                        lr/=self.lr_factor
                        print(' lr={:.1e}'.format(lr),end='')
                        if lr<self.lr_min:
                            print()
                            break
                        patience=self.lr_patience
                        self.optimizer=self._get_optimizer(lr,which_type)
                print()

            # Restore best
            utils.set_model_(self.model,best_model)

        return



    def train_epoch(self,t,data,iter_bar,which_type):
        self.model.train()
        # Loop batches
        for step, batch in enumerate(iter_bar):
            batch = [
                bat.to(self.device) if bat is not None else None for bat in batch]
            input_ids, segment_ids, input_mask, targets,_= batch
            s=(self.smax-1/self.smax)*step/len(data)+1/self.smax
            with torch.no_grad():
                task=torch.autograd.Variable(torch.LongTensor([t]).cuda())

            # Forward
            outputs=self.model.forward(task,input_ids, segment_ids, input_mask,which_type,s)
            output=outputs[0][t]
            loss=self.criterion(output,targets)
            
            iter_bar.set_description('Train Iter (loss=%5.3f)' % loss.item())

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # if t>0 and which_type=='mcl':
            task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),volatile=False)
            mask=self.model.ac.mask(task,s=self.smax)
            mask = torch.autograd.Variable(mask.data.clone(),requires_grad=False)
            # Commented out the masking operation - start
            # for n,p in self.model.named_parameters():
                # if n in rnn_weights:
                    # # print('n: ',n)
                    # # print('p: ',p.grad.size())
                    # p.grad.data*=self.model.get_view_for(n,mask)
                    # Commented out the masking operation - end

            # Compensate embedding gradients
            # for n,p in self.model.ac.named_parameters():
            #     if 'ac.e' in n:
            #         num=torch.cosh(torch.clamp(s*p.data,-self.thres_cosh,self.thres_cosh))+1
            #         den=torch.cosh(p.data)+1
            #         p.grad.data*=self.smax/s*num/den


            torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.clipgrad)
            self.optimizer.step()

            # Constrain embeddings
            for n,p in self.model.ac.named_parameters():
                if 'ac.e' in n:
                    p.data=torch.clamp(p.data,-self.thres_emb,self.thres_emb)

        return

    def eval(self,t,data,which_type,my_debug=0,input_tokens=None):
        which_type='test'
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()

        example_id = -1
        for step, batch in enumerate(data):
            batch = [
                bat.to(self.device) if bat is not None else None for bat in batch]
            input_ids, segment_ids, input_mask, targets,_= batch
            real_b=input_ids.size(0)
            with torch.no_grad():
                task=torch.autograd.Variable(torch.LongTensor([t]).cuda())
            outputs = self.model.forward(task,input_ids, segment_ids, input_mask, which_type, s=self.smax)
            output=outputs[0][t]
            loss=self.criterion(output,targets)

            _,pred=output.max(1)
            hits=(pred==targets).float()

            # Log
            total_loss+=loss.data.cpu().numpy().item()*real_b
            total_acc+=hits.sum().data.cpu().numpy().item()
            total_num+=real_b

            # Extract attributions only for the training data, once the model has been trained
            if my_debug==1:
                # # Initialize the attribution algorithm with the model
                # # The model needs to be in training mode for the backward call for gradient computation (per error message)
                # # TODO: Is this really needed? Documentation for example does not have this.
                # self.model.train()
                # integrated_gradients = IntegratedGradients(self.model)
                # # Ask the algorithm to attribute our output target to input features
                # # TODO: Increase n_steps, identify baseline
                # # print('input:',input_ids[0:1])
                # # print('additional fwd args:', task, input_ids[0:1], segment_ids[0:1], input_mask[0:1], which_type, self.smax)
                # # print('target:',targets[0:1])
                # input_embedding = self.model.bert(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)['last_hidden_state']
                # # Construct custom baseline per input sample
                # for i,example_input_embedding in enumerate(input_embedding):
                    # # Count the example id to index the tokens for getting token length excl CLS, SEP and padding tokens
                    # example_id += 1
                    # # print(example_input_embedding[0,:].shape)
                    # # print(torch.zeros([len(input_tokens[example_id]),768]).shape)
                    # # print(example_input_embedding[len(input_tokens[example_id])+1:,:].shape)
                    # # print('\n example input embedding shape:',example_input_embedding.shape)
                    # example_baseline_embedding = torch.cat((example_input_embedding[0,:].reshape([1,768]) # CLS token
                                                            # ,torch.zeros([len(input_tokens[example_id]),768]).to("cuda:0") # Actual input tokens
                                                            # ,example_input_embedding[len(input_tokens[example_id])+1:,:]) # SEP and PAD tokens
                                                        # , axis=0).reshape([1,128,768])
                    # if i==0:
                        # baseline_embedding = example_baseline_embedding
                    # else:
                        # baseline_embedding = torch.cat((baseline_embedding,example_baseline_embedding), axis=0)
                # # print(input_embedding[0])
                # # print(baseline_embedding[0])
                # # print('Input:',input_embedding.shape, input_embedding.dtype)
                # # print('Baseline:',baseline_embedding.shape, baseline_embedding.dtype)
                # attributions_ig_b = integrated_gradients.attribute(inputs=input_embedding
                                                                    # # Note: Attributions are not computed with respect to these additional arguments
                                                                    # , additional_forward_args=(task, segment_ids, input_mask
                                                                                                # , which_type, self.smax
                                                                                                # , -1, 1)
                                                                    # , target=pred, n_steps=10
                                                                    # ,baselines=(baseline_embedding))
                # # Get the max attribution across embeddings per token
                # # attributions_ig_b = torch.sum(attributions_ig_b, dim=2)
                # attributions_ig_b, attributions_ig_indices_b = torch.max(attributions_ig_b, dim=2) # Note: If multiple max exists, only the first is returned
                # # print('IG attributions:',attributions_ig.shape)
                
                # # interpretable_emb = configure_interpretable_embedding_layer(self.model, 'bert')
                # print(list(self.model.children()))
                # lrp = LRP(torch.nn.Sequential(*list(self.model.children())[1:]))
                # # input_emb = interpretable_emb.indices_to_embeddings(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
                # # print('Input:',input_emb.shape, input_emb.dtype)
                # attributions_lrp = lrp.attribute(inputs=input_embedding['last_hidden_state']
                                                # # Note: Attributions are not computed with respect to these additional arguments
                                                # , additional_forward_args=(task, segment_ids, input_mask
                                                                            # , which_type, self.smax
                                                                            # , -1, 1)
                                                # , target=pred)
                # # remove_interpretable_embedding_layer(self.model, interpretable_emb)
                # print('LRP attributions:',attributions_lrp.shape)
                
                # occlusion = Occlusion(self.model)
                # input_embedding=self.model.bert(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)['last_hidden_state']
                # sliding_window_shapes=(1,768)
                # # baseline_embedding = self.model.bert()
                # attributions_occ_b = occlusion.attribute(inputs=input_embedding
                                                        # # Note: Attributions are not computed with respect to these additional arguments
                                                        # , additional_forward_args=(task, segment_ids, input_mask
                                                                                    # , which_type, self.smax
                                                                                    # , -1, 1)
                                                        # , target=pred
                                                        # , sliding_window_shapes=sliding_window_shapes
                                                        # )
                # print(attributions_occ_b[0,0,:50])
                # attributions_occ_b = torch.sum(attributions_occ_b, dim=2)
                # # attributions_occ_b, attributions_occ_indices_b = torch.max(attributions_occ_b, dim=2) # Note: If multiple max exists, only the first is returned
                
                # My Occlusion-1
                # print(input_ids.shape)
                # print(len(input_ids[0]))
                # print('--------------------------')
                print('step:',step)
                occ_mask = torch.ones((input_ids.shape[1]-2,input_ids.shape[1])).to('cuda:0')
                for token in range(input_ids.shape[1]-2):
                    occ_mask[token,token+1] = 0 # replace with padding token

                for i in range(len(input_ids)): # loop through each input in the batch
                    temp_input_ids = input_ids[i:i+1,:].detach().clone().to('cuda:0') # using input_ids[:1,:] instead of input_ids[0] maintains the 2D shape of the tensor
                    my_input_ids = (temp_input_ids*occ_mask).long()
                    my_segment_ids = segment_ids[i:i+1,:].repeat(segment_ids.shape[1]-2,1)
                    my_input_mask = input_mask[i:i+1,:].repeat(input_mask.shape[1]-2,1)
                    # print('--------------------------')
                    # print(input_ids.shape)
                    occ_output = self.model.forward(task,my_input_ids, my_segment_ids, my_input_mask, which_type, s=self.smax)[0][t]
                    occ_output = torch.nn.Softmax(dim=1)(occ_output)
                    actual_output = self.model.forward(task,input_ids[i:i+1,:], segment_ids[i:i+1,:], input_mask[i:i+1,:], which_type, s=self.smax)[0][t]
                    actual_output = torch.nn.Softmax(dim=1)(actual_output)
                    occ_output = torch.cat((actual_output,occ_output,actual_output), axis=0) # placeholder for CLS and SEP such that their attribution scores are 0
                    _,actual_pred = actual_output.max(1)
                    _,pred=output.max(1)
                    # print(occ_output)
                    # print(actual_output)
                    attributions_occ1_b = torch.subtract(actual_output,occ_output)[:,[actual_pred.item()]] # attributions towards the predicted class
                    attributions_occ1_b = torch.transpose(attributions_occ1_b, 0, 1)
                    attributions_occ1_b = attributions_occ1_b.detach().cpu()
                    
                    # # My Occlusion-2
                    # for token1 in range(1,len(input_ids[i])-1): # loop through all tokens except CLS and SEP
                        # for token2 in range(token1+1,len(input_ids[i])-1): # loop through all subsequent tokens except SEP
                            # temp_input_ids = input_ids[i:i+1,:].detach().clone() # using input_ids[:1,:] instead of input_ids[0] maintains the 2D shape of the tensor
                            # temp_input_ids[0,token1] = 0 # replace with padding token
                            # temp_input_ids[0,token2] = 0 # replace with padding token
                            # # if i==0 and token1==2:
                                # # print('Occ-2')
                                # # print(temp_input_ids)
                            # if token1==1 and token2==2:
                                # my_input_ids = temp_input_ids
                                # my_segment_ids = segment_ids[i:i+1,:]
                                # my_input_mask = input_mask[i:i+1,:]
                            # else:
                                # my_input_ids = torch.cat((my_input_ids,temp_input_ids), axis=0)
                                # my_segment_ids = torch.cat((my_segment_ids,segment_ids[i:i+1,:]), axis=0)
                                # my_input_mask = torch.cat((my_input_mask,input_mask[i:i+1,:]), axis=0)
                    # # print('--------------------------')
                    # # print(input_ids.shape)
                    # print(len(my_input_ids))
                    # outputs = self.model.forward(task, my_input_ids, my_segment_ids, my_input_mask, which_type, s=self.smax)
                    # output=outputs[0][t]
                    # attributions_occ2_b=output
                    # _,pred=output.max(1)
                
                
                    if step==0 and i==0:
                        # attributions_ig = attributions_ig_b
                        # attributions_ig_indices = attributions_ig_indices_b
                        attributions_occ1 = attributions_occ1_b
                        # attributions_occ2 = attributions_occ2_b
                        # attributions_occ_indices = attributions_occ_indices_b
                        predictions = pred
                        class_targets = targets
                    else:
                        # attributions_ig = torch.cat((attributions_ig,attributions_ig_b), axis=0)
                        # attributions_ig_indices = torch.cat((attributions_ig_indices,attributions_ig_indices_b), axis=0)
                        attributions_occ1 = torch.cat((attributions_occ1,attributions_occ1_b), axis=0)
                        # attributions_occ2 = torch.cat((attributions_occ2,attributions_occ2_b), axis=0)
                        # attributions_occ_indices = torch.cat((attributions_occ_indices,attributions_occ_indices_b), axis=0)
                        predictions = torch.cat((predictions,pred), axis=0)
                        class_targets = torch.cat((class_targets,targets), axis=0)
                    # break
            # break

            else:
                if step==0:
                    predictions = pred
                    class_targets = targets
                else:
                    predictions = torch.cat((predictions,pred), axis=0)
                    class_targets = torch.cat((class_targets,targets), axis=0)

        if my_debug==1:    
            # print('IG attributions:',attributions_ig.shape, attributions_ig_indices.shape)
            # print('IG attributions:',attributions_occ.shape, attributions_occ_indices.shape)
            # print('Predictions:',predictions.shape)
            return class_targets, predictions, attributions_occ1 #, attributions_occ2 #, attributions_occ_indices #attributions_ig, attributions_ig_indices, attributions_lrp

        return total_loss/total_num,total_acc/total_num, f1_score(class_targets, predictions, average='macro', zero_division=1)
