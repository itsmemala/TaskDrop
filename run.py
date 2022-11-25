import sys,os,argparse,time
import numpy as np
import pickle
import torch
from config import set_args
import utils
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import logging
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, ConcatDataset
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
# os.environ['CUDA_VISIBLE_DEVICES'] = "2"
tstart=time.time()

# Arguments


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

args = set_args()

args.output='TaskDrop/res/'+'dis_'+args.experiment+'_'+args.approach+'_'+str(args.note)+'.txt'
performance_output=args.output+'_performance'
performance_output_forward=args.output+'_forward_performance'
performance_saving=args.output+'.pt'
print('='*100)
print('Arguments =')
for arg in vars(args):
    print('\t'+arg+':',getattr(args,arg))
print('='*100)

########################################################################################################################

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
else: print('[CUDA unavailable]'); sys.exit()

# Args -- Experiment
if args.experiment=='dis':
    args.ntasks=6
    from dataloaders import bert_dis as dataloader
elif args.experiment=='sim':
    args.ntasks=6
    from dataloaders import bert_sim as dataloader
elif args.experiment=='mix':
    args.ntasks=24
    from dataloaders import bert2 as dataloader

# Args -- Approach
if args.approach=='taskdrop':
    from approaches import taskdrop as approach

# # Args -- Network
if args.approach=='taskdrop':
    from networks import taskdrop as network

########################################################################################################################

# Load
print('Load data...')
data,taskcla=dataloader.get(logger=logger,args=args)

print('\nTask info =',taskcla)

# Inits
print('Inits...')
net=network.Net(taskcla,args=args).cuda()


appr=approach.Appr(net,logger=logger,args=args)

# Loop tasks
acc=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
lss=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)

my_save_path = '/content/gdrive/MyDrive/nomask_myocc1occ2_attributions/'
# my_save_path = '/content/gdrive/MyDrive/taskdrop_test_attributions/'

for t,ncla in taskcla:

    print('*'*100)
    print('Task {:2d} ({:s})'.format(t,data[t]['name']))
    print('*'*100)

    # if t>1: exit()

    if 'mtl' in args.approach:
        if t==0:
            train=data[t]['train']
            valid=data[t]['valid']
            num_train_steps=data[t]['num_train_steps']

        else:
            train = ConcatDataset([train,data[t]['train']])
            valid = ConcatDataset([valid,data[t]['valid']])
            num_train_steps+=data[t]['num_train_steps']
        task=t

        if t < len(taskcla)-1: continue #only want the last one

    else:
        train=data[t]['train']
        valid=data[t]['valid']
        num_train_steps=data[t]['num_train_steps']
        task=t

    train_sampler = RandomSampler(train)
    train_dataloader = DataLoader(train, sampler=train_sampler, batch_size=args.train_batch_size)

    valid_sampler = SequentialSampler(valid)
    valid_dataloader = DataLoader(valid, sampler=valid_sampler, batch_size=args.eval_batch_size)

    # Save train data tokens
    # print(len(data[t]['train_tokens'][0]))
    # print(data[t]['train_tokens'][0])
    # print(len(data[t]['train_tokens']))
    # print(len(data[t]['train']))
    with open(my_save_path+"inputtokens_task"+str(t)+".txt", "wb") as internal_filename:
        pickle.dump(data[t]['train_tokens'], internal_filename)

    if t>0:
        # Retain the order of the dataset, i.e. no shuffling, for comparing attributions at different points
        train_sampler_saving = SequentialSampler(train)
        train_dataloader_saving = DataLoader(train, sampler=train_sampler_saving, batch_size=args.train_batch_size)

        # Calculate attributions on new tasks before training
        targets, predictions, attributions_occ1, attributions_occ2 = appr.eval(t,train_dataloader_saving,'mcl'
                                                                                ,my_debug=1,input_tokens=data[t]['train_tokens'])
        np.savez_compressed(my_save_path+'attributions_model'+str(t-1)+'task'+str(t)
                            ,targets=targets.cpu()
                            ,predictions=predictions.cpu()
                            ,attributions_occ1=attributions_ig.cpu()
                            ,attributions_occ2=attributions_ig.cpu()
                            # ,attributions_ig=attributions_occ.cpu()
                            # ,attributions_ig_indices=attributions_ig_indices.cpu()
                            #,attributions_lrp=attributions_lrp
                            )

    # Train
    appr.train(task,train_dataloader,valid_dataloader,args)
    print('-'*100)


    # Test
    for u in range(t+1):
        test=data[u]['test']
        test_sampler = SequentialSampler(test)
        test_dataloader = DataLoader(test, sampler=test_sampler, batch_size=args.eval_batch_size)
        
        test_loss,test_acc=appr.eval(u,test_dataloader,'mcl')
        # if 'kan' in args.approach:
        #     test_loss,test_acc=appr.eval(u,test_dataloader,'mcl')
        # else:
        #     test_loss,test_acc=appr.eval(u,test_dataloader)
        print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u,data[u]['name'],test_loss,100*test_acc))
        acc[t,u]=test_acc
        lss[t,u]=test_loss
        
        # Calculate attributions on all previous tasks and current task after training
        train = data[u]['train']
        train_sampler = SequentialSampler(train) # Retain the order of the dataset, i.e. no shuffling
        train_dataloader = DataLoader(train, sampler=train_sampler, batch_size=args.train_batch_size)
        targets, predictions, attributions_occ1, attributions_occ2 = appr.eval(u,train_dataloader,'mcl'
                                                                                ,my_debug=1,input_tokens=data[u]['train_tokens'])
        np.savez_compressed(my_save_path+'attributions_model'+str(t)+'task'+str(u)
                            ,targets=targets.cpu()
                            ,predictions=predictions.cpu()
                            ,attributions_occ1=attributions_ig.cpu()
                            ,attributions_occ2=attributions_ig.cpu()
                            # ,attributions_ig=attributions_ig.detach().cpu()
                            # ,attributions_ig_indices=attributions_ig_indices.cpu()
                            #,attributions_lrp=attributions_lrp
                            )

    # Save
    print('Save at '+args.output)
    np.savetxt(args.output,acc,'%.4f',delimiter='\t')

    # appr.decode(train_dataloader)
    # if t==0:
        # break

# Done
print('*'*100)
print('Accuracies =')
for i in range(acc.shape[0]):
    print('\t',end='')
    for j in range(acc.shape[1]):
        print('{:5.1f}% '.format(100*acc[i,j]),end='')
    print()
print('*'*100)
print('Done!')

print('[Elapsed time = {:.1f} h]'.format((time.time()-tstart)/(60*60)))

# Save results
torch.save((acc,lss),performance_saving)

if args.save_model=='yes':
    torch.save(appr,'TaskDrop/res/model/'+args.approach+'_'+str(args.note)+'.pt')
