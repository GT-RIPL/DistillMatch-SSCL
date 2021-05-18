from __future__ import print_function
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from types import MethodType
import models
from utils.metric import accuracy, AverageMeter, Timer
import numpy as np
from torch.optim import Optimizer
import contextlib
from models.layers import CosineScaling

class NormalNN(nn.Module):
    """
    consider citing the benchmarking environment this was built on top of

    git url: https://github.com/GT-RIPL/Continual-Learning-Benchmark

    @article{hsu2018re,
        title={Re-evaluating continual learning scenarios: A categorization and case for strong baselines},
        author={Hsu, Yen-Chang and Liu, Yen-Cheng and Ramasamy, Anita and Kira, Zsolt},
        journal={arXiv preprint arXiv:1810.12488},
        year={2018}
    }
    """
    def __init__(self, learner_config):

        super(NormalNN, self).__init__()
        self.log = print
        self.config = learner_config
        self.model = self.create_model()
        self.reset_optimizer = True

        # class balancing
        self.dw = self.config['DW']
        self.dw_thresh = 10.0 # never let gradient weight get above this number

        # supervised criterion
        if self.dw:
            self.criterion_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
        else:
            self.criterion_fn = nn.CrossEntropyLoss()
        
        # cuda gpu
        if learner_config['gpuid'][0] >= 0:
            self.cuda()
            self.gpu = True
        else:
            self.gpu = False
        
        # highest class index from past task
        self.last_valid_out_dim = 0 

        # highest class index from current task
        self.valid_out_dim = 0

        # replay memory parameters
        self.memory_size = self.config['memory']
        self.task_count = 0

        # ssl
        self.weight_aux = self.config['weight_aux']
        
        # set up schedules
        self.schedule_type = self.config['schedule_type']
        self.ft = self.config['FT']
        if self.memory_size == 0: self.ft = False
        if self.ft:
            if self.schedule_type == 'decay':
                self.fschedule = [10, 15, 20]
                self.schedule = self.config['schedule'][:-1]
        else:
            # no finetuning schedule
            self.fschedule = [0]
            self.schedule = self.config['schedule']

        # initialize optimizer
        self.init_optimizer()

    # sets model optimizers
    def init_optimizer(self):

        # parse optimizer args
        optimizer_arg = {'params':self.model.parameters(),
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        finetune_optimizer_arg = {'params':filter(lambda p: p.requires_grad, self.model.last.parameters()),
                         'lr':self.config['lr']/10.0,
                         'weight_decay':self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD','RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
            finetune_optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
            finetune_optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            finetune_optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'
        elif self.config['optimizer'] == 'Adam':
            optimizer_arg['betas'] = (self.config['momentum'],0.999)
            finetune_optimizer_arg['betas'] = (self.config['momentum'],0.999)

        # create optimizers
        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        self.finetune_optimizer = torch.optim.__dict__[self.config['optimizer']](**finetune_optimizer_arg)
        
        # create schedules
        if self.schedule_type == 'decay':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.schedule, gamma=0.1)
            self.finetune_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.finetune_optimizer, milestones=self.fschedule, gamma=0.1)

    # returns optimizer for passed model
    def new_optimizer(self, model):

        # parse optimizer args
        optimizer_arg = {'params':model.parameters(),
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        finetune_optimizer_arg = {'params':filter(lambda p: p.requires_grad, model.last.parameters()),
                         'lr':self.config['lr']/10.0,
                         'weight_decay':self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD','RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
            finetune_optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
            finetune_optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            finetune_optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'
        elif self.config['optimizer'] == 'Adam':
            optimizer_arg['betas'] = (self.config['momentum'],0.999)
            finetune_optimizer_arg['betas'] = (self.config['momentum'],0.999)

        # create optimizers
        optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        finetune_optimizer = torch.optim.__dict__[self.config['optimizer']](**finetune_optimizer_arg)

        # create schedules
        if self.schedule_type == 'decay':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.schedule, gamma=0.1)
            finetune_scheduler = torch.optim.lr_scheduler.MultiStepLR(finetune_optimizer, milestones=self.fschedule, gamma=0.1)

        return optimizer, scheduler, finetune_optimizer, finetune_scheduler

    def create_model(self):
        cfg = self.config

        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']]()

        # Apply network surgery to the backbone
        # Create the heads for tasks (It can be single task or multi-task)
        n_feat = model.last.in_features
        if isinstance(model.last, CosineScaling):
            model.last =  CosineScaling(n_feat, cfg['out_dim'])
        else:
            model.last = nn.Linear(n_feat, cfg['out_dim'])

        return model

    def create_ood_model(self):
        cfg = self.config

        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        model = models.__dict__[cfg['model_type']].__dict__[cfg['ood_model_name']]()

        # Apply network surgery to the backbone
        # Create the heads for tasks (It can be single task or multi-task)
        n_feat = model.last.in_features
        if isinstance(model.last, CosineScaling):
            model.last =  CosineScaling(n_feat, cfg['out_dim'])
        else:
            model.last = nn.Linear(n_feat, cfg['out_dim'])

        return model

    
    def reset_model(self):
        self.model.apply(weight_reset)

    def forward(self, x, ood=False):
        if ood:
            return self.ood_model.forward(x)[:, :self.valid_out_dim]
        else:
            return self.model.forward(x)[:, :self.valid_out_dim]

    def predict(self, inputs):
        self.model.eval()
        out = self.forward(inputs)
        return out

    def validation(self, dataloader, model=None, task_in = None):

        if model is None: model = self.model

        # This function doesn't distinguish tasks.
        batch_timer = Timer()
        acc = AverageMeter()
        batch_timer.tic()

        orig_mode = model.training
        model.eval()
        for i, (input, target, task) in enumerate(dataloader):

            if self.gpu:
                with torch.no_grad():
                    input = input.cuda()
                    target = target.cuda()
            if task_in is None:
                output = model.forward(input)[:, :self.valid_out_dim]
                acc = accumulate_acc(output, target, task, acc)
            else:
                output = model.forward(input)[:, task_in]
                acc = accumulate_acc(output, target-task_in[0], task, acc)
            
        model.train(orig_mode)

        self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
                 .format(acc=acc, time=batch_timer.toc()))
        return acc.avg

    def criterion(self, logits, targets, **kwargs):

        # labeled loss
        if self.dw:
            dw_cls     = self.dw_d['seen'][targets.long()]
            loss_supervised = (self.criterion_fn(logits, targets.long()) * dw_cls).mean()
        else:
            loss_supervised = self.criterion_fn(logits, targets.long())

        # return loss
        return loss_supervised 

    # data weighting
    def data_weighting(self, dataset, num_seen=None):

        # count number of examples in dataset per class
        print('*************************\n\n\n')
        if num_seen is None:
            labels = [int(dataset[i][1]) for i in range(len(dataset))]
            labels = np.asarray(labels, dtype=np.int64)
            num_seen = np.asarray([len(labels[labels==k]) for k in range(self.valid_out_dim)], dtype=np.float32)
        print('num seen:' + str(num_seen))
        
        # in case a zero exists in PL...
        num_seen += 1

        # local
        local = np.ones(self.valid_out_dim - self.last_valid_out_dim + 1, dtype=np.float32)
        local = torch.tensor(local)
        stats_local = num_seen[self.last_valid_out_dim:self.valid_out_dim]
        local_dw = np.ones(self.valid_out_dim - self.last_valid_out_dim + 1, dtype=np.float32)
        local_dw[:self.valid_out_dim - self.last_valid_out_dim ] = stats_local.sum() / (stats_local * len(stats_local))
        local_dw[local_dw > self.dw_thresh] = self.dw_thresh
        local_dw= torch.tensor(local_dw)

        # previous seen
        prev = np.ones(self.last_valid_out_dim + 1, dtype=np.float32)
        prev = torch.tensor(prev)
        stats_prev = num_seen[:self.last_valid_out_dim]
        prev_dw = np.ones(self.last_valid_out_dim + 1, dtype=np.float32)
        prev_dw[:self.last_valid_out_dim] = stats_prev.sum() / (stats_prev * len(stats_prev))
        prev_dw[prev_dw > self.dw_thresh] = self.dw_thresh
        prev_dw = torch.tensor(prev_dw)

        # all seen
        seen = np.ones(self.valid_out_dim + 1, dtype=np.float32)
        seen = torch.tensor(seen)
        seen_dw = np.ones(self.valid_out_dim + 1, dtype=np.float32)
        seen_dw[:self.valid_out_dim] = num_seen.sum() / (num_seen * len(num_seen))
        seen_dw[seen_dw > self.dw_thresh] = self.dw_thresh
        seen_dw = torch.tensor(seen_dw)

        # maps
        seen_map = np.full(self.valid_out_dim+1, -1)
        for i in range(self.valid_out_dim): seen_map[i] = i
        seen_map = torch.tensor(seen_map)
        prev_map = np.full(self.valid_out_dim+1, -1)
        for i in range(self.last_valid_out_dim): prev_map[i] = i
        prev_map = torch.tensor(prev_map)
        cur_map =  np.full(self.valid_out_dim+1, -1)
        for i in range(self.last_valid_out_dim,self.valid_out_dim): cur_map[i] = i - self.last_valid_out_dim
        cur_map = torch.tensor(cur_map)

        dw_c = {'seen':None,'prev':None,'local':local}
        if self.dw:
            dw_d = {'seen':seen_dw,'prev':prev_dw,'local':local_dw}
        else:
            dw_d = {'seen':seen,'prev':prev,'local':local}
        print('dw_c')
        print(dw_c)
        print('dw_d')
        print(dw_d)
        print(seen_map)
        print(prev_map)
        print(cur_map)
        print('\n\n\n*************************')

        # cuda
        if self.cuda:
            for key in dw_c: 
                if dw_c[key] is not None: dw_c[key] = dw_c[key].cuda()
            for key in dw_d: 
                if dw_d[key] is not None: dw_d[key] = dw_d[key].cuda()
            prev_map = prev_map.cuda()
            cur_map = cur_map.cuda()
            seen_map = seen_map.cuda()
        
        self.dw_c = dw_c
        self.dw_d = dw_d
        self.prev_map = prev_map
        self.cur_map = cur_map
        self.seen_map = seen_map

    def update_model(self, inputs_labeled, targets):
        
        logits_labeled = self.forward(inputs_labeled)
        total_loss, loss_supervised, loss_unsupervised = self.criterion(logits_labeled, targets.long())

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), logits_labeled, loss_unsupervised.detach(), loss_supervised.detach()

    def learn_batch(self, train_loader, train_dataset, val_loader=None, **kwargs):
        
        if self.reset_optimizer:  # Reset optimizer before learning each task
            self.log('Optimizer is reset!')
            self.init_optimizer()

        self.data_weighting(train_dataset)
        
        # Evaluate the performance of current task
        if val_loader is not None:
            self.validation(val_loader)
        
        for epoch in range(self.config['schedule'][-1]):
            
            self.epoch=epoch
            data_timer = Timer()
            batch_timer = Timer()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            losses_ul = AverageMeter()
            losses_ce = AverageMeter()
            acc = AverageMeter()

            # Config the model and optimizer
            self.log('Epoch:{0}'.format(epoch+1))
            self.model.train()
            self.scheduler.step(epoch)
            for param_group in self.optimizer.param_groups:
                self.log('LR:', param_group['lr'])

            # Learning with mini-batch
            data_timer.tic()
            batch_timer.tic()
            for i, (xl, y, xul, yul, task)  in enumerate(train_loader):

                data_time.update(data_timer.toc())  # measure data loading time
                self.batch_xl = y.size(0)

                prev_preds = torch.empty([], requires_grad=False)
                if self.gpu:
                    xl = [xl[k].cuda() for k in range(len(xl))]
                    y = y.cuda()
                
                # model update
                loss, output, lu, ls = self.update_model(xl[0], y)
                y = y.detach()

                # measure accuracy and record loss
                accumulate_acc(output, y, task, acc)
                losses.update(loss,  y.size(0))   
                losses_ul.update(lu, y.size(0))   
                losses_ce.update(ls, y.size(0))   
                batch_time.update(batch_timer.toc())  # measure elapsed time
                data_timer.toc()

            self.log(' * Loss {loss.avg:.3f} | CE {loss_ce.avg:.3f} | Ul {loss_ul.avg:.3f}'.format(loss=losses,
                        loss_ce=losses_ce,loss_ul=losses_ul))
            self.log(' * Train Acc {acc.avg:.3f}'.format(acc=acc))

            # Evaluate the performance of current task
            if val_loader is not None:
                self.validation(val_loader)

        self.model.eval()

        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False

        # Extend memory
        self.task_count += 1
        if self.memory_size > 0:
            train_dataset.update_coreset(self.memory_size, np.arange(self.last_valid_out_dim))

    def add_valid_output_dim(self, dim=0):
        # This function is kind of ad-hoc, but it is the simplest way to support incremental class learning
        self.log('Incremental class: Old valid output dimension:', self.valid_out_dim)
        self.valid_out_dim += dim
        self.log('Incremental class: New Valid output dimension:', self.valid_out_dim)
        return self.valid_out_dim

    def count_parameter(self):
        return sum(p.numel() for p in self.model.parameters())

    def save_model(self, filename):
        model_state = self.model.state_dict()
        if isinstance(self.model, torch.nn.DataParallel):
            # Get rid of 'module' before the name of states
            model_state = self.model.module.state_dict()
        for key in model_state.keys():  # Always save it to cpu
            model_state[key] = model_state[key].cpu()
        print('=> Saving model to:', filename)
        torch.save(model_state, filename + '.pth')
        print('=> Save Done')

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def accumulate_acc(output, target, task, meter):
    labeled_inds = (target >= 0).nonzero().flatten()  # only use labeled data
    labeled_target = target[labeled_inds]
    labeled_output = output[labeled_inds]
    if len(labeled_target) > 0:
        meter.update(accuracy(labeled_output, labeled_target), len(labeled_target))
    return meter

