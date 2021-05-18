import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
import models
import math
from utils.metric import accuracy, AverageMeter, Timer
import numpy as np
import contextlib
from torch.optim import Optimizer
import dataloaders
from models.layers import CosineScaling

class GD(nn.Module):
    """
    adaptation of Global Distillation Method and Baselines E2E and DR

    git url: https://github.com/kibok90/iccv2019-inc

    @inproceedings{lee2019overcoming,
        title={Overcoming catastrophic forgetting with unlabeled data in the wild},
        author={Lee, Kibok and Lee, Kimin and Shin, Jinwoo and Lee, Honglak},
        booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
        pages={312--321},
        year={2019}
    }
    """

    # standard init
    def __init__(self, learner_config):
        super(GD, self).__init__()
        self.config = learner_config
        self.first_task = True
        self.prior_task_distill_data = 'all'
        self.log = print

        # declare model (M), current_model (C), and previous model (P)
        self.model = self.create_model()
        self.current_model = self.create_model()
        self.prior_model = self.create_model()

        # declare losses
        self.criterion = {'cl': nn.CrossEntropyLoss(reduction='none', ignore_index=-1), 'kl': nn.KLDivLoss(reduction='none')}
        self.distill_loss = self.config['distill_loss']

        # replay memory parameters
        self.memory_size = self.config['memory']
        self.task_count = 0

        # distillation constants (may later add to experiment configs)
        self.finetuning = False
        self.dw = self.config['DW']
        if self.memory_size == 0: self.dw = False
        self.ft = self.config['FT']
        if self.memory_size == 0: self.ft = False
        self.kdr = 1.0
        self.T = 2.0
        self.qT = 1.0
        self.schedule_type = self.config['schedule_type']
        if self.ft:
            # finetuning schedule
            if self.schedule_type == 'decay':
                self.fschedule = [10, 15, 20]
                self.mschedule = self.config['schedule'][:-1]
        else:
            # no finetuning schedule
            self.fschedule = [0]
            self.mschedule = self.config['schedule']

        self.unknown_boundary_class = -1
       
        # send models and loosses to gpu
        if learner_config['gpuid'][0] >= 0:
            self.cuda()
            self.gpu = True
        else:
            self.gpu = False

        # will be initialized in add_valid_out_dim, called in main
        self.valid_out_dim = 0

        # classes up to which P knows
        self.last_valid_out_dim = 0 

        # retain past tasks
        self.past_tasks = []

        # confidence
        self.co = learner_config['co']

        # no unlabeled data
        self.no_unlabeled_data = learner_config['no_unlabeled_data']

    # create model
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

    # init optimizers for classification (C), distill (M), and finetuning (M)
    # called at start of each task
    def optimizer_init(self):
        
        # C
        self.optimizer_arg = {'params':self.current_model.parameters(),
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        # M
        self.distill_optimizer_arg = {'params':self.model.parameters(),
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        # M linear heads - smaller learning rate
        if ('Q' in self.distill_loss):
            self.finetune_optimizer_arg = {'params':filter(lambda p: p.requires_grad, self.model.last.parameters()),
                            'lr':self.config['lr']/10.0,
                            'weight_decay':self.config['weight_decay']}
        else:
            self.finetune_optimizer_arg = {'params':self.model.parameters(),
                            'lr':self.config['lr']/10.0,
                            'weight_decay':self.config['weight_decay']}
        
        # other optimization arguments
        if self.config['optimizer'] in ['SGD','RMSprop']:
            self.optimizer_arg['momentum'] = self.config['momentum']
            self.distill_optimizer_arg['momentum'] = self.config['momentum']
            self.finetune_optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            self.optimizer_arg.pop('weight_decay')
            self.distill_optimizer_arg.pop('weight_decay')
            self.finetune_optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            self.optimizer_arg['amsgrad'] = True
            self.distill_optimizer_arg['amsgrad'] = True
            self.finetune_optimizer_arg['amsgrad'] = True

        # create optimizers
        self.current_optimizer = torch.optim.__dict__[self.config['optimizer']](**self.optimizer_arg)
        self.distill_optimizer = torch.optim.__dict__[self.config['optimizer']](**self.distill_optimizer_arg)
        self.finetune_optimizer = torch.optim.__dict__[self.config['optimizer']](**self.finetune_optimizer_arg)
        
        # create schedules
        if self.schedule_type == 'decay':
            self.current_schedule = torch.optim.lr_scheduler.MultiStepLR(self.current_optimizer, milestones=self.config['schedule'], gamma=0.1)
            self.distill_schedule = torch.optim.lr_scheduler.MultiStepLR(self.distill_optimizer, milestones=self.mschedule, gamma=0.1)
            self.distill_schedule_finetune = torch.optim.lr_scheduler.MultiStepLR(self.finetune_optimizer, milestones=self.fschedule, gamma=0.1)  
          

    # updates to current model C
    def global_distill_update0(self, batch_inputs_all, batch_targets, tasks, optimizer, dw, **kwargs):

        # init loss
        loss = torch.zeros((1,), requires_grad=True).cuda() 
        optimizer.zero_grad()

        # labelled vs unlabelled indexes
        labeled_mask = (batch_targets > self.unknown_boundary_class)
        labeled_ind = labeled_mask.nonzero().view(-1)

        # current task location
        cur_mask = (batch_targets >= self.last_valid_out_dim)
        cur_ind = cur_mask.nonzero().view(-1)

        # confidence task location
        co_mask = (batch_targets < self.last_valid_out_dim)
        co_ind = co_mask.nonzero().view(-1)

        # to save in logging
        task_all = np.arange(0,self.valid_out_dim)
        final_logits_out = self.current_model.forward(batch_inputs_all[0])[:, task_all]

        # soft augmentations
        batch_inputs = batch_inputs_all[0]

        # simple classification using current model
        #
        # we only compute over the current task dimensions
        # therefore, we need to shift the targets!
        logits_out = self.current_model.forward(batch_inputs)
        if len(cur_ind) > 0:
            class_out = logits_out[cur_ind , self.last_valid_out_dim:self.valid_out_dim]
            dw_cls     = dw['local'][-1][batch_targets[cur_ind ]-self.last_valid_out_dim]
            loss += (self.criterion['cl'](class_out, batch_targets[cur_ind ]-self.last_valid_out_dim) * dw_cls).mean()

        # confidence loss
        if (self.co > 0) and (len(co_ind) > 0) and (not self.first_task):
            output_conf = logits_out[co_ind, self.last_valid_out_dim:self.valid_out_dim]
            loss -= (output_conf.log_softmax(dim=1).mean() + math.log(self.valid_out_dim-self.last_valid_out_dim)) * self.co

        # step optimizer
        loss.backward()
        optimizer.step()
        return loss[0].detach(), final_logits_out
        
    # updates to model M
    def global_distill_update(self, batch_inputs_all, batch_targets, tasks, optimizer, dw, prev_map, cur_map, seen_map, **kwargs):

        # remove hard augmentations
        batch_inputs = batch_inputs_all[0]
        optimizer.zero_grad()

        # init loss
        loss = torch.zeros((1,), requires_grad=True).cuda() 

        # other outputs
        prior_logits = self.prior_model.forward(batch_inputs).detach()
        new_logits = self.current_model.forward(batch_inputs).detach()
        final_logits = self.model.forward(batch_inputs)

        # dimensions for previous task, new task, and combined task
        task_n = np.arange(self.last_valid_out_dim,self.valid_out_dim)
        task_p = np.arange(0,self.last_valid_out_dim)
        task_all = np.arange(0,self.valid_out_dim)

        # labelled vs unlabelled indexes
        unlabeled_mask = (batch_targets == self.unknown_boundary_class)
        unlabeled_ind = unlabeled_mask.nonzero().view(-1)
        labeled_mask = (batch_targets > self.unknown_boundary_class)
        labeled_ind = labeled_mask.nonzero().view(-1)

        # to save in logging
        final_logits_out = final_logits[:, task_all]
        
        # compute loss
        class_out = self.model.forward(batch_inputs_all[0][labeled_ind])[:, task_all]
        dw_cls     = dw['seen'][batch_targets[labeled_ind]]
        loss_class = (self.criterion['cl'](class_out, batch_targets[labeled_ind]) * dw_cls).sum()
        
        # distillation from Q (ensemble of P and C)
        unlabeled_mask = (batch_targets == self.unknown_boundary_class)
        unlabeled_ind = unlabeled_mask.nonzero().view(-1)
        if ('Q' in self.distill_loss) and (len(unlabeled_ind) > 0):
            output_q  = (final_logits  [unlabeled_ind][:,task_all]).log_softmax(dim=1)
            starget_q = (prior_logits[unlabeled_ind][:,task_p]).softmax(dim=1)
            target_q  = seen_map[batch_targets[unlabeled_ind]]
            dw_q      = dw['seen'][target_q][:,None]
            starget_t = (prior_logits[unlabeled_ind][:,task_n]).softmax(dim=1)
            starget_q = concat_target(starget_q, starget_t, task_n, task_all, task_p)
            loss_class += (self.criterion['kl'](output_q, starget_q) * dw_q).sum(dim=1).sum() * (self.qT**2)

            # normalize global losses together
            loss_class /= (labeled_mask.to(torch.float).sum() + unlabeled_mask.to(torch.float).sum())
        else:
            loss_class /= labeled_mask.to(torch.float).sum()
        
        loss += loss_class
        

        # distillation distance loss
        if not self.finetuning and ('P' in self.distill_loss or 'C' in self.distill_loss or 'L' in self.distill_loss):            
            num_local = len(task_n) + len(task_p)
            loss_kl = torch.zeros((1,)).cuda()  

            # P
            if 'P' in self.distill_loss:
                output_pgd  = (final_logits  [:,task_p] / self.T).log_softmax(dim=1)
                starget_pgd = (prior_logits[:,task_p] / self.T).softmax(dim=1)
                bloss_pgd   = len(task_p) / num_local
                ptargets = prev_map[batch_targets]
                dw_pgd      = dw['prev'][ptargets][:,None]
                loss += (self.criterion['kl'](output_pgd, starget_pgd) * dw_pgd).sum(dim=1).mean() * (self.T**2) * bloss_pgd

            # C
            if 'C' in self.distill_loss:
                output_cdst  = (final_logits  [:,task_n] / self.T).log_softmax(dim=1)
                starget_cdst = (new_logits[:,task_n] / self.T).softmax(dim=1)
                bloss_cdst   = len(task_n) / num_local
                ctargets = cur_map[-1][batch_targets]
                dw_cdst      = dw['local'][-1][ctargets][:,None]
                loss += (self.criterion['kl'](output_cdst, starget_cdst) * dw_cdst).sum(dim=1).mean() * (self.T**2) * bloss_cdst

            # L
            if 'L' in self.distill_loss:
                for s, task_l in enumerate(self.past_tasks):

                    output_pld  = (final_logits  [:,task_l] / self.T).log_softmax(dim=1)
                    starget_pld = (prior_logits[:,task_l] / self.T).softmax(dim=1)
                    bloss_pld   = len(task_l) / num_local
                    ltargets = cur_map[s][batch_targets]
                    dw_pld      = dw['local'][s][ltargets][:,None]
                    loss += (self.criterion['kl'](output_pld, starget_pld) * dw_pld).sum(dim=1).mean() * (self.T**2) * bloss_pld

        # step optimizer
        loss.backward()
        optimizer.step()
        return loss[0].detach(), final_logits_out

    # data weighting
    def data_weighting(self, dataset):

        # count number of examples in dataset per class
        print('*************************\n\n\n')
        labels = [int(dataset[i][1]) for i in range(len(dataset))]
        labels = np.asarray(labels, dtype=np.int64)
        num_seen = np.asarray([len(labels[labels==k]) for k in range(self.valid_out_dim)], dtype=np.float32)
        print('num seen:' + str(num_seen))
        print('num unlabeled: ' + str(len(labels[labels==-1])))

        # local
        local = np.ones(self.valid_out_dim - self.last_valid_out_dim + 1, dtype=np.float32)
        local = [torch.tensor(local) for t in range(len(self.past_tasks) + 1)]
        if self.dw:
            local_dw = []
            for task in self.past_tasks:
                stats_local = num_seen[task[0]:task[-1]]
                local_dw_t = np.ones(task[-1] - task[0] + 1, dtype=np.float32)
                local_dw_t[:task[-1] - task[0] ] = stats_local.sum() / (stats_local * len(stats_local))
                local_dw.append(torch.tensor(local_dw_t))
            stats_local = num_seen[self.last_valid_out_dim:self.valid_out_dim]
            local_dw_t = np.ones(self.valid_out_dim - self.last_valid_out_dim + 1, dtype=np.float32)
            local_dw_t[:self.valid_out_dim - self.last_valid_out_dim ] = stats_local.sum() / (stats_local * len(stats_local))
            local_dw.append(torch.tensor(local_dw_t))

        # previous seen
        prev = np.ones(self.last_valid_out_dim + 1, dtype=np.float32)
        prev = torch.tensor(prev)
        stats_prev = num_seen[:self.last_valid_out_dim]
        if self.dw:
            prev_dw = np.ones(self.last_valid_out_dim + 1, dtype=np.float32)
            prev_dw[:self.last_valid_out_dim] = stats_prev.sum() / (stats_prev * len(stats_prev))
            prev_dw = torch.tensor(prev_dw)

        # all seen
        seen = np.ones(self.valid_out_dim + 1, dtype=np.float32)
        seen = torch.tensor(seen)
        if self.dw:
            seen_dw = np.ones(self.valid_out_dim + 1, dtype=np.float32)
            seen_dw[:self.valid_out_dim] = num_seen.sum() / (num_seen * len(num_seen))
            seen_dw = torch.tensor(seen_dw)

        # maps
        seen_map = np.full(self.valid_out_dim+1, -1)
        for i in range(self.valid_out_dim): seen_map[i] = i
        seen_map = torch.tensor(seen_map)
        prev_map = np.full(self.valid_out_dim+1, -1)
        for i in range(self.last_valid_out_dim): prev_map[i] = i
        prev_map = torch.tensor(prev_map)
        cur_map = []
        for task in self.past_tasks:
            cur_map_t =  np.full(self.valid_out_dim+1, -1)
            for i in task: cur_map_t[i] = i - task[0]
            cur_map_t = torch.tensor(cur_map_t)
            cur_map.append(cur_map_t)

        dw_ft = None
        dw_c = {'seen':None,'prev':None,'local':local}
        if self.dw:
            if self.ft:
                dw_d = {'seen':seen,'prev':prev,'local':local}
                dw_ft = {'seen':seen_dw,'prev':None,'local':None}
            else:
                dw_d = {'seen':seen_dw,'prev':prev_dw,'local':local_dw}
        else:
            dw_d = {'seen':seen,'prev':prev,'local':local}
            if self.ft: dw_ft = {'seen':seen,'prev':None,'local':None}
        print('dw_c')
        print(dw_c)
        print('dw_d')
        print(dw_d)
        if self.ft:
            print('dw_ft')
            print(dw_ft)
        print(seen_map)
        print(prev_map)
        print(cur_map)
        print('\n\n\n*************************')

        # cuda
        if self.cuda:
            for key in dw_c: 
                if dw_c[key] is not None:
                    if key is 'local':
                        dw_c[key] = [dw_c[key][k].cuda() for k in range(len(dw_c[key]))]
                    else:
                        dw_c[key] = dw_c[key].cuda()
            for key in dw_d: 
                if dw_d[key] is not None:
                    
                    if key is 'local':
                        dw_d[key] = [dw_d[key][k].cuda() for k in range(len(dw_d[key]))]
                    else:
                        dw_d[key] = dw_d[key].cuda()
                        
            if self.ft:
                for key in dw_ft: 
                    if dw_ft[key] is not None:
                        if key is 'local':
                            dw_ft[key] = [dw_ft[key][k].cuda() for k in range(len(dw_ft[key]))]
                        else:
                            dw_ft[key] = dw_ft[key].cuda()
            prev_map = prev_map.cuda()
            cur_map = [cur_map[k].cuda() for k in range(len(cur_map))]
            seen_map = seen_map.cuda()

        return dw_c, dw_d, dw_ft, prev_map, cur_map, seen_map

    # learn from task
    def learn_batch(self, train_loader, train_dataset, prev, val_loader=None, val_dataset=None):
        
        # do phase 1?
        do_phase_one = ((self.first_task) or ('C' in self.distill_loss))

        # update requires_grad
        # only updated heads of new task for current model C and 
        # valid classes (p+n) for model M
        task_n = np.arange(self.last_valid_out_dim,self.valid_out_dim)
        print(task_n)
        task_all = np.arange(0,self.valid_out_dim)

        # optimizer init
        self.optimizer_init()

        # get data weighting
        dw_c, dw_d, dw_ft, prev_map, cur_map, seen_map = self.data_weighting(train_dataset)
        
        ###
        # Learning phase 1: Cross Entropy for Current Model
        ###
        # validation dataset without previous classes
        val_dataset.load_dataset(prev, self.task_count, train=True)
        if do_phase_one:
            self.log(' --- Current Task Training --- ')
            for epoch in range(self.config['schedule'][-1]):

                data_timer = Timer()
                batch_timer = Timer()
                batch_time = AverageMeter()
                data_time = AverageMeter()
                losses = AverageMeter()
                acc = AverageMeter()
                self.log('Epoch:{0}'.format(epoch))

                # Config the model and optimizer
                self.model.eval()
                self.prior_model.eval()
                self.current_model.train()
                self.current_schedule.step(epoch)
                for param_group in self.current_optimizer.param_groups:
                    self.log('LR:', param_group['lr'])

                # Learning with mini-batch
                data_timer.tic()
                batch_timer.tic()
                for i, (xl, y, xul, _, task) in enumerate(train_loader):

                    # go from ssl data to gd data
                    input, target = self.collect_data(xl, y, xul)
                    data_time.update(data_timer.toc())  # measure data loading time

                    if self.gpu:
                        input = [input[k].cuda() for k in range(len(input))]
                        target = target.cuda()

                    loss, output = self.global_distill_update0(input, target, task, self.current_optimizer, dw_c)
                    target = target.detach()

                    # measure accuracy and record loss
                    accumulate_acc(output, target, task, acc)
                    losses.update(loss, target.size(0))

                    batch_time.update(batch_timer.toc())  # measure elapsed time
                    data_timer.toc()

                self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses,acc=acc))
                self.log(' * Task Train Acc {acc.avg:.3f}'.format(acc=acc))

                # Evaluate the performance of current task
                if val_loader is not None:
                    batch_timer = Timer()
                    acc = AverageMeter()
                    batch_timer.tic()

                    orig_mode = self.current_model.training
                    self.current_model.eval()
                    for i, (input, target, task) in enumerate(val_loader):

                        if self.gpu:
                            with torch.no_grad():
                                input = input.cuda()
                                target = target.cuda()
                        output = self.current_model.forward(input)[:, task_all]

                        # Summarize the performance of all tasks, or 1 task, depends on dataloader.
                        # Calculated by total number of data.
                        acc = accumulate_acc(output, target, task, acc)

                    self.current_model.train(orig_mode)

                    self.log(' * Task Val Acc {acc.avg:.3f}, Total time {time:.2f}'
                                .format(acc=acc, time=batch_timer.toc()))
        
        ###
        # Learning phase 2: Global Distillation for model
        ###
        if not self.first_task:

            # validation with previous classes
            val_dataset.load_dataset(prev, self.task_count, train=False)

            self.log(' --- Global Distillation --- ')
            for epoch in range(self.mschedule[-1]):

                data_timer = Timer()
                batch_timer = Timer()
                batch_time = AverageMeter()
                data_time = AverageMeter()
                losses = AverageMeter()
                acc = AverageMeter()
                self.log('Epoch:{0}'.format(epoch))

                # Config the model and optimizer
                self.model.train()
                self.prior_model.eval()
                self.current_model.eval()
                self.distill_schedule.step(epoch)
                for param_group in self.distill_optimizer.param_groups:
                    self.log('LR:', param_group['lr'])

                # Learning with mini-batch
                data_timer.tic()
                batch_timer.tic()
                for i, (xl, y, xul, _, task)  in enumerate(train_loader):

                    # go from ssl data to gd data
                    input, target = self.collect_data(xl, y, xul)
                    data_time.update(data_timer.toc())  # measure data loading time

                    if self.gpu:
                        input = [input[k].cuda() for k in range(len(input))]
                        target = target.cuda()

                    loss, output = self.global_distill_update(input, target, task, self.distill_optimizer, dw_d, prev_map, cur_map, seen_map)
                    target = target.detach()

                    # measure accuracy and record loss
                    accumulate_acc(output, target, task, acc)
                    losses.update(loss, target.size(0))  # TODO: Non-deterministic bug here
                    batch_time.update(batch_timer.toc())  # measure elapsed time
                    data_timer.toc()

                self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses,acc=acc))

                self.log(' * Distill Train Acc {acc.avg:.3f}'.format(acc=acc))

                # Evaluate the performance of current task
                if val_loader is not None:
                    batch_timer = Timer()
                    acc = AverageMeter()
                    batch_timer.tic()

                    orig_mode = self.model.training
                    self.model.eval()
                    for i, (input, target, task) in enumerate(val_loader):

                        if self.gpu:
                            with torch.no_grad():
                                input = input.cuda()
                                target = target.cuda()
                        output = self.model.forward(input)[:, task_all]

                        # Summarize the performance of all tasks, or 1 task, depends on dataloader.
                        # Calculated by total number of data.
                        acc = accumulate_acc(output, target, task, acc)

                    self.model.train(orig_mode)

                    self.log(' * Distill Val Acc {acc.avg:.3f}, Total time {time:.2f}'
                             .format(acc=acc, time=batch_timer.toc()))

            ###
            # Learning phase 2b: Global Distillation finetuning for model
            ###
            self.finetuning = True
            epoch_start = epoch+1
            self.log(' --- Finetuning --- ')
            for epoch in range(self.fschedule[-1]):
                
                data_timer = Timer()
                batch_timer = Timer()
                batch_time = AverageMeter()
                data_time = AverageMeter()
                losses = AverageMeter()
                acc = AverageMeter()
                self.log('Epoch:{0}'.format(epoch+epoch_start))

                # Config the model and optimizer
                self.model.train()
                self.prior_model.eval()
                self.current_model.eval()
                self.distill_schedule_finetune.step(epoch)
                for param_group in self.finetune_optimizer.param_groups:
                    self.log('LR:', param_group['lr'])

                # Learning with mini-batch
                data_timer.tic()
                batch_timer.tic()
                for i, (xl, y, xul, _, task)  in enumerate(train_loader):

                    # go from ssl data to gd data
                    input, target = self.collect_data(xl, y, xul)
                    data_time.update(data_timer.toc())  # measure data loading time

                    if self.gpu:
                        input = [input[k].cuda() for k in range(len(input))]
                        target = target.cuda()

                    loss, output = self.global_distill_update(input, target, task, self.finetune_optimizer, dw_ft, prev_map, cur_map, seen_map)
                    target = target.detach()

                    # measure accuracy and record loss
                    accumulate_acc(output, target, task, acc)
                    losses.update(loss, target.size(0))

                    batch_time.update(batch_timer.toc())  # measure elapsed time
                    data_timer.toc()

                self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses,acc=acc))
                self.log(' * FT Distill Train Acc {acc.avg:.3f}'.format(acc=acc))

                # Evaluate the performance of current task
                if val_loader is not None:
                    batch_timer = Timer()
                    acc = AverageMeter()
                    batch_timer.tic()

                    orig_mode = self.model.training
                    self.model.eval()
                    for i, (input, target, task) in enumerate(val_loader):

                        if self.gpu:
                            with torch.no_grad():
                                input = input.cuda()
                                target = target.cuda()
                        output = self.model.forward(input)[:, task_all]

                        # Summarize the performance of all tasks, or 1 task, depends on dataloader.
                        # Calculated by total number of data.
                        acc = accumulate_acc(output, target, task, acc)

                    self.model.train(orig_mode)

                    self.log(' * FT Distill Val Acc {acc.avg:.3f}, Total time {time:.2f}'
                             .format(acc=acc, time=batch_timer.toc()))
            self.finetuning = False

        else:

            # if this is first task, load parameters of C into M
            self.model.load_state_dict(self.current_model.state_dict())  # copy state
            self.first_task = False

        # append new task
        self.past_tasks.append(np.arange(self.last_valid_out_dim,self.valid_out_dim))

        # load M into C and P
        self.prior_model.load_state_dict(self.model.state_dict())  # copy state
        self.current_model.load_state_dict(self.model.state_dict())  # copy state
        self.model.eval() # prepare for eval
        self.last_valid_out_dim = self.valid_out_dim # update classes seen

        # Extend memory
        self.task_count += 1
        if self.memory_size > 0:

            train_dataset.update_coreset(self.memory_size, np.arange(self.last_valid_out_dim))

    # concatenate labelled and unlabelled data
    def collect_data(self, xl, y, xul):

        if self.no_unlabeled_data:
            return xl, y
            
        else:
            # images
            x_new_all = []
            for k in range(len(xl)):
                x_new= []
                x_new.append(xl[k])
                x_new.append(xul[k])
                x_new= torch.cat(x_new)
                x_new_all.append(x_new)

            # targets
            y_new = []
            y_new.append(y)
            a = -1 * np.ones((len(xul[0]),))
            y_new.append(torch.from_numpy(a).long())
            y_new = torch.cat(y_new)

            return x_new_all, y_new

    # update valid output dimensions of model given classes seen
    def add_valid_output_dim(self, dim=0):
        # This function is kind of ad-hoc, but it is the simplest way to support incremental class learning
        self.log('Incremental class: Old valid output dimension:', self.valid_out_dim)
        self.valid_out_dim += dim
        self.log('Incremental class: New Valid output dimension:', self.valid_out_dim)
        return self.valid_out_dim

    # number of parameters in model M
    def count_parameter(self):
        return sum(p.numel() for p in self.model.parameters())

    # validation to be called by main
    def validation(self, dataloader, task_in=None):
        # This function doesn't distinguish tasks.
        batch_timer = Timer()
        acc = AverageMeter()
        batch_timer.tic()

        orig_mode = self.training
        self.eval()
        for i, (input, target, task) in enumerate(dataloader):

            if self.gpu:
                with torch.no_grad():
                    input = input.cuda()
                    target = target.cuda()
            output = self.model.forward(input)
            if task_in is None:
                output = output[:,np.arange(self.valid_out_dim)]
                acc = accumulate_acc(output, target, task, acc)
            else:
                output = output[:,task_in]
                acc = accumulate_acc(output, target-task_in[0], task, acc)

        self.train(orig_mode)

        self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
                 .format(acc=acc, time=batch_timer.toc()))
        return acc.avg

    # get logits of model M
    def predict(self, inputs):
        self.model.eval()
        out = self.model.forward(inputs)
        return out.detach()

    # push models and loss functions to cuda
    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.prior_model = self.prior_model.cuda()
        self.current_model = self.current_model.cuda()
        self.criterion['cl'] =  self.criterion['cl'].cuda()
        self.criterion['kl'] =  self.criterion['kl'].cuda()

        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self

# update accuracy saving
def accumulate_acc(output, target, task, meter):
    labeled_inds = (target >= 0).nonzero().flatten()  # only use labeled data
    labeled_target = target[labeled_inds]
    labeled_output = output[labeled_inds]
    if len(labeled_target) > 0:
        meter.update(accuracy(labeled_output, labeled_target), len(labeled_target))
    return meter

# concat target q
def concat_target(p_out, c_out, cur, seen, prev):

    # target return
    target_full = p_out.new_zeros(p_out.size(0), len(seen))
    
    # scale based on non-target probs
    len_p, len_q = len(prev), len(seen)
    len_c = len(cur) # r-l

    # calculate pmax
    target_max, target_amax = p_out.max(dim=1, keepdim=True)

    # calculate epsilon
    eps = (1.-target_max) * len_c / (len_q-1)
    
    # y in prev tasks
    target_full[:,prev] = ((1. - target_max - eps)/(1 - target_max)) * p_out

    # y in current tasks
    target_full[:,cur] = eps * c_out

    # overwrite location at max
    target_full[range(target_full.size(0)),target_amax[:,0]] = target_max[:,0]

    return target_full