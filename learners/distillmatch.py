import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import numpy as np
from .default import NormalNN, accumulate_acc
from utils.metric import accuracy, AverageMeter, Timer
from torch.autograd import Variable
import random
import models
import dataloaders
import math

class DistillMatch(NormalNN):

    def __init__(self, learner_config):

        # name of ood detector model
        if learner_config['ood_model_name'] == None: learner_config['ood_model_name'] = learner_config['model_name']

        # simple init calls
        super(DistillMatch, self).__init__(learner_config)
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}  # For convenience
        self.first_task = True
        self.criterion_ood = nn.CrossEntropyLoss()
        self.oodtpr = learner_config['oodtpr']
        self.tpr = learner_config['tpr']
        self.num_classes = learner_config['num_classes']

        # use pseudolabels
        self.pl_flag = learner_config['pl_flag']
        self.prob_threshold_class = 0.0
        self.prob_threshold_ood = 0.0

        # fm parameters
        self.fm = {'thresh':0.85}

        # ood model
        self.ood_model = self.create_ood_model()
        self.ood_model_past = self.create_ood_model()
        self.ood_model = self.ood_model.cuda()
        self.ood_model_past = self.ood_model_past.cuda()
            
        # get optimizer for ood model - only need to train half as long
        self.schedule = [int(i / 2.0) for i in self.schedule]
        self.ood_epochs = self.schedule[-1]
        self.ood_optimizer, self.ood_scheduler, _, _ = self.new_optimizer(self.ood_model)
        self.schedule = [int(i * 2.0) for i in self.schedule]
        
        # pl model when using distillation or binary ood
        self.copy_model = super(DistillMatch, self).create_model() 
        self.copy_model = self.copy_model.cuda()

        # distance function for distillation
        self.distf = nn.KLDivLoss(reduction='none')

        # num tasks for repeats
        self.tasks = 0

        # retain past tasks
        self.past_tasks = []

        # distillation hyperparameter
        self.T = 2.0

        # some manual settings
        self.ood_holdout_ratio = 0.5 # amount of holdout training data for ood detection model
        self.dc_eps_values=[0.0025, 0.005, 0.001, 0.002, 0.004, 0.08] # epsilon values to iteratate over in DC tuning
        self.grad_clip=1 # gradient clipping in ood model

        # for ood calibration algorithm
        self.num_deltas = 100
        self.num_delta_loop = 10



    ##########################################
    #           MODEL TRAINING               #
    ##########################################

    # main training function
    def learn_batch(self, train_loader, train_dataset, train_dataset_ul, model_dir, val_loader=None):
        
        self.tasks += 1
        if self.reset_optimizer:  # Reset optimizer before learning each task
            self.log('Optimizer is reset!')
            self.init_optimizer()

        # get data weighting
        if self.first_task:
            self.data_weighting(train_dataset)
        else:
            self.data_weighting(train_dataset, num_seen=self.pseudolabel_DW_dataset(train_loader))

        # Evaluate the performance of current task
        if val_loader is not None:
            self.validation(val_loader)

        # load model if saved one exists
        try:
            self.load_models(model_dir)
            need_train = False
        except:
            need_train = True

        if need_train:
            self.log("Training classification model...")

            # classification model
            for epoch in range(self.schedule[-1]):
                self.epoch=epoch
                data_timer = Timer()
                batch_timer = Timer()
                batch_time = AverageMeter()
                data_time = AverageMeter()
                losses = AverageMeter()
                losses_ul = AverageMeter()
                losses_ce = AverageMeter()
                acc = AverageMeter()
                fpr = AverageMeter()
                tpr = AverageMeter()
                detection_error = AverageMeter()

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

                    if self.gpu:
                        xl = [xl[k].cuda() for k in range(len(xl))]
                        y = y.cuda()
                        xul = [xul[k].cuda() for k in range(len(xul))]
                    
                    xu_ind = {}
                    if not self.first_task and self.pl_flag:
                        xl, y, stats, xu_ind = self.pseudolabel_batch(xl, y, xul, yul)

                        # update ood det analysis
                        fpr.update(stats[0], stats[3][1])
                        tpr.update(stats[1], stats[3][0])
                        detection_error.update(stats[2], yul.size(0))
                    
                    # model update
                    loss, output, lu, ls = self.update_model(xl, y, xul, xu_ind)

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
                if not self.first_task and self.pl_flag:
                    self.log(' * FPR {fpr.avg:.3f} | TPR {tpr.avg:.3f} | DE {detection_error.avg:.3f}'.format(fpr=fpr,
                            tpr = tpr,detection_error=detection_error))
                self.log(' * Train Acc {acc.avg:.3f}'.format(acc=acc))

                # Evaluate the performance of current task
                if val_loader is not None:
                    self.validation(val_loader)

            # finetuning
            if not self.first_task: self.data_weighting(train_dataset)
            for epoch in range(self.fschedule[-1]):
                self.epoch=epoch
                data_timer = Timer()
                batch_timer = Timer()
                batch_time = AverageMeter()
                data_time = AverageMeter()
                losses = AverageMeter()
                acc = AverageMeter()

                # Config the model and optimizer
                self.log('Finetuning Epoch:{0}'.format(epoch+1))
                self.model.train()
                self.finetune_scheduler.step(epoch)
                for param_group in self.finetune_optimizer.param_groups:
                    self.log('LR:', param_group['lr'])

                # Learning with mini-batch
                data_timer.tic()
                batch_timer.tic()
                for i, (xl, y, xul, yul, task)  in enumerate(train_loader):

                    data_time.update(data_timer.toc())  # measure data loading time
                    if self.gpu:
                        xl = [xl[k].cuda() for k in range(len(xl))]
                        y = y.cuda()
                    
                    # loss
                    logits = self.forward(xl[0])
                    loss = self.criterion(logits, y.long())
                                
                    # step optimizer
                    self.finetune_optimizer.zero_grad()
                    loss.backward()
                    self.finetune_optimizer.step()

                    y = y.detach()

                    # measure accuracy and record loss
                    accumulate_acc(logits.detach(), y, task, acc)
                    losses.update(loss,  y.size(0))   
                    batch_time.update(batch_timer.toc())  # measure elapsed time
                    data_timer.toc()

                self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses,acc=acc))

                # Evaluate the performance of current task
                if val_loader is not None:
                    self.validation(val_loader)

            # ood model
            self.model.eval()
            if self.pl_flag:
                self.log("Training OOD detection model...")
                # sample dataset
                train_dataset.sample_dataset(0.0, self.ood_holdout_ratio)
                train_dataset_ul.sample_dataset(0.0, self.ood_holdout_ratio)
                if not self.first_task: self.data_weighting(train_dataset, num_seen=self.pseudolabel_DW_dataset(train_loader, ood_type='ood'))
                for epoch in range(self.ood_epochs):
                    self.epoch=epoch
                    losses_ood = AverageMeter()
                    acc_ood = AverageMeter()
                    fpr = AverageMeter()
                    tpr = AverageMeter()
                    detection_error = AverageMeter()

                    # Config the model and optimizer
                    self.log('Epoch:{0}'.format(epoch+1))
                    self.ood_model.train()
                    self.ood_scheduler.step(epoch)
                    for param_group in self.ood_optimizer.param_groups:
                        self.log('LR:', param_group['lr'])

                    # Learning with mini-batch
                    for i, (xl, y, xul, yul, task)  in enumerate(train_loader):

                        self.batch_xl = y.size(0)
                        if self.gpu:
                            xl = [xl[k].cuda() for k in range(len(xl))]
                            y = y.cuda()
                            xul = [xul[k].cuda() for k in range(len(xul))]
                        
                        xu_ind = {}
                        if not self.first_task and self.pl_flag:
                            xl, y, stats, xu_ind = self.pseudolabel_batch(xl, y, xul, yul, ood_type='ood')

                            # update ood det analysis
                            fpr.update(stats[0], stats[3][1])
                            tpr.update(stats[1], stats[3][0])
                            detection_error.update(stats[2], yul.size(0))
                        
                        # model update
                        loss, output, lu, ls = self.update_ood_model(xl, y, xul, xu_ind)
                        accumulate_acc(output, y, task, acc_ood)
                        losses_ood.update(loss, y.size(0))

                    # evaluate ood model
                    self.log(' * OOD Loss {loss.avg:.3f} | OOD Train Acc {acc.avg:.3f}'.format(loss=losses_ood, acc=acc_ood))

                    if not self.first_task:
                        self.log(' * FPR {fpr.avg:.3f} | TPR {tpr.avg:.3f} | DE {detection_error.avg:.3f}'.format(fpr=fpr,
                                tpr = tpr,detection_error=detection_error))

                    # if ood model is NOT copying classification model
                    self.log(' OOD validation...')
                    self.validation(val_loader, model = self.ood_model)

                # sample dataset
                train_dataset.sample_dataset()
                train_dataset_ul.sample_dataset()
        
            # turn off training
            self.ood_model.eval()

            # save models
            self.save_models(model_dir)

        # load copy model for distillation
        self.copy_model.load_state_dict(self.model.state_dict())  # copy state
        self.copy_model.eval()

        # load frozen ood model
        self.ood_model_past.load_state_dict(self.ood_model.state_dict())  # copy state
        self.ood_model_past.eval()

        # append new task
        self.past_tasks.append(np.arange(self.last_valid_out_dim,self.valid_out_dim))

        # get ready for next task!
        self.valid_out_dim_past_past = self.last_valid_out_dim
        self.last_valid_out_dim = self.valid_out_dim
        self.callibrate_ood_model(train_loader, train_dataset, train_dataset_ul)
        self.first_task = False

        # Extend memory
        self.task_count += 1
        if self.memory_size > 0:
            train_dataset.update_coreset(self.memory_size, np.arange(self.last_valid_out_dim))

    # apply update to class model
    def update_model(self, inputs_labeled, targets, inputs_unlabeled, unlabeled_ind):
        
        # unsupervised loss
        loss_unsupervised = torch.zeros((1,), requires_grad=True).cuda()    
        if (not self.first_task) and self.config['fm_loss']:
            tasks = [
                    np.arange(0,self.valid_out_dim)
                    ]
            task_idx = [np.arange(len(inputs_unlabeled[0]))]
            loss_unsupervised = self.fm_loss(tasks, inputs_unlabeled, task_idx)   

        # supervised loss  
        logits_labeled = self.forward(inputs_labeled[0]) 
        loss_supervised = self.criterion(logits_labeled, targets.long())

        # total loss
        total_loss = loss_supervised + self.weight_aux*loss_unsupervised

        # distillation loss
        if (not self.first_task):
            inputs_distill = []
            inputs_distill.append(inputs_labeled[0])
            inputs_distill.append(inputs_unlabeled[0])
            logits_distill_past = self.copy_model.forward(torch.cat(inputs_distill)).detach()
            logits_distill_current = self.forward(torch.cat(inputs_distill))
            for task in self.past_tasks:
                pout_current = (logits_distill_current[:,task]/self.T).log_softmax(dim=1)
                pout_past = (logits_distill_past[:,task]/self.T).softmax(dim=1)
                loss_weight = len(task) / self.valid_out_dim
                total_loss += (self.distf(pout_current, pout_past)).sum(dim=1).mean() * loss_weight
            
        # apply update
        self.optimizer.zero_grad()
        total_loss.backward()

        # need grad clipping for hard aug loss
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()
        return total_loss.detach(), logits_labeled, loss_unsupervised.detach(), loss_supervised.detach()

    # apply update to ood model
    def update_ood_model(self, inputs_labeled, targets, inputs_unlabeled, unlabeled_ind):
        
        loss_unsupervised = torch.zeros((1,), requires_grad=True).cuda()               
        logits_labeled = self.forward(inputs_labeled[0],ood=True)
        loss_supervised = self.criterion(logits_labeled, targets.long())
        total_loss = loss_supervised + self.weight_aux*loss_unsupervised
            
        # distillation loss
        if (not self.first_task):
            inputs_distill = []
            inputs_distill.append(inputs_labeled[0])
            inputs_distill.append(inputs_unlabeled[0])
            logits_distill_current = self.ood_model.forward(torch.cat(inputs_distill))
            logits_distill_past = self.ood_model_past.forward(torch.cat(inputs_distill))
            for task in self.past_tasks:
                pout_current = (logits_distill_current[:,task]/self.T).log_softmax(dim=1)
                pout_past = (logits_distill_past[:,task]/self.T).softmax(dim=1).detach()
                loss_weight = len(task) / self.valid_out_dim
                total_loss += (self.distf(pout_current, pout_past)).sum(dim=1).mean() * loss_weight

        # update
        self.ood_optimizer.zero_grad()
        total_loss.backward()

        # need grad clipping for hard aug loss
        torch.nn.utils.clip_grad_norm_(self.ood_model.parameters(), self.grad_clip)

        self.ood_optimizer.step()
        return total_loss.detach(), logits_labeled, loss_unsupervised.detach(), loss_supervised.detach()
        
    
    # fixmatch loss for semi-supervised learning
    def fm_loss(self, tasks, task_data, task_idx):
        """
        @article{sohn2020fixmatch,
            title={Fixmatch: Simplifying semi-supervised learning with consistency and confidence},
            author={Sohn, Kihyuk and Berthelot, David and Li, Chun-Liang and Zhang, Zizhao and Carlini, Nicholas and Cubuk, Ekin D and Kurakin, Alex and Zhang, Han and Raffel, Colin},
            journal={arXiv preprint arXiv:2001.07685},
            year={2020}
        }
        """
        
        # free up memory
        loss_con = torch.zeros((1,), requires_grad=True).cuda()[0]

        task_data = torch.stack(task_data)
        task_data = task_data.permute(1,0,2,3,4)
        bsu, ku, c = len(task_data), task_data.size(1), self.valid_out_dim

        # forward pass on unlabelled data
        logits_xu_all = self.forward(task_data.reshape(-1, *task_data.shape[2:])).reshape(bsu, ku, c)

        for t in range(len(tasks)):
            task = tasks[t]
            logits_xu = logits_xu_all[task_idx[t]]
            if len(logits_xu) > 0:
                bsu = len(logits_xu)

                # compute pseudo label
                prob_xu = torch.softmax(logits_xu[:, 0, task].detach(), dim=1)
                wu, yu = torch.max(prob_xu, dim=1, keepdim=True)
                yu = yu.repeat(1, ku-1) + task[0]
                wu = (wu > self.fm['thresh']).repeat(1, ku-1)

                # consistency loss
                loss_con_sum = F.cross_entropy(logits_xu[:, 1].reshape(-1, c), yu.flatten(), reduction='none')
                loss_con += torch.sum(loss_con_sum[wu.flatten()])/bsu/(ku-1) * (len(task) / self.valid_out_dim)

        return loss_con


    ##########################################
    #            OOD DETECTION               #
    ##########################################    

    # out of distribution detection with decomposed confidence
    def ood(self, xul, ood_dim, ood_type='class'):
        """
        @inproceedings{hsu2020generalized,
            title =     {Generalized ODIN: Detecting Out-of-distribution Image without Learning from Out-of-distribution Data},
            author =    {Yen-Chang Hsu and Yilin Shen and Hongxia Jin and Zsolt Kira},
            booktitle = {CVPR},
            year =      {2020},
        }
        """

        # get pseudolabels from copy model
        logits_new = self.copy_model.forward(xul).detach()[:,:ood_dim]
        _, pseudo_labels_return = logits_new.max(dim=1)

        if ood_type == 'class':
            thresh = self.prob_threshold_class
        elif ood_type == 'ood':
            thresh = self.prob_threshold_ood

        # decomposed confidence
        # calculate pertubation
        # first, get scaled logit
        inputs = Variable(xul, requires_grad = True)
        H, d = self.ood_model_past.ood_forward(inputs)
        H = H[:,:ood_dim]
        outputs = H
        
        loss, _ = self.criterion_ood(outputs)
        loss = -1 * loss.mean()
        loss.backward()
        
        # Normalizing the gradient to binary in {0, 1}
        gradient =  torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        
        # Adding small perturbations to images
        tempInputs = torch.add(inputs.detach(),  -self.oodeps, gradient)

        H, d = self.ood_model_past.ood_forward(tempInputs)
        H = H[:,:ood_dim]
        H.detach()
        d.detach()
        _, pseudo_labels = H.max(dim=1)
        scores_max, output = self.criterion_ood(H)

        pl_idxs, no_pl_idxs = self.pl_decision(output, thresh)

        return pl_idxs, no_pl_idxs, pseudo_labels_return, scores_max

    # ood logits and scores
    def criterion_ood(self, logits):
        score, _ = logits.max(dim=1)
        return score, logits

    # given scores and thresholds, return pl indexes
    def pl_decision(self, scores, thresh):
        scores_max, _ = scores.max(dim=1)
        pl_idx_mask = scores_max > (thresh)
        pl_idxs = pl_idx_mask.nonzero().view(-1)
        no_pl_idx_mask = scores_max <= (thresh)
        no_pl_idxs = no_pl_idx_mask.nonzero().view(-1)
        return pl_idxs, no_pl_idxs

    # calibrate ood model
    def callibrate_ood_model(self, train_loader, train_dataset, train_dataset_ul):

        orig_mode = self.training
        orig_mode_past = self.ood_model_past.training
        self.eval()
        self.ood_model_past.eval()

        # find input pertubation epsilon
        if len(self.dc_eps_values) == 1:
            self.oodeps = self.dc_eps_values[0]
        else:
            magnitude_list = self.dc_eps_values
            self.log('Searching the best perturbation magnitude on in-domain data. Magnitude:', magnitude_list)
            loss_list = {}
            for m in magnitude_list:

                loss_meter = AverageMeter()
                for i, (xl, y, xul, yul, task)  in enumerate(train_loader):
                    
                    if self.gpu:
                        inputs = xl[0].cuda()
                    
                    inputs = Variable(inputs, requires_grad = True)
                    H, d = self.ood_model_past.ood_forward(inputs)
                    H = H[:,:self.valid_out_dim]
                    outputs = H
                    
                    loss, _ = self.criterion_ood(outputs)
                    loss = -1*loss.mean()
                    loss.backward()

                    gradient = torch.ge(inputs.grad.data, 0)
                    gradient = (gradient.float() - 0.5) * 2
                    modified_input = torch.add(inputs.detach(), -m, gradient)
                    H.detach()
                    d.detach()

                    H, d = self.ood_model_past.ood_forward(modified_input)
                    H = H[:,:self.valid_out_dim]
                    H.detach()
                    d.detach()

                    _, pseudo_labels = H.max(dim=1)
                    loss, _ = self.criterion_ood(H)
                    loss_meter.update(loss.mean().detach().cpu(), len(loss))

                loss_list[m] = loss_meter.avg
                self.log('Magnitude:', m, 'loss:', loss_list[m])
            best_m = min(loss_list, key=(lambda key: loss_list[key]))
            self.oodeps = best_m / 2.
        
        # sample dataset for next part
        train_dataset.sample_dataset(self.ood_holdout_ratio, 1.0)
        train_dataset_ul.sample_dataset(self.ood_holdout_ratio, 1.0)

        # get scores from dataset
        scores_max_all = []
        for loop in range(self.num_delta_loop): 
            for i, (xl, y, xul, yul, task)  in enumerate(train_loader):

                if self.gpu:
                    xl = [xl[k].cuda() for k in range(len(xl))]
                    y = y.cuda()

                # ood detection scores
                _, _, _, scores_max = self.ood(xl[0], self.valid_out_dim)
                scores_max_all.extend(scores_max.cpu().detach())

        # take out any nan
        scores_max_all = np.asarray(scores_max_all)
        scores_max_all = scores_max_all[~np.isnan(scores_max_all)]

        # get class data
        scores_k = scores_max_all
        ood_thresh = self.oodtpr
        class_thresh = self.tpr

        # first for ood network
        start = 1.0 / self.valid_out_dim
        end = 10.0
        best_thresh = start
        for loop in range(self.num_delta_loop):   
            gap = (end - start)/self.num_deltas
            tpr_best = None
            for delta in np.arange(start, end, gap):

                # calculate pl indexes
                TP = sum(scores_k > (delta))
                FN = sum(scores_k <= (delta))

                # tpr
                if TP > 0:
                    tpr = TP / (TP + FN)
                else:
                    tpr = 0

                if tpr < ood_thresh:
                    if tpr_best is not None:
                        best_thresh = delta-gap/2.0
                    else:
                        best_thresh = delta-gap/2.0
                        tpr_best = tpr
                    start = delta-gap
                    end = delta
                    break

                tpr_best = tpr

        self.prob_threshold_ood = best_thresh
        self.log('New Threshold OOD: {thresh:.10f} | TPR: {tpr:.4f}'.format(thresh=self.prob_threshold_ood, tpr=tpr_best))

        # second for classification network
        start = 1.0 / self.valid_out_dim
        end = 10.0
        best_thresh = start
        for loop in range(self.num_delta_loop):   
            gap = (end - start)/self.num_deltas
            tpr_best = None
            for delta in np.arange(start, end, gap):

                # calculate pl indexes
                TP = sum(scores_k > (delta))
                FN = sum(scores_k <= (delta))

                # tpr
                if TP > 0:
                    tpr = TP / (TP + FN)
                else:
                    tpr = 0

                if tpr < class_thresh:
                    if tpr_best is not None:
                        best_thresh = delta-gap/2.0
                    else:
                        best_thresh = delta-gap/2.0
                        tpr_best = tpr
                    start = delta-gap
                    end = delta
                    break

                tpr_best = tpr
        self.prob_threshold_class = best_thresh
        self.log('New Threshold Class: {thresh:.10f} | TPR: {tpr:.4f}'.format(thresh=self.prob_threshold_class, tpr=tpr_best))
        self.train(orig_mode)
        self.ood_model_past.train(orig_mode_past)

        # make sure entire train dataset loaded
        train_dataset.sample_dataset()
        train_dataset_ul.sample_dataset()
        
    ##########################################
    #                 OTHER                  #
    ##########################################   

    # estimate number of classes present for class-balanced gradient weighting
    def pseudolabel_DW_dataset(self, train_loader, ood_type='class'):

        labels=[]
        pseudo_labels=[]
        num_l = 0
        num_ul = 0

        # number of times to iterate over training
        # dataloader when estimating class distribution of unlabeled data
        maxj = 5 
        for j in range(maxj):
            for i, (xl, y, xul, yul, task)  in enumerate(train_loader):

                if self.gpu:
                    xl = [xl[k].cuda() for k in range(len(xl))]
                    y = y.cuda()
                    xul = [xul[k].cuda() for k in range(len(xul))]
                    yul = yul.cuda()
                    labels.extend(y.cpu().detach().numpy())
                    bsl = len(y)
                    num_l += bsl
                if not self.first_task and self.pl_flag:
                    xl, y, _, _ = self.pseudolabel_batch(xl, y, xul, yul, ood_type=ood_type)
                    pseudo_labels.extend(y[bsl:].cpu().detach().numpy())
                    num_ul += len(yul)

        labels = np.asarray(labels, dtype=np.int64) 
        pseudo_labels = np.asarray(pseudo_labels, dtype=np.int64)  
        num_seen_l = np.asarray([len(labels[labels==k]) for k in range(self.valid_out_dim)], dtype=np.float32)
        num_seen_pl = np.asarray([len(pseudo_labels[pseudo_labels==k]) for k in range(self.valid_out_dim)], dtype=np.float32)
        if self.memory_size > 0:
            num_seen_total = num_seen_l
        else:
            num_seen_total = num_seen_l + num_seen_pl
        return num_seen_total    

    # add psuedolabeled data into training data
    def pseudolabel_batch(self, xl, y, xul, orig_label=None, ood_type='class'):

        # ood detection
        pl_idxs, no_pl_idxs, pseudo_labels, _ = self.ood(xul[0], self.last_valid_out_dim, ood_type=ood_type)

        # get new pl data
        xl_new, y_new = [],[y]
        for i in range(len(xl)):
            xl_k = []
            xl_k.append(xl[i])
            xl_k.append(xul[i][pl_idxs])
            xl_new.append(torch.cat(xl_k))
        y_new.append(pseudo_labels[pl_idxs])
        y_new = torch.cat(y_new)

        # remove pl from unlabeled data
        xu_ind = {'ID':pl_idxs,'OOD':pl_idxs}

        # calculate ood detection statistics
        if orig_label is not None:
            stats = []

            # don't care if PL correct, just distribution!
            correct_pl = orig_label[pl_idxs] < self.last_valid_out_dim
            TP = correct_pl.nonzero().view(-1).size(0)
            FP = pl_idxs.size(0) - TP

            # negatives
            true_rejects = orig_label[no_pl_idxs] >= self.last_valid_out_dim
            TN = true_rejects.nonzero().view(-1).size(0)
            FN = no_pl_idxs.size(0) - TN

            # stats
            if FP > 0:
                fpr = FP / (FP + TN)
            else:
                fpr = 0
            stats.append(fpr)
            if TP > 0:
                tpr = TP / (TP + FN)
            else:
                tpr = 0
            stats.append(tpr)
            stats.append(0.5*(1-stats[1]) + 0.5*(stats[0]))
            stats.append([TP + FN, FP + TN])

            return xl_new, y_new, stats, xu_ind
        else:
            return xl_new, y_new, xu_ind

    # save models
    def save_models(self, filename):
        model_state = self.model.state_dict()
        for key in model_state.keys():  # Always save it to cpu
            model_state[key] = model_state[key].cpu()
        self.log('=> Saving class model to:', filename)
        torch.save(model_state, filename + 'class.pth')
        model_state = self.ood_model.state_dict()
        for key in model_state.keys():  # Always save it to cpu
            model_state[key] = model_state[key].cpu()
        self.log('=> Saving ood model to:', filename)
        torch.save(model_state, filename + 'ood.pth')
        self.log('=> Save Done')

    # load models
    def load_models(self, filename):
        self.model.load_state_dict(torch.load(filename + 'class.pth'))
        self.ood_model.load_state_dict(torch.load(filename + 'ood.pth'))
        self.log('=> Load Done')
        if self.gpu:
            self.model = self.model.cuda()
            self.ood_model = self.ood_model.cuda()
        self.model.eval()
        self.ood_model.eval()
 
    # validation with ood detection statistics
    def validation_pl(self, dataloader):
        
        stats = [AverageMeter(), AverageMeter(), AverageMeter()]
        orig_mode = self.training
        self.eval()
        for i, (input, target, task) in enumerate(dataloader):

            if self.gpu:
                with torch.no_grad():
                    input = input.cuda()
                    target = target.cuda()

            # ood detection
            pl_idxs, no_pl_idxs, pl, scores_max = self.ood(input, self.valid_out_dim)
            
            # don't care if PL correct, just distribution!
            correct_pl = target[pl_idxs] < self.valid_out_dim
            TP = correct_pl.nonzero().view(-1).size(0)
            FP = pl_idxs.size(0) - TP

            # negatives
            true_rejects = target[no_pl_idxs] >= self.valid_out_dim
            TN = true_rejects.nonzero().view(-1).size(0)
            FN = no_pl_idxs.size(0) - TN

            # stats
            if FP > 0:
                fpr = FP / (FP + TN)
            else:
                fpr = 0
            if TP > 0:
                tpr = TP / (TP + FN)
            else:
                tpr = 0
            de = 0.5*(1-tpr) + 0.5*(fpr)

            stats[0].update(fpr, FP + TN)
            stats[1].update(tpr, TP + FN)
            stats[2].update(de, target.size(0))
 
        self.train(orig_mode)

        self.log(' * FPR {fpr.avg:.3f} | TPR {tpr.avg:.3f} | DE {detection_error.avg:.3f}'.format(fpr=stats[0],
                tpr = stats[1],detection_error=stats[2]))

        return [stats[i].avg for i in range(3)]