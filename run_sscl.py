from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import sys
import argparse
import torch
import numpy as np
import yaml
import json
import random
from random import shuffle
from collections import OrderedDict
import dataloaders
from dataloaders.utils import *
from torch.utils.data import DataLoader
import learners

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
def run(args, seed):
    if not os.path.exists('outputs'):
        os.mkdir('outputs')

    # prepare dataloader
    Dataset = None
    if args.dataset == 'CIFAR10':
        Dataset = dataloaders.iCIFAR10
        num_classes = 10
    elif args.dataset == 'CIFAR100':
        Dataset = dataloaders.iCIFAR100
        num_classes = 100
    elif args.dataset == 'TinyIMNET':
        Dataset = dataloaders.iTinyIMNET
        num_classes = 200
    else:
        Dataset = dataloaders.H5Dataset
        num_classes = 100

    # load tasks
    class_order = np.arange(num_classes).tolist()
    class_order_logits = np.arange(num_classes).tolist()
    if seed > 0 and args.rand_split:
        print('=============================================')
        print('Shuffling....')
        print('pre-shuffle:' + str(class_order))
        random.seed(seed)
        random.shuffle(class_order)
        print('post-shuffle:' + str(class_order))
        print('=============================================')
    tasks = []
    tasks_logits = []
    p = 0
    while p < num_classes:
        inc = args.other_split_size if p > 0 else args.first_split_size
        tasks.append(class_order[p:p+inc])
        tasks_logits.append(class_order_logits[p:p+inc])
        p += inc
    num_tasks = len(tasks)
    task_names = [str(i+1) for i in range(num_tasks)]
    
    # number of transforms per image
    k = 1
    if args.fm_loss: 
        k = 2
    ky = 1

    # datasets and dataloaders
    train_transform = dataloaders.utils.get_transform(dataset=args.dataset, phase='train', aug=args.train_aug)
    train_transformb = dataloaders.utils.get_transform(dataset=args.dataset, phase='train', aug=args.train_aug, hard_aug=True)
    test_transform  = dataloaders.utils.get_transform(dataset=args.dataset, phase='test', aug=args.train_aug)
    
    train_dataset = Dataset(args.dataroot, args.dataset, args.labeled_samples, args.unlabeled_task_samples, train=True, lab = True,
                            download=True, transform=TransformK(train_transform, train_transform, ky), l_dist=args.l_dist, ul_dist=args.ul_dist,
                            tasks=tasks, seed=seed, rand_split=args.rand_split, validation=args.validation, kfolds=args.repeat)
    train_dataset_ul = Dataset(args.dataroot, args.dataset, args.labeled_samples, args.unlabeled_task_samples, train=True, lab = False,
                            download=True, transform=TransformK(train_transform, train_transformb, k), l_dist=args.l_dist, ul_dist=args.ul_dist,
                            tasks=tasks, seed=seed, rand_split=args.rand_split, validation=args.validation, kfolds=args.repeat)
    test_dataset  = Dataset(args.dataroot, args.dataset, train=False,
                            download=False, transform=test_transform, l_dist=args.l_dist, ul_dist=args.ul_dist,
                            tasks=tasks, seed=seed, rand_split=args.rand_split, validation=args.validation, kfolds=args.repeat)

    # in case tasks reset...
    tasks = train_dataset.tasks

    # Prepare the Learner (model)
    learner_config = {'num_classes': num_classes,
                      'lr': args.lr,
                      'ul_batch_size': args.ul_batch_size,
                      'tpr': args.tpr,
                      'oodtpr': args.oodtpr,
                      'momentum': args.momentum,
                      'weight_decay': args.weight_decay,
                      'schedule': args.schedule,
                      'schedule_type': args.schedule_type,
                      'model_type': args.model_type,
                      'model_name': args.model_name,
                      'ood_model_name': args.ood_model_name,
                      'out_dim': args.force_out_dim,
                      'optimizer': args.optimizer,
                      'gpuid': args.gpuid,
                      'pl_flag': args.pl_flag,
                      'fm_loss': args.fm_loss,
                      'weight_aux': args.weight_aux,
                      'memory': args.memory,
                      'distill_loss': args.distill_loss,
                      'co': args.co,
                      'FT': args.FT,
                      'DW': args.DW,
                      'num_labeled_samples': args.labeled_samples,
                      'num_unlabeled_samples': args.unlabeled_task_samples,
                      'super_flag': args.l_dist == "super",
                      'no_unlabeled_data': args.no_unlabeled_data
                      }
    learner = learners.__dict__[args.learner_type].__dict__[args.learner_name](learner_config)
    print(learner.model)
    print('#parameter of model:', learner.count_parameter())

    acc_table = OrderedDict()
    acc_table_pt = OrderedDict()
    if args.learner_name == 'DistillMatch' and len(task_names) > 1 and not args.oracle_flag:
        run_ood = {}
    else:
        run_ood = None
    save_table_ssl = []
    save_table = []
    save_table_pc = -1*np.ones((num_tasks,num_tasks))
    pl_table = [[],[],[],[]]
    temp_dir = args.log_dir + '/temp'
    if not os.path.exists(temp_dir): os.makedirs(temp_dir)

    # for oracle
    out_dim_add = 0

    ###
    # Training
    ###
    # Feed data to learner and evaluate learner's performance
    if args.max_task > 0:
        max_task = min(args.max_task, len(task_names))
    else:
        max_task = len(task_names)
    for i in range(max_task):

        # set seeds
        random.seed(seed*100 + i)
        np.random.seed(seed*100 + i)
        torch.manual_seed(seed*100 + i)
        torch.cuda.manual_seed(seed*100 + i)

        train_name = task_names[i]
        print('======================', train_name, '=======================')

        # load dataset for task
        task = tasks_logits[i]
        prev = sorted(set([k for task in tasks_logits[:i] for k in task]))

        # if oracle
        if args.oracle_flag:
            train_dataset.load_dataset(prev, i, train=False)
            train_dataset_ul.load_dataset(prev, i, train=False)
            learner = learners.__dict__[args.learner_type].__dict__[args.learner_name](learner_config)
            out_dim_add += len(task)
        else:
            train_dataset.load_dataset(prev, i, train=True)
            train_dataset_ul.load_dataset(prev, i, train=True)
            out_dim_add = len(task)

        # load dataset with memory
        train_dataset.append_coreset(only=False)

        # load dataloader
        train_loader_l = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=int(args.workers / 2))
        train_loader_ul = DataLoader(train_dataset_ul, batch_size=args.ul_batch_size, shuffle=True, drop_last=False, num_workers=int(args.workers / 2))
        train_loader_ul_task = DataLoader(train_dataset_ul, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=int(args.workers / 2))
        train_loader = dataloaders.SSLDataLoader(train_loader_l, train_loader_ul)

        # add valid class to classifier
        learner.add_valid_output_dim(out_dim_add)

        # Learn
        test_dataset.load_dataset(prev, i, train=False)
        test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.workers)
        if args.learner_type == 'distillation':
            learner.learn_batch(train_loader, train_dataset, prev, test_loader, test_dataset)
        else:
            model_save_dir = args.log_dir + '/models/repeat-'+str(seed+1)+'/task-'+task_names[i]+'/'
            if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
            learner.learn_batch(train_loader, train_dataset, train_dataset_ul, model_save_dir, test_loader)

        # Evaluate
        acc_table[train_name] = OrderedDict()
        acc_table_pt[train_name] = OrderedDict()
        for j in range(i+1):
            val_name = task_names[j]
            print('validation split name:', val_name)
            test_dataset.load_dataset(prev, j, train=True)
            test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.workers)
            acc_table[val_name][train_name] = learner.validation(test_loader)
            save_table_pc[i,j] = acc_table[val_name][train_name]
            acc_table_pt[val_name][train_name] = learner.validation(test_loader, task_in = tasks_logits[j])
        save_table.append(np.mean([acc_table[task_names[j]][train_name] for j in range(i+1)]))

        # Evaluate PL
        if args.learner_name == 'DistillMatch' and i+1 < len(task_names) and not args.oracle_flag:
            test_dataset.load_dataset(prev, len(task_names)-1, train=False)
            test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.workers)
            stats = learner.validation_pl(test_loader)
            names = ['stats-fpr','stats-tpr','stats-de']
            for ii in range(3):
                pl_table[ii].append(stats[ii])
                save_file = temp_dir + '/'+names[ii]+'_table.csv'
                np.savetxt(save_file, np.asarray(pl_table[ii]), delimiter=",", fmt='%.2f')
            run_ood['tpr'] = pl_table[1]
            run_ood['fpr'] = pl_table[0]
            run_ood['de'] = pl_table[2]

        # save temporary results
        save_file = temp_dir + '/acc_table.csv'
        np.savetxt(save_file, np.asarray(save_table), delimiter=",", fmt='%.2f')
        save_file_pc = temp_dir + '/acc_table_pc.csv'
        np.savetxt(save_file_pc, np.asarray(save_table_pc), delimiter=",", fmt='%.2f')

    return acc_table, acc_table_pt, task_names, run_ood

def create_args():
    
    # This function prepares the variables shared across demo.py
    parser = argparse.ArgumentParser()

    # Standard Args
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                         help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--log_dir', type=str, default="outputs/out",
                         help="Save experiments results in dir for future plotting!")
    parser.add_argument('--model_type', type=str, default='mlp', help="The type (mlp|lenet|vgg|resnet) of backbone network")
    parser.add_argument('--model_name', type=str, default='MLP', help="The name of actual model for the backbone")
    parser.add_argument('--force_out_dim', type=int, default=2, help="Set 0 to let the task decide the required output dimension")
    parser.add_argument('--learner_type', type=str, default='default', help="The type (filename) of learner")
    parser.add_argument('--learner_name', type=str, default='NormalNN', help="The class name of learner")
    parser.add_argument('--optimizer', type=str, default='SGD', help="SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad|Adamax ...")
    parser.add_argument('--dataroot', type=str, default='data', help="The root folder of dataset or downloaded data")
    parser.add_argument('--dataset', type=str, default='CIFAR100', help="CIFAR10|CIFAR100|TinyIMNET")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--schedule', nargs="+", type=int, default=[2],
                        help="The list of epoch numbers to reduce learning rate by factor of 0.1. Last number is the end epoch")
    parser.add_argument('--schedule_type', type=str, default='cosine',
                        help="decay, cosine")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--ul_batch_size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=8, help="#Thread for dataloader")
    parser.add_argument('--validation', default=False, action='store_true', help='Evaluate on fold of training dataset rather than testing data')
    parser.add_argument('--FT', default=False, action='store_true', help='finetune distillation')
    parser.add_argument('--repeat', type=int, default=1, help="Repeat the experiment N times")

    # OOD Args
    parser.add_argument('--ood_model_name', type=str, default=None, help="The name of actual model for the backbone ood")
    parser.add_argument('--tpr', type=float, default=0.95, help="tpr for ood calibration of class network")
    parser.add_argument('--oodtpr', type=float, default=0.95, help="tpr for ood calibration of ood network")

    # SSL Args
    parser.add_argument('--weight_aux', type=float, default=1.0, help="Auxillery weight, usually used for trading off unsupervised and supervised losses")
    parser.add_argument('--labeled_samples', type=int, default=50000, help='Number of labeled samples in ssl')
    parser.add_argument('--unlabeled_task_samples', type=int, default=0, help='Number of unlabeled samples in each task in ssl')
    parser.add_argument('--fm_loss', default=False, action='store_true', help='Use fix-match loss with classifier (WARNING: currently only pseudolabel)')
    parser.add_argument('--pl_flag', default=False, action='store_true', help='use pseudo-labeled ul data for DM')
    
    # GD Args
    parser.add_argument('--no_unlabeled_data', default=False, action='store_true')
    parser.add_argument('--distill_loss', nargs="+", type=str, default='C', help='P, C, Q')
    parser.add_argument('--co', type=float, default=1., metavar='R',
                    help='out-of-distribution confidence loss ratio (default: 0.)')

    # CL Args
    parser.add_argument('--first_split_size', type=int, default=2, help="size of first CL task")
    parser.add_argument('--other_split_size', type=int, default=2, help="size of remaining CL tasks")              
    parser.add_argument('--train_aug', dest='train_aug', default=False, action='store_true',
                        help="Allow data augmentation during training")
    parser.add_argument('--rand_split', dest='rand_split', default=False, action='store_true',
                        help="Randomize the classes in splits")
    parser.add_argument('--l_dist', type=str, default='vanilla', help="vanilla|super")
    parser.add_argument('--ul_dist', type=str, default=None, help="none|vanilla|super - if none, copy l dist")
    parser.add_argument('--oracle_flag', default=False, action='store_true', help='Upper bound for oracle')
    parser.add_argument('--max_task', type=int, default=-1, help="number tasks to perform; if -1, then all tasks")
    parser.add_argument('--memory', type=int, default=0, help="size of memory for replay")
    parser.add_argument('--DW', default=False, action='store_true', help='dataset balancing')

    return parser

def get_args(argv):
    parser=create_args()
    args = parser.parse_args(argv)
    return args

# want to save everything printed to outfile
class Logger(object):
    def __init__(self, name):
        self.terminal = sys.stdout
        self.log = open(name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.log.flush()

if __name__ == '__main__':
    args = get_args(sys.argv[1:])

    # determinstic backend
    torch.backends.cudnn.deterministic=True

    # duplicate output stream to output file
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    log_out = args.log_dir + '/output.log'
    sys.stdout = Logger(log_out)

    # save args
    json.dump(
            vars(args), open(args.log_dir + '/args.yaml', "w")
        )

    avg_final_acc = np.zeros(args.repeat)
    avg_ood = {'tpr': {}, 'fpr': {}, 'de': {}, 'roc-auc': {}}


    # load results
    if os.path.exists(args.log_dir + '/results.yaml'):
        
        # load yaml results
        save_file = args.log_dir + '/results.yaml'
        with open(save_file, 'r') as yaml_file:
            yaml_result = yaml.safe_load(yaml_file)
            avg_acc_all = np.asarray(yaml_result['avg_acc_history'])
        save_file = args.log_dir + '/results_pt.yaml'
        with open(save_file, 'r') as yaml_file:
            yaml_result = yaml.safe_load(yaml_file)
            avg_acc_table = np.asarray(yaml_result['acc_pt_history'])
        save_file = args.log_dir + '/results_pt_bounded.yaml'
        with open(save_file, 'r') as yaml_file:
            yaml_result = yaml.safe_load(yaml_file)
            avg_acc_table_pt = np.asarray(yaml_result['acc_pt_history'])

        # load ood results
        if os.path.exists(args.log_dir + '/results_ood-tpr.yaml'):
            for key in avg_ood.keys():
                save_file = args.log_dir + '/results_ood-'+key+'.yaml'
                with open(save_file, 'r') as yaml_file:
                    yaml_result = yaml.safe_load(yaml_file)
                    avg_ood[key] = np.asarray(yaml_result['history'])

        # next repeat needed
        start_r = avg_acc_all.shape[1]

        # extend if more repeats left
        if start_r < args.repeat:
            max_task = avg_acc_all.shape[0]
            avg_acc_table = np.append(avg_acc_table, np.zeros((max_task, max_task, args.repeat-start_r)), axis=-1)
            avg_acc_table_pt = np.append(avg_acc_table_pt, np.zeros((max_task, max_task, args.repeat-start_r)), axis=-1)
            avg_acc_all = np.append(avg_acc_all, np.zeros((max_task,args.repeat-start_r)), axis=-1)
            if os.path.exists(args.log_dir + '/results_ood-tpr.yaml'):
                for key in avg_ood.keys(): avg_ood[key] = np.append(avg_ood[key], np.zeros((avg_ood[key].shape[0],args.repeat-start_r)), axis=-1)

    else:
        start_r = 0

    for r in range(start_r, args.repeat):

        print('************************************')
        print('* STARTING TRIAL ' + str(r+1))
        print('************************************')

        # set random seeds
        seed = r
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        ####### Run the experiment #######
        acc_table, acc_table_pt, task_names, run_ood = run(args, seed)
        print(acc_table)

        # number of tasks to be performed
        if args.max_task > 0:
            max_task = min(args.max_task, len(task_names))
        else:
            max_task = len(task_names)

        # init total run store
        if r == 0: 
            avg_acc_table = np.zeros((max_task, max_task, args.repeat))
            avg_acc_table_pt = np.zeros((max_task, max_task, args.repeat))
            avg_acc_all = np.zeros((max_task,args.repeat))
            for key in avg_ood.keys(): avg_ood[key] = np.zeros((min(max_task,len(task_names)-1),args.repeat))

        # Calculate average performance across tasks
        # Customize this part for a different performance metric
        avg_acc_history = [0] * max_task
        for i in range(max_task):
            train_name = task_names[i]
            cls_acc_sum = 0
            for j in range(i+1):
                val_name = task_names[j]
                cls_acc_sum += acc_table[val_name][train_name]
                avg_acc_table[j,i,r] = acc_table[val_name][train_name]
                avg_acc_table_pt[j,i,r] = acc_table_pt[val_name][train_name]
            avg_acc_history[i] = cls_acc_sum / (i + 1)
            print('Task', train_name, 'average acc:', avg_acc_history[i])

        # Gather the final avg accuracy
        avg_final_acc[r] = avg_acc_history[-1]
        avg_acc_all[:,r] = avg_acc_history
        

        # Print the summary so far
        print('===Summary of experiment repeats:', r+1, '/', args.repeat, '===')
        print('The last avg acc of all repeats:', avg_final_acc[:r+1])
        print('mean:', avg_final_acc[:r+1].mean(), 'std:', avg_final_acc[:r+1].std())

        # save results in yml files
        yaml_results = {}
        yaml_results['avg_acc_mean'] = avg_acc_all[:,:r+1].mean(axis=1).tolist()
        if r>1: yaml_results['avg_acc_std'] = avg_acc_all[:,:r+1].std(axis=1).tolist()
        yaml_results['avg_acc_history'] = avg_acc_all[:,:r+1].tolist()
        save_file = args.log_dir + '/results.yaml'
        with open(save_file, 'w') as yaml_file:
            yaml.dump(yaml_results, yaml_file, default_flow_style=False)

        # save per task results
        yaml_results = {}
        yaml_results['acc_pt_mean'] = avg_acc_table[:,:,:r+1].mean(axis=2).tolist()
        yaml_results['acc_pt_history'] = avg_acc_table[:,:,:r+1].tolist()
        save_file = args.log_dir + '/results_pt.yaml'
        with open(save_file, 'w') as yaml_file:
            yaml.dump(yaml_results, yaml_file, default_flow_style=False)

        # task indexes known
        yaml_results = {}
        yaml_results['acc_pt_mean'] = avg_acc_table_pt[:,:,:r+1].mean(axis=2).tolist()
        yaml_results['acc_pt_history'] = avg_acc_table_pt[:,:,:r+1].tolist()
        save_file = args.log_dir + '/results_pt_bounded.yaml'
        with open(save_file, 'w') as yaml_file:
            yaml.dump(yaml_results, yaml_file, default_flow_style=False)

        # save ood results
        if run_ood is not None:
            for key in avg_ood.keys(): avg_ood[key][:,r] = run_ood[key]
            for key in avg_ood.keys():
                yaml_results = {}
                yaml_results['mean'] = avg_ood[key][:,:r+1].mean(axis=1).tolist()
                if r>1: yaml_results['std'] = avg_ood[key][:,:r+1].std(axis=1).tolist()
                yaml_results['history'] = avg_ood[key][:,:r+1].tolist()
                save_file = args.log_dir + '/results_ood-'+key+'.yaml'
                with open(save_file, 'w') as yaml_file:
                    yaml.dump(yaml_results, yaml_file, default_flow_style=False)