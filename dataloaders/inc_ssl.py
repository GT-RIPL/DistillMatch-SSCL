from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch
import torch.utils.data as data
from .utils import download_url, check_integrity
import random

"""
This file heavily adapts data-loading from Global Distillation

git url: https://github.com/kibok90/iccv2019-inc

@inproceedings{lee2019overcoming,
    title={Overcoming catastrophic forgetting with unlabeled data in the wild},
    author={Lee, Kibok and Lee, Kimin and Shin, Jinwoo and Lee, Honglak},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
    pages={312--321},
    year={2019}
}
"""

class iCIFAR10(data.Dataset):
    
    # CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }
    super_to_mega = None

    def __init__(self, root, dataset, num_labeled=None, num_unlabeled_pt=0,
                train=True, lab=True, transform=None, l_dist='vanilla', ul_dist=None,
                download=False, tasks=None, seed=-1,rand_split=False, validation=False, kfolds=3):

        # number of unlabeled classes per task in random dist
        self.num_classes_rand_dist = 20 

        # if unlabeled distribution not declared, then same as labeled
        self.lab = lab
        if ul_dist is None: ul_dist = l_dist

        # process rest of args
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train or validation:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []
        self.course_targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
                if 'coarse_labels' in entry:
                    self.course_targets.extend(entry['coarse_labels'])
                
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self._load_meta()
        self.num_classes = len(np.unique(self.targets))

        # resample tasks if not vanilla task
        self.seed = seed
        self.t = -1
        self.l_dist = l_dist
        self.ul_dist = ul_dist
        if self.ul_dist == 'rand':
            self.valid_ul = [np.arange(self.num_classes) for t in range(int(len(tasks) * self.num_classes_rand_dist / self.num_classes))]
            for class_list in self.valid_ul:
                np.random.shuffle(class_list)
            self.valid_ul = np.asarray(self.valid_ul)
            self.valid_ul = self.valid_ul.reshape(len(tasks),self.num_classes_rand_dist)
            self.valid_ul = self.valid_ul.tolist()
            for class_list in self.valid_ul:
                if len(np.unique(np.asarray(class_list))) < self.num_classes_rand_dist:
                    raise ValueError('multiple classes sampled for random task')
        else:
            self.valid_ul = [np.arange(self.num_classes) for t in range(len(tasks))]
        if self.l_dist == 'super':
            self.tasks = []
            if self.ul_dist == 'super' or self.ul_dist == 'neg':
                self.valid_ul = []
            shuffled_superclasses = list(self.super_to_mega.values())
            if self.ul_dist == 'neg':
                shuffle_complete = False
                while not shuffle_complete:
                    random.shuffle(shuffled_superclasses)
                    shuffle_dic = dict(zip(self.super_to_mega.keys(), shuffled_superclasses)) 
                    shuffle_complete = True
                    for key, value in self.super_to_mega.items():
                        if shuffle_dic[key] == value:
                            shuffle_complete = False

                self.super_to_mega = dict(zip(self.super_to_mega.keys(), shuffled_superclasses))           
            for super_k in np.unique(self.course_targets):
                ind_task = np.where(self.course_targets == super_k)[0]
                ind_task_labels = [self.targets[ind_task[i]] for i in range(len(ind_task))]
                classes_in_task = np.unique(ind_task_labels)
                self.tasks.append(classes_in_task.tolist())
                if self.ul_dist == 'super' or self.ul_dist == 'neg':
                    valid_ul_ = []
                    for super_k_embedded in np.unique(self.course_targets):
                        if self.super_to_mega[super_k_embedded] == self.super_to_mega[super_k]:
                            ind_task = np.where(self.course_targets == super_k_embedded)[0]
                            ind_task_labels = [self.targets[ind_task[i]] for i in range(len(ind_task))]
                            classes_in_task = np.unique(ind_task_labels)
                            valid_ul_.extend(classes_in_task.tolist())
                    self.valid_ul.append(valid_ul_) 
            # shuffle order of tasks...
            if rand_split:
                print('=============================================')
                print('Shuffling in Super task (ignore previous shuffle!!!)....')
                print('pre-shuffle:' + str(self.tasks))
                random.seed(self.seed)
                ctasks = list(zip(self.tasks, self.valid_ul))
                random.shuffle(ctasks)
                self.tasks[:], self.valid_ul[:] = zip(*ctasks)
                print('post-shuffle:' + str(self.tasks))
                print('=============================================')
            
        else:
            self.tasks = tasks

        # remap labels to match task order
        c = 0
        self.class_mapping = {}
        self.class_mapping[-1] = -1
        for task in self.tasks:
            for k in task:
                self.class_mapping[k] = c
                c += 1

        # targets as numpy.array
        self.targets = np.array(self.targets)

        # if testing
        if not self.train and not validation:
            self.archive = []
            for task in self.tasks:
                locs = np.isin(self.targets, task).nonzero()[0]
                self.archive.append((self.data[locs].copy(), self.targets[locs].copy()))
            self.unlabeled = None

        # if validation data
        else:
            if validation:

                # get locations of training and validation for kfolds validation
                num_data_per_fold = int(len(self.targets) / kfolds)
                start = 0
                stop = num_data_per_fold
                locs_train = []
                locs_val = []
                for f in range(kfolds):
                    if self.seed == f:
                        locs_val.extend(np.arange(start,stop))
                    else:
                        locs_train.extend(np.arange(start,stop))
                    start += num_data_per_fold
                    stop += num_data_per_fold

                # sample validation data
                self.archive = []
                for task in self.tasks:
                    locs = np.isin(self.targets[locs_val], task).nonzero()[0]
                    self.archive.append((self.data[locs_val][locs].copy(), self.targets[locs_val][locs].copy()))
                self.unlabeled = None

                # rest is training data
                self.data = self.data[locs_train]
                self.targets = self.targets[locs_train]

            if self.train:
                # num labeled examples per class for sampling
                num_labeled_pc = int(num_labeled / self.num_classes)

                # split dataset
                self.archive = []
                ul_data = []
                ul_targets = []

                for task in self.tasks:
                    data_k = []
                    targets_k = []
                    for k in task:
                        # get indexes of dataset corresponding to k
                        locs = (self.targets == k).nonzero()[0]

                        # shuffle with given seed - remove risk
                        locs_ind = locs[np.random.RandomState(seed=self.seed).permutation(len(locs))]

                        # sample locations for labeled and unlabeled data
                        locs_labeled = locs_ind[:num_labeled_pc]
                        locs_unlabeled = locs_ind[num_labeled_pc:]

                        # append labeled data
                        data_k.extend(self.data[locs_labeled].copy())
                        targets_k.extend(self.targets[locs_labeled].copy())

                        # append unlabeled data
                        ul_data.extend(self.data[locs_unlabeled].copy())
                        ul_targets.extend(self.targets[locs_unlabeled].copy())
                    
                    # append task
                    self.archive.append((data_k,targets_k))
                
                # combine unlabeled data
                ul_data = np.asarray(ul_data)
                ul_targets = np.asarray(ul_targets)

                # first find the tasks for which each class is present
                class_presense = [[] for k in range(self.num_classes)]
                for t in range(len(self.tasks)):
                    valid_classes = self.valid_ul[t]
                    for k in valid_classes: class_presense[k].append(t)
                
                # next, sample unlabeled data for each class
                ul_data_samples = [[] for k in range(len(self.tasks))]
                for k in range(self.num_classes):

                    # get indexes of dataset corresponding to k
                    locs = (ul_targets == k).nonzero()[0]
                    num_k = len(locs)

                    # shuffle with given seed - remove risk
                    locs_ind = locs[np.random.RandomState(seed=self.seed).permutation(len(locs))]

                    # number of samples allowed per task for this class
                    num_sample_pt = int(num_k / len(class_presense[k]))

                    # append samples to each valid task
                    start = 0
                    stop = num_sample_pt
                    for t in class_presense[k]:
                        ul_data_samples[t].extend(locs_ind[start:stop])
                        start += num_sample_pt
                        stop += num_sample_pt
                    ul_data_samples_len = [len(np.unique(np.asarray(ul_targets[samples]))) for samples in ul_data_samples]
        
                # finally, form unlabeled tuples
                self.unlabeled = []
                for t in range(len(self.tasks)):
                    self.unlabeled.append((ul_data[ul_data_samples[t]],ul_targets[ul_data_samples[t]]))

                self.num_sample_ul = num_unlabeled_pt

        if self.train:
            self.coreset = (np.zeros(0, dtype=self.data.dtype), np.zeros(0, dtype=self.targets.dtype))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        
        return img, self.class_mapping[target], self.t

    def load_dataset(self, prev, t, train=True, included_unlabeled = True, start = 0.0, end = 1.0):
        if train:
            if self.lab:
                self.data, self.targets = self.archive[t]
            else:
                self.load_ul(t)   
        else:
            self.data    = np.concatenate([self.archive[s][0] for s in range(t+1)], axis=0)
            self.targets = np.concatenate([self.archive[s][1] for s in range(t+1)], axis=0)

        self.t = t

        
        print('*********')
        print('training - ' + str(self.train))
        print('labeled - ' + str(self.lab))
        print('classes - ' + str(np.unique(self.targets)))
        print('num data - ' + str(len(self.targets)))
        print('*********')

        # for backup
        shuffle_indexes = np.arange(len(self.targets))
        np.random.shuffle(shuffle_indexes)
        self.data = [self.data[loc] for loc in shuffle_indexes]
        self.targets = [self.targets[loc] for loc in shuffle_indexes]
        self.data_backup = self.data
        self.targets_backup = self.targets

    def sample_dataset(self, start = 0.0, end = 1.0, last_class = 0):

        # if labeled, split evenly by classes
        if self.lab:
            self.data = []
            self.targets = []

            for k in range(self.num_classes):

                # get indexes of dataset corresponding to k
                locs = (np.asarray(self.targets_backup) == k).nonzero()[0]
                num_k = len(locs)

                if k < last_class:
                    if end < 1:
                        self.data.extend([self.data_backup[locs[i]] for i in range(0, num_k)])
                        self.targets.extend([self.targets_backup[locs[i]] for i in range(0, num_k)])
                else:
                    start_index = int(start * num_k)
                    end_index = int(end * num_k)
                    self.data.extend([self.data_backup[locs[i]] for i in range(start_index, end_index)])
                    self.targets.extend([self.targets_backup[locs[i]] for i in range(start_index, end_index)])
        
        # if not, random sample!
        else:
            # sample start and end
            start_index = int(start * len(self.targets_backup))
            end_index = int(end * len(self.targets_backup))
            self.data = [self.data_backup[loc] for loc in range(start_index, end_index)]
            self.targets = [self.targets_backup[loc] for loc in range(start_index, end_index)]


    def append_coreset(self, only=False, interp=False):
        if self.train and (len(self.coreset[0]) > 0):
            if only:
                self.data, self.targets = self.coreset
            else:
                self.data = np.concatenate([self.data, self.coreset[0]], axis=0)
                self.targets = np.concatenate([self.targets, self.coreset[1]], axis=0)

        # for backup
        shuffle_indexes = np.arange(len(self.targets))
        np.random.shuffle(shuffle_indexes)
        self.data = [self.data[loc] for loc in shuffle_indexes]
        self.targets = [self.targets[loc] for loc in shuffle_indexes]
        self.data_backup = self.data
        self.targets_backup = self.targets

    def load_ul(self, t):
        
        # sample unlabeled examples
        if self.num_sample_ul == 0:
            self.data = []
            self.targets = []
        else:
            # now, sample and load all!
            locs_chosen = np.arange(len(self.unlabeled[t][0]))
            self.data = self.unlabeled[t][0][locs_chosen]
            self.targets = self.unlabeled[t][1][locs_chosen]

    # Note - all methods update the coreset in the same way
    # We use random coreset updating for a fair comparison between DM, GD, DR, and E2E
    def update_coreset(self, coreset_size, seen):
        num_data_per = coreset_size // len(seen)
        remainder = coreset_size % len(seen)
        data = []
        targets = []

        # random coreset management; latest classes take memory remainder
        # coreset selection without affecting RNG state
        state = np.random.get_state()
        np.random.seed(self.seed*10000+self.t)
        for k in reversed(seen):
            mapped_targets = [self.class_mapping[self.targets[i]] for i in range(len(self.targets))]
            locs = (mapped_targets == k).nonzero()[0]
            if (remainder > 0) and (len(locs) > num_data_per):
                num_data_k = num_data_per + 1
                remainder -= 1
            else:
                num_data_k = min(len(locs), num_data_per)
            locs_chosen = locs[np.random.choice(len(locs), num_data_k, replace=False)]
            data.append([self.data[loc] for loc in locs_chosen])
            targets.append([self.targets[loc] for loc in locs_chosen])
        self.coreset = (np.concatenate(list(reversed(data)), axis=0), np.concatenate(list(reversed(targets)), axis=0))
        np.random.set_state(state)

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=self.root)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class iCIFAR100(iCIFAR10):

    # CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    super_to_mega = {
        0:0, 1:0, 2:1, 3:2, 
        4:1, 5:2, 6:2, 7:3, 
        8:4, 9:5, 10:5, 11:4, 
        12:4, 13:3, 14:6, 15:4, 
        16:4, 17:1, 18:7, 19:7
        }


class iTinyIMNET(iCIFAR10):

    # tiny imagenet https://github.com/rmccorm4/Tiny-Imagenet-200
    # wget http://cs231n.stanford.edu/tiny-imagenet-200.zip

    def __init__(self, root, dataset, num_labeled=None, num_unlabeled_pt=0,
                train=True, lab=True, transform=None, l_dist='vanilla', ul_dist=None,
                download=False, tasks=None, seed=-1,rand_split=False, validation=False, kfolds=3):

        # number of unlabeled classes per task in random dist
        self.num_classes_rand_dist = 20 

        # if unlabeled distribution not declared, then same as labeled
        self.lab = lab
        if ul_dist is None: ul_dist = l_dist

        # process rest of args
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train  # training set or test set

        from os import path
        root = os.path.expanduser(root)
        FileNameEnd = 'JPEG'
        train_dir = path.join(root, 'tiny-imagenet-200/train')
        self.class_names = sorted(os.listdir(train_dir))
        self.names2index = {v: k for k, v in enumerate(self.class_names)}
        self.data = []
        self.targets = []

        if self.train:
            for label in self.class_names:
                d = path.join(root, 'tiny-imagenet-200/train', label)
                for directory, _, names in os.walk(d):
                    for name in names:
                        filename = path.join(directory, name)
                        if filename.endswith(FileNameEnd):
                            self.data.append(filename)
                            self.targets.append(self.names2index[label])
        else:
            val_dir = path.join(root, 'tiny-imagenet-200/val')
            with open(path.join(val_dir, 'val_annotations.txt'), 'r') as f:
                infos = f.read().strip().split('\n')
                infos = [info.strip().split('\t')[:2] for info in infos]
                self.data = [path.join(val_dir, 'images', info[0])for info in infos]
                self.targets = [self.names2index[info[1]] for info in infos]

        self.num_classes = len(np.unique(self.targets))

        # resample tasks if not vanilla task
        self.seed = seed
        self.t = -1
        self.l_dist = l_dist
        self.ul_dist = ul_dist
        self.valid_ul = [np.arange(self.num_classes) for t in range(len(tasks))]
        self.tasks = tasks

        # remap labels to match task order
        c = 0
        self.class_mapping = {}
        self.class_mapping[-1] = -1
        for task in self.tasks:
            for k in task:
                self.class_mapping[k] = c
                c += 1

        # targets as numpy.array
        self.targets = np.array(self.targets)

        # if testing
        if not self.train and not validation:
            self.archive = []
            for task in self.tasks:
                locs = np.isin(self.targets, task).nonzero()[0]
                self.archive.append(([self.data[ind] for ind in locs], self.targets[locs].copy()))
            self.unlabeled = None

        # if validation data
        else:
            if validation:

                # get locations of training and validation for kfolds validation
                num_data_per_fold = int(len(self.targets) / kfolds)
                start = 0
                stop = num_data_per_fold
                locs_train = []
                locs_val = []
                for f in range(kfolds):
                    if self.seed == f:
                        locs_val.extend(np.arange(start,stop))
                    else:
                        locs_train.extend(np.arange(start,stop))
                    start += num_data_per_fold
                    stop += num_data_per_fold

                # sample validation data
                self.archive = []
                for task in self.tasks:
                    locs = np.isin(self.targets[locs_val], task).nonzero()[0]
                    self.archive.append(([self.data[locs_val][ind] for ind in locs], self.targets[locs_val][locs].copy()))
                self.unlabeled = None

                # rest is training data
                self.data = self.data[locs_train]
                self.targets = self.targets[locs_train]

            if self.train:
                # num labeled examples per class for sampling
                num_labeled_pc = int(num_labeled / self.num_classes)

                # split dataset
                self.archive = []
                ul_data = []
                ul_targets = []

                for task in self.tasks:
                    data_k = []
                    targets_k = []
                    for k in task:
                        # get indexes of dataset corresponding to k
                        locs = (self.targets == k).nonzero()[0]

                        # shuffle with given seed - remove risk
                        locs_ind = locs[np.random.RandomState(seed=self.seed).permutation(len(locs))]

                        # sample locations for labeled and unlabeled data
                        locs_labeled = locs_ind[:num_labeled_pc]
                        locs_unlabeled = locs_ind[num_labeled_pc:]

                        # append labeled data
                        data_k.extend([self.data[ind] for ind in locs_labeled])
                        # data_k.extend(self.data[locs_labeled].copy())
                        targets_k.extend(self.targets[locs_labeled].copy())

                        # append unlabeled data
                        ul_data.extend([self.data[ind] for ind in locs_unlabeled])
                        # ul_data.extend(self.data[locs_unlabeled].copy())
                        ul_targets.extend(self.targets[locs_unlabeled].copy())
                    
                    # append task
                    self.archive.append((data_k,targets_k))
                
                # combine unlabeled data
                ul_data = np.asarray(ul_data)
                ul_targets = np.asarray(ul_targets)

                # first find the tasks for which each class is present
                class_presense = [[] for k in range(self.num_classes)]
                for t in range(len(self.tasks)):
                    valid_classes = self.valid_ul[t]
                    for k in valid_classes: class_presense[k].append(t)
                
                # next, sample unlabeled data for each class
                ul_data_samples = [[] for k in range(len(self.tasks))]
                for k in range(self.num_classes):

                    # get indexes of dataset corresponding to k
                    locs = (ul_targets == k).nonzero()[0]
                    num_k = len(locs)

                    # shuffle with given seed - remove risk
                    locs_ind = locs[np.random.RandomState(seed=self.seed).permutation(len(locs))]

                    # number of samples allowed per task for this class
                    num_sample_pt = int(num_k / len(class_presense[k]))

                    # append samples to each valid task
                    start = 0
                    stop = num_sample_pt
                    for t in class_presense[k]:
                        ul_data_samples[t].extend(locs_ind[start:stop])
                        start += num_sample_pt
                        stop += num_sample_pt
                    ul_data_samples_len = [len(np.unique(np.asarray(ul_targets[samples]))) for samples in ul_data_samples]
        
                # finally, form unlabeled tuples
                self.unlabeled = []
                for t in range(len(self.tasks)):
                    self.unlabeled.append((ul_data[ul_data_samples[t]],ul_targets[ul_data_samples[t]]))

                self.num_sample_ul = num_unlabeled_pt

        if self.train:
            self.coreset = ([], np.zeros(0, dtype=self.targets.dtype))

    def __getitem__(self, index):
        path, target= self.data[index], self.targets[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        return img, self.class_mapping[target], self.t

    def load_dataset(self, prev, t, train=True, included_unlabeled = True, start = 0.0, end = 1.0):
        if train:
            if self.lab:
                self.data, self.targets = self.archive[t]
            else:
                self.load_ul(t)   
        else:
            self.data    = np.concatenate([self.archive[s][0] for s in range(t+1)], axis=0)
            self.targets = np.concatenate([self.archive[s][1] for s in range(t+1)], axis=0)

        self.t = t

        
        print('*********')
        print('training - ' + str(self.train))
        print('labeled - ' + str(self.lab))
        print('classes - ' + str(np.unique(self.targets)))
        print('num data - ' + str(len(self.targets)))
        print('*********')

        # for backup
        shuffle_indexes = np.arange(len(self.targets))
        np.random.shuffle(shuffle_indexes)
        self.data = [self.data[loc] for loc in shuffle_indexes]
        self.targets = [self.targets[loc] for loc in shuffle_indexes]
        self.data_backup = self.data
        self.targets_backup = self.targets

    def sample_dataset(self, start = 0.0, end = 1.0, last_class = 0):

        # if labeled, split evenly by classes
        if self.lab:
            self.data = []
            self.targets = []

            for k in range(self.num_classes):

                # get indexes of dataset corresponding to k
                locs = (np.asarray(self.targets_backup) == k).nonzero()[0]
                num_k = len(locs)

                if k < last_class:
                    if end < 1:
                        self.data.extend([self.data_backup[locs[i]] for i in range(0, num_k)])
                        self.targets.extend([self.targets_backup[locs[i]] for i in range(0, num_k)])
                else:
                    start_index = int(start * num_k)
                    end_index = int(end * num_k)
                    self.data.extend([self.data_backup[locs[i]] for i in range(start_index, end_index)])
                    self.targets.extend([self.targets_backup[locs[i]] for i in range(start_index, end_index)])
        
        # if not, random sample!
        else:
            # sample start and end
            start_index = int(start * len(self.targets_backup))
            end_index = int(end * len(self.targets_backup))
            self.data = [self.data_backup[loc] for loc in range(start_index, end_index)]
            self.targets = [self.targets_backup[loc] for loc in range(start_index, end_index)]


    def append_coreset(self, only=False, interp=False):
        if self.train and (len(self.coreset[0]) > 0):
            if only:
                self.data, self.targets = self.coreset
            else:
                self.data = np.concatenate([self.data, self.coreset[0]], axis=0)
                self.targets = np.concatenate([self.targets, self.coreset[1]], axis=0)

        # for backup
        shuffle_indexes = np.arange(len(self.targets))
        np.random.shuffle(shuffle_indexes)
        self.data = [self.data[loc] for loc in shuffle_indexes]
        self.targets = [self.targets[loc] for loc in shuffle_indexes]
        self.data_backup = self.data
        self.targets_backup = self.targets

    def load_ul(self, t):
        
        # sample unlabeled examples
        if self.num_sample_ul == 0:
            self.data = []
            self.targets = []
        else:
            # now, sample and load all!
            locs_chosen = np.arange(len(self.unlabeled[t][0]))
            self.data = self.unlabeled[t][0][locs_chosen]
            self.targets = self.unlabeled[t][1][locs_chosen]

    # Note - all methods update the coreset in the same way
    # We use random coreset updating for a fair comparison between DM, GD, DR, and E2E
    def update_coreset(self, coreset_size, seen):
        num_data_per = coreset_size // len(seen)
        remainder = coreset_size % len(seen)
        data = []
        targets = []

        # random coreset management; latest classes take memory remainder
        # coreset selection without affecting RNG state
        state = np.random.get_state()
        np.random.seed(self.seed*10000+self.t)
        for k in reversed(seen):
            mapped_targets = [self.class_mapping[self.targets[i]] for i in range(len(self.targets))]
            locs = (mapped_targets == k).nonzero()[0]
            if (remainder > 0) and (len(locs) > num_data_per):
                num_data_k = num_data_per + 1
                remainder -= 1
            else:
                num_data_k = min(len(locs), num_data_per)
            locs_chosen = locs[np.random.choice(len(locs), num_data_k, replace=False)]
            data.append([self.data[loc] for loc in locs_chosen])
            targets.append([self.targets[loc] for loc in locs_chosen])
        self.coreset = (np.concatenate(list(reversed(data)), axis=0), np.concatenate(list(reversed(targets)), axis=0))
        np.random.set_state(state)

class SSLDataLoader(object):
    def __init__(self, labeled_dset, unlabeled_dset):
        self.labeled_dset = labeled_dset
        self.unlabeled_dset = unlabeled_dset

        self.labeled_iter = iter(self.labeled_dset)
        self.unlabeled_iter = iter(self.unlabeled_dset)

    def __iter__(self):
        self.labeled_iter = iter(self.labeled_dset)
        return self

    def __len__(self):
        return len(self.labeled_dset)

    def __next__(self):
        
        # labeled
        xl, yl, task = next(self.labeled_iter)
        shuffle_idx = torch.randperm(len(yl), device=yl.device)
        xl, yl, task = [xl[k][shuffle_idx] for k in range(len(xl))], yl[shuffle_idx], task[shuffle_idx]

        # unlabeled
        try:
            xu, yul, _ = next(self.unlabeled_iter)
        except:
            self.unlabeled_iter = iter(self.unlabeled_dset)
            xu, yul, _ = next(self.unlabeled_iter)
        shuffle_idx = torch.randperm(len(yul), device=yul.device)
        xu, yul = [xu[k][shuffle_idx] for k in range(len(xu))], yul[shuffle_idx]

        return xl, yl, xu, yul, task