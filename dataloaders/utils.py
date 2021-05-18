import os
import os.path
import hashlib
import errno
import torch
from torchvision import transforms
import numpy as np
import random
import PIL
from PIL import Image, ImageEnhance, ImageOps
from torchvision import transforms as T
import cv2

dataset_stats = {
    'CIFAR10' : {'mean': (0.49139967861519607, 0.48215840839460783, 0.44653091444546567),
                 'std' : (0.2470322324632819, 0.24348512800005573, 0.26158784172796434),
                 'size' : 32},
    'CIFAR100': {'mean': (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                 'std' : (0.2673342858792409, 0.25643846291708816, 0.2761504713256834),
                 'size' : 32},
    'TinyIMNET': {'mean': (0.4389, 0.4114, 0.3682),
                 'std' : (0.2402, 0.2350, 0.2268),
                 'size' : 64},
}
    
# k transormations 
class TransformK:
    def __init__(self, transform, transformb, k):
        self.transform = transform
        self.transformb = transformb
        self.k = k

    def __call__(self, inp):
        x = [self.transform(inp)]
        for _ in range(self.k-1): x.append(self.transformb(inp))
        return x

# transformations
def get_transform(dataset='cifar100', phase='test', aug=True, hard_aug=False):
    transform_list = []
    if phase == 'train' and not ('mnist' in dataset) and aug:
        if hard_aug:
            transform_list.extend([
                transforms.ColorJitter(brightness=63/255, contrast=0.8),
                RandomAugment(),
                transforms.ToTensor(), \
                transforms.Normalize(dataset_stats[dataset]['mean'], dataset_stats[dataset]['std']),
                Cutout()
                                ])
        else:
            transform_list.extend([
                transforms.ColorJitter(brightness=63/255, contrast=0.8),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(dataset_stats[dataset]['size'], padding=4),
                transforms.ToTensor(),
                transforms.Normalize(dataset_stats[dataset]['mean'], dataset_stats[dataset]['std']),
                                ])
    else:
        transform_list.extend([
                transforms.ToTensor(),
                transforms.Normalize(dataset_stats[dataset]['mean'], dataset_stats[dataset]['std']),
                                ])
    
    return transforms.Compose(transform_list)

def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def download_url(url, root, filename, md5):
    from six.moves import urllib

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath)
        except:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath)

def list_dir(root, prefix=False):
    """List all directories at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
        filter(
            lambda p: os.path.isdir(os.path.join(root, p)),
            os.listdir(root)
        )
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories

def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files


"""
Code adapted from
https://github.com/4uiiurz1/pytorch-auto-augment/blob/master/auto_augment.py
"""
class RandomAugment:
    """
    Random aggressive data augmentation transformer.
    """
    def __init__(self, N=2, M=9):
        """
        :param N: int, [1, #ops]. max number of operations
        :param M: int, [0, 9]. max magnitude of operations
        """
        self.operations = {
            'Identity': lambda img, magnitude: self.identity(img, magnitude),
            'ShearX': lambda img, magnitude: self.shear_x(img, magnitude),
            'ShearY': lambda img, magnitude: self.shear_y(img, magnitude),
            'TranslateX': lambda img, magnitude: self.translate_x(img, magnitude),
            'TranslateY': lambda img, magnitude: self.translate_y(img, magnitude),
            'Rotate': lambda img, magnitude: self.rotate(img, magnitude),
            'Mirror': lambda img, magnitude: self.mirror(img, magnitude),
            'AutoContrast': lambda img, magnitude: self.auto_contrast(img, magnitude),
            'Equalize': lambda img, magnitude: self.equalize(img, magnitude),
            'Solarize': lambda img, magnitude: self.solarize(img, magnitude),
            'Posterize': lambda img, magnitude: self.posterize(img, magnitude),
            'Invert': lambda img, magnitude: self.invert(img, magnitude),
            'Contrast': lambda img, magnitude: self.contrast(img, magnitude),
            'Color': lambda img, magnitude: self.color(img, magnitude),
            'Brightness': lambda img, magnitude: self.brightness(img, magnitude),
            'Sharpness': lambda img, magnitude: self.sharpness(img, magnitude)
        }

        self.N = np.clip(N, a_min=1, a_max=len(self.operations))
        self.M = np.clip(M, a_min=0, a_max=9)

    def identity(self, img, magnitude):
        return img

    def transform_matrix_offset_center(self, matrix, x, y):
        o_x = float(x) / 2 + 0.5
        o_y = float(y) / 2 + 0.5
        offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
        reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
        transform_matrix = offset_matrix @ matrix @ reset_matrix
        return transform_matrix

    def shear_x(self, img, magnitude):
        img = img.transpose(Image.TRANSPOSE)
        magnitudes = np.random.choice([-1.0, 1.0]) * np.linspace(0, 0.3, 11)
        transform_matrix = np.array([[1, random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]), 0],
                                     [0, 1, 0],
                                     [0, 0, 1]])
        transform_matrix = self.transform_matrix_offset_center(transform_matrix, img.size[0], img.size[1])
        img = img.transform(img.size, Image.AFFINE, transform_matrix.flatten()[:6], Image.BICUBIC)
        img = img.transpose(Image.TRANSPOSE)
        return img

    def shear_y(self, img, magnitude):
        img = img.transpose(Image.TRANSPOSE)
        magnitudes = np.random.choice([-1.0, 1.0]) * np.linspace(0, 0.3, 11)
        transform_matrix = np.array([[1, 0, 0],
                                     [random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]), 1, 0],
                                     [0, 0, 1]])
        transform_matrix = self.transform_matrix_offset_center(transform_matrix, img.size[0], img.size[1])
        img = img.transform(img.size, Image.AFFINE, transform_matrix.flatten()[:6], Image.BICUBIC)
        img = img.transpose(Image.TRANSPOSE)
        return img

    def translate_x(self, img, magnitude):
        img = img.transpose(Image.TRANSPOSE)
        magnitudes = np.random.choice([-1.0, 1.0]) * np.linspace(0, 0.3, 11)
        transform_matrix = np.array([[1, 0, 0],
                                     [0, 1, img.size[1]*random.uniform(magnitudes[magnitude], magnitudes[magnitude+1])],
                                     [0, 0, 1]])
        transform_matrix = self.transform_matrix_offset_center(transform_matrix, img.size[0], img.size[1])
        img = img.transform(img.size, Image.AFFINE, transform_matrix.flatten()[:6], Image.BICUBIC)
        img = img.transpose(Image.TRANSPOSE)
        return img

    def translate_y(self, img, magnitude):
        img = img.transpose(Image.TRANSPOSE)
        magnitudes = np.random.choice([-1.0, 1.0]) * np.linspace(0, 0.3, 11)
        transform_matrix = np.array([[1, 0, img.size[0]*random.uniform(magnitudes[magnitude], magnitudes[magnitude+1])],
                                     [0, 1, 0],
                                     [0, 0, 1]])
        transform_matrix = self.transform_matrix_offset_center(transform_matrix, img.size[0], img.size[1])
        img = img.transform(img.size, Image.AFFINE, transform_matrix.flatten()[:6], Image.BICUBIC)
        img = img.transpose(Image.TRANSPOSE)
        return img

    def rotate(self, img, magnitude):
        img = img.transpose(Image.TRANSPOSE)
        magnitudes = np.random.choice([-1.0, 1.0]) * np.linspace(0, 30, 11)
        theta = np.deg2rad(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
        transform_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                     [np.sin(theta), np.cos(theta), 0],
                                     [0, 0, 1]])
        transform_matrix = self.transform_matrix_offset_center(transform_matrix, img.size[0], img.size[1])
        img = img.transform(img.size, Image.AFFINE, transform_matrix.flatten()[:6], Image.BICUBIC)
        img = img.transpose(Image.TRANSPOSE)
        return img

    def mirror(self, img, magnitude):
        img = ImageOps.mirror(img)
        return img

    def auto_contrast(self, img, magnitude):
        img = ImageOps.autocontrast(img)
        return img

    def equalize(self, img, magnitude):
        img = ImageOps.equalize(img)
        return img

    def solarize(self, img, magnitude):
        magnitudes = np.linspace(0, 256, 11)
        img = ImageOps.solarize(img, random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
        return img

    def posterize(self, img, magnitude):
        magnitudes = np.linspace(4, 8, 11)
        img = ImageOps.posterize(img, int(round(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))))
        return img

    def invert(self, img, magnitude):
        img = ImageOps.invert(img)
        return img

    def contrast(self, img, magnitude):
        magnitudes = 1.0 + np.random.choice([-1.0, 1.0])*np.linspace(0.1, 0.9, 11)
        img = ImageEnhance.Contrast(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
        return img

    def color(self, img, magnitude):
        magnitudes = 1.0 + np.random.choice([-1.0, 1.0])*np.linspace(0.1, 0.9, 11)
        img = ImageEnhance.Color(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
        return img

    def brightness(self, img, magnitude):
        magnitudes = 1.0 + np.random.choice([-1.0, 1.0])*np.linspace(0.1, 0.9, 11)
        img = ImageEnhance.Brightness(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
        return img

    def sharpness(self, img, magnitude):
        magnitudes = 1.0 + np.random.choice([-1.0, 1.0])*np.linspace(0.1, 0.9, 11)
        img = ImageEnhance.Sharpness(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
        return img

    def __call__(self, img):
        ops = np.random.choice(list(self.operations.keys()), self.N)
        for op in ops:
            mag = random.randint(0, self.M)
            img = self.operations[op](img, mag)

        return img

class Cutout:
    def __init__(self, M=0.5, fill=0.0):
        self.M = np.clip(M, a_min=0.0, a_max=1.0)
        self.fill = fill

    def __call__(self, x):
        """
        Ref https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
        """
        _, h, w = x.shape
        lh, lw = int(round(self.M * h)), int(round(self.M * w))

        cx, cy = np.random.randint(0, h), np.random.randint(0, w)
        x1 = np.clip(cx - lh // 2, 0, h)
        x2 = np.clip(cx + lh // 2, 0, h)
        y1 = np.clip(cy - lw // 2, 0, w)
        y2 = np.clip(cy + lw // 2, 0, w)
        x[:, x1: x2, y1: y2] = self.fill

        return x