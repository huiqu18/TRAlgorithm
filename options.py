import os
import numpy as np
import argparse
from torchvision import transforms


class Options:
    def __init__(self):
        self.dataset = 'CIFAR10'
        self.random_seed = -1

        self.model_name = 'ResNet18'
        self.num_epoch = 200
        self.batch_size = 8
        self.weight_decay = 1e-4

        self.optimizer = 'TRSGD'    # the optimizer
        self.inner_S = 10       # S: the number of iteration in the inner loop
        self.lr = 0.1
        self.lambda_w = 0.1 * self.lr
        self.momentum = 0.9
        self.weight_decay = 1e-4

        self.log_interval = 500
        self.workers = 2
        self.gpu = [0, ]

        self.data_dir = './data'     # path to the images
        self.save_dir = './experiments/{:s}'.format(self.dataset.lower())   # path to save results
        self.checkpoint = None
        self.start_epoch = 0

        # data transform
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    def parse(self):
        """ Parse the options, replace the default value if there is a new input """
        parser = argparse.ArgumentParser(description='')
        parser.add_argument('--optimizer', type=str, default=self.optimizer, help='choose optimizer')
        parser.add_argument('--dataset', type=str, default=self.dataset, help='select dataset (CIFAR10/CIFAR100)')
        parser.add_argument('--random-seed', type=int, default=self.random_seed, help='set the random seed')
        parser.add_argument('--model', type=str, default=self.model_name, help='specify the model')
        parser.add_argument('--epochs', type=int, default=self.num_epoch, help='training epochs')
        parser.add_argument('--lr', type=float, help='learning rate')
        parser.add_argument('--batch-size', type=int, default=self.batch_size, help='training batch size')
        parser.add_argument('--gpu', type=list, default=self.gpu, help='GPU for training')
        parser.add_argument('--log-interval', type=int, default=self.log_interval, help='how many batches to wait before logging training status')
        parser.add_argument('--checkpoint', type=str, default=self.checkpoint, help='directory to load a checkpoint')
        parser.add_argument('--data-dir', type=str, default=self.data_dir, help='directory of training data')
        parser.add_argument('--save-dir', type=str, default=self.save_dir, help='directory to save training results')
        args = parser.parse_args()

        self.optimizer = args.optimizer
        if args.lr is not None:
            self.lr = args.lr
        elif 'sgd' in self.optimizer.lower():
            self.lr = 0.1
        else:
            self.lr = 0.001

        self.lambda_w = self.lr * 0.1
        self.dataset = args.dataset
        self.model_name = args.model
        self.random_seed = args.random_seed
        self.num_epoch = args.epochs
        self.batch_size = args.batch_size
        self.gpu = args.gpu
        self.log_interval = args.log_interval
        self.checkpoint = args.checkpoint
        self.data_dir = args.data_dir
        self.save_dir = args.save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def print_options(self, logger=None):
        message = '\n'
        message += '------------------------- Options -----------------------\n'
        for k, v in sorted(self.__dict__.items()):
            if 'transform' in str(k):
                v = str(v).replace('\n', '\n                 ')
            message += '{:>15}: {:<50}\n'.format(str(k), str(v))
        message += '------------------------- End ---------------------------'
        if not logger:
            print(message)
        else:
            logger.info(message)

    def save_options(self):
        file = open('{:s}/options.txt'.format(self.save_dir), 'w')
        file.write('# ------------------------- Options ----------------------- #\n\n')
        for k, v in sorted(self.__dict__.items()):
            if 'transform' in str(k):
                v = str(v).replace('\n', '\n                 ')
            file.write('{:>15}: {:<50}\n'.format(str(k), str(v)))
        file.close()