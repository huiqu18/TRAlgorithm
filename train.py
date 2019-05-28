from __future__ import print_function
import os, shutil
import numpy as np
import logging
import torch
import torch.nn as nn
from torchvision import datasets
from tensorboardX import SummaryWriter
import models
from options import Options
from TRSGD import TRSGD
from TRAdam import TRAdam


def main():
    global opt, tb_writer, logger, logger_results
    
    opt = Options()
    opt.parse()
    opt.save_options()
    
    tb_writer = SummaryWriter('{:s}/tb_logs'.format(opt.save_dir))

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.gpu)

    # set up logger
    logger, logger_results = setup_logger(opt)
    opt.print_options(logger)

    if opt.random_seed >= 0:
        # logger.info("=> Using random seed {:d}".format(opt.random_seed))
        torch.manual_seed(opt.random_seed)
        torch.cuda.manual_seed(opt.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(opt.random_seed)
    else:
        torch.backends.cudnn.benchmark = True

    # define models and same initialization
    if opt.model_name not in models.__model_names__:
        raise NotImplementedError()
    num_classes = 10 if opt.dataset.lower() == 'cifar10' else 100
    model = models.__dict__[opt.model_name](num_classes=num_classes)
    # logger.info('=> Model: {:s}'.format(opt.model_name))
    model = torch.nn.DataParallel(model).cuda()

    # define optimizers
    if opt.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    elif opt.optimizer.lower() == 'trsgd':
        optimizer = TRSGD(model.parameters(), lambda_w=opt.lambda_w, lr=opt.lr, momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    elif opt.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.weight_decay)
    elif opt.optimizer.lower() == 'tradam':
        optimizer = TRAdam(model.parameters(), opt.lambda_w, lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.weight_decay)
    else:
        raise NotImplementedError()

    # define loss function
    global criterion
    criterion = nn.CrossEntropyLoss()

    # define transformation and load dataset
    transform_train, transform_test = opt.transform_train, opt.transform_test
    dataset = datasets.CIFAR10 if opt.dataset.lower() == 'cifar10' else datasets.CIFAR100
    trainset = dataset(root=opt.data_dir, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=1)
    testset = dataset(root=opt.data_dir, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=1)

    # load a checkpoint to resume training if exists
    if opt.checkpoint:
        if os.path.isfile(opt.checkpoint):
            logger.info("=> loading checkpoint '{}'".format(opt.checkpoint))
            checkpoint = torch.load(opt.checkpoint)
            opt.start_epoch = checkpoint['epoch'] - 1
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at {}".format(opt.checkpoint))

    # train and test
    logger.info("=> Optimizer: {:s}".format(opt.optimizer))
    for epoch in range(opt.start_epoch + 1, opt.num_epoch + 1):  # 1 base
        adjust_learning_rate(opt, optimizer, epoch, opt.lr)

        train_stats = train(opt, model, train_loader, optimizer, epoch)
        test_stats = test(opt, model, test_loader, epoch)

        # save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, opt.save_dir)

        # save training results
        logger_results.info('{:<6d}| {r1[0]:<12.4f}{r1[1]:<12.4f}| {r2[0]:<12.4f}{r2[1]:<12.4f}'
                            .format(epoch, r1=train_stats, r2=test_stats))

        # tensorboard logs
        tb_writer.add_scalars('epoch_losses',
                              {'train_loss': train_stats[0], 'test_loss': test_stats[0]}, epoch)
        tb_writer.add_scalars('epoch_accuracies',
                              {'train_acc': train_stats[1], 'test_acc': test_stats[1]}, epoch)
        
    tb_writer.close()
    for i in list(logger.handlers):
        logger.removeHandler(i)
        i.flush()
        i.close()
    for i in list(logger_results.handlers):
        logger_results.removeHandler(i)
        i.flush()
        i.close()


def train(opt,  model, train_loader, optimizer, epoch):
    # loss, acc
    train_result = AverageMeter(2)

    model.train()
    counter = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # if batch_idx == 0:
        #     logger.info('=> lr: {:.3g}'.format(optimizer.param_groups[0]['lr']))

        data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        if opt.optimizer.lower() == 'trsgd' or opt.optimizer.lower() == 'tradam':
            optimizer.step(counter)
            counter = counter + 1 if counter < opt.inner_S - 1 else 0
        else:
            optimizer.step()
        pred = output.max(1, keepdim=True)[1]  # get the index of the max probability
        acc = pred.eq(target.view_as(pred)).sum().item() / float(opt.batch_size)

        result = [loss.item(), acc]
        train_result.update(result, data.size(0))

        if batch_idx % opt.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {r[0]:.4f}\tAcc: {r[1]:.4f}'
                        .format(epoch, batch_idx * len(data),  len(train_loader.dataset),
                                100. * batch_idx / len(train_loader), r=train_result.avg))

    logger.info('=> Train Epoch: {}\tLoss: {r[0]:.4f}\tAcc: {r[1]:.4f}'.format(epoch, r=train_result.avg))
    return train_result.avg


def test(opt, model, test_loader, epoch):
    # loss, acc
    test_result = AverageMeter(2)

    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = criterion(output, target)
            pred = output.max(1, keepdim=True)[1]  # get the index of the max probability
            acc = pred.eq(target.view_as(pred)).sum().item() / float(opt.batch_size)

            result = [loss.item(), acc]
            test_result.update(result, data.size(0))

        logger.info('=> Test Epoch: {}\tLoss: {r[0]:.4f}\tAcc: {r[1]:.4f}\n'.format(epoch, r=test_result.avg))

    return test_result.avg


def adjust_learning_rate(opt, optimizer, epoch, initial_lr):
    # step decay
    if epoch <= opt.num_epoch * 0.5:
        lr = initial_lr
    elif epoch <= opt.num_epoch * 0.75:
        lr = 0.1 * initial_lr
    else:
        lr = 0.01 * initial_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    filename = '{:s}/checkpoint.pth.tar'.format(save_dir)
    torch.save(state, filename)


def setup_logger(opt):
    mode = 'a' if opt.checkpoint else 'w'

    # create logger for training information
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.DEBUG)
    # create console handler and file handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler('{:s}/train_log.txt'.format(opt.save_dir), mode=mode)
    file_handler.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s\t%(message)s', datefmt='%m-%d %I:%M')
    # add formatter to handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    # add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # create logger for epoch results
    logger_results = logging.getLogger('results')
    logger_results.setLevel(logging.DEBUG)

    # create logger for iteration results
    logger_iter_results = logging.getLogger('results_iteration')
    logger_iter_results.setLevel(logging.DEBUG)

    # set up logger for each result
    file_handler2 = logging.FileHandler('{:s}/epoch_results.txt'.format(opt.save_dir), mode=mode)
    file_handler2.setFormatter(logging.Formatter('%(message)s'))
    logger_results.addHandler(file_handler2)

    logger.info('***** Training starts *****')
    logger.info('save directory: {:s}'.format(opt.save_dir))
    if mode == 'w':
        logger_results.info('epoch | Train_loss  Train_acc   | Test_loss   Test_acc')

    return logger, logger_results


class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self, shape=1):
        self.shape = shape
        self.reset()

    def reset(self):
        self.val = np.zeros(self.shape)
        self.avg = np.zeros(self.shape)
        self.sum = np.zeros(self.shape)
        self.count = np.zeros(self.shape)

    def update(self, val, n=1):
        val = np.array(val)
        assert val.shape == self.val.shape
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
