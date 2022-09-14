import argparse
import os
import random
import time
import torch
import numpy as np
import wandb

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from easydict import EasyDict as edict

import utils.train_utils as tutils
from utils.config import cfg, cfg_from_yaml_file
from utils.kitti_dataloader import create_train_dataloader, create_val_dataloader
from utils.losses import ComputeLSTMLoss
from utils.logger import Logger
from utils.wandb_utils import Wandb
from models.lstm_refiner import dLSTM

# hyperparams_default = {
#   "dropout": 0.25,
#   "batch_size": 32,
#   "num_layers": 2,
#   "hidden_dim": 10,
#   "learning_rate": 0.001,
#   "optimizer_type": "adam"
# }

def main(configs):
    wandb_logger = Wandb(configs)
    # Access all hyperparameter values through wandb.config
    hconfig = wandb.config
    # Reproduce results
    if configs.seed is not None:
        random.seed(configs.seed)
        np.random.seed(configs.seed)
        torch.manual_seed(configs.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if configs.gpu_idx is not None:
        print('You have chosen a specific GPU. This will completely disable data parallelism.')
    
    gpu_idx = configs.gpu_idx
    logger = Logger(configs.logs_dir, configs.name)
    logger.info('>>> Created a new logger')
    logger.info('>>> configs: {}'.format(configs))
    tb_writer = SummaryWriter(log_dir=os.path.join(configs.logs_dir, 'tensorboard'))

    configs.device = torch.device('cpu' if configs.gpu_idx is None else 'cuda:{}'.format(configs.gpu_idx))
    
    # model
    model = dLSTM(
        hconfig.batch_size,
        configs.input_size,
        hconfig.hidden_dim,
        configs.device,
        hconfig.dropout)
    print(model)
    
    # resume weights of model from a checkpoint
    if configs.resume_path is not None:
        assert os.path.isfile(configs.resume_path), "no checkpoint found at '{}'".format(configs.resume_path)
        model.load_state_dict(torch.load(configs.resume_path, map_location='cpu'))
        if logger is not None:
            logger.info('resume training model from checkpoint {}'.format(configs.resume_path))
    
    model = model.to(configs.device)

    # Make sure to create optimizer after moving the model to cuda
    optimizer = tutils.create_optimizer(configs, model, hconfig)
    lr_scheduler = tutils.create_lr_scheduler(optimizer, configs)
    configs.step_lr_in_epoch = False if configs.lr_type in ['multi_step', 'cosin'] else True

    # resume optimizer, lr_scheduler from a checkpoint
    if configs.resume_path is not None:
        utils_path = configs.resume_path.replace('Model_', 'Utils_')
        assert os.path.isfile(utils_path), "=> no checkpoint found at '{}'".format(utils_path)
        utils_state_dict = torch.load(utils_path, map_location='cuda:{}'.format(configs.gpu_idx))
        optimizer.load_state_dict(utils_state_dict['optimizer'])
        lr_scheduler.load_state_dict(utils_state_dict['lr_scheduler'])
        configs.start_epoch = utils_state_dict['epoch'] + 1
    
    if logger is not None:
        logger.info(">>> Loading dataset & getting dataloader...")
    
    # Create dataloader
    train_dataloader = create_train_dataloader(configs)
    if configs.evaluate:
        val_dataloader = create_val_dataloader(configs)
        val_loss = validate(val_dataloader, model, configs)
        print('val_loss: {:.4e}'.format(val_loss))
        return
    
    for epoch in range(configs.start_epoch, configs.epochs + 1):
        if logger is not None:
            logger.info('{}'.format('#=' * 35))
            logger.info('{} {}/{} {}'.format('-' * 30, epoch, configs.epochs, '-' * 30))
            logger.info('{}'.format('#=' * 35))
            logger.info('>>>> Epoch: [{}/{}]'.format(epoch, configs.epochs))
        
        # train for one epoch
        train_one_epoch(
            train_dataloader,
            model,
            optimizer,
            lr_scheduler,
            epoch,
            configs,
            logger,
            tb_writer,
            wandb_logger
        )
        
        if (not configs.no_val) and (epoch % configs.checkpoint_freq == 0):
            val_dataloader = create_val_dataloader(configs)
            print('number of batches in val_dataloader: {}'.format(len(val_dataloader)))
            val_loss = validate(val_dataloader, model, configs)
            print('val_loss: {:.4e}'.format(val_loss))
            if tb_writer is not None:
                tb_writer.add_scalar('Val_loss', val_loss, epoch)
                wandb_logger.log({'val_loss': val_loss}, epoch)
        
        # Save checkpoint
        if epoch % configs.checkpoint_freq == 0:
            model_state_dict, utils_state_dict = tutils.get_saved_state(model, optimizer, lr_scheduler, epoch, configs)
            tutils.save_checkpoint(configs.checkpoints_dir, configs.name, model_state_dict, utils_state_dict, epoch)
        
        if not configs.step_lr_in_epoch:
            lr_scheduler.step()
            if tb_writer is not None:
                tb_writer.add_scalar('LR', lr_scheduler.get_lr()[0], epoch)
                wandb_logger.log({'LR': lr_scheduler.get_lr()[0]}, epoch)
    if tb_writer is not None:
        tb_writer.close()


def train_one_epoch(train_dataloader, model, optimizer, lr_scheduler, epoch, configs, logger, tb_writer, wandb_logger):
    batch_time = tutils.AverageMeter('Time', ':6.3f')
    data_time = tutils.AverageMeter('Data', ':6.3f')
    losses = tutils.AverageMeter('Loss', ':.4e')
    progress = tutils.ProgressMeter(len(train_dataloader), [batch_time, data_time, losses],
                             prefix="Train - Epoch: [{}/{}]".format(epoch, configs.epochs))

    criterion = ComputeLSTMLoss()
    num_iters_per_epoch = len(train_dataloader)
    model.train()
    start_time = time.time()
    for batch_idx, batch_data in enumerate(tqdm(train_dataloader)):
        data_time.update(time.time() - start_time)
        hc = model.init_hidden(configs.device)
        inputs, targets = batch_data
        batch_size = inputs.size(0)
        global_step = num_iters_per_epoch * (epoch - 1) + batch_idx + 1
        inputs = inputs.to(configs.device)
        targets = targets.to(configs.device)
        inputs = inputs.permute(1, 0, 2)
        outputs = model(inputs, hc)
        total_loss, loss_stats = criterion(outputs, targets)
        total_loss.backward()
        optimizer.step()
        # zero the parameter gradients
        optimizer.zero_grad()
        # Adjust learning rate
        if configs.step_lr_in_epoch:
            lr_scheduler.step()
            if tb_writer is not None:
                tb_writer.add_scalar('LR', lr_scheduler.get_lr()[0], global_step)
                wandb_logger.log({'LR': lr_scheduler.get_lr()[0]}, global_step)
        reduced_loss = total_loss.data
        losses.update(tutils.to_python_float(reduced_loss), batch_size)
        # measure elapsed time
        # torch.cuda.synchronize()
        batch_time.update(time.time() - start_time)
        if tb_writer is not None:
            if (global_step % configs.tensorboard_freq) == 0:
                loss_stats['avg_loss'] = losses.avg
                tb_writer.add_scalars('Train', loss_stats, global_step)
                wandb_logger.log(loss_stats, global_step)
        # Log message
        if logger is not None:
            if (global_step % configs.print_freq) == 0:
                logger.info(progress.get_message(batch_idx))
        start_time = time.time()
        print('train_loss: {:.4e}'.format(tutils.to_python_float(reduced_loss)))
    wandb_logger.log({'loss_avg_epoch': losses.avg}, epoch)

def validate(val_dataloader, model, configs):
    losses = tutils.AverageMeter('Loss', ':.4e')
    criterion = ComputeLSTMLoss()

    model.eval()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(val_dataloader)):
            inputs, targets = batch_data
            hc = model.init_hidden(configs.device)
            batch_size = inputs.size(0)
            inputs = inputs.to(configs.device)
            targets = targets.to(configs.device)
            inputs = inputs.permute(1, 0, 2)
            outputs = model(inputs, hc)
            total_loss, loss_stats = criterion(outputs, targets)
            reduced_loss = total_loss.data
            losses.update(tutils.to_python_float(reduced_loss), batch_size)
    return losses.avg


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--seed', type=int, default=2020,
                        help='re-produce the results with seed random')
    parser.add_argument('--cfg_file', type=str, default="/opt/vineet-workspace/lstm-yolo/config/kitti_lstm.yaml",
                        help='specify the config for tracking')
    parser.add_argument('--name', type=str, default='dlstm', metavar='FN',
                        help='The name using for saving logs, models,...')     
    parser.add_argument('--entity', type=str, default='makaveli', metavar='FN',
                        help='The name using for saving logs, models,...')                    
    ####################################################################
    ##############     Dataloader and Running configs            #######
    ####################################################################
    parser.add_argument('--hflip_prob', type=float, default=0.5,
                        help='The probability of horizontal flip')
    parser.add_argument('--no-val', action='store_true',
                        help='If true, dont evaluate the model on the val set')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='mini-batch size (default: 16), this is the total'
                             'batch size of all GPUs on the current node when using'
                             'Data Parallel or Distributed Data Parallel')
    parser.add_argument('--print_freq', type=int, default=50, metavar='N',
                        help='print frequency (default: 50)')
    parser.add_argument('--tensorboard_freq', type=int, default=50, metavar='N',
                        help='frequency of saving tensorboard (default: 50)')
    parser.add_argument('--checkpoint_freq', type=int, default=100, metavar='N',
                        help='frequency of saving checkpoints (default: 5)')
    ####################################################################
    ##############     Training strategy            ####################
    ####################################################################

    parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                        help='the starting epoch')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--dropout', type=float, default=0.4, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--hidden_dim', type=int, default=20, metavar='N',
                        help='number of hidden dims in lstms')
    parser.add_argument('--lr_type', type=str, default='cosin',
                        help='the type of learning rate scheduler (cosin or multi_step or one_cycle)')
    parser.add_argument('--learning_rate', type=float, default=0.001, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--minimum_lr', type=float, default=1e-7, metavar='MIN_LR',
                        help='minimum learning rate during training')
    parser.add_argument('--momentum', type=float, default=0.949, metavar='M',
                        help='momentum')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0., metavar='WD',
                        help='weight decay (default: 0.)')
    parser.add_argument('--optimizer_type', type=str, default='adam', metavar='OPTIMIZER',
                        help='the type of optimizer, it can be sgd or adam')
    parser.add_argument('--steps', nargs='*', default=[150, 180],
                        help='number of burn in step')
    parser.add_argument('--gpu_idx', default=0, type=int,
                        help='GPU index to use.')
    parser.add_argument('--evaluate', action='store_true',
                        help='only evaluate the model, not training')
    parser.add_argument('--resume_path', type=str, default=None, metavar='PATH',
                        help='the path of the resumed checkpoint')
    args = edict(vars(parser.parse_args()))
    configs = cfg_from_yaml_file(args.cfg_file, args)
    configs.checkpoints_dir = os.path.join('checkpoints', configs.name)
    configs.logs_dir = os.path.join('logs', configs.name)
    if not os.path.isdir(configs.checkpoints_dir):
        os.makedirs(configs.checkpoints_dir)
    if not os.path.isdir(configs.logs_dir):
        os.makedirs(configs.logs_dir)

    main(configs)