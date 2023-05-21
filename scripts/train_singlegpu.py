import os, sys

from zmq import device
os.chdir('/data/hanhc/GazeEstimation/GSAL_hhc')
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.split(__file__)[0])
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '12345'
from utils.dataloader import GazeDataset, loader
from utils.eval_metrics import *
from model.build_model import Model, Gelossop, Delossop

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP
from torch.utils.tensorboard import SummaryWriter
import cv2
from datetime import datetime
import argparse
from easydict import EasyDict as edict
import yaml
import logging
from tqdm import tqdm

def eval_net(config: edict) -> None: 
    pass

def train_net(config: edict, args=None) -> None:

    # Setting up ----------------
    attentionmap = cv2.imread(config.map, 0) / 255
    attentionmap = torch.from_numpy(attentionmap).type(torch.FloatTensor)
    data = config.data
    params = config.params
    save = config.save
    # device_list = list(config.device.split(','))
    # device_ids = [int(i) for i in device_list]
    # device_ids = args.local_rank
    # print(f'device_ids: {device_ids}')
    exp_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    
    # torch.distributed.init_process_group(backend='nccl', init_method='env://', 
    #     world_size=len(device_ids), rank=device_ids[0])
    
    # torch.cuda.set_device(device_ids)
    # torch.distributed.init_process_group(backend='nccl')

    # Path to save events, logs, and checkpoints
    train_save = os.path.join(save.path, 'train', f'{save.exp}-{exp_time}')
    train_cp_save = os.path.join(train_save, 'checkpoints')
    if not os.path.exists(train_save):
        os.makedirs(train_save)
    if not os.path.exists(train_cp_save):
        os.makedirs(train_cp_save)
    
    # torch.cuda.set_device(f'cuda:{config.device}')

    logging.basicConfig(filename=os.path.join(train_save, f'EXP_{save.exp}_LR_{params.lr}_BS_{params.batch_size}.log'), 
        filemode= 'w', 
        level=logging.INFO, 
        format='%(levelname)s: %(message)s')
    writer = SummaryWriter(log_dir=train_save, comment=f'_EXP_{save.exp}_LR_{params.lr}_BS_{params.batch_size}')
    
    # Preparing data ----------------
    logging.info('Preparing data...')
    trainset = GazeDataset(data, augmentation=False)
    # train_sampler = DistributedSampler(trainset)
    # train_loader = DataLoader(trainset, batch_size=params.batch_size, shuffle=False, 
    #     num_workers=16, sampler=train_sampler)
    train_loader = DataLoader(trainset, batch_size=params.batch_size, shuffle=False, 
        num_workers=16)
    # trainset = loader(data, batch_size=params.batch_size, shuffle=False, 
    #     augmentation=False, num_workers=16, sampler=train_sampler)
    

    # Buiding model ----------------
    logging.info('Buiding model...')
    net = Model()

    # Single GPU
    # net.feature = net.feature.cuda()
    # net.gazeEs = net.gazeEs.cuda()
    # net.deconv = net.deconv.cuda()

    # Multi-GPU Parallel
    net.feature = DP(net.feature.cuda())
    net.gazeEs = DP(net.gazeEs.cuda())
    net.deconv = DP(net.deconv.cuda())

    # Multi-GPU DistributedDataParallel (more efficient)
    # net.feature = DDP(net.feature.cuda(), device_ids=[device_ids])
    # net.gazeEs = DDP(net.gazeEs.cuda(), device_ids=[device_ids])
    # net.deconv = DDP(net.deconv.cuda(), device_ids=[device_ids])

    if config.pretrain:
        net.load_state_dict('PRETRAIN_PATH', strict=False)
    
    # Building optimizer ----------------
    logging.info('Building optimizer...')
    geloss_op = Gelossop(attentionmap, w1=3, w2=1)
    deloss_op = Delossop()
    geloss_op = geloss_op
    deloss_op = deloss_op

    ge_optimizer = torch.optim.RMSprop(net.feature.parameters(), 
            lr=params.lr, weight_decay=1e-4, momentum=0.9)
    
    ga_optimizer = torch.optim.RMSprop(net.gazeEs.parameters(), 
            lr=params.lr, weight_decay=1e-4, momentum=0.9)
    
    de_optimizer = torch.optim.RMSprop(net.deconv.parameters(), 
            lr=params.lr, momentum=0.9)
    
    ge_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(ge_optimizer, 'min', patience=5)
    ga_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(ga_optimizer, 'min', patience=5)
    de_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(de_optimizer, 'min', patience=5)

    # Training ----------------
    logging.info('Training...')
    num_train = len(train_loader) * params.batch_size
    epochs = params.epoch
    global_step = 0

    for epoch in range(epochs):
        net.train()
        epoch_geloss = 0.0
        epoch_deloss = 0.0
        
        with tqdm(total=num_train, desc=f'Epoch {epoch+1}/{epochs}', unit='imgs') as pbar:
            for batch in train_loader:

                # Get data
                (data_x, label) = batch
                data_x.face = data_x.face.cuda()
                label = label.cuda()

                # Foward
                gaze, img = net(data_x)
                ge_optimizer.zero_grad()
                ga_optimizer.zero_grad()
                de_optimizer.zero_grad()
                
                # Calculate loss
                for param in net.deconv.parameters():
                    param.requires_grad = False
                geloss = geloss_op(gaze, img, label, data_x.face)
                geloss.backward(retain_graph=True)
                for param in net.deconv.parameters():
                    param.requires_grad = True

                for param in net.feature.parameters():
                    param.requires_grad = False
                deloss = deloss_op(img, data_x.face)
                deloss.backward()
                for param in net.feature.parameters():
                    param.requires_grad = True
                
                epoch_geloss += geloss.item()
                epoch_deloss += deloss.item()
                writer.add_scalar('geloss/train', geloss.item(), global_step)
                writer.add_scalar('deloss/train', deloss.item(), global_step)         

                ge_optimizer.step()
                ga_optimizer.step()
                de_optimizer.step()
                
                pbar.update(params.batch_size)
                global_step += 1

        for tag, value in net.named_parameters():
            tag = tag.replace('.', '/')           
            writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
            try:
                writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)  # if freezed, grads has no data
            except:
                pass

        train_geloss = epoch_geloss / num_train
        train_deloss = epoch_deloss / num_train
        valid_geloss = 0.0
        valid_deloss = 0.0

        logging.info(f'''Epoch {epoch+1}: {datetime.now().strftime('%H:%M:%S')}
          Training: geloss: {train_geloss}, deloss: {train_deloss}
          Valisation: geloss: {valid_geloss}, deloss: {valid_deloss}''')

        ge_scheduler.step(train_geloss)
        ga_scheduler.step(train_geloss + train_deloss)
        de_scheduler.step(train_deloss)

        if epoch % save.step == 0:
            torch.save(net.state_dict(), os.path.join(train_cp_save, f'Iter_{epoch}.pth'))
                
    writer.close()
    return None


def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Basic Model Training')

    parser.add_argument('-c', '--config', type=str,
                        help='Path to the config file.')
    parser.add_argument("--local_rank", default=-1, type=int)
    return parser.parse_args()

if __name__ == '__main__':

    args = get_args()

    config_path = '/data/hanhc/GazeEstimation/GSAL_hhc/config/train/config_eth_train.yaml'
    # config = edict(yaml.load(open(args.config), Loader=yaml.FullLoader))
    config = edict(yaml.load(open(config_path), Loader=yaml.FullLoader))
    train_net(config, args)
    # train_net(args.config)

    
     
