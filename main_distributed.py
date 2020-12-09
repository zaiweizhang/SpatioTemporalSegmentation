# Change dataloader multiprocess start method to anything not fork
import open3d as o3d

import torch.multiprocessing as mp
try:
  mp.set_start_method('forkserver')  # Reuse process created
except RuntimeError:
  pass

import os
import sys
import json
import logging
from easydict import EasyDict as edict

# Torch packages
import torch

# Train deps
from config import get_config

from lib.test import test
from lib.train import train
from lib.utils import load_state_with_same_shape, get_torch_device, count_parameters
from lib.dataset import initialize_data_loader
from lib.datasets import load_dataset

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from checkpoint import init_model_from_weights
from models import load_model, load_wrapper

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format=os.uname()[1].split('.')[0] + ' %(asctime)s %(message)s',
    datefmt='%m/%d %H:%M:%S',
    handlers=[ch])


def main():
  config = get_config()
  if config.resume:
    json_config = json.load(open(config.resume + '/config.json', 'r'))
    json_config['resume'] = config.resume
    config = edict(json_config)

  ### Distributed
  if config.dist_url == "env://" and config.world_size == -1:
    config.world_size = int(os.environ["WORLD_SIZE"])

  ngpus_per_node = torch.cuda.device_count()

  config.distributed = config.world_size > 1 or config.multiprocessing_distributed
  if config.multiprocessing_distributed:
    # Since we have ngpus_per_node processes per node, the total world_size
    # needs to be adjusted accordingly
    config.world_size = ngpus_per_node * config.world_size
    # Use torch.multiprocessing.spawn to launch distributed processes: the
    # main_worker process function
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
  else:
    # Simply call main_worker function
    main_worker(config.gpu, ngpus_per_node, config)

def main_worker(gpu, ngpus_per_node, config):
  config.gpu = gpu
  
  #if config.is_cuda and not torch.cuda.is_available():
  #  raise Exception("No GPU found")
  if config.gpu is not None:
    print("Use GPU: {} for training".format(config.gpu))
  device = get_torch_device(config.is_cuda)

  if config.distributed:
    if config.dist_url == "env://" and config.rank == -1:
      config.rank = int(os.environ["RANK"])
    if config.multiprocessing_distributed:
      # For multiprocessing distributed training, rank needs to be the
      # global rank among all the processes
      config.rank = config.rank * ngpus_per_node + gpu
    dist.init_process_group(backend=config.dist_backend, init_method=config.dist_url,
                            world_size=config.world_size, rank=config.rank)
  
  logging.info('===> Configurations')
  dconfig = vars(config)
  for k in dconfig:
    logging.info('    {}: {}'.format(k, dconfig[k]))

  DatasetClass = load_dataset(config.dataset)
  if config.test_original_pointcloud:
    if not DatasetClass.IS_FULL_POINTCLOUD_EVAL:
      raise ValueError('This dataset does not support full pointcloud evaluation.')

  if config.evaluate_original_pointcloud:
    if not config.return_transformation:
      raise ValueError('Pointcloud evaluation requires config.return_transformation=true.')

  if (config.return_transformation ^ config.evaluate_original_pointcloud):
    raise ValueError('Rotation evaluation requires config.evaluate_original_pointcloud=true and '
                     'config.return_transformation=true.')
  
  logging.info('===> Initializing dataloader')
  if config.is_train:
    train_data_loader,train_sampler = initialize_data_loader(
        DatasetClass,
        config,
        phase=config.train_phase,
        num_workers=config.num_workers,
        augment_data=True,
        shuffle=True,
        repeat=True,
        batch_size=config.batch_size,
        limit_numpoints=config.train_limit_numpoints)

    val_data_loader,val_sampler = initialize_data_loader(
        DatasetClass,
        config,
        num_workers=config.num_val_workers,
        phase=config.val_phase,
        augment_data=False,
        shuffle=True,
        repeat=False,
        batch_size=config.val_batch_size,
        limit_numpoints=False)
    if train_data_loader.dataset.NUM_IN_CHANNEL is not None:
      num_in_channel = train_data_loader.dataset.NUM_IN_CHANNEL
    else:
      num_in_channel = 3  # RGB color

    num_labels = train_data_loader.dataset.NUM_LABELS
  else:
    test_data_loader,val_sampler = initialize_data_loader(
        DatasetClass,
        config,
        num_workers=config.num_workers,
        phase=config.test_phase,
        augment_data=False,
        shuffle=False,
        repeat=False,
        batch_size=config.test_batch_size,
        limit_numpoints=False)
    if test_data_loader.dataset.NUM_IN_CHANNEL is not None:
      num_in_channel = test_data_loader.dataset.NUM_IN_CHANNEL
    else:
      num_in_channel = 3  # RGB color

    num_labels = test_data_loader.dataset.NUM_LABELS
    
  logging.info('===> Building model')
  NetClass = load_model(config.model)
  if config.wrapper_type == 'None':
    model = NetClass(num_in_channel, num_labels, config)
    logging.info('===> Number of trainable parameters: {}: {}'.format(NetClass.__name__,
                                                                      count_parameters(model)))
  else:
    wrapper = load_wrapper(config.wrapper_type)
    model = wrapper(NetClass, num_in_channel, num_labels, config)
    logging.info('===> Number of trainable parameters: {}: {}'.format(
        wrapper.__name__ + NetClass.__name__, count_parameters(model)))

  logging.info(model)

  if config.weights == 'modelzoo':  # Load modelzoo weights if possible.
    logging.info('===> Loading modelzoo weights')
    model.preload_modelzoo()
  # Load weights if specified by the parameter.
  elif config.weights.lower() != 'none':
    logging.info('===> Loading weights: ' + config.weights)
    state = torch.load(config.weights)
    if config.weights_for_inner_model:
      model.model.load_state_dict(state['state_dict'])
    else:
      if config.lenient_weight_loading:
        matched_weights = load_state_with_same_shape(model, state['state_dict'])
        model_dict = model.state_dict()
        model_dict.update(matched_weights)
        model.load_state_dict(model_dict)
      else:
        init_model_from_weights(model, state, freeze_bb=False)

  if config.distributed:
    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    if config.gpu is not None:
      torch.cuda.set_device(config.gpu)
      model.cuda(config.gpu)
      # When using a single GPU per process and per
      # DistributedDataParallel, we need to divide the batch size
      # ourselves based on the total number of GPUs we have
      config.batch_size = int(config.batch_size / ngpus_per_node)
      config.num_workers = int((config.num_workers + ngpus_per_node - 1) / ngpus_per_node)
      model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu])
    else:
      model.cuda()
      # DistributedDataParallel will divide and allocate batch_size to all
      # available GPUs if device_ids are not set
      model = torch.nn.parallel.DistributedDataParallel(model)
      
  #model = model.to(device)
  if config.is_train:
    train(model, train_data_loader, val_data_loader, config, train_sampler=train_sampler, ngpus_per_node=ngpus_per_node)
  else:
    test(model, test_data_loader, config)


if __name__ == '__main__':
  __spec__ = None
  main()
