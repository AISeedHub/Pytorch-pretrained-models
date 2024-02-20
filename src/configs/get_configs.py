import torch
from torch import nn
import torch.optim as optim
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from typing import Union, Tuple, List, Dict


def get_lr_scheduler_config(monitor: str,
                            lr_scheduler: str='step',
                            **kwargs) -> Dict[str, Union[optim.lr_scheduler._LRScheduler, str, str, int]]:
    '''
    Set up learning rate scheduler configuration.
    Args:
        optimizer: optimizer
        lr_scheduler: type of learning rate scheduler
        lr_step: step size for scheduler
        lr_decay: decay factor for scheduler
        metric: metric for scheduler monitoring
    Returns:
        lr_scheduler_config: learning rate scheduler configuration
    '''
    scheduler_mapping = {
        'step': lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, **kwargs),
        'multistep': lambda optimizer: torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=10, **kwargs),
        'plateau': lambda optimizer: torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    }

    scheduler_creator = scheduler_mapping.get(lr_scheduler)

    if scheduler_creator is not None:
        scheduler = scheduler_creator
    else:
        raise NotImplementedError

    return {
        'scheduler': scheduler,
        'monitor': monitor,
        'interval': 'epoch',
        'frequency': 1,
    }

def get_optimizer(  optimizer: str='adam',
                    lr: int=0.0001,
                    **kwargs) -> optim.Optimizer:
    """
    Set up learning optimizer
    Args:
        parameters: model's parameters
        settings: settings hyperparameters
    Returns:
        optimizer: optimizer
    """
    optimizer_mapping = {
        'adam': lambda params: optim.Adam(params, lr=lr, **kwargs),
        'sgd': lambda params: optim.SGD(params, lr=lr, **kwargs),
    }

    optimizer_creator = optimizer_mapping.get(optimizer)

    if optimizer_creator is not None:
        return optimizer_creator
    else:
        raise NotImplementedError()

class NoLoss(nn.Module):
    '''
    Customized loss function for pytorch-lightning
    '''
    def __init__(self):
        super().__init__()

    def forward(self, y_hat, y):
        '''
        Compute loss
        Args:
            y_hat: predicted output
            y: ground truth
        Returns:
            loss: loss value
        '''
        
        return -1

def get_loss_function(loss_type: str) -> nn.Module:
    """
    Set up loss function
    Args:
        loss_type: loss function type
    Returns:
        loss: loss function
    """
    loss_mapping = {
        'ce': nn.CrossEntropyLoss(),
        'bce': nn.BCELoss(),
        'mse': nn.MSELoss(),
        'none': NoLoss(),  # Only for task == detection
    }

    loss_function = loss_mapping.get(loss_type)

    if loss_function is not None:
        return loss_function
    else:
        raise NotImplementedError()

def get_gpu_settings(gpu_ids: list[int]) -> Tuple[str, int, str]:
    '''
    Get GPU settings for PyTorch Lightning Trainer:
    Args:
        gpu_ids (list[int])
        n_gpu (int)
    Returns:
        tuple[str, int, str]: accelerator, devices, strategy
    '''
    if not torch.cuda.is_available():
        return "cpu", None, None

    n_gpu = len(gpu_ids)
    mapping = {
        'devices': gpu_ids if gpu_ids is not None else n_gpu if n_gpu is not None else 1,
        'strategy': 'ddp' if (gpu_ids or n_gpu) and (len(gpu_ids) > 1 or n_gpu > 1) else 'auto'
    }

    return "gpu", mapping['devices'], mapping['strategy']


def get_basic_callbacks(monitor: str,
                        ckpt_path: str, 
                        early_stopping: bool = False,
                        **kwargs) -> List[Union[LearningRateMonitor, ModelCheckpoint, EarlyStopping]]:
    '''
    Get basic callbacks for PyTorch Lightning Trainer.
    Args:
        metric_name: name of the metric
        ckpt_path: path to save the checkpoints
        early_stopping: flag for early stopping callback
    Returns:
        callbacks: list of callbacks
    '''
    if "loss" in monitor:
        mode = "min"
    else:
        mode = "max"
 
    callbacks_mapping = {
        'last': ModelCheckpoint(dirpath=ckpt_path, filename='{epoch:03d}', monitor=None, **kwargs),
        'best': ModelCheckpoint(dirpath=ckpt_path, filename='{epoch:03d}', monitor=monitor, mode=mode, **kwargs),
        'lr': LearningRateMonitor(logging_interval='epoch', **kwargs),
        'early_stopping': EarlyStopping(monitor=monitor, mode=mode, patience=5, **kwargs),
    }

    callbacks = [callbacks_mapping[key] for key in ['last', 'best', 'lr']]
    
    if early_stopping:
        callbacks.append(callbacks_mapping['early_stopping'])

    return callbacks
    
def get_trainer(gpu_ids: list[int]=None,
                monitor: str='val_loss',
                ckpt_path: str='checkpoints',
                max_epochs: int=10,
                early_stopping: bool=False,
                logger=None,
                **kwargs) -> Trainer:
    '''
    Get trainer and logging for pytorch-lightning trainer:
    Args: 
        settings: hyperparameter settings
        task: task to run training
    Returns:
        trainer: trainer object
        logger: neptune logger object
    '''

    callbacks = get_basic_callbacks(monitor, ckpt_path, early_stopping, **kwargs)
    accelerator, devices, strategy = get_gpu_settings(gpu_ids)

    return Trainer(
        logger=logger,
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        callbacks=callbacks,
    )