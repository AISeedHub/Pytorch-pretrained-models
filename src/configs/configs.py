from dataclasses import dataclass
from typing import List
import yaml

@dataclass
class CapsSettings:
    mode: int
    cap_dims: int
    routing: str
    iteration: int
    lambda_val: float
    fuzzy: float

@dataclass
class DatasetSettings:
    name: str
    path: str
    num_workers: int
    batch_size: int
    n_cls: int=-1

@dataclass
class TrainingSettings:
    gpu_ids: List[int]
    loss: str
    metrics: List[str]
    ckpt_path: str
    log_dir: str
    max_epochs: int
    optimizer: str
    lr_scheduler: str
    lr: float
    early_stopping: bool
    monitor: str

@dataclass
class ArchitectSettings:
    backbone: str
    is_full: bool
    is_freeze: bool
    is_caps: bool
    caps: CapsSettings

@dataclass
class Config:
    architect_settings: ArchitectSettings
    dataset_settings: DatasetSettings
    training_settings: TrainingSettings

    def __init__(self,
                 architect_settings: ArchitectSettings, 
                 dataset_settings: DatasetSettings, 
                 training_settings: TrainingSettings):
        
        self.architect_settings = architect_settings
        self.dataset_settings = dataset_settings
        self.training_settings = training_settings

    def parse_from_yaml(self, yml_path: str):
        
        with open(yml_path, 'r') as file:
            data = yaml.safe_load(file)

            self.architect_settings = ArchitectSettings(
                data['architect_settings']['backbone'],
                data['architect_settings']['is_full'],
                data['architect_settings']['is_freeze'],
                data['architect_settings']['is_caps'],
                CapsSettings(
                    data['architect_settings']['caps']['mode'],
                    data['architect_settings']['caps']['cap_dims'],
                    data['architect_settings']['caps']['routing'],
                    data['architect_settings']['caps']['iteration'],
                    data['architect_settings']['caps']['lambda'],
                    data['architect_settings']['caps']['fuzzy']
                )
            )
            self.dataset_settings = DatasetSettings(
                data['dataset_settings']['name'],
                data['dataset_settings']['path'],
                data['dataset_settings']['n_cls'],
                data['dataset_settings']['num_workers'],
                data['dataset_settings']['batch_size']
            )
            self.training_settings = TrainingSettings(
                data['training_settings']['gpu_ids'],
                data['training_settings']['loss'],
                data['training_settings']['metrics'],
                data['training_settings']['ckpt_path'],
                data['training_settings']['log_dir'],
                data['training_settings']['max_epochs'],
                data['training_settings']['optimizer'],
                data['training_settings']['lr_scheduler'],
                data['training_settings']['lr'],
                data['training_settings']['early_stopping'],
                data['training_settings']['monitor']
            )

    def to_dict(self):
        return {
            'architect_settings': {
                'backbone': self.architect_settings.backbone,
                'is_full': self.architect_settings.is_full,
                'is_freeze': self.architect_settings.is_freeze,
                'is_caps': self.architect_settings.is_caps,
                'caps': {
                    'mode': self.architect_settings.caps.mode,
                    'cap_dims': self.architect_settings.caps.cap_dims,
                    'routing': self.architect_settings.caps.routing,
                    'iteration': self.architect_settings.caps.iteration,
                    'lambda': self.architect_settings.caps.lambda_val,
                    'fuzzy': self.architect_settings.caps.fuzzy
                }
            },
            'dataset_settings': {
                'name': self.dataset_settings.name,
                'path': self.dataset_settings.path,
                'num_workers': self.dataset_settings.num_workers,
                'batch_size': self.dataset_settings.batch_size
            },
            'training_settings': {
                'gpu_ids': self.training_settings.gpu_ids,
                'loss': self.training_settings.loss,
                'metrics': self.training_settings.metrics,
                'ckpt_path': self.training_settings.ckpt_path,
                'log_dir': self.training_settings.log_dir,
                'max_epochs': self.training_settings.max_epochs,
                'optimizer': self.training_settings.optimizer,
                'lr_scheduler': self.training_settings.lr_scheduler,
                'lr': self.training_settings.lr,
                'early_stopping': self.training_settings.early_stopping,
                'monitor': self.training_settings.monitor
            }
        }
        


        