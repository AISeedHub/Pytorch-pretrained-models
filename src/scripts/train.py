from configs.configs import Config
from models.Model import WrapModel

from configs.get_configs import get_optimizer, get_loss_function, get_lr_scheduler_config, get_trainer
from configs.get_metrics import get_metrics
from configs.get_overlay import get_class_overlay, get_bboxes_overlay, get_masks_overlay
from pytorch_lightning.loggers import TensorBoardLogger

from models.wrappers.Segment import CapsuleWrappingSegment
from models.wrappers.Segment import MODEL as SEGMENT_MODEL
from models.wrappers.Segment import WEIGHTS as SEGMENT_WEIGHTS
from models.wrappers.Detector import CapsuleWrappingDetector
from models.wrappers.Detector import MODEL as DETECTOR_MODEL
from models.wrappers.Detector import WEIGHTS as DETECTOR_WEIGHTS
from models.wrappers.Classifier import CapsuleWrappingClassifier
from models.wrappers.Classifier import MODEL as CLASSIFIER_MODEL
from models.wrappers.Classifier import WEIGHTS as CLASSIFIER_WEIGHTS

from data.custom_dataset import PennFudanDataset, collate_fn_dict, LungCTscan
from data.default_dataset import CIFAR10, Mnist, affNist, SmallNorb
from data.data_module import DataModule


MODELS = {**CLASSIFIER_MODEL, **DETECTOR_MODEL, **SEGMENT_MODEL}
WEIGHTS = {**CLASSIFIER_WEIGHTS, **DETECTOR_WEIGHTS, **SEGMENT_WEIGHTS}
TASKS = {
        'segment': CapsuleWrappingSegment,
        'classifier': CapsuleWrappingClassifier,
        'detector': CapsuleWrappingDetector
}
OVERLAY_FN = {
        'segment': get_masks_overlay,
        'classifier': get_class_overlay,
        'detector': get_bboxes_overlay
}


def load_model(config: Config):

        if config.architect_settings.backbone in DETECTOR_MODEL:
                task = 'detector'
        elif config.architect_settings.backbone in SEGMENT_MODEL:
                task = 'segment'
        elif config.architect_settings.backbone in CLASSIFIER_MODEL:
                task = 'classifier'
        
        model = TASKS[task](model=MODELS[config.architect_settings.backbone],
                                weight=WEIGHTS[config.architect_settings.backbone].DEFAULT,
                                is_freeze=config.architect_settings.is_freeze,
                                is_full=config.architect_settings.is_full,
                                n_cls=config.dataset_settings.n_cls,
                                is_caps=config.architect_settings.is_caps,
                                mode=config.architect_settings.caps.mode,
                                cap_dims=config.architect_settings.caps.cap_dims,
                                routing=config.architect_settings.caps.routing,
                                iteration=config.architect_settings.caps.iteration,
                                lambda_val=config.architect_settings.caps.lambda_val,
                                fuzzy=config.architect_settings.caps.fuzzy)
        
        return task, model



def load_data(config: Config):

        if not config.dataset_settings.path:
                if config.dataset_settings.name == 'CIFAR10':
                        train_dataset = CIFAR10(data_path=r'./datasets', train=True)
                        test_dataset = CIFAR10(data_path=r'./datasets', train=False)
                        collate_fn = None
                elif config.dataset_settings.name == 'Mnist':
                        train_dataset = Mnist(data_path=r'./datasets', train=True)
                        test_dataset = Mnist(data_path=r'./datasets', train=False)
                        collate_fn = None
                elif config.dataset_settings.name == 'affNist':
                        train_dataset = affNist(data_path=r'./datasets/affNist', train=True)
                        test_dataset = affNist(data_path=r'./datasets/affNist', train=False)
                        collate_fn = None
                elif config.dataset_settings.name == 'SmallNorb':
                        train_dataset = SmallNorb(data_path=r'./datasets/SmallNorb', train=True)
                        test_dataset = SmallNorb(data_path=r'./datasets/SmallNorb', train=False)
                        collate_fn = None
                elif config.dataset_settings.name == 'LungCTscan':
                        train_dataset = LungCTscan(data_path=r'./datasets/LungCTscan', train=True)
                        test_dataset = LungCTscan(data_path=r'./datasets/LungCTscan', train=False)
                        collate_fn = None
                elif config.dataset_settings.name == 'PennFudan':
                        train_dataset = PennFudanDataset(data_path=r'./datasets/PennFudanPed', train=True)
                        test_dataset = PennFudanDataset(data_path=r'./datasets/PennFudanPed', train=False)
                        collate_fn = collate_fn_dict
                else:
                        raise ValueError('Dataset not found')

        data_loader = DataModule(train_dataset,
                                test_dataset,
                                batch_size=config.dataset_settings.batch_size,
                                num_workers=config.dataset_settings.num_workers,
                                collate_fn=collate_fn)
        n_cls = data_loader.n_cls
        return n_cls, data_loader


def train(config: Config):
        
        logger = None
        if len(config.training_settings.log_dir) > 0:
                logger = TensorBoardLogger(config.training_settings.log_dir, name="")
                logger.log_hyperparams(config.to_dict())
       
        n_cls, data_loader = load_data(config)
        config.dataset_settings.n_cls = n_cls
        
        task, model = load_model(config)
        
        # create model
        wrap_model = WrapModel(model=model,
                        optimizer_fn=get_optimizer(config.training_settings.optimizer,
                                                config.training_settings.lr),
                        loss_fn=get_loss_function(config.training_settings.loss),
                        train_metrics_fn=get_metrics(config.training_settings.metrics,
                                                data_loader.n_cls,
                                                prefix='train_'),
                        val_metrics_fn=get_metrics(config.training_settings.metrics,
                                                data_loader.n_cls,
                                                prefix='val_'),
                        lr_scheduler_fn=get_lr_scheduler_config(config.training_settings.monitor,
                                                                config.training_settings.lr_scheduler),
                        log_fn=OVERLAY_FN[task])
        
        # create model
        trainer = get_trainer(logger=logger,
                    gpu_ids=config.training_settings.gpu_ids,
                    monitor=config.training_settings.monitor,
                    max_epochs=config.training_settings.max_epochs,
                    ckpt_path=config.training_settings.ckpt_path,
                    early_stopping=config.training_settings.early_stopping)
        
        trainer.fit(wrap_model, data_loader)
