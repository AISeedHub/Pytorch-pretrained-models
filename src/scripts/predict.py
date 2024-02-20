import torch
from PIL import Image
from torchvision.transforms import ToTensor
from models.wrappers.Segment import CapsuleWrappingSegment
from models.wrappers.Segment import MODEL as SEGMENT_MODEL
from models.wrappers.Segment import WEIGHTS as SEGMENT_WEIGHTS
from models.wrappers.Detector import CapsuleWrappingDetector
from models.wrappers.Detector import MODEL as DETECTOR_MODEL
from models.wrappers.Detector import WEIGHTS as DETECTOR_WEIGHTS
from models.wrappers.Classifier import CapsuleWrappingClassifier
from models.wrappers.Classifier import MODEL as CLASSIFIER_MODEL
from models.wrappers.Classifier import WEIGHTS as CLASSIFIER_WEIGHTS
from models.Model import WrapModel
from typing import Union
from configs.get_overlay import get_bboxes_overlay, get_masks_overlay, get_class_overlay


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

def load_model(model: str, ckpt: str=None) -> tuple[str, Union[CapsuleWrappingSegment, 
                                               CapsuleWrappingClassifier, 
                                               CapsuleWrappingDetector]]:
        
        if model in DETECTOR_MODEL:
                task = 'detector'
        elif model in SEGMENT_MODEL:
                task = 'segment'
        elif model in CLASSIFIER_MODEL:
                task = 'classifier'

        if ckpt:
                backbone = WrapModel.load_from_checkpoint(ckpt.name)
                backbone = backbone.model.cpu()
        else:
                backbone = TASKS[task](model=MODELS[model],
                                        weight=WEIGHTS[model].DEFAULT,
                                        is_freeze=True,
                                        is_full=True)
        backbone.eval()
        
        return task, backbone

def predict(img, model, ckpt: str=None):
    
        task, backbone = load_model(model, ckpt)
        original_img = ToTensor()(img)
        original_img = original_img.unsqueeze(0)
        with torch.no_grad():
                img_tensor = backbone.preprocess(img)
                img_tensor = img_tensor.unsqueeze(0)
                output = backbone(img_tensor)
        overlay = OVERLAY_FN[task](images=original_img, predictions=output)
        overlay = Image.fromarray(overlay.numpy().transpose(1, 2, 0))
        
        return overlay