import torch.nn.functional as F
from torchvision.utils import make_grid, draw_bounding_boxes, draw_segmentation_masks
import cv2
import numpy as np
from pytorch_lightning.loggers import Logger
import torch
from typing import List


def get_masks_overlay(images: torch.Tensor,
                       predictions: torch.Tensor,
                       targets: torch.Tensor=None,
                       class_names: List[str]=None,
                       phase: str='train',
                       num_of_images: int=16,
                       threshold: float=0.9,
                       logger: Logger=None) -> torch.Tensor:
    '''
    Get bounding boxes overlay for images
    Args:
        images: images
        targets: ground truth bounding boxes
        predictions: predicted bounding boxes
        class_names: list of class names
        threshold: threshold for IoU
    Returns:
        images: images with bounding boxes overlay
    '''
    images = (images.clone().cpu() * 255).to(torch.uint8)
    n = min(num_of_images, images.shape[0])
    images = images[:n]
    predictions = predictions.clone().detach().cpu()[:n]
    predictions = torch.softmax(predictions, dim=1)

    boolean_masks = [out > threshold for out in predictions]
    reconstructions = [draw_segmentation_masks(image, mask, alpha=0.5)
                            for image, mask in zip(images, boolean_masks)]
    
    reconstructions = torch.stack([F.interpolate(img.unsqueeze(0), size=(224, 224))
                                    for img in reconstructions]).squeeze(1) 
    reconstructions = make_grid(reconstructions, nrow= int(n ** 0.5))
    # reconstructions = reconstructions.numpy().transpose(1, 2, 0) / 255

    if logger:
        logger.experiment.add_image(phase, reconstructions, 0)

    return reconstructions

def get_bboxes_overlay(images: torch.Tensor,
                      predictions: torch.Tensor,
                      targets: torch.Tensor=None,
                      class_names: List[str]=None,
                      logger: Logger=None,
                      phase: str='train',
                      num_of_images: int=4,
                      threshold: float=0.8) -> torch.Tensor:
    '''
    Get masks overlay for images
    Args:
        images: images
        targets: ground truth masks
        predictions: predicted masks
        class_names: list of class names
        threshold: threshold for IoU
    Returns:
        images: images with masks overlay
    '''
    images = [(image.clone().cpu() * 255).to(torch.uint8) for image in images]

    n = min(num_of_images, len(images))
    predictions = [{k: v.cpu() for k, v in t.items()} for t in predictions[:n]]
    images = images[:n]

    boxes = [out['boxes'][out['scores'] > threshold] for out in predictions]
    reconstructions = [draw_bounding_boxes(image, box, width=4, colors='red')
                                    for image, box in zip(images, boxes)]
    
    reconstructions = torch.stack([F.interpolate(img.unsqueeze(0), size=(224, 224))
                                    for img in reconstructions]).squeeze(1) 
    reconstructions = make_grid(reconstructions, nrow= int(n ** 0.5))
    # reconstructions = reconstructions.numpy().transpose(1, 2, 0)

    if logger:
        logger.experiment.add_image(phase, reconstructions, 0)

    return reconstructions
 

def get_class_overlay(images: torch.Tensor,
                    predictions: torch.Tensor,
                    targets: torch.Tensor=None,
                    logger: Logger=None,
                    phase: str='train',
                    class_names: List[str]=None,
                    num_of_images: int=16,
                    **kwargs) -> torch.Tensor:
    '''
    Get class overlay for images
    Args:
        images: images
        targets: ground truth class
        predictions: predicted class
        class_names: list of class names
    Returns:
        images: images with class overlay
    '''
    
    images = ((images.clone().cpu() * 255).to(torch.uint8).permute(0, 2, 3, 1)).numpy()
    predictions = predictions.clone().detach().cpu()
    n = min(num_of_images, images.shape[0])

    predictions = torch.argmax(predictions, dim=-1)
    images = [cv2.resize(image, (224, 224)) for image in images]
    draw_images = np.array([cv2.putText(image, "class: " + str(label.item()) if class_names is None else class_names[int(label.item())], 
                                        (112, 112), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                                    for image, label in zip(images[:n], predictions[:n])])

    reconstructions = torch.from_numpy(draw_images.transpose(0, 3, 1, 2))
    reconstructions = make_grid(reconstructions, nrow= int(n ** 0.5))
    # reconstructions = reconstructions.numpy().transpose(1, 2, 0)

    if logger:
        logger.experiment.add_image(phase, reconstructions, 0)

    return reconstructions