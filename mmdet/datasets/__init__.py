# Copyright (c) OpenMMLab. All rights reserved.
from .base_det_dataset import BaseDetDataset
from .cityscapes import CityscapesDataset
from .coco import CocoDataset, CocoDataset70ShuffleV1Base, CocoDataset70ShuffleV1Task2
from .coco_increase import CocoDataset40Order, CocoDataset40_20Order, CocoDataset40_20_20Order, CocoDataset70Order, \
    CocoDataset70_10Order
from .coco_panoptic import CocoPanopticDataset
from .crowdhuman import CrowdHumanDataset
from .dataset_wrappers import MultiImageMixDataset
from .deepfashion import DeepFashionDataset
from .lvis import LVISDataset, LVISV1Dataset, LVISV05Dataset
from .objects365 import Objects365V1Dataset, Objects365V2Dataset
from .openimages import OpenImagesChallengeDataset, OpenImagesDataset
from .samplers import (AspectRatioBatchSampler, ClassAwareSampler,
                       GroupMultiSourceSampler, MultiSourceSampler)
from .utils import get_loading_pipeline
from .voc import VOCDataset, VOCDataset5, VOCDataset10, VOCDataset12, VOCDataset14, VOCDataset15, VOCDataset16, \
    VOCDataset17, VOCDataset18, VOCDataset19, class_dior, class_voc, class_dota
from .wider_face import WIDERFaceDataset
from .xml_style import XMLDataset

__all__ = [
    'XMLDataset', 'CocoDataset', 'DeepFashionDataset', 'VOCDataset',
    'CityscapesDataset', 'LVISDataset', 'LVISV05Dataset', 'LVISV1Dataset',
    'WIDERFaceDataset', 'get_loading_pipeline', 'CocoPanopticDataset',
    'MultiImageMixDataset', 'OpenImagesDataset', 'OpenImagesChallengeDataset',
    'AspectRatioBatchSampler', 'ClassAwareSampler', 'MultiSourceSampler',
    'GroupMultiSourceSampler', 'BaseDetDataset', 'CrowdHumanDataset',
    'Objects365V1Dataset', 'Objects365V2Dataset',
    'CocoDataset70ShuffleV1Base', 'CocoDataset70ShuffleV1Task2',
    'CocoDataset40Order', 'CocoDataset40_20Order', 'CocoDataset40_20_20Order',
    'VOCDataset10', 'VOCDataset5', 'VOCDataset15', 'CocoDataset70Order', 'CocoDataset70_10Order',
    'VOCDataset',
    'VOCDataset12',
    'VOCDataset14'
    'VOCDataset16',
    'VOCDataset17',
    'VOCDataset18',
    'VOCDataset19',
    'class_dior',
    'class_voc',
    'class_dota'
]
