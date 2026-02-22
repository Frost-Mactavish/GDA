# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module()
class VOCDataset(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {
        'classes':
        ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
         'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
                  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157), (106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157)]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'VOC2007' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2007'
        elif 'VOC2012' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2012'
        else:
            self._metainfo['dataset_type'] = None

@DATASETS.register_module()
class VOCDataset11(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {
        'classes':
        ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
         'chair', 'cow', 'diningtable'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157), (106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157)]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'VOC2007' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2007'
        elif 'VOC2012' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2012'
        else:
            self._metainfo['dataset_type'] = None
@DATASETS.register_module()
class VOCDataset12(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {
        'classes':
        ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
         'chair', 'cow', 'diningtable', 'dog'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157), (106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157)]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'VOC2007' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2007'
        elif 'VOC2012' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2012'
        else:
            self._metainfo['dataset_type'] = None

@DATASETS.register_module()
class VOCDataset13(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {
        'classes':
        ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
         'chair', 'cow', 'diningtable', 'dog', 'horse'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157), (106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157)]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'VOC2007' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2007'
        elif 'VOC2012' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2012'
        else:
            self._metainfo['dataset_type'] = None

@DATASETS.register_module()
class VOCDataset14(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {
        'classes':
        ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
         'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157), (106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157)]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'VOC2007' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2007'
        elif 'VOC2012' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2012'
        else:
            self._metainfo['dataset_type'] = None


@DATASETS.register_module()
class VOCDataset16(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {
        'classes':
        ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
         'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
                  'pottedplant'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157), (106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157)]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'VOC2007' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2007'
        elif 'VOC2012' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2012'
        else:
            self._metainfo['dataset_type'] = None


@DATASETS.register_module()
class VOCDataset17(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {
        'classes':
        ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
         'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
                  'pottedplant', 'sheep'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157), (106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157)]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'VOC2007' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2007'
        elif 'VOC2012' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2012'
        else:
            self._metainfo['dataset_type'] = None

@DATASETS.register_module()
class VOCDataset18(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {
        'classes':
        ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
         'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
                  'pottedplant', 'sheep', 'sofa'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157), (106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157)]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'VOC2007' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2007'
        elif 'VOC2012' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2012'
        else:
            self._metainfo['dataset_type'] = None

@DATASETS.register_module()
class VOCDataset10_2(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {
        'classes':
        ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
         'chair', 'cow', 'diningtable', 'dog'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157), (106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157)]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'VOC2007' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2007'
        elif 'VOC2012' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2012'
        else:
            self._metainfo['dataset_type'] = None

@DATASETS.register_module()
class VOCDataset12_2(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {
        'classes':
        ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
         'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157), (106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157)]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'VOC2007' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2007'
        elif 'VOC2012' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2012'
        else:
            self._metainfo['dataset_type'] = None

@DATASETS.register_module()
class VOCDataset14_2(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {
        'classes':
        ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
         'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
                  'pottedplant'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157), (106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157)]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'VOC2007' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2007'
        elif 'VOC2012' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2012'
        else:
            self._metainfo['dataset_type'] = None

@DATASETS.register_module()
class VOCDataset16_2(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {
        'classes':
            ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
             'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
             'pottedplant', 'sheep', 'sofa'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157), (106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157)]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'VOC2007' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2007'
        elif 'VOC2012' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2012'
        else:
            self._metainfo['dataset_type'] = None

@DATASETS.register_module()
class VOCDataset18_2(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {
        'classes':
            ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
             'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
             'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157), (106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157)]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'VOC2007' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2007'
        elif 'VOC2012' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2012'
        else:
            self._metainfo['dataset_type'] = None

@DATASETS.register_module()
class VOCDataset19_1(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {
        'classes':
        ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
         'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
                  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157), (106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157)]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'VOC2007' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2007'
        elif 'VOC2012' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2012'
        else:
            self._metainfo['dataset_type'] = None

@DATASETS.register_module()
class VOCDataset19(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {
        'classes':
            ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
             'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
             'pottedplant', 'sheep', 'sofa', 'train'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157), (106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1)]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'VOC2007' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2007'
        elif 'VOC2012' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2012'
        else:
            self._metainfo['dataset_type'] = None


@DATASETS.register_module()
class VOCDataset15(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {
        'classes':
        ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
         'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157), (106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255)]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'VOC2007' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2007'
        elif 'VOC2012' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2012'
        else:
            self._metainfo['dataset_type'] = None


@DATASETS.register_module()
class VOCDataset5(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {
        'classes':
        ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255)]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'VOC2007' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2007'
        elif 'VOC2012' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2012'
        else:
            self._metainfo['dataset_type'] = None

@DATASETS.register_module()
class VOCDataset10(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {
        'classes':
        ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
         'chair', 'cow'),
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157)]
    }

    def __init__(self, **kwargs):
        super().__init__(
            **kwargs)
        if 'VOC2007' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2007'
        elif 'VOC2012' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2012'
        else:
            self._metainfo['dataset_type'] = None


@DATASETS.register_module()
class VOCDataset10_10(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {
        'classes':
        ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
         'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
                  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157), (106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157)]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'VOC2007' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2007'
        elif 'VOC2012' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2012'
        else:
            self._metainfo['dataset_type'] = None


@DATASETS.register_module()
class VOCDataset10_5(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {
        'classes':
        ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
         'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person'
        ),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157), (106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255)]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'VOC2007' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2007'
        elif 'VOC2012' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2012'
        else:
            self._metainfo['dataset_type'] = None


@DATASETS.register_module()
class VOCDataset10_5_5(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {
        'classes':
        ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
         'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
                  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157), (106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157)]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'VOC2007' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2007'
        elif 'VOC2012' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2012'
        else:
            self._metainfo['dataset_type'] = None


palette_dior = [
    (106, 0, 228),
    (119, 11, 32),
    (165, 42, 42),
    (0, 0, 192),
    (197, 226, 255),
    (0, 60, 100),
    (0, 0, 142),
    (255, 77, 255),
    (153, 69, 1),
    (120, 166, 157),
    (106, 0, 228),
    (119, 11, 32),
    (165, 42, 42),
    (0, 0, 192),
    (197, 226, 255),
    (0, 60, 100),
    (0, 0, 142),
    (255, 77, 255),
    (153, 69, 1),
    (120, 166, 157),
]

category_dior = (
    "airplane",
    "baseballfield",
    "bridge",
    "groundtrackfield",
    "vehicle",
    "ship",
    "tenniscourt",
    "airport",
    "chimney",
    "dam",
    "basketballcourt",
    "Expressway-Service-area",
    "Expressway-toll-station",
    "golffield",
    "harbor",
    "overpass",
    "stadium",
    "storagetank",
    "trainstation",
    "windmill",
)

palette_dota = palette_dior[:15]
category_dota = (
    "plane",
    "baseball-diamond",
    "bridge",
    "ground-track-field",
    "small-vehicle",
    "large-vehicle",
    "ship",
    "tennis-court",
    "basketball-court",
    "storage-tank",
    "soccer-ball-field",
    "roundabout",
    "harbor",
    "swimming-pool",
    "helicopter",
)


@DATASETS.register_module()
class DIORDataset(XMLDataset):
    METAINFO = {
        "classes": category_dior,
        # palette is a list of color tuples, which is used for visualization.
        "palette": palette_dior,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "VOC2007" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2007"
        elif "VOC2012" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2012"
        else:
            self._metainfo["dataset_type"] = None


@DATASETS.register_module()
class DIORDataset5(XMLDataset):
    METAINFO = {
        'classes': category_dior[:5],
        # palette is a list of color tuples, which is used for visualization.
        'palette': palette_dior[:5]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'VOC2007' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2007'
        elif 'VOC2012' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2012'
        else:
            self._metainfo['dataset_type'] = None


@DATASETS.register_module()
class DIORDataset10(XMLDataset):
    METAINFO = {
        "classes": category_dior[:10],
        # palette is a list of color tuples, which is used for visualization.
        "palette": palette_dior[:10],
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "VOC2007" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2007"
        elif "VOC2012" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2012"
        else:
            self._metainfo["dataset_type"] = None


@DATASETS.register_module()
class DIORDataset12(XMLDataset):
    METAINFO = {
        "classes": category_dior[:12],
        # palette is a list of color tuples, which is used for visualization.
        "palette": palette_dior[:12],
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "VOC2007" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2007"
        elif "VOC2012" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2012"
        else:
            self._metainfo["dataset_type"] = None


@DATASETS.register_module()
class DIORDataset14(XMLDataset):
    METAINFO = {
        "classes": category_dior[:14],
        # palette is a list of color tuples, which is used for visualization.
        "palette": palette_dior[:14],
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "VOC2007" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2007"
        elif "VOC2012" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2012"
        else:
            self._metainfo["dataset_type"] = None


@DATASETS.register_module()
class DIORDataset15(XMLDataset):
    METAINFO = {
        "classes": category_dior[:15],
        # palette is a list of color tuples, which is used for visualization.
        "palette": palette_dior[:15],
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "VOC2007" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2007"
        elif "VOC2012" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2012"
        else:
            self._metainfo["dataset_type"] = None


@DATASETS.register_module()
class DIORDataset16(XMLDataset):
    METAINFO = {
        "classes": category_dior[:16],
        # palette is a list of color tuples, which is used for visualization.
        "palette": palette_dior[:16],
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "VOC2007" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2007"
        elif "VOC2012" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2012"
        else:
            self._metainfo["dataset_type"] = None


@DATASETS.register_module()
class DIORDataset17(XMLDataset):
    METAINFO = {
        "classes": category_dior[:17],
        # palette is a list of color tuples, which is used for visualization.
        "palette": palette_dior[:17],
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "VOC2007" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2007"
        elif "VOC2012" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2012"
        else:
            self._metainfo["dataset_type"] = None


@DATASETS.register_module()
class DIORDataset18(XMLDataset):
    METAINFO = {
        "classes": category_dior[:18],
        # palette is a list of color tuples, which is used for visualization.
        "palette": palette_dior[:18],
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "VOC2007" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2007"
        elif "VOC2012" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2012"
        else:
            self._metainfo["dataset_type"] = None


@DATASETS.register_module()
class DIORDataset19(XMLDataset):
    METAINFO = {
        "classes": category_dior[:19],
        # palette is a list of color tuples, which is used for visualization.
        "palette": palette_dior[:19],
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "VOC2007" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2007"
        elif "VOC2012" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2012"
        else:
            self._metainfo["dataset_type"] = None


@DATASETS.register_module()
class DOTADataset(XMLDataset):
    METAINFO = {
        "classes": category_dota,
        # palette is a list of color tuples, which is used for visualization.
        "palette": palette_dota,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "VOC2007" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2007"
        elif "VOC2012" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2012"
        else:
            self._metainfo["dataset_type"] = None


@DATASETS.register_module()
class DOTADataset5(XMLDataset):
    METAINFO = {
        "classes": category_dota[:5],
        # palette is a list of color tuples, which is used for visualization.
        "palette": palette_dota[:5],
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "VOC2007" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2007"
        elif "VOC2012" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2012"
        else:
            self._metainfo["dataset_type"] = None


@DATASETS.register_module()
class DOTADataset8(XMLDataset):
    METAINFO = {
        "classes": category_dota[:8],
        # palette is a list of color tuples, which is used for visualization.
        "palette": palette_dota[:8],
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "VOC2007" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2007"
        elif "VOC2012" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2012"
        else:
            self._metainfo["dataset_type"] = None


@DATASETS.register_module()
class DOTADataset10(XMLDataset):
    METAINFO = {
        "classes": category_dota[:10],
        # palette is a list of color tuples, which is used for visualization.
        "palette": palette_dota[:10],
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "VOC2007" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2007"
        elif "VOC2012" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2012"
        else:
            self._metainfo["dataset_type"] = None


@DATASETS.register_module()
class DOTADataset11(XMLDataset):
    METAINFO = {
        "classes": category_dota[:11],
        # palette is a list of color tuples, which is used for visualization.
        "palette": palette_dota[:11],
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "VOC2007" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2007"
        elif "VOC2012" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2012"
        else:
            self._metainfo["dataset_type"] = None


@DATASETS.register_module()
class DOTADataset12(XMLDataset):
    METAINFO = {
        "classes": category_dota[:12],
        # palette is a list of color tuples, which is used for visualization.
        "palette": palette_dota[:12],
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "VOC2007" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2007"
        elif "VOC2012" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2012"
        else:
            self._metainfo["dataset_type"] = None


@DATASETS.register_module()
class DOTADataset13(XMLDataset):
    METAINFO = {
        "classes": category_dota[:13],
        # palette is a list of color tuples, which is used for visualization.
        "palette": palette_dota[:13],
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "VOC2007" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2007"
        elif "VOC2012" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2012"
        else:
            self._metainfo["dataset_type"] = None


@DATASETS.register_module()
class DOTADataset14(XMLDataset):
    METAINFO = {
        "classes": category_dota[:14],
        # palette is a list of color tuples, which is used for visualization.
        "palette": palette_dota[:14],
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "VOC2007" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2007"
        elif "VOC2012" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2012"
        else:
            self._metainfo["dataset_type"] = None