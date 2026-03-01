# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS
from .xml_style import XMLDataset


palette = [
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

class_voc = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

class_dior = (
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

class_dota = (
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
class VOCDataset(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {"classes": class_voc, "palette": palette}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "VOC2007" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2007"
        elif "VOC2012" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2012"
        else:
            self._metainfo["dataset_type"] = None


@DATASETS.register_module()
class VOCDataset5(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {"classes": class_voc[:5], "palette": palette[:5]}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "VOC2007" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2007"
        elif "VOC2012" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2012"
        else:
            self._metainfo["dataset_type"] = None


@DATASETS.register_module()
class VOCDataset10(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {"classes": class_voc[:10], "palette": palette[:10]}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "VOC2007" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2007"
        elif "VOC2012" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2012"
        else:
            self._metainfo["dataset_type"] = None


@DATASETS.register_module()
class VOCDataset12(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {"classes": class_voc[:12], "palette": palette[:12]}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "VOC2007" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2007"
        elif "VOC2012" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2012"
        else:
            self._metainfo["dataset_type"] = None


@DATASETS.register_module()
class VOCDataset14(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {"classes": class_voc[:14], "palette": palette[:14]}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "VOC2007" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2007"
        elif "VOC2012" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2012"
        else:
            self._metainfo["dataset_type"] = None


@DATASETS.register_module()
class VOCDataset15(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {"classes": class_voc[:15], "palette": palette[:15]}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "VOC2007" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2007"
        elif "VOC2012" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2012"
        else:
            self._metainfo["dataset_type"] = None


@DATASETS.register_module()
class VOCDataset16(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {"classes": class_voc[:16], "palette": palette[:16]}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "VOC2007" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2007"
        elif "VOC2012" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2012"
        else:
            self._metainfo["dataset_type"] = None


@DATASETS.register_module()
class VOCDataset17(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {"classes": class_voc[:17], "palette": palette[:17]}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "VOC2007" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2007"
        elif "VOC2012" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2012"
        else:
            self._metainfo["dataset_type"] = None


@DATASETS.register_module()
class VOCDataset18(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {"classes": class_voc[:18], "palette": palette[:18]}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "VOC2007" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2007"
        elif "VOC2012" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2012"
        else:
            self._metainfo["dataset_type"] = None


@DATASETS.register_module()
class VOCDataset19(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {"classes": class_voc[:19], "palette": palette[:19]}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "VOC2007" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2007"
        elif "VOC2012" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2012"
        else:
            self._metainfo["dataset_type"] = None


@DATASETS.register_module()
class DIORDataset(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {"classes": class_dior, "palette": palette}

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
    METAINFO = {"classes": class_dior[:5], "palette": palette[:5]}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "VOC2007" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2007"
        elif "VOC2012" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2012"
        else:
            self._metainfo["dataset_type"] = None


@DATASETS.register_module()
class DIORDataset10(XMLDataset):
    METAINFO = {"classes": class_dior[:10], "palette": palette[:10]}

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
    METAINFO = {"classes": class_dior[:12], "palette": palette[:12]}

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
    METAINFO = {"classes": class_dior[:14], "palette": palette[:14]}

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
    METAINFO = {"classes": class_dior[:15], "palette": palette[:15]}

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
    METAINFO = {"classes": class_dior[:16], "palette": palette[:16]}

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
    METAINFO = {"classes": class_dior[:17], "palette": palette[:17]}

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
    METAINFO = {"classes": class_dior[:18], "palette": palette[:18]}

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
    METAINFO = {"classes": class_dior[:19], "palette": palette[:19]}

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
    METAINFO = {"classes": class_dota, "palette": palette[:15]}

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
    METAINFO = {"classes": class_dota[:5], "palette": palette[:5]}

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
    METAINFO = {"classes": class_dota[:8], "palette": palette[:8]}

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
    METAINFO = {"classes": class_dota[:10], "palette": palette[:10]}

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
    METAINFO = {"classes": class_dota[:11], "palette": palette[:11]}

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
    METAINFO = {"classes": class_dota[:12], "palette": palette[:12]}

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
    METAINFO = {"classes": class_dota[:13], "palette": palette[:13]}

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
    METAINFO = {"classes": class_dota[:14], "palette": palette[:14]}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "VOC2007" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2007"
        elif "VOC2012" in self.sub_data_root:
            self._metainfo["dataset_type"] = "VOC2012"
        else:
            self._metainfo["dataset_type"] = None
