_base_ = ["../_base_/dota.py"]

dataset_type = "DOTADataset8"
data_root = "/data/my_code/dataset/DOTA_xml"
work_dir = "log/DOTA/8+7/STEP0"

backend_args = None

model = dict(roi_head=dict(bbox_head=dict(num_classes=8)))

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", scale=(1333, 1024), keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs"),
]

train_dataloader = dict(
    dataset=dict(
        dataset=dict(
            datasets=[
                dict(
                    type=dataset_type,
                    data_root=data_root,
                    ann_file=data_root + "/ImageSets/Main/trainval.txt",
                    data_prefix=dict(sub_data_root=data_root + "/Annotations"),
                    filter_cfg=dict(filter_empty_gt=True, min_size=5, bbox_min_size=5),
                    pipeline=train_pipeline,
                    backend_args=backend_args,
                )
            ]
        )
    )
)
val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
    ),
)
test_dataloader = val_dataloader
