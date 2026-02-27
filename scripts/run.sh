python tools/train.py configs/gfl_increment_voc/gfl_r50_fpn_1x_coco_fd_stage_0_tal.py

# 40+20X stage 0
python tools/train.py configs/gfl_increment/gfl_r50_fpn_1x_coco_fd_stage_0_tal.py
# 40+20X stage 1
python tools/train.py configs/gfl_increment/gfl_r50_fpn_1x_coco_fd_40_20X_stage_1_adg_tal.py
# 40+20X stage 2
python tools/train.py configs/gfl_increment/gfl_r50_fpn_1x_coco_fd_40_20X_stage_2_adg_tal.py