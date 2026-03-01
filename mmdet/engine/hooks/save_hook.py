import os
import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class SaveBestResultHook(Hook):
    def __init__(self, key_indicator="pascal_voc/mAP", save_file="best_result.txt"):
        self.key_indicator = key_indicator
        self.save_file = save_file
        self.best_score = None

    def after_val_epoch(self, runner, metrics):
        if self.key_indicator not in metrics:
            return
        current_score = metrics[self.key_indicator]
        if self.best_score is None or current_score > self.best_score:
            self.best_score = current_score

            old_map = None
            new_map = None
            task = str(runner.cfg.get("task", ""))
            if "+" in task:
                prefix = self.key_indicator.rsplit("/", 1)[0]
                classwise_ap = metrics[f"{prefix}/classwise_AP"]
                old_num_classes = int(task.split("+")[0])
                num_classes = int(runner.cfg.num_classes)
                new_num_classes = num_classes - old_num_classes
                old_map = sum(classwise_ap[:old_num_classes]) / old_num_classes
                new_map = sum(classwise_ap[old_num_classes:num_classes]) / new_num_classes

            save_path = os.path.join(runner.work_dir, self.save_file)
            with open(save_path, "w") as f:
                f.write(f"Best Epoch: {runner.epoch}\n")
                f.write(f"Eval Results:\n")
                for k, v in metrics.items():
                    if isinstance(v, float):
                        f.write(f"  {k}: {v * 100:.1f}\n")
                    else:
                        f.write(f"  {k}: {v}\n")
                if old_map is not None:
                    f.write(f"  old_classes_mAP: {old_map * 100:.1f}\n")
                    f.write(f"  new_classes_mAP: {new_map * 100:.1f}\n")

            # save necessary state_dict and meta.dataset_meta for storage efficiency
            model = runner.model
            if hasattr(model, "module"):
                model = model.module

            checkpoint = {
                "meta": {"dataset_meta": runner.train_dataloader.dataset.metainfo},
                "state_dict": model.state_dict(),
            }

            ckpt_path = os.path.join(runner.work_dir, "best_model.pth")
            torch.save(checkpoint, ckpt_path)
