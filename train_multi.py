import os
import sys
import yaml

sys.path.append(".")

import torch
import torch.distributed as dist
import matplotlib.pyplot as plt

from libs.datasets.data_loader import DatasetLoader
from libs.models.aw_sdm import AWSDM


config = {
    "config_path": os.getenv("CONFIG_PATH"),
    "pretrained_path": os.getenv("PRETRAINED_PATH"),
    "train_anno_path": os.getenv("TRAIN_ANNO_PATH"),
    "val_anno_path": os.getenv("VAL_ANNO_PATH"),
    "image_root": os.getenv("IMAGE_ROOT"),
    "feature_root": os.getenv("FEATURE_ROOT"),
    "feature_subdir_prefix": os.getenv("FEATURE_SUBDIR_PREFIX"),
    "save_dir": os.getenv("SAVE_DIR"),
    "epoch": os.getenv("EPOCH"),
    "val_freq": os.getenv("VAL_FREQ"),
}


def train(rank, world_size):
    """Training function for each GPU process.

    Args:
        rank (int): The rank of the current process (one per GPU).
        world_size (int): Total number of processes.
    """
    # Initialize the process group for distributed training
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    with open(config["config_path"]) as f:
        config_yaml = yaml.load(f, Loader=yaml.FullLoader)
    # print(config_yaml)
    model = AWSDM(
        params=config_yaml,
        pretrained_path=config["pretrained_path"],
        dist_model=True,
        rank=rank,
    )
    model.switch_to("train")

    train_loader = DatasetLoader(
        config["train_anno_path"],
        config["feature_root"],
        config["feature_subdir_prefix"],
    )

    val_loader = DatasetLoader(
        config["val_anno_path"],
        config["feature_root"],
        config["feature_subdir_prefix"],
    )

    if not os.path.exists(config["save_dir"]):
        os.makedirs(config["save_dir"])

    best_mIoU = 0

    for epoch in range(int(config["epoch"])):
        for i, data in enumerate(train_loader):
            visible_mask, invisible_mask, final_mask, bbox, sd_feats, _ = data
            # model.set_input(
            #     rgb=sd_feats, mask=visible_mask, target=final_mask, rank=rank
            # )
            model.set_input(
                rgb=None, mask=visible_mask, target=final_mask, rank=rank
            )
            loss = model.step()
            if i % config_yaml["trainer"]["print_freq"] == 0:
                print(f"Epoch: {epoch}, step: {i+1}, loss: {loss}")

        if epoch % int(config["val_freq"]) == 0:
            total_iou = 0
            percents_iou = dict()
            result_count = 5
            results = []
            for data in val_loader:
                (
                    visible_mask,
                    invisible_mask,
                    final_mask,
                    bbox,
                    sd_feats,
                    percent,
                ) = data
                # iou, predict = model.evaluate(
                #     rgb=sd_feats,
                #     mask=visible_mask,
                #     bbox=bbox,
                #     target=final_mask,
                #     rank=rank,
                # )
                iou, predict = model.evaluate(
                    mask=visible_mask,
                    bbox=bbox,
                    target=final_mask,
                    rank=rank,
                )

                if percent not in percents_iou:
                    percents_iou[percent] = [0, 0]

                percents_iou[percent][0] += 1
                percents_iou[percent][1] += iou
                total_iou += iou

                if len(results) < result_count:
                    results.append([predict, final_mask])

            nrows = 2
            ncols = result_count
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 2))
            plt.suptitle(f"EPOCH : {epoch}")

            for i, result in enumerate(results):
                axes[0][i].imshow(result[0], cmap="gray")
                axes[0][i].axis("off")
                axes[1][i].imshow(result[1], cmap="gray")
                axes[1][i].axis("off")

            plt.show()

            fig.savefig(f"{config['save_dir']}/epoch_{epoch}_result.jpg")

            for percent, value in percents_iou.items():
                percents_iou[percent] = value[1] / value[0]

            mIoU = total_iou / val_loader.anno_len

            if best_mIoU < mIoU and rank == 0:
                print("Saving new best checkpoint...")
                best_mIoU = mIoU
                model.save_state(path=config["save_dir"], epoch=epoch)
                print("Saving done!")

                print(f"\nEpoch: {epoch}, best mIoU {best_mIoU}, mIoU: {mIoU}")
                print("IoU per percent:")
                for percent, miou in percents_iou.items():
                    print(f"{percent} percent: {miou} (mIoU)")

        dist.barrier()

    # Clean up
    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    local_rank = int(os.environ["LOCAL_RANK"])
    train(local_rank, world_size)
