import os
import time

import torch
import torch.distributed as dist
import matplotlib.pyplot as plt

from src.options.train_options import TrainOptions
from src.datasets.data_loader import DatasetLoader
from src.models import create_model


def train(rank, world_size, opt):
    """Training function for each GPU process.

    Args:
        rank (int): The rank of the current process (one per GPU).
        world_size (int): Total number of processes.
    """
    # Initialize the process group for distributed training
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    train_dataset = DatasetLoader(anno_path=opt.train_anno_path)
    val_dataset = DatasetLoader(anno_path=opt.val_anno_path)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()
        epoch_iter = 0              
        model.update_learning_rate()    
        for data in train_dataset:
            visible_mask, invisible_mask, final_mask, _ = data
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print(f"saving the model at the end of epoch {epoch}")
            model.save_networks('latest')
            model.save_networks(epoch)

        print(f"End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} \t Time Taken: {time.time() - epoch_start_time} sec")

        if epoch % opt.val_freq == 0:
            percents_iou = {}
            for data in val_dataset:
                visible_mask, invisible_mask, final_mask, percent = data

                model.set_input(input=visible_mask, target= final_mask)
                model.forward_only()

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

    dist.destroy_process_group()


if __name__ == "__main__":
    opt = TrainOptions().parse()
    world_size = torch.cuda.device_count()
    local_rank = int(os.environ["LOCAL_RANK"])
    train(local_rank, world_size, opt)

        
