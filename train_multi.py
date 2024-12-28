import os
import time
from PIL import Image

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt

from src.datasets.custom_dataset import CustomDataset
from src.options.custom_options import parse_args
from src.models import create_model
from src.utils.util import tensor2im


def validate(model, val_dataset, val_loader, result_count=5):
    total_iou = 0
    percents_iou = {}
    results = []
    for data in val_loader:
        
        input_datas, final_masks, percents = data
        predict_masks = model.forward_only(input_datas)

        for predict_mask, final_mask, percent in zip(predict_masks, final_masks, percents):
            predict_mask = tensor2im(predict_mask).squeeze()
            final_mask = tensor2im(final_mask).squeeze()
            intersection = ((predict_mask == 1) & (final_mask == 1)).sum()
            predict_area = (predict_mask == 1).sum()
            target_area = (final_mask == 1).sum()
            iou = intersection / (predict_area + target_area - intersection)

            percent = percent.item()
            if percent not in percents_iou:
                percents_iou[percent] = [0, 0]

            percents_iou[percent][0] += 1
            percents_iou[percent][1] += iou
            total_iou += iou

            if len(results) < result_count:
                results.append([predict_mask, final_mask])

    m_iou = total_iou / len(val_dataset)
    for percent, values in percents_iou.items():
        percents_iou[percent] = values[1] / values[0]

    return m_iou, results, percents_iou


def train(rank, world_size, opt):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    checkpoint_save_path = f"{opt.save_path}/{opt.name}"

    opt.isTrain = True
    opt.rank = rank
    opt.is_ddp = True

    train_dataset = CustomDataset(opt.train_anno_path, opt)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=opt.is_shuffle, drop_last=False)
    val_dataset = CustomDataset(opt.val_anno_path, opt)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=opt.is_shuffle)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)

    model = create_model(opt)
    model.setup(opt)

    best_miou = 0
    result_count = 5
    if opt.model_generator_path is not None:
        m_iou, _, percents_iou = validate(model, val_dataset, val_loader, result_count)
        if rank == 0:
            save_best = False
            if best_miou < m_iou:
                best_miou = m_iou

            print(f"Best mean IoU: {best_miou}, this epoch mean IoU: {m_iou}")
            for percent, m_iou in percents_iou.items():
                print(f"percent {percent}, mean IoU: {m_iou}")

    if not os.path.exists(opt.plot_save_path) and rank == 0:
        os.makedirs(opt.plot_save_path)

    if not os.path.exists(checkpoint_save_path) and rank == 0:
        os.makedirs(checkpoint_save_path)

    epoch_losses = {"gen": [], "dis": []}
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()          
            
        this_epoch_losses = {"num": 0, "gen": 0, "dis": 0}

        is_discrim_backprop = ((epoch % opt.discrim_backprop_freq) == 0)
        for data in train_loader:
            input_datas, final_masks, _ = data
            model.set_input(input=input_datas, target=final_masks)
            gen_loss, dis_loss = model.optimize_parameters(is_discrim_backprop)

            this_epoch_losses["num"] += 1
            this_epoch_losses["gen"] += gen_loss
            this_epoch_losses["dis"] += dis_loss

        epoch_losses["gen"].append(this_epoch_losses["gen"]/this_epoch_losses["num"])
        epoch_losses["dis"].append(this_epoch_losses["dis"]/this_epoch_losses["num"])

        if rank == 0:
            fig, axs = plt.subplots(1, 1)
            axs.set_xlabel("epoch")
            axs.set_ylabel("loss")
            axs.set_title("Loss per epoch")

            epoch_nums = range(1, len(epoch_losses["gen"])+1)
            axs.plot(epoch_nums, epoch_losses["gen"], label="generator_loss")
            axs.plot(epoch_nums, epoch_losses["dis"], label="discriminator_loss")
            axs.xaxis.get_major_locator().set_params(integer=True)
            axs.legend()
            fig.savefig(f"{opt.plot_save_path}/losses_plot.jpg")
            plt.close(fig)

        model.update_learning_rate()

        if epoch % opt.val_freq == 0:
            m_iou, results, percents_iou = validate(model, val_dataset, val_loader, result_count)

            if rank == 0:
                save_best = False
                if best_miou < m_iou:
                    best_miou = m_iou
                    save_best = True
                nrows = 2
                ncols = result_count
                fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 2))
                plt.suptitle(f"EPOCH : {epoch}")
                
                for i, result in enumerate(results):
                    axes[0][i].imshow(Image.fromarray(result[0]), cmap="gray")
                    axes[0][i].axis("off")
                    axes[1][i].imshow(Image.fromarray(result[1]), cmap="gray")
                    axes[1][i].axis("off")

                fig.savefig(f"{opt.plot_save_path}/epoch_{epoch}_result.jpg")
                plt.close(fig)

                print(f"Best mean IoU: {best_miou}, this epoch mean IoU: {m_iou}")
                for percent, m_iou in percents_iou.items():
                    print(f"percent {percent}, mean IoU: {m_iou}")

                if save_best:
                    print(f"saving the model at the end of epoch {epoch}")
                    model.save_networks("best")

                model.save_networks("last")

            

        if rank == 0:
            print(f"End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} \t Time Taken: {time.time() - epoch_start_time} sec")

        dist.barrier()

    # Clean up
    dist.destroy_process_group()


if __name__ == "__main__":
    opt = parse_args()
    world_size = torch.cuda.device_count()
    local_rank = int(os.environ["LOCAL_RANK"])
    train(local_rank, world_size, opt)

        
