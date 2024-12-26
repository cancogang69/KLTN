import os
import time
from PIL import Image

import torch
import torch.distributed as dist
import matplotlib.pyplot as plt

from src.datasets.data_loader import DatasetLoader
from src.options.custom_options import parse_args
from src.models import create_model
from src.utils.util import tensor2im


def train(rank, world_size, opt):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    checkpoint_save_path = "checkpoints/mask_generator"
    opt.isTrain = True
    opt.rank = rank
    opt.is_ddp = True
    train_dataset = DatasetLoader(opt.train_anno_path, opt)
    val_dataset = DatasetLoader(opt.val_anno_path, opt)

    model = create_model(opt)
    model.setup(opt)
    model.load_networks(opt.model_generator_path, opt.model_discriminator_path, opt.isTrain)

    if not os.path.exists(opt.plot_save_path) and rank == 0:
        os.makedirs(opt.plot_save_path)

    if not os.path.exists(checkpoint_save_path) and rank == 0:
        os.makedirs(checkpoint_save_path)

    best_miou = 0
    result_count = 5
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()          
            
        for data in train_dataset:
            visible_mask, final_mask, _ = data
            model.set_input(input=visible_mask, target=final_mask)
            model.optimize_parameters()

        model.update_learning_rate()

        if epoch % opt.val_freq == 0:
            total_iou = 0
            percents_iou = {}
            results = []
            for data in val_dataset:
                
                visible_mask, final_mask, percent = data
                predict_mask = model.forward_only(visible_mask)

                predict_mask = tensor2im(predict_mask).squeeze()
                final_mask = tensor2im(final_mask).squeeze()
                intersection = ((predict_mask == 1) & (final_mask == 1)).sum()
                predict_area = (predict_mask == 1).sum()
                target_area = (final_mask == 1).sum()
                iou = intersection / (predict_area + target_area - intersection)
                if percent not in percents_iou:
                    percents_iou[percent] = [0, 0]

                percents_iou[percent][0] += 1
                percents_iou[percent][1] += iou
                total_iou += iou

                if len(results) < result_count:
                    results.append([predict_mask, final_mask])

            m_iou = total_iou / val_dataset.anno_len
            for percent, values in percents_iou.items():
                percents_iou[percent] = values[1] / values[0]

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

        
