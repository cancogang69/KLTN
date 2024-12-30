import os
import time
from PIL import Image

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.datasets.custom_dataset import CustomDataset
from src.options.custom_options import parse_args
from src.models import create_model
from src.utils.util import tensor2im

torch.backends.cudnn.benchmark = True


def evaluate(rank, opt):
    opt.isTrain = False
    opt.rank = rank
    opt.is_ddp = False

    val_dataset = CustomDataset(opt.val_anno_path, opt, sdf_root=opt.sdf_val_root)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True)

    model = create_model(opt)
    model.setup(opt)

    if not os.path.exists(plot_save_path) and rank == 0:
        os.makedirs(plot_save_path)

    if not os.path.exists(checkpoint_save_path) and rank == 0:
        os.makedirs(checkpoint_save_path)

    total_iou = 0
    percents_iou = {}
    results = []
    for data in val_loader:
        
        input_datas, final_masks, percents = data
        predict_masks = model.forward_only(input_datas)

        for predict_mask, final_mask, percent in zip(predict_masks, final_masks, percents):
            predict_mask = tensor2im(predict_mask, is_sdf=opt.sdf).squeeze()
            final_mask = tensor2im(final_mask, is_sdf=opt.sdf).squeeze()
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

    m_iou = total_iou / len(val_dataset)
    for percent, values in percents_iou.items():
        percents_iou[percent] = values[1] / values[0]

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

    fig.savefig(f"{plot_save_path}/epoch_{epoch}_result.jpg")
    plt.close(fig)

    print(f"Mean IoU: {m_iou}")
    for percent, m_iou in percents_iou.items():
        print(f"percent {percent}, mean IoU: {m_iou}")

    print(f"End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} \t Time Taken: {time.time() - epoch_start_time} sec")


if __name__ == "__main__":
    opt = parse_args()
    evaluate(local_rank, world_size, opt)