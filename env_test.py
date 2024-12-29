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


def validate(model, val_dataset, val_loader, result_count=5, is_sdf=False):
    total_iou = 0
    percents_iou = {}
    results = []
    for data in val_loader:
        pass

def train(rank, opt):
    checkpoint_save_path = f"{opt.save_path}/{opt.name}"
    plot_save_path = f"{opt.plot_save_path}/{opt.name}"

    opt.isTrain = True
    opt.rank = rank
    opt.is_ddp = False

    train_dataset = CustomDataset(opt.train_anno_path, opt)
    val_dataset = CustomDataset(opt.val_anno_path, opt)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=opt.shuffle, num_workers=opt.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True)

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

    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)

    if not os.path.exists(checkpoint_save_path):
        os.makedirs(checkpoint_save_path)

    epoch_losses = {"gen": [], "dis": []}
    for epoch in range(1):
        epoch_start_time = time.time()          
            
        this_epoch_losses = {"num": 0, "gen": 0, "dis": 0}

        is_discrim_backprop = ((epoch % opt.discrim_backprop_freq) == 0)
        for data in train_loader:
            pass
        
        validate(model, val_dataset, val_loader, result_count)

        print(f"End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} \t Time Taken: {time.time() - epoch_start_time} sec")


if __name__ == "__main__":
    opt = parse_args()
    train(0, opt)