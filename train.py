"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from src.options.custom_options import parse_args
from src.datasets.data_loader import DatasetLoader
from src.models import create_model


if __name__ == '__main__':
    opt = parse_args()
    opt.isTrain = True
    opt.rank = "0"
    train_dataset = DatasetLoader(anno_path=opt.train_anno_path)
    val_dataset = DatasetLoader(anno_path=opt.val_anno_path)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations

    best_miou = 0
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()          
        model.update_learning_rate()    
        for data in train_dataset:
            visible_mask, invisible_mask, final_mask, _ = data
            model.set_input(input=visible_mask, target=final_mask)
            model.optimize_parameters()

            iter_data_time = time.time()

        if epoch % opt.val_freq == 0:
            total_iou = 0
            percents_iou = {}
            for data in val_dataset:
                visible_mask, invisible_mask, final_mask, percent = data
                
                predict_mask = model.forward_only(visible_mask)

                intersection = ((predict_mask == 1) & (final_mask == 1)).sum()
                predict_area = (predict_mask == 1).sum()
                target_area = (final_mask == 1).sum()
                iou = intersection / (predict_area + target_area - intersection)
                if percent not in percents_iou:
                    percents_iou[percent] = [0, 0]

                percents_iou[percent][0] += 1
                percents_iou[percent][1] += iou
                total_iou += iou

            m_iou = total_iou / val_dataset.anno_len
            for percent, values in percents_iou.items():
                percents_iou[percent] = values[1] / values[0]

            if best_miou < m_iou:
                best_miou = m_iou

            print(f"Best mean IoU: {best_miou}, this epoch mean IoU: {m_iou}")
            for percent, m_iou in percents_iou.items():
                print(f"percent {percent}, mean IoU: {m_iou}")

            if epoch % opt.save_epoch_freq == 0:
                print(f"saving the model at the end of epoch {epoch}")
                model.save_networks('latest')
                model.save_networks(epoch)

        print(f"End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} \t Time Taken: {time.time() - epoch_start_time} sec")
