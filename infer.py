import os
import torch
from torch.utils.data import DataLoader

from src.datasets.custom_dataset import CustomDataset
from src.options.custom_options import parse_args
from src.models import create_model
from src.utils.util import tensor2im
from PIL import Image

torch.backends.cudnn.benchmark = True


def evaluate(rank, opt):
    opt.isTrain = False
    opt.rank = rank
    opt.is_ddp = False

    val_dataset = CustomDataset(opt.val_anno_path, opt, sdf_root=opt.sdf_val_root)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True)

    model = create_model(opt)

    save_root = "image_mask"
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    for data in val_loader:
        input_datas, final_masks, expand_regions, percents, img_names, anno_ids = data
        predict_masks = model.predict(input_datas)

        for predict_mask, _, expand_region, percent, img_name, anno_id in zip(predict_masks, final_masks, expand_regions, percents, img_names, anno_ids):
            expand_predict = predict_mask.mul(expand_region)

            predict_mask = tensor2im(predict_mask, is_sdf=opt.sdf).squeeze()

            expand_predict_mask = tensor2im(expand_predict, is_sdf=opt.sdf).squeeze()

            if not os.path.exists(f"{save_root}/{percent}"):
              os.makedirs(f"{save_root}/{percent}")

            predict_mask_path = f"{save_root}/{percent}/{img_name.split(".")[0]}_{anno_id}_a.png"
            Image.fromarray(predict_mask).save(predict_mask_path)
            predict_mask_path = f"{save_root}/{percent}/{img_name.split(".")[0]}_{anno_id}_e.png"
            Image.fromarray(predict_mask).save(expand_predict_mask)


if __name__ == "__main__":
    opt = parse_args()
    rank = 0
    evaluate(rank, opt)