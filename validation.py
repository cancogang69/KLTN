import torch
from torch.utils.data import DataLoader

from src.datasets.custom_dataset import CustomDataset
from src.options.custom_options import parse_args
from src.models import create_model
from src.utils.util import tensor2im, get_iou

torch.backends.cudnn.benchmark = True


def evaluate(rank, opt):
    opt.isTrain = False
    opt.rank = rank
    opt.is_ddp = False

    val_dataset = CustomDataset(opt.val_anno_path, opt, sdf_root=opt.sdf_val_root)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True)

    model = create_model(opt)

    total_iou = 0
    total_expand_iou = 0
    percents_iou = {}
    percents_expand_iou = {}
    for data in val_loader:
        input_datas, final_masks, expand_regions, percents = data[:4]
        predict_masks = model.predict(input_datas)

        for predict_mask, final_mask, expand_region, percent in zip(predict_masks, final_masks, expand_regions, percents):
            expand_predict = predict_mask.mul(expand_region)
            expand_final = final_mask.mul(expand_region)

            predict_mask = tensor2im(predict_mask, is_sdf=opt.sdf).squeeze()
            final_mask = tensor2im(final_mask, is_sdf=opt.sdf).squeeze()

            expand_predict_mask = tensor2im(expand_predict, is_sdf=opt.sdf).squeeze()
            expand_final_mask = tensor2im(expand_final, is_sdf=opt.sdf).squeeze()

            image_iou = get_iou(predict_mask, final_mask)
            expand_iou = get_iou(expand_predict_mask, expand_final_mask)

            percent = percent.item()
            if percent not in percents_iou:
                percents_iou[percent] = [0, 0]
                percents_expand_iou[percent] = [0, 0]

            percents_iou[percent][0] += 1
            percents_iou[percent][1] += image_iou
            total_iou += image_iou

            percents_expand_iou[percent][0] += 1
            percents_expand_iou[percent][1] += expand_iou
            total_expand_iou += expand_iou

    m_iou = total_iou / len(val_dataset)
    for percent, values in percents_iou.items():
        percents_iou[percent] = values[1] / values[0]

    m_expand_iou = total_expand_iou / len(val_dataset)
    for percent, values in percents_expand_iou.items():
        percents_expand_iou[percent] = values[1] / values[0]


    print(f"Mean IoU: {m_iou}")
    for percent, m_iou in percents_iou.items():
        print(f"percent {percent}, mean IoU: {m_iou}")
        
    print(f"Expand mean IoU: {m_expand_iou}")
    for percent, m_iou in percents_expand_iou.items():
        print(f"percent {percent}, mean expand IoU: {m_iou}")

if __name__ == "__main__":
    opt = parse_args()
    rank = 0
    evaluate(rank, opt)