import numpy as np
import json
import cv2
from PIL import Image

import torch

from src.data.base_dataset import get_transform, get_label_segment_transform


class CustomDataset(object):
    def __init__(self, anno_path, opt):
        with open(anno_path, "r") as anno_file:
            data = json.load(anno_file)
        self.opt = opt
        images_info = dict(
            [[img_info["id"], img_info] for img_info in data["images"]]
        )

        self.categories = [cate["id"] for cate in data["categories"]]
        self.annos = []
        for anno in data["annotations"]:
            img_info = images_info[anno["image_id"]]
            black_start = 0
            black_end = 0
            if anno["last_col"] > 0:
                black_start = anno["last_col"]
                black_end = img_info["width"]
            else:
                black_end = anno["last_col"] + 1

            self.annos.append(
                {
                    "mask": anno,
                    "image_file_name": img_info["file_name"],
                    "image_height": img_info["height"],
                    "image_width": img_info["width"],
                    "black_start": black_start,
                    "black_end": black_end,
                    "percent": anno["percent"],
                    "category_id": anno["category_id"]
                }
            )

        self.transform_img = get_transform(self.opt, None, grayscale=(self.opt.input_nc == 1))
        self.transform_label_mask = get_label_segment_transform(opt.load_size)

    def __len__(self):
        return len(self.annos)

    def __get_mask(self, height, width, polygons):
        mask = np.zeros([height, width])
        for polygon in polygons:
            polygon = np.array([polygon]).reshape(1, -1, 2)
            mask = cv2.fillPoly(
                mask, np.array(polygon), color=[255, 255, 255]
            )
        if self.opt.is_gray:
            mask[mask>1] = 1
        mask = mask.astype(np.uint8)
        return mask
    
    def __get_label_segment(self, mask, label_id):
        label_segment = mask.copy()
        label_segment[label_segment > 128] = label_id
        return label_segment

    def __getitem__(self, idx):
        anno = self.annos[idx]

        img_path = f"{self.opt.image_root}/{anno['image_file_name']}"
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image_h, image_w = img.shape[:2]
        white_img = np.full_like(img, 255)

        visible_mask = self.__get_mask(
            image_h, image_w, anno["mask"]["visible_segmentations"]
        )
        label_segment = self.__get_label_segment(visible_mask, anno["category_id"])
        label_segment = self.transform_label_mask(label_segment)

        if self.opt.is_gray:
            visible_mask = cv2.bitwise_and(img, white_img, mask=visible_mask)

        visible_mask = self.transform_img(Image.fromarray(visible_mask))

        input_data = torch.cat((visible_mask, label_segment), 0)

        final_mask = self.__get_mask(
            image_h, image_w, anno["mask"]["segmentations"]
        )

        if self.opt.is_gray:
            final_mask = cv2.bitwise_and(img, white_img, mask=final_mask)

        final_mask = self.transform_img(Image.fromarray(final_mask))

        percent = anno["percent"]

        return [
            input_data,
            final_mask,
            percent,
        ]