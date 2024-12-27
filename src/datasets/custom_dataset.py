import os
import json
import cv2
from PIL import Image
import numpy as np

import torch

from src.data.base_dataset import get_transform, get_label_segment_transform


class CustomDataset(object):
    def __init__(self, anno_path, opt):
        with open(anno_path, "r") as anno_file:
            data = json.load(anno_file)
        self.opt = opt
        self.images_info = dict(
            [[img_info["id"], img_info] for img_info in data["images"]]
        )

        self.categories = [cate["id"] for cate in data["categories"]]
        self.annos_info = data["annotations"]

        self.is_grayscale = (self.opt.input_nc == 1)

        self.transform_img = get_transform(self.opt, None, grayscale=self.is_grayscale)
        self.transform_grayscale_img = get_transform(self.opt, None, grayscale=True)
        self.transform_label_mask = get_label_segment_transform(opt.load_size)

    def __len__(self):
        return len(self.annos_info)

    def __get_mask(self, height, width, polygons):
        mask = np.zeros([height, width])
        for polygon in polygons:
            polygon = np.array([polygon]).reshape(1, -1, 2)
            mask = cv2.fillPoly(
                mask, np.array(polygon), color=[255, 255, 255]
            )
            
        mask = mask.astype(np.uint8)
        return mask
    
    def __get_label_segment(self, mask, label_id):
        label_segment = mask.copy()
        label_segment[label_segment > 128] = label_id
        return label_segment
    
    def __get_expand_map(self, height, width, last_col):
        expand_map = np.zeros([height, width])
        if last_col > 0:
            expand_map[:, :last_col] = 255
        else:
            expand_map[:, last_col:] = 255

        return expand_map
    
    def __get_object(self, image, object_mask):
        white_image = np.full_like(image, fill_value=(255, 255, 255))
        object_mask[object_mask > 127] = 1 
        masked = cv2.bitwise_and(image, white_image, mask=object_mask)

        return masked


    def __getitem__(self, idx):
        anno = self.annos_info[idx]
        image_info = self.images_info[anno["image_id"]]
        image_h, image_w = image_info["height"], image_info["width"]

        visible_mask = self.__get_mask(
            image_h, image_w, anno["visible_segmentations"]
        )            

        if self.opt.use_extra_info:
            label_segment = self.__get_label_segment(visible_mask, anno["category_id"])
            label_segment = self.transform_label_mask(Image.fromarray(label_segment))

            expand_map = self.__get_expand_map(image_h, image_w, anno["last_col"])
            expand_map = self.transform_grayscale_img(Image.fromarray(expand_map))

        if self.is_grayscale:
            image_path = f"{self.opt.image_root}/{image_info['file_name']}"
            assert os.path.exists(image_path), f"{image_path} doesn't exist"
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            visible_mask = self.__get_object(img, visible_mask)
        
        visible_mask = self.transform_img(Image.fromarray(visible_mask))

        if self.opt.use_extra_info:
            input_data = torch.cat((visible_mask, label_segment, expand_map), 0)
        else:
            input_data = visible_mask

        final_mask = self.__get_mask(
            image_h, image_w, anno["segmentations"]
        )

        final_mask = self.transform_grayscale_img(Image.fromarray(final_mask))

        percent = anno["percent"]

        return [
            input_data,
            final_mask,
            percent,
        ]