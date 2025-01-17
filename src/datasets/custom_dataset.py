import os
import json
import cv2
from PIL import Image
import numpy as np

import torch
import skfmm

from src.data.base_dataset import get_transform, get_label_segment_transform, input_resize


class CustomDataset(object):
    def __init__(self, anno_path, opt, sdf_root):
        with open(anno_path, "r") as anno_file:
            data = json.load(anno_file)
        self.opt = opt
        self.images_info = dict(
            [[img_info["id"], img_info] for img_info in data["images"]]
        )

        self.categories = [cate["id"] for cate in data["categories"]]

        if opt.split_percent != -1:
            self.annos_info = [anno for anno in data["annotations"] 
                               if anno["percent"] == opt.split_percent]
        else:
            self.annos_info = data["annotations"]

        self.is_grayscale = (self.opt.input_nc == 1)

        self.transform_img = get_transform(self.opt, None, grayscale=self.is_grayscale)
        self.transform_grayscale_img = get_transform(self.opt, None, grayscale=True)
        self.transform_label_mask = get_label_segment_transform(self.opt.load_size)
        self.input_resize = input_resize(self.opt.load_size)
        if opt.sdf:
            assert os.path.exists(sdf_root), f"{sdf_root} path not exists"
            self.sdf_root = sdf_root

        self.first_time = True

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

        return np.expand_dims(label_segment, axis=0)
    
    def __get_expand_map(self, height, width, last_col):
        expand_map = np.zeros([height, width])
        if last_col > 0:
            expand_map[:, :last_col] = 1
            expand_map[:, last_col:] = -1
        else:
            expand_map[:, last_col:] = 1
            expand_map[:, :last_col] = -1

        return np.expand_dims(expand_map, axis=0).astype(np.float16)
    
    def __get_object(self, image, object_mask):
        white_image = np.full_like(image, fill_value=(255, 255, 255))
        object_mask[object_mask > 127] = 1 
        masked = cv2.bitwise_and(image, white_image, mask=object_mask)

        return masked
    
    def __get_sdf_map(self, mask):
        mask_rbga = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2RGBA)
        phi = np.int64(np.any(mask_rbga[:, :, :3], axis = 2))
        phi = np.where(phi, 0, -1) + 0.5
        sdf_map = skfmm.distance(phi, dx = 1)
        return np.expand_dims(sdf_map, axis=0).astype(np.float16)

    def __getitem__(self, idx):
        anno = self.annos_info[idx]
        image_info = self.images_info[anno["image_id"]]
        image_h, image_w = image_info["height"], image_info["width"]

        visible_mask = self.__get_mask(
            image_h, image_w, anno["visible_segmentations"]
        )

        if "expand" in self.opt.extra_info:
            expand_map = self.__get_expand_map(image_h, image_w, anno["last_col"])
            expand_region = expand_map.copy()
            expand_region[expand_region == -1] = 0
            expand_region = self.input_resize(torch.as_tensor(expand_region))
            expand_map = self.input_resize(torch.as_tensor(expand_map))
        if "label" in self.opt.extra_info:
            label_segment = self.__get_label_segment(visible_mask, anno["category_id"])
            label_segment = self.transform_label_mask(torch.Tensor(label_segment))

        if not self.is_grayscale:
            image_path = f"{self.opt.image_root}/{image_info['file_name']}"
            assert os.path.exists(image_path), f"{image_path} doesn't exist"
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            visible_mask = self.__get_object(img, visible_mask)

        # if self.opt.sdf:
        #     if self.opt.use_precalculate_sdf:
        #         sdf_path = f"{self.sdf_root}/{anno['id']}_input.npy"
        #         assert os.path.exists(sdf_path), f"{sdf_path} doesn't exist"
        #         visible_mask = np.expand_dims(np.load(sdf_path), 0)
        #     else:
        #         visible_mask = self.__get_sdf_map(visible_mask)
        #     visible_mask[visible_mask < -100] = -100
        #     visible_mask = self.input_resize(torch.Tensor(visible_mask))
        # else:
        visible_mask = self.transform_img(Image.fromarray(visible_mask))

        input_data = visible_mask
        if "expand" in self.opt.extra_info:
            input_data = torch.cat((input_data, expand_map), 0)
        if "label" in self.opt.extra_info:
            input_data = torch.cat((input_data, label_segment), 0)

        final_mask = self.__get_mask(
            image_h, image_w, anno["segmentations"]
        )

        if self.opt.sdf:
            if self.opt.use_precalculate_sdf:
                sdf_path = f"{self.sdf_root}/{anno['id']}_target.npy"
                assert os.path.exists(sdf_path), f"{sdf_path} doesn't exist"
                final_mask = np.expand_dims(np.load(sdf_path), 0)
            else:
                final_mask = self.__get_sdf_map(final_mask)
            # final_mask[final_mask < -100] = -100
            final_mask = self.input_resize(torch.Tensor(final_mask))
        else:
            final_mask = self.transform_grayscale_img(Image.fromarray(final_mask))

        percent = anno["percent"]

        return [
            input_data,
            final_mask,
            expand_region,
            percent,
            image_info["file_name"],
            anno["id"],
        ]