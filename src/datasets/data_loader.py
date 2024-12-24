import numpy as np
import json
import cv2
from PIL import Image

from src.data.base_dataset import get_transform


class DatasetLoader(object):
    def __init__(self, anno_path, opt):
        with open(anno_path, "r") as anno_file:
            data = json.load(anno_file)
        self.opt = opt
        images_info = dict(
            [[img_info["id"], img_info] for img_info in data["images"]]
        )

        annos_info = data["annotations"]

        self.transform = get_transform(self.opt, None, grayscale=(self.opt.input_nc == 1))

        self.annos = []
        for anno in annos_info:
            img_info = images_info[anno["image_id"]]
            black_start = 0
            black_end = 0
            if anno["last_col"] > 0:
                black_start = anno["last_col"]
                black_end = img_info["width"]
            else:
                black_end = anno["last_col"] + 1

            feature_file_name = (
                f"{img_info['file_name'].split('.')[0]}_{anno['id']}.pt"
            )

            self.annos.append(
                {
                    "mask": anno,
                    "image_file_name": img_info["file_name"],
                    "feature_file_name": feature_file_name,
                    "image_height": img_info["height"],
                    "image_width": img_info["width"],
                    "black_start": black_start,
                    "black_end": black_end,
                    "percent": anno["percent"],
                }
            )

        self.anno_len = len(self.annos)

    def __iter__(self):
        self.curr_idx = 0
        return self

    def __get_mask(self, height, width, polygons):
        mask = np.zeros([height, width])
        for polygon in polygons:
            polygon = np.array([polygon]).reshape(1, -1, 2)
            mask = cv2.fillPoly(
                mask, np.array(polygon), color=[255, 255, 255]
            )

        mask[mask>1] = 1
        return mask

    def __next__(self):
        if self.curr_idx == self.anno_len:
            raise StopIteration

        anno = self.annos[self.curr_idx]
        self.curr_idx += 1

        img_path = f"{self.opt.image_root}/{anno['image_file_name']}"
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image_h, image_w = img.shape[:2]
        white_img = np.full_like(img, 255)

        visible_mask = self.__get_mask(
            image_h, image_w, anno["mask"]["visible_segmentations"]
        )
        
        visible_mask = cv2.bitwise_and(img, white_img, mask=visible_mask)
        visible_mask = self.transform(Image.fromarray(visible_mask)).unsqueeze(0)

        invisible_mask = self.__get_mask(
            image_h, image_w, anno["mask"]["invisible_segmentations"]
        )

        final_mask = self.__get_mask(
            image_h, image_w, anno["mask"]["segmentations"]
        )

        final_mask = cv2.bitwise_and(img, white_img, mask=final_mask)
        final_mask = self.transform(Image.fromarray(final_mask)).unsqueeze(0)

        percent = anno["percent"]

        return [
            visible_mask,
            invisible_mask,
            final_mask,
            percent,
        ]
