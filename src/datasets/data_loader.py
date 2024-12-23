import numpy as np
import json
import cv2


class DatasetLoader(object):
    def __init__(self, anno_path):
        with open(anno_path) as anno_file:
            data = json.load(anno_path)
        
        images_info = dict(
            [[img_info["id"], img_info] for img_info in data["images"]]
        )
        
        annos_info = data["annotations"]

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
            mask = cv2.fillPoly(
                mask, np.array([polygon]), color=[255, 255, 255]
            )
        mask[mask > 0] = 1
        return mask

    def __next__(self):
        if self.curr_idx == self.anno_len:
            raise StopIteration

        anno = self.annos[self.curr_idx]
        self.curr_idx += 1
        image_h, image_w = anno["image_height"], anno["image_width"]

        visible_mask = self.__get_mask(
            image_h, image_w, anno["mask"]["visible_segmentations"]
        )
        invisible_mask = self.__get_mask(
            image_h, image_w, anno["mask"]["invisible_segmentations"]
        )
        final_mask = self.__get_mask(
            image_h, image_w, anno["mask"]["segmentations"]
        )
        bbox = mask_to_bbox(visible_mask)
        sd_feats = self.__get_feature_from_save(anno["feature_file_name"])
        sd_feats = self.__combime_mask_with_sd_features(
            image_height=anno["image_height"],
            image_width=anno["image_width"],
            bbox=bbox,
            sd_features=sd_feats,
        )

        percent = anno["percent"]

        return [
            visible_mask,
            invisible_mask,
            final_mask,
            bbox,
            sd_feats,
            percent,
        ]
