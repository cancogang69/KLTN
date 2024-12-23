import os
import copy
import json
from shapely import Polygon
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm

import cv2


def box_padding(tl_point, br_point, img_width, img_height):
    tl_point[0] = float(tl_point[0]) - 1
    tl_point[1] = float(tl_point[1]) - 1
    br_point[0] = float(br_point[0]) + 1
    br_point[1] = float(br_point[1]) + 1

    tl_point[0] = 0 if tl_point[0] < 0 else tl_point[0]
    tl_point[1] = 0 if tl_point[1] < 0 else tl_point[1]
    tl_point[0] = img_width if tl_point[0] > img_width else tl_point[0]
    tl_point[1] = img_height if tl_point[1] > img_height else tl_point[1]

    return tl_point, br_point


def get_box_pairs(annos, img_width, img_height) -> list:
    border_boxes = []
    object_boxes = []

    for anno in annos:
        if anno.tag == "box":
            tl_point, br_point = box_padding(
                [anno.attrib["xtl"], anno.attrib["ytl"]],
                [anno.attrib["xbr"], anno.attrib["ybr"]],
                img_width,
                img_height,
            )
            border_points = get_retangle_points(tl_point, br_point)
            border_info = {
                "label": anno.attrib["label"],
                "points": border_points,
            }
            border_boxes.append(border_info)
        else:
            object_points = get_segmentation_points(anno.attrib["points"])
            object_info = {
                "label": anno.attrib["label"],
                "points": object_points,
            }
            object_boxes.append(object_info)

    if len(border_boxes) == 0:
        image_border_points = get_retangle_points(
            [0, 0], [img_width, img_height]
        )
        image_border_info = {
            "label": object_boxes[0]["label"],
            "points": image_border_points,
        }
        border_boxes.append(image_border_info)

    count = 0
    box_pairs = []
    for border_info in border_boxes:
        border_points = border_info["points"]
        border_poly = Polygon(border_points)
        border_object_map = [border_points]
        for object_info in object_boxes:
            object_points = object_info["points"]
            object_poly = Polygon(object_points)
            if border_info["label"] == object_info[
                "label"
            ] and border_poly.contains(object_poly):
                object_points = map_points(border_points[0], object_points)
                border_object_map.append(object_points)
                count += 1

        box_pairs.append(border_object_map)

    return box_pairs, count == len(object_boxes)


def get_segmentation_points(points_str):
    points_coor = points_str.split(";")
    points = [list(map(float, coor.split(","))) for coor in points_coor]
    points = np.array(points, dtype=int)
    return points


def get_retangle_points(top_left, bottom_right):
    points = [
        top_left,
        [bottom_right[0], top_left[1]],
        bottom_right,
        [top_left[0], bottom_right[1]],
    ]

    return np.array(points, dtype=float).astype(int)


def map_points(root_coor, object_points) -> list:
    maped_object_points = []
    for point in object_points:
        new_point = [point[0] - root_coor[0], point[1] - root_coor[1]]
        maped_object_points.append(new_point)

    return maped_object_points


def is_in_range(start, end, number):
    return start <= number and number <= end


def get_mask_area(mask, threshold):
    return np.sum(mask > threshold)


def convert_mask_to_polygon(mask):
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    polygons = [contour.squeeze().reshape(-1).tolist() for contour in contours]
    return polygons


def split_mask(mask, split_percents, error_percent=0.05):
    h, w = mask.shape[:2]
    left_mask = np.zeros([h, w])
    right_mask = mask.copy()
    mask_area = get_mask_area(mask, 128)

    split_result = {}
    curr_masks = {
        "left_mask": [],
        "right_mask": [],
        "percent": 0,
        "last_col": 0,
    }

    curr_pos = 0
    curr_percent = split_percents[curr_pos]
    last_left_percent = 0
    for col in range(w):
        left_mask[0:h, col][right_mask[0:h, col] > 128] = 255
        right_mask[0:h, col] = 0

        left_mask_area = get_mask_area(left_mask, 128)
        left_mask_area_percent = left_mask_area / mask_area

        low_threshold = curr_percent - error_percent
        high_threshold = curr_percent + error_percent
        if left_mask_area_percent < low_threshold:
            continue

        if left_mask_area_percent > high_threshold:
            split_result[curr_percent] = copy.deepcopy(curr_masks)
            curr_pos += 1
            if curr_pos == len(split_percents):
                break
            curr_percent = split_percents[curr_pos]
            continue

        diff_last = abs(curr_percent - last_left_percent)
        diff_curr = abs(curr_percent - left_mask_area_percent)

        if diff_last > diff_curr:
            last_left_percent = left_mask_area_percent
            curr_masks["left_mask"] = left_mask.copy()
            curr_masks["right_mask"] = right_mask.copy()
            curr_masks["percent"] = left_mask_area_percent
            curr_masks["last_col"] = col

    return split_result


if __name__ == "__main__":
    annos_path = "dataset/annotations/annotations.xml"
    anno_save_path = "test.json"
    images_path = "dataset/KLTN.v2i.coco/train/processed"
    save_img_path = "dataset/processed/crop"
    split_percents = [0.3, 0.5, 0.7]
    content = {"images": [], "annotations": []}
    image_idx = 1
    anno_idx = 1

    start_idx = 0
    end_idx = 2999

    with open(annos_path, "r", encoding="utf-8") as anno_file:
        tree = ET.parse(anno_file)
        root = tree.getroot()

    if not os.path.exists(save_img_path):
        os.makedirs(save_img_path)

    for item in tqdm(root.findall("image")):
        image_id = int(item.attrib["id"])
        image_name = item.attrib["name"].split("/")[-1]
        if not is_in_range(start_idx, end_idx, image_id):
            continue

        img_path = f"{images_path}/{image_name}"
        img = cv2.imread(img_path)

        box_pairs, flag = get_box_pairs(
            item, int(item.attrib["width"]), int(item.attrib["height"])
        )

        if not flag:
            print(f"Something is wrong here {image_id=}")

        for i, pair in enumerate(box_pairs):
            border_points = pair[0]
            bd_top_left = border_points[0]
            bd_bottom_right = border_points[2]
            new_img = img[
                bd_top_left[1] : bd_bottom_right[1],
                bd_top_left[0] : bd_bottom_right[0],
            ]

            mask = np.zeros(
                (
                    bd_bottom_right[1] - bd_top_left[1],
                    bd_bottom_right[0] - bd_top_left[0],
                )
            )
            for i, points in enumerate(pair):
                if i == 0:
                    continue

                mask = cv2.fillPoly(
                    mask, np.array([points]), color=[255, 255, 255]
                )

            if len(pair) == 1:
                print(image_id)

            object_image_name = f"{image_name.split('.')[0]}_{i}.jpg"
            save_object_img = f"{save_img_path}/{object_image_name}"
            cv2.imwrite(save_object_img, new_img)

            _, mask = cv2.threshold(mask, 127, 255, 0)

            image_info = {
                "id": image_idx,
                "file_name": image_name,
                "width": mask.shape[1],
                "height": mask.shape[0],
            }

            content["images"].append(image_info)

            results = split_mask(mask, split_percents)

            all_polygons = convert_mask_to_polygon(mask)

            for key, value in results.items():
                left_polygons = convert_mask_to_polygon(value["left_mask"])
                right_polygons = convert_mask_to_polygon(value["right_mask"])

                anno = {
                    "id": anno_idx,
                    "image_id": image_idx,
                    "percent": key,
                    "last_col": value["last_col"],
                    "segmentations": all_polygons,
                    "visible_segmentations": left_polygons,
                    "invisible_segmentations": right_polygons,
                }
                content["annotations"].append(anno)
                anno_idx += 1

                anno = {
                    "id": anno_idx,
                    "image_id": image_idx,
                    "percent": key,
                    "last_col": -value["last_col"],
                    "segmentations": all_polygons,
                    "visible_segmentations": right_polygons,
                    "invisible_segmentations": left_polygons,
                }
                content["annotations"].append(anno)
                anno_idx += 1

            image_idx += 1

    with open(anno_save_path, "w+") as anno_file:
        json.dump(content, anno_file)
