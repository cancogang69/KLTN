import pickle
import json


if __name__ == "__main__":
    pkl_path = "D:/HocTap/KLTN/dataset/mp3d/MP3D_seleted_eval_10.29.pkl"
    json_path = "test.json"

    with open(pkl_path, "rb") as pkl_file:
        data = pickle.load(pkl_file)

    new_annotations = []
    for anno in data["annotations"]:
        new_regions = []
        for region in anno["regions"]:
            region["area"] = int(region["area"])
            region["occlusion_rate"] = float(region["occlusion_rate"])

            region["segmentation"]["counts"] = region["segmentation"][
                "counts"
            ].decode()
            region["visible_mask"]["counts"] = region["visible_mask"][
                "counts"
            ].decode()
            new_regions.append(region)

        anno["regions"] = new_regions
        new_annotations.append(anno)

    data["annotations"] = new_annotations

    with open(json_path, "w") as json_file:
        json.dump(data, json_file)
