import sys
using_colab = False

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
from PIL import Image
from sam2_repo.sam2.build_sam import build_sam2
from sam2_repo.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import copy
from skimage.segmentation import slic

def load_img(image_path=None):
    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))

    height = image.shape[0]
    width = image.shape[1]
    return image, height, width

# loas SAM2
def load_sam2(min_segment, device=None):
    sam2_checkpoint = "./sam2_repo/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "../../sam2_repo/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"

    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=64,
        points_per_batch=256,
        pred_iou_thresh=0.75,
        stability_score_thresh=0.85,
        stability_score_offset=1.0,
        mask_threshold=0.1,
        box_nms_thresh=0.6,
        crop_n_layers=0,
        crop_nms_thresh=0.5,
        crop_overlap_ratio=0.85,
        min_mask_region_area=min_segment,
    )

    return mask_generator

# Determine whether there is a containment relationship between concepts
def judge_contrain(masks, threshold=0.90):
    contain_tag = np.zeros([len(masks), len(masks)])
    for i in range(len(masks)):
        area_i = masks[i]['area']
        father = -1
        for j in range(len(masks)):
            area_j = masks[j]['area']
            if i == j:
                continue
            cover_idx = masks[i]['segmentation'] & masks[j]['segmentation']
            if (np.sum(cover_idx) / area_i >= threshold and area_j > area_i and
                    (father == -1 or area_j > masks[father]['area'])):
                father = j
        if father != -1:
            contain_tag[father][i] = 1
    return contain_tag

# removal of overlapping areas from the larger concept
def segment_big(image, masks, contain_tag):
    small_masks = copy.deepcopy(masks)
    for i in range(len(small_masks)):
        if sum(contain_tag[i]) > 0:
            idxs = np.where(contain_tag[i] == 1)[0]
            for idx in idxs:
                small_masks[i]['segmentation'][small_masks[idx]['segmentation']] = False
            small_masks[i]['area'] = np.sum(small_masks[i]['segmentation'])
    return small_masks

# assigning overlapping pixels to concepts with more similar colors
def overlap_reassignment(image, masks):
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if np.sum(masks[i]['segmentation'] & masks[j]['segmentation']) > 0:
                segment_colors_i = np.mean(image[masks[i]['segmentation']], axis=0)
                segment_colors_j = np.mean(image[masks[j]['segmentation']], axis=0)
                overlap_colors = np.mean(image[masks[i]['segmentation'] & masks[j]['segmentation']], axis=0)
                distance_i = np.linalg.norm(segment_colors_i - overlap_colors)
                distance_j = np.linalg.norm(segment_colors_j - overlap_colors)
                if distance_i < distance_j:
                    masks[j]['segmentation'] = masks[j]['segmentation'] & ~masks[i]['segmentation']
                else:
                    masks[i]['segmentation'] = masks[i]['segmentation'] & ~masks[j]['segmentation']
    return masks

# using SLIC clustering as a background
def blank_reassignment_SLIC(image, masks, height, width):
    axis_include = np.zeros([height, width]).astype(np.bool_)
    for mask in masks:
        axis_include = axis_include | mask['segmentation']

    segments_slic = SLIC(image, axis_include)
    unique_segments = np.unique(segments_slic)
    num_segments = len(unique_segments)
    add_masks = []
    for i in np.arange(1, num_segments):
        temp_mask = copy.deepcopy(masks[0])
        temp_mask['segmentation'] = segments_slic == i
        temp_mask['area'] = np.sum(temp_mask['segmentation'])
        add_masks.append(temp_mask)
        masks = np.insert(masks, -1, temp_mask)
    return masks, add_masks

def SLIC(image, axis_include):
    black_non_blank_image = image.__copy__()
    black_non_blank_image[axis_include, :] = 0

    segments_slic = slic(black_non_blank_image, n_segments=10, compactness=10,
                         start_label=1,
                         mask=~axis_include)

    return segments_slic

def concept_segmentation_pipeline(image_path=None, mask_generator=None, contain_threshold=0.90):
    # load the explained img x
    image, height, width = load_img(image_path)

    # SAM2 segmentation
    masks = mask_generator.generate(image)

    # Determine whether there is a containment relationship between concepts
    contain_tag = judge_contrain(masks, threshold=contain_threshold)

    # Handling of overlapping areas above a threshold: removal of overlapping areas from the larger concept
    masks = segment_big(image, masks, contain_tag)

    # Handling of overlapping areas not exceeding the threshold: assigning overlapping pixels to concepts with more similar colors
    masks = overlap_reassignment(image, masks)

    # Dealing with unassigned areas: using SLIC clustering as a background
    masks, add_small_masks = blank_reassignment_SLIC(image, masks, height, width)

    return masks
