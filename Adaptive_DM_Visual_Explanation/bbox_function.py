import cv2
import numpy as np

def save_prototype_original_img_with_bbox(img_rgb,
                                          bbox_height_start, bbox_height_end,
                                          bbox_width_start, bbox_width_end, color=(0, 255, 255)):

    p_img_bgr = img_rgb[..., ::-1]
    cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1), color, thickness=2)
    p_img_rgb = p_img_bgr[..., ::-1]
    p_img_rgb = np.float32(p_img_rgb) / 255

    return p_img_rgb

def save_prototype_original_img_with_bbox1(img_rgb,
                                          bbox_height_start, bbox_height_end,
                                          bbox_width_start, bbox_width_end, color=(255, 255, 0)):

    p_img_bgr = img_rgb - np.min(img_rgb)
    cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1), color, thickness=2)
    p_img_rgb = p_img_bgr
    p_img_rgb = np.float32(p_img_rgb) / 255

    return p_img_rgb