import os
import cv2 as cv
import numpy as np

import bbox
from config import ConfigManager
import draw
import preprocess

IMG_PATHS = [
    "images/kormchaya.jpg",
    "images/jatsimirskii_49.tif",
    "images/illarion.png",
    "images/vkl.png",
    "images/kormchaya_cut.png",
]

CURRENT_IMAGE = 4

class ExtractedInfo:
    def __init__(self):
        self.chars = []
        self.plots = []
        

def extract_characters_from_image(
    img_path=None,
    should_visualize=True,
    should_save=False
):
    if img_path is None:
        img_path = IMG_PATHS[CURRENT_IMAGE]
    get_config_from_image = True

    extracted_info = ExtractedInfo()
    img = cv.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Не удалось открыть изображение по пути: {img_path}")

    if get_config_from_image:
        config = ConfigManager.get_config_by_image(img)
    else:
        config = ConfigManager.get_constant_config()

    orig, gray, gray_clahe, binary, binary_inv = preprocess.preprocess_image(img, config)
    blocks_morph = preprocess.morphology_for_blocks(binary_inv, config)
    lines_morph = preprocess.morphology_for_lines(binary_inv, config)
    chars_morph = preprocess.morphology_for_chars(binary_inv, config)

    blocks, dilated_blocks = bbox.find_text_blocks(blocks_morph, config)
    lines, dilated_lines = bbox.find_lines(blocks, lines_morph, config)
    chars = bbox.find_chars(lines, chars_morph, config)

    extracted_info.chars = chars
    
    if should_visualize:
        extracted_info.plots = [
            draw.get_preprocessed_plot(
                orig, gray, gray_clahe, binary, binary_inv, blocks_morph, lines_morph, chars_morph, dilated_blocks, dilated_lines,
            ),
            draw.get_boxes_plot(orig, blocks, lines, chars),
        ]

    if should_save:    
        for i, (x, y, w, h) in enumerate(chars):
                char_img = orig[y:y + h, x:x + w]
                if char_img.size == 0:
                    print("FAIL")
                    continue
                filename = os.path.join(config.out_dir, f"char_{i:04d}.jpg")
                cv.imwrite(filename, char_img)
                print(f"Saved {filename}")

    return extracted_info
        


def extract_characters_from_image_directory(path):
    for file_path in os.listdir(path):
        if os.path.isfile(file_path):
            extract_characters_from_image(file_path, should_visualize=True)

