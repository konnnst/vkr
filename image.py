import os
import cv2 as cv
import numpy as np

import bbox
from config import ConfigManager
import draw
import preprocess

img_paths = [
    "images/kormchaya.jpg",
    "images/jatsimirskii_49.tif",
    "images/illarion.png",
]


def extract_characters_from_image(
    img_path=None,
    should_visualize=True,
    should_save=False
):
    if img_path is None:
        img_path = img_paths[1]
    get_config_from_image = True

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
    chars = bbox.find_chars(lines, chars_morph)

    if should_visualize:
        draw.draw_preprocessed(
            orig, gray, gray_clahe, binary, binary_inv, blocks_morph, lines_morph, chars_morph, dilated_blocks, dilated_lines,
        )
        draw.draw_boxes(orig, blocks, lines, chars)

    if should_save:    
        for i, (x, y, w, h) in enumerate(chars):
                char_img = orig[y:y + h, x:x + w]
                if char_img.size == 0:
                    print("FAIL")
                    continue
                filename = os.path.join(config.out_dir, f"char_{i:04d}.jpg")
                cv.imwrite(filename, char_img)
                print(f"Saved {filename}")


def extract_characters_from_image_directory(path):
    for file_path in os.listdir(path):
        if os.path.isfile(file_path):
            extract_characters_from_image(file_path, should_visualize=True)
    

