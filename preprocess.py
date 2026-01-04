import cv2 as cv


def preprocess_image(img, config):
    h, w = img.shape[:2]

    dx = int(w * config.crop_ratio)
    dy = int(h * config.crop_ratio)
    dx = max(dx, 1)
    dy = max(dy, 1)
    img_cropped = img[dy:h - dy, dx:w - dx].copy()

    gray = cv.cvtColor(img_cropped, cv.COLOR_BGR2GRAY)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    _, binary = cv.threshold(gray_clahe, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    binary_inv = 255 - binary

    return img_cropped, gray, gray_clahe, binary, binary_inv


def morphology_for_blocks(binary_inv, config):
    return morphology_by_diameter(binary_inv, config.block_morph_kernel_diameter)


def morphology_for_lines(binary_inv, config):
    return morphology_by_diameter(binary_inv, config.line_morph_kernel_diameter)


def morphology_for_chars(binary_inv, config):
    return morphology_by_diameter(binary_inv, config.char_morph_kernel_diameter)


def morphology_by_diameter(binary_inv, d):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (d, d))
    morph = cv.morphologyEx(binary_inv, cv.MORPH_OPEN, kernel, iterations=1)
    return morph