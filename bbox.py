import cv2 as cv
import numpy as np
from sklearn.cluster import DBSCAN

def find_text_blocks(morph_img, config):
    kernel_block = cv.getStructuringElement(
        cv.MORPH_RECT, (config.block_kernel_width, config.block_kernel_height),
    )
    dilated = cv.dilate(morph_img, kernel_block, iterations=1)

    contours, _ = cv.findContours(
        dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )

    blocks = []
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        if w >= config.min_block_width and h >= config.min_block_height:
            blocks.append((x, y, w, h))

    blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
    return blocks, dilated


def find_lines_in_block(morph_img, block_rect, kernel_line, config):
    x, y, w, h = block_rect
    roi = morph_img[y:y + h, x:x + w]

    dilated = cv.dilate(roi, kernel_line, iterations=1)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (300, 3))
    dilated = cv.morphologyEx(dilated, cv.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv.findContours(
        dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )

    lines = []
    for cnt in contours:
        lx, ly, lw, lh = cv.boundingRect(cnt)
        if lw >= config.min_line_width:
            lines.append((x + lx, y + ly, lw, lh))

    lines = sorted(lines, key=lambda r: r[1])
    return lines, dilated


def find_lines(blocks, morph, config):
    dilated_lines_canvas = np.zeros_like(morph)

    lines = []

    kernel_line = cv.getStructuringElement(cv.MORPH_RECT, (config.line_kernel_width, config.line_kernel_height))

    for b_idx, block in enumerate(blocks):
        x, y, w, h = block
        print(f"\nBlock {b_idx}: x={x}, y={y}, w={w}, h={h}")

        lines, dilated_roi = find_lines_in_block(morph, block, kernel_line, config)
        lines.extend(lines)

        roi_canvas = dilated_lines_canvas[y:y + h, x:x + w]
        dilated_lines_canvas[y:y + h, x:x + w] = cv.bitwise_or(
            roi_canvas, dilated_roi
        )

    return lines, dilated_lines_canvas


def find_chars_in_line_dbscan(
    morph_img,
    line_rect,
    config,
    eps=3.0,
    min_samples=5,
    min_area_ratio=0.002,
    max_area_ratio=0.3,
):
    """
    Обнаружение символов в строке с помощью DBSCAN.

    morph_img  – бинарное изображение (белый текст на чёрном фоне),
                  после инверсии и морфологии.
    line_rect  – (x, y, w, h) строки в координатах всего изображения.
    eps        – радиус кластеризации DBSCAN (в пикселях).
    min_samples – минимальное число точек в кластере.
    """

    x, y, w, h = line_rect
    roi = morph_img[y:y + h, x:x + w]

    # координаты белых пикселей (текста)
    ys, xs = np.where(roi > 0)
    if len(xs) == 0:
        return []

    points = np.vstack((xs, ys)).T  # N x 2, (x, y)

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(points)

    chars = []
    line_area = w * h

    for label in np.unique(labels):
        if label == -1:
            # -1 = шум
            continue

        cluster_points = points[labels == label]
        if cluster_points.size == 0:
            continue

        min_x = int(cluster_points[:, 0].min())
        max_x = int(cluster_points[:, 0].max())
        min_y = int(cluster_points[:, 1].min())
        max_y = int(cluster_points[:, 1].max())

        cw = max_x - min_x + 1
        ch = max_y - min_y + 1

        if cw < config.min_char_width or ch < config.min_char_height:
            continue

        # перевод координат ROI -> глобальные
        gx = x + min_x
        gy = y + min_y
        chars.append((gx, gy, cw, ch))

    # сортировка символов слева направо
    chars = sorted(chars, key=lambda r: r[0])
    return chars

def find_chars(lines, morph, config):
    chars = []

    for l_idx, line in enumerate(lines):
        lx, ly, lw, lh = line
        print(f"  Line {l_idx}: x={lx}, y={ly}, w={lw}, h={lh}")

        line_chars = find_chars_in_line_dbscan(
            morph, line, config,
            eps=3.0,
            min_samples=5,
            min_area_ratio=0.002,
            max_area_ratio=0.3,
        )
        chars.extend(line_chars)

        for c_idx, ch in enumerate(chars):
            cx, cy, cw, ch_h = ch
            #print(f"    Char {c_idx}: x={cx}, y={cy}, w={cw}, h={ch_h}")
    
    return chars

