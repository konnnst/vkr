import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import DBSCAN

IMAGE_PATH = "jatsimirskii_49.tif"
CROP_RATIO = 0.08
MIN_BLOCK_AREA = 100_000
MIN_LINE_LENGTH = 500
MORPH_KERNEL_DIAMETER = 3
BLOCK_KERNEL_WIDTH = 100
BLOCK_KERNEL_HEIGHT = 70 

def preprocess_image(path, crop_ratio, kernel_diameter):
    img = cv.imread(path)
    if img is None:
        raise FileNotFoundError(f"Не удалось открыть изображение по пути: {path}")

    h, w = img.shape[:2]

    dx = int(w * crop_ratio)
    dy = int(h * crop_ratio)
    dx = max(dx, 1)
    dy = max(dy, 1)
    img_cropped = img[dy:h - dy, dx:w - dx].copy()

    gray = cv.cvtColor(img_cropped, cv.COLOR_BGR2GRAY)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    _, binary = cv.threshold(gray_clahe, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    binary_inv = 255 - binary

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_diameter, kernel_diameter))
    morph = cv.morphologyEx(binary_inv, cv.MORPH_OPEN, kernel, iterations=1)

    return img_cropped, gray, gray_clahe, binary, binary_inv, morph


def find_text_blocks(morph_img, min_area):
    kernel_block = cv.getStructuringElement(cv.MORPH_RECT, (MORPH_KERNEL_DIAMETER, 20))
    dilated = cv.dilate(morph_img, kernel_block, iterations=1)

    contours, _ = cv.findContours(
        dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )

    blocks = []
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        if w * h >= min_area:
            blocks.append((x, y, w, h))

    blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
    return blocks, dilated


def find_lines_in_block(morph_img, block_rect, kernel_line, min_length=MIN_LINE_LENGTH):
    x, y, w, h = block_rect
    roi = morph_img[y:y + h, x:x + w]

    dilated = cv.dilate(roi, kernel_line, iterations=1)

    contours, _ = cv.findContours(
        dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )

    lines = []
    for cnt in contours:
        lx, ly, lw, lh = cv.boundingRect(cnt)
        if lw >= min_length:
            lines.append((x + lx, y + ly, lw, lh))

    lines = sorted(lines, key=lambda r: r[1])
    return lines, dilated


def find_chars_in_line_dbscan(
    morph_img,
    line_rect,
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

    # кластеризация DBSCAN
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
        area = cw * ch

        # фильтры по площади относительно всей строки
        if area < line_area * min_area_ratio:
            continue
        if area > line_area * max_area_ratio:
            continue

        # перевод координат ROI -> глобальные
        gx = x + min_x
        gy = y + min_y
        chars.append((gx, gy, cw, ch))

    # сортировка символов слева направо
    chars = sorted(chars, key=lambda r: r[0])
    return chars


def draw_preprocessing(orig, gray, gray_clahe, binary, binary_inv, morph, dilated_blocks, dilated_lines):
    imgs = [
        cv.cvtColor(orig, cv.COLOR_BGR2RGB),
        gray,
        gray_clahe,
        binary,
        binary_inv,
        morph,
        dilated_blocks,
        dilated_lines,
    ]
    titles = [
        "Обрезанный оригинал",
        "Серое",
        "CLAHE",
        "Бинаризация",
        "Инвертированная бинаризация",
        "Морфология (open)",
        "Дилатация (блоки)",
        "Дилатация (строки)",
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, (img, title) in enumerate(zip(imgs, titles)):
        if len(img.shape) == 2:
            axes[i].imshow(img, cmap="gray")
        else:
            axes[i].imshow(img)
        axes[i].set_title(title)
        axes[i].axis("off")

    if len(imgs) < len(axes):
        for j in range(len(imgs), len(axes)):
            axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def draw_result(orig, blocks, all_lines, all_chars):
    vis = orig.copy()

    for (x, y, w, h) in blocks:
        cv.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 4)

    for (x, y, w, h) in all_lines:
        cv.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 3)

    for (x, y, w, h) in all_chars:
        cv.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 3)

    plt.figure(figsize=(10, 14))
    plt.imshow(cv.cvtColor(vis, cv.COLOR_BGR2RGB))
    plt.axis("off")
    block_patch = mpatches.Patch(color='green', label='Блок текста')
    line_patch = mpatches.Patch(color='blue', label='Строка текста')
    char_patch = mpatches.Patch(color='red', label='Символ')

    plt.legend(
        handles=[block_patch, line_patch, char_patch],
        loc='lower right',
        framealpha=0.8,
        fontsize=10,
    )
    plt.show()


def main():
    orig, gray, gray_clahe, binary, binary_inv, morph = preprocess_image(
        IMAGE_PATH, CROP_RATIO, KERNEL_DIAMETER,
    )

    blocks, dilated_blocks = find_text_blocks(morph, min_area=MIN_BLOCK_AREA)

    dilated_lines_canvas = np.zeros_like(morph)

    all_lines = []
    all_chars = []

    kernel_line = cv.getStructuringElement(cv.MORPH_RECT, (50, 3))

    for b_idx, block in enumerate(blocks):
        x, y, w, h = block
        print(f"\nBlock {b_idx}: x={x}, y={y}, w={w}, h={h}")

        lines, dilated_roi = find_lines_in_block(morph, block, kernel_line, min_length=MIN_LINE_LENGTH)
        all_lines.extend(lines)

        roi_canvas = dilated_lines_canvas[y:y + h, x:x + w]
        dilated_lines_canvas[y:y + h, x:x + w] = cv.bitwise_or(
            roi_canvas, dilated_roi
        )

        for l_idx, line in enumerate(lines):
            lx, ly, lw, lh = line
            print(f"  Line {l_idx}: x={lx}, y={ly}, w={lw}, h={lh}")

            # Символы в строке через DBSCAN
            chars = find_chars_in_line_dbscan(
                morph, line,
                eps=3.0,
                min_samples=5,
                min_area_ratio=0.002,
                max_area_ratio=0.3,
            )
            all_chars.extend(chars)

            for c_idx, ch in enumerate(chars):
                cx, cy, cw, ch_h = ch
                print(f"    Char {c_idx}: x={cx}, y={cy}, w={cw}, h={ch_h}")

    draw_preprocessing(orig, gray, gray_clahe, binary, binary_inv, morph, dilated_blocks, dilated_lines_canvas)
    draw_result(orig, blocks, all_lines, all_chars)


if __name__ == "__main__":
    main()
