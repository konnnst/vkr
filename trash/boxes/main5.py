import cv2
import numpy as np
import matplotlib.pyplot as plt


def preprocess_image(path, crop_ratio=0.05):
    # 1. Чтение изображения
    img = cv2.imread(path)  # BGR
    if img is None:
        raise FileNotFoundError(f"Не удалось открыть изображение по пути: {path}")

    h, w = img.shape[:2]

    # 2. Обрезка по 5% с каждой стороны
    dx = int(w * crop_ratio)
    dy = int(h * crop_ratio)
    dx = max(dx, 1)
    dy = max(dy, 1)
    img_cropped = img[dy:h - dy, dx:w - dx].copy()

    # 3. Оттенки серого
    gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)

    # 4. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    # 5. Бинаризация (Оцу)
    _, binary = cv2.threshold(
        gray_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 6. Инверсия (чёрный фон, белый текст)
    binary_inv = 255 - binary

    # 7. Морфология (очистка шума)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel, iterations=1)

    return img_cropped, gray, gray_clahe, binary, binary_inv, morph


def find_text_blocks(morph_img, min_area=1000000):
    kernel_block = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 20))
    dilated = cv2.dilate(morph_img, kernel_block, iterations=1)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    blocks = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h >= min_area:
            print("ok")
            blocks.append((x, y, w, h))
        else:
            print("not ok")

    blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
    return blocks, dilated


def find_lines_in_block(morph_img, block_rect, kernel_line, min_height=10):
    x, y, w, h = block_rect
    roi = morph_img[y:y + h, x:x + w]

    dilated = cv2.dilate(roi, kernel_line, iterations=1)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    lines = []
    for cnt in contours:
        lx, ly, lw, lh = cv2.boundingRect(cnt)
        if lh >= min_height:
            lines.append((x + lx, y + ly, lw, lh))

    lines = sorted(lines, key=lambda r: r[1])
    return lines, dilated


def find_chars_in_line(morph_img, line_rect,
                       min_area_ratio=0.01, max_area_ratio=0.5):
    x, y, w, h = line_rect
    roi = morph_img[y:y + h, x:x + w]

    contours, _ = cv2.findContours(
        roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    chars = []
    line_area = w * h

    for cnt in contours:
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if area < line_area * min_area_ratio:
            continue
        if area > line_area * max_area_ratio:
            continue

        chars.append((x + cx, y + cy, cw, ch))

    chars = sorted(chars, key=lambda r: r[0])
    return chars


def main():
    image_path = "pages/page.tif"

    # Предобработка
    orig, gray, gray_clahe, binary, binary_inv, morph = preprocess_image(
        image_path, crop_ratio=0.05
    )

    # Блоки текста + дилатация для блоков
    blocks, dilated_blocks = find_text_blocks(morph)

    # Холст для дилатации строк
    dilated_lines_canvas = np.zeros_like(morph)

    all_lines = []
    all_chars = []

    kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))

    print("==== Блоки, строки и символы ====")
    for b_idx, block in enumerate(blocks):
        x, y, w, h = block
        print(f"\nBlock {b_idx}: x={x}, y={y}, w={w}, h={h}")

        lines, dilated_roi = find_lines_in_block(morph, block, kernel_line)
        all_lines.extend(lines)

        roi_canvas = dilated_lines_canvas[y:y + h, x:x + w]
        dilated_lines_canvas[y:y + h, x:x + w] = cv2.bitwise_or(
            roi_canvas, dilated_roi
        )

        for l_idx, line in enumerate(lines):
            lx, ly, lw, lh = line
            print(f"  Line {l_idx}: x={lx}, y={ly}, w={lw}, h={lh}")

            chars = find_chars_in_line(morph, line)
            all_chars.extend(chars)

            for c_idx, ch in enumerate(chars):
                cx, cy, cw, ch_h = ch
                print(f"    Char {c_idx}: x={cx}, y={cy}, w={cw}, h={ch_h}")

    # -------- Визуализация этапов препроцессинга + дилатации (2 ряда) --------
    imgs = [
        cv2.cvtColor(orig, cv2.COLOR_BGR2RGB),
        gray,
        gray_clahe,
        binary,
        morph,
        dilated_blocks,
        dilated_lines_canvas,
    ]
    titles = [
        "Обрезанный оригинал",
        "Серое",
        "CLAHE",
        "Бинаризация",
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

    # -------- Визуализация блоков, строк и символов --------
    vis = orig.copy()

    # Блоки – зелёный, толщина 4 px
    for (x, y, w, h) in blocks:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 4)

    # Строки – синий, толщина 3 px
    for (x, y, w, h) in all_lines:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 3)

    # Символы – красный, толщина 3 px
    for (x, y, w, h) in all_chars:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 3)

    plt.figure(figsize=(10, 14))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Блоки (зелёный, 4px), строки (синий, 3px), символы (красный, 3px)")
    plt.show()


if __name__ == "__main__":
    main()
