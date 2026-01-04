import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_PATH = "page.jpg"          # входное изображение
BINARY_OUT_PATH = "page_binary.png"  # ЧБ-изображение без рамок


def choose_best_binary(binary_a, binary_b, img_area):
    def count_good_components(bin_img):
        contours, _ = cv2.findContours(
            bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        count = 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area < 30:
                continue
            if area > 0.2 * img_area:
                continue
            count += 1
        return count

    count_a = count_good_components(binary_a)
    count_b = count_good_components(binary_b)
    return binary_a if count_a >= count_b else binary_b


def preprocess_image(img_bgr, show_steps=True):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray_blur)

    _, binary_raw = cv2.threshold(
        gray_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    h, w = binary_raw.shape[:2]
    img_area = h * w
    binary_inv = 255 - binary_raw

    binary_chosen = choose_best_binary(binary_raw, binary_inv, img_area)

    kernel = np.ones((2, 2), np.uint8)
    binary_clean = cv2.morphologyEx(
        binary_chosen, cv2.MORPH_OPEN, kernel, iterations=1
    )

    # оставляем текст белым на чёрном
    white_pixels = np.sum(binary_clean == 255)
    black_pixels = np.sum(binary_clean == 0)
    if white_pixels > black_pixels:
        binary_clean = 255 - binary_clean

    if show_steps:
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 3, 1)
        plt.title("Исходное (уменьш.)")
        small = cv2.resize(
            cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
            None,
            fx=0.5,
            fy=0.5,
            interpolation=cv2.INTER_AREA,
        )
        plt.imshow(small)
        plt.axis("off")

        plt.subplot(2, 3, 2)
        plt.title("Серое + blur")
        plt.imshow(gray_blur, cmap="gray")
        plt.axis("off")

        plt.subplot(2, 3, 3)
        plt.title("После CLAHE")
        plt.imshow(gray_clahe, cmap="gray")
        plt.axis("off")

        plt.subplot(2, 3, 4)
        plt.title("Binary (Otsu)")
        plt.imshow(binary_raw, cmap="gray")
        plt.axis("off")

        plt.subplot(2, 3, 5)
        plt.title("Binary (инверсия)")
        plt.imshow(binary_inv, cmap="gray")
        plt.axis("off")

        plt.subplot(2, 3, 6)
        plt.title("Выбранное + MORPH_OPEN (инверт.)")
        plt.imshow(binary_clean, cmap="gray")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    return binary_clean


def get_character_boxes(binary_img):
    h_img, w_img = binary_img.shape[:2]
    img_area = h_img * w_img

    contours, _ = cv2.findContours(
        binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        if area < 40:
            continue

        if area > 0.2 * img_area:
            continue

        boxes.append((x, y, w, h))

    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    return boxes


def draw_boxes_on_binary(binary_img, boxes):
    # переводим в BGR для цветных рамок
    img_color = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in boxes:
        # бирюзовый: BGR = (255, 255, 0), толщина = 4
        cv2.rectangle(img_color, (x, y), (x + w, y + h), (255, 255, 0), 4)
    return img_color


def main():
    img_bgr = cv2.imread(IMAGE_PATH)
    if img_bgr is None:
        print("Не удалось открыть изображение:", IMAGE_PATH)
        return

    # предобработка
    binary = preprocess_image(img_bgr, show_steps=True)

    # СОХРАНЯЕМ последнее ЧБ-изображение (без bounding boxes)
    cv2.imwrite(BINARY_OUT_PATH, binary)
    print(f"ЧБ-изображение сохранено в файл: {BINARY_OUT_PATH}")

    # поиск боксов
    boxes = get_character_boxes(binary)

    print("Найдено боксов:", len(boxes))
    for i, (x, y, w, h) in enumerate(boxes, start=1):
        print(f"{i}: x={x}, y={y}, w={w}, h={h}")

    # отрисовка рамок
    binary_with_boxes = draw_boxes_on_binary(binary, boxes)

    plt.figure(figsize=(8, 8))
    plt.title("Инвертированное бинарное изображение с бирюзовыми bounding boxes (толщина 4)")
    plt.imshow(cv2.cvtColor(binary_with_boxes, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
