import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_PATH = "page.jpg"
MIN_SIZE_THRESHOLD = 10

def preprocess_image(img_bgr, show_steps=True):
    # 1. В оттенки серого
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Убираем шум
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # 3. Повышаем контраст (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray_blur)

    # 4. Бинаризация (Otsu)
    _, binary = cv2.threshold(
        gray_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 5. Инверсия, чтобы символы были светлее фона
    if np.mean(binary) > 127:
        binary = 255 - binary

    # 6. Морфология для удаления мелкого шума
    kernel = np.ones((2, 2), np.uint8)
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    if show_steps:
        plt.figure(figsize=(10, 8))

        plt.subplot(2, 2, 1)
        plt.title("Серое + blur")
        plt.imshow(gray_blur, cmap="gray")
        plt.axis("off")

        plt.subplot(2, 2, 2)
        plt.title("После CLAHE (контраст)")
        plt.imshow(gray_clahe, cmap="gray")
        plt.axis("off")

        plt.subplot(2, 2, 3)
        plt.title("Бинаризация (Otsu)")
        plt.imshow(binary, cmap="gray")
        plt.axis("off")

        plt.subplot(2, 2, 4)
        plt.title("После морфологии (очистка)")
        plt.imshow(binary_clean, cmap="gray")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    return binary_clean


def get_character_boxes(binary_img):
    contours, _ = cv2.findContours(
        binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # отбрасываем слишком маленькие объекты (мусор)
        if w * h < MIN_SIZE_THRESHOLD:
            continue

        boxes.append((x, y, w, h))

    # сортировка: сверху вниз, слева направо
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    return boxes


def draw_boxes(img_bgr, boxes):
    img_out = img_bgr.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(img_out, (x, y), (x + w, y + h), (0, 0, 255), 3)
    return img_out


def main():
    # 1. Читаем изображение из константы
    img_bgr = cv2.imread(IMAGE_PATH)
    if img_bgr is None:
        print("Не удалось открыть изображение:", IMAGE_PATH)
        return

    # Показываем исходное изображение
    plt.figure(figsize=(6, 6))
    plt.title("Исходное изображение")
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    # 2. Предобработка (с промежуточными шагами)
    binary = preprocess_image(img_bgr, show_steps=True)

    # 3. Находим bounding boxes
    boxes = get_character_boxes(binary)

    # 4. Печатаем координаты
    for i, (x, y, w, h) in enumerate(boxes, start=1):
        print(f"{i}: x={x}, y={y}, w={w}, h={h}")

    # 5. Рисуем bounding boxes и показываем изображение
    img_with_boxes = draw_boxes(img_bgr, boxes)

    plt.figure(figsize=(8, 8))
    plt.title("Символы с bounding boxes")
    plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
