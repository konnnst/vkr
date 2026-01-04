import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import matplotlib.pyplot as plt

# !!! НУЖЕН tesseract-ocr !!!
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

IMAGE_PATH = "page.jpg"
UPSCALE_FACTOR = 2.0   # во сколько раз увеличиваем изображение


# ---------------------- PREPROCESSING ---------------------- #

def choose_best_binary(bin1, bin2, img_area):
    """
    Выбираем между bin1 и bin2 ту бинаризацию, где доля "чернил" (меньшинства пикселей)
    выглядит более правдоподобной (обычно 1–40% площади).
    """
    def score_binary(b):
        white = np.sum(b == 255)
        black = img_area - white
        minority = min(white, black)
        frac = minority / img_area  # доля предполагаемого текста
        # хотим около 0.05–0.2, но допустим 0.01–0.5
        if frac < 0.01 or frac > 0.5:
            return 0.0
        # чем ближе к 0.1, тем лучше
        return 1.0 - abs(frac - 0.1)

    s1 = score_binary(bin1)
    s2 = score_binary(bin2)
    return bin1 if s1 >= s2 else bin2


def show_preprocess_steps(
    img_bgr,
    img_up,
    gray,
    gray_blur,
    gray_clahe,
    binary_raw,
    binary_inv,
    binary_chosen,
    binary_clean,
):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_up_rgb = cv2.cvtColor(img_up, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(16, 10))

    plt.subplot(2, 5, 1)
    plt.title("Оригинал")
    plt.imshow(img_rgb)
    plt.axis("off")

    plt.subplot(2, 5, 2)
    plt.title("Upscaled")
    plt.imshow(img_up_rgb)
    plt.axis("off")

    plt.subplot(2, 5, 3)
    plt.title("Gray")
    plt.imshow(gray, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 5, 4)
    plt.title("Gray blur")
    plt.imshow(gray_blur, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 5, 5)
    plt.title("CLAHE")
    plt.imshow(gray_clahe, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 5, 6)
    plt.title("Binary (Otsu)")
    plt.imshow(binary_raw, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 5, 7)
    plt.title("Binary inverted")
    plt.imshow(binary_inv, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 5, 8)
    plt.title("Chosen binary")
    plt.imshow(binary_chosen, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 5, 9)
    plt.title("Morph CLOSE (clean)")
    plt.imshow(binary_clean, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def preprocess_image(img_bgr, show_steps=True):
    """
    Препроцессинг по заданной схеме:
    0. Upscale
    1. Gray
    2. Gaussian blur
    3. CLAHE
    4. Otsu binarization
    5. Выбор лучшей версии (raw / inverted)
    6. Morph CLOSE
    7. Гарантируем: текст белый, фон чёрный (white-on-black)
    """
    # 0. Upscale
    img_up = cv2.resize(
        img_bgr,
        None,
        fx=UPSCALE_FACTOR,
        fy=UPSCALE_FACTOR,
        interpolation=cv2.INTER_CUBIC,
    )

    # 1. Gray
    gray = cv2.cvtColor(img_up, cv2.COLOR_BGR2GRAY)

    # 2. Blur
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # 3. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray_blur)

    # 4. Otsu
    _, binary_raw = cv2.threshold(
        gray_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    h, w = binary_raw.shape[:2]
    img_area = h * w
    binary_inv = 255 - binary_raw

    # 5. Выбор лучшей бинаризации
    binary_chosen = choose_best_binary(binary_raw, binary_inv, img_area)

    # 6. Лёгкая морфология: CLOSE соединяет разорванные штрихи
    kernel = np.ones((2, 2), np.uint8)
    binary_clean = cv2.morphologyEx(
        binary_chosen, cv2.MORPH_CLOSE, kernel, iterations=1
    )

    # 7. Гарантируем: текст – меньшинство пикселей и он белый на чёрном фоне
    white_pixels = np.sum(binary_clean == 255)
    black_pixels = np.sum(binary_clean == 0)
    if white_pixels > black_pixels:
        binary_clean = 255 - binary_clean

    if show_steps:
        show_preprocess_steps(
            img_bgr,
            img_up,
            gray,
            gray_blur,
            gray_clahe,
            binary_raw,
            binary_inv,
            binary_chosen,
            binary_clean,
        )

    # Для OCR Tesseract лучше чёрный текст на белом -> инвертируем
    bin_for_ocr = 255 - binary_clean  # текст чёрный, фон белый

    return img_up, bin_for_ocr


# ---------------------- OCR / LAYOUT ---------------------- #

def detect_layout(img_bin, lang="rus+eng"):
    """
    Структура страницы через Tesseract:
    - блоки и строки: image_to_data (уровни 2 и 4)
    - символы: image_to_boxes
    """
    data = pytesseract.image_to_data(
        img_bin,
        lang=lang,
        output_type=Output.DICT
    )
    boxes_str = pytesseract.image_to_boxes(img_bin, lang=lang)
    return data, boxes_str


def draw_structure(orig_bgr_up, img_bin, data, boxes_str):
    """
    Рисует:
    - блоки (зелёным)
    - строки (синим)
    - символы (красным)
    """
    img = orig_bgr_up.copy()
    h_img, w_img = img_bin.shape[:2]

    # Блоки и строки
    n = len(data["level"])
    for i in range(n):
        level = data["level"][i]
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]

        if level == 2:      # блок
            color = (0, 255, 0)   # зелёный
            thickness = 3
        elif level == 4:    # строка
            color = (255, 0, 0)   # синий (BGR)
            thickness = 2
        else:
            continue

        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

    # Символы
    for line in boxes_str.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        ch, x1, y1, x2, y2 = parts[0], int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])

        # У Tesseract начало координат внизу слева; перевод в OpenCV (сверху слева)
        y1_new = h_img - y1
        y2_new = h_img - y2

        cv2.rectangle(img, (x1, y2_new), (x2, y1_new), (0, 0, 255), 1)  # красный

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 16))
    plt.title("Блоки (зелёный), строки (синий), символы (красный)")
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()


def extract_char_bboxes(boxes_str, img_height):
    """
    Возвращает список bounding boxes символов:
    [{"char": 'А', "x1":..., "y1":..., "x2":..., "y2":...}, ...]
    """
    chars = []
    for line in boxes_str.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        ch, x1, y1, x2, y2 = parts[0], int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
        y1_new = img_height - y1
        y2_new = img_height - y2
        chars.append({
            "char": ch,
            "x1": x1,
            "y1": y2_new,  # верх
            "x2": x2,
            "y2": y1_new,  # низ
        })
    return chars


# -------------------------- MAIN -------------------------- #

def main():
    img_bgr = cv2.imread(IMAGE_PATH)
    if img_bgr is None:
        raise FileNotFoundError(f"Не удалось открыть изображение: {IMAGE_PATH}")

    # 1–2. Препроцессинг по заданной схеме + показ промежуточных стадий
    img_up, img_bin = preprocess_image(img_bgr, show_steps=True)

    # 3–5. Обнаруживаем блоки, строки, символы и bounding boxes
    data, boxes_str = detect_layout(img_bin, lang="rus+eng")

    # Пример: список bbox символов
    char_bboxes = extract_char_bboxes(boxes_str, img_height=img_bin.shape[0])
    for ch_info in char_bboxes[:10]:
        print(ch_info)

    # 6. Рисуем структуру
    draw_structure(img_up, img_bin, data, boxes_str)


if __name__ == "__main__":
    main()
