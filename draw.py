import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def get_preprocessed_plot(orig, gray, gray_clahe, binary, binary_inv, blocks_morph, lines_morph, chars_morph, dilated_blocks, dilated_lines):
    imgs = [
        cv.cvtColor(orig, cv.COLOR_BGR2RGB),
        gray,
        gray_clahe,
        binary,
        binary_inv,
        blocks_morph,
        lines_morph,
        chars_morph,
        dilated_blocks,
        dilated_lines,
    ]
    titles = [
        "Обрезанный оригинал",
        "Серое",
        "CLAHE",
        "Бинаризация",
        "Инвертированная бинаризация",
        "Морфология (open, блоки)",
        "Морфология (open, строки)",
        "Морфология (open, графемы)",
        "Дилатация (блоки)",
        "Дилатация (строки)",
    ]

    fig, axes = plt.subplots(3, 4, figsize=(16, 8))
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

    return fig 


def get_boxes_plot(orig, blocks, all_lines, all_chars):
    vis = orig.copy()

    for (x, y, w, h) in blocks:
        cv.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 3)

    for (x, y, w, h) in all_lines:
        cv.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 4)

    for (x, y, w, h) in all_chars:
        cv.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 3)

    fig = plt.figure(figsize=(10, 14))
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

    return fig

