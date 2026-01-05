import cv2 as cv

# Constants for constant config
OUT_DIR = "extracted"
CROP_RATIO = 0.08
MIN_LINE_WIDTH = 500
BLOCK_MORPH_KERNEL_DIAMETER = 10
LINE_MORPH_KERNEL_DIAMETER = 3
CHAR_MORPH_KERNEL_DIAMETER = 3
MIN_BLOCK_HEIGHT = 1000
MIN_BLOCK_WIDTH = 1000
BLOCK_KERNEL_WIDTH = 100
BLOCK_KERNEL_HEIGHT = 100 

# Constants for dynamic image config
BLOCK_MORPH_KERNEL_DIAMETER_RATIO = 0.035
ONE_PAGE_RATIO = 0.5
TWO_PAGE_RATIO = 0.25
BLOCK_KERNEL_RATIO = 0.6

class Config:
    def __init__(
        self,
        out_dir,
        crop_ratio,
        min_line_width,
        block_morph_kernel_diameter,
        line_morph_kernel_diameter,
        char_morph_kernel_diameter,
        block_kernel_width,
        block_kernel_height,
        min_block_height,
        min_block_width
    ):
        self.out_dir = out_dir

        self.crop_ratio = crop_ratio
        
        self.block_morph_kernel_diameter = block_morph_kernel_diameter
        self.line_morph_kernel_diameter = line_morph_kernel_diameter
        self.char_morph_kernel_diameter = char_morph_kernel_diameter

        self.min_block_height = min_block_height
        self.min_block_width = min_block_width
        self.block_kernel_width = block_kernel_width
        self.block_kernel_height = block_kernel_height

        self.min_line_width = min_line_width

class ConfigManager:
    @staticmethod
    def get_constant_config():
        return Config(
            OUT_DIR,
            CROP_RATIO,
            MIN_LINE_WIDTH,
            BLOCK_MORPH_KERNEL_DIAMETER,
            LINE_MORPH_KERNEL_DIAMETER,
            CHAR_MORPH_KERNEL_DIAMETER,
            BLOCK_KERNEL_WIDTH,
            BLOCK_KERNEL_HEIGHT,
            MIN_BLOCK_HEIGHT,
            MIN_BLOCK_WIDTH,
        )

    @staticmethod
    def get_config_by_image(img):
        config = ConfigManager.get_constant_config()
        h, w = img.shape[:2]
        cropped_h, cropped_w = int(h * config.crop_ratio), int(w * config.crop_ratio)

        if cropped_h > cropped_w:
            ratio = ONE_PAGE_RATIO
        else:
            ratio = TWO_PAGE_RATIO

        config.block_morph_kernel_diameter = int(cropped_h * BLOCK_MORPH_KERNEL_DIAMETER_RATIO)

        config.min_block_height = int(cropped_h * ratio)
        config.min_block_width = int(cropped_w * ratio)
        config.block_kernel_height = int(cropped_h * BLOCK_KERNEL_RATIO * ratio)
        config.block_kernel_width = int(cropped_w * BLOCK_KERNEL_RATIO * ratio)

        config.min_line_width = config.min_block_width

        return config
