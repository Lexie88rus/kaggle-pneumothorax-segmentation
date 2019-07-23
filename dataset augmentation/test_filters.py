import numpy as np

from PIL import Image
from PIL import ImageEnhance

def main():
    # read png file with image
    image = Image.open('../xray.png')
    image = image.convert('RGB')

    image.save('image_original.png')

    # create sharpened image
    enhancer = ImageEnhance.Sharpness(image)
    factor = 5
    enhanced = enhancer.enhance(factor)
    enhanced.save('xray_sharpened_' + str(factor) + '.png')

    # color adjustment
    enhancer_color = ImageEnhance.Color(image)
    factor = 5
    enhanced = enhancer_color.enhance(factor)
    enhanced.save('xray_color_' + str(factor) + '.png')

    # color adjustment
    enhancer_constrast = ImageEnhance.Contrast(image)
    factor = 1.2
    enhanced = enhancer_constrast.enhance(factor)
    enhanced.save('xray_contrast_' + str(factor) + '.png')

if __name__ == '__main__':
    main()
