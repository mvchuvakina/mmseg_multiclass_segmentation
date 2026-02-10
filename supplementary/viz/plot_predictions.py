import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def plot_image_and_prediction(image_path, mask_path, pred_path):
    image = np.array(Image.open(image_path))
    mask = np.array(Image.open(mask_path))
    pred = np.array(Image.open(pred_path))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title('Input Image')
    axes[1].imshow(mask)
    axes[1].set_title('True Mask')
    axes[2].imshow(pred)
    axes[2].set_title('Predicted Mask')
    plt.show()

# Пример использования
plot_image_and_prediction('image.png', 'mask.png', 'prediction.png')
