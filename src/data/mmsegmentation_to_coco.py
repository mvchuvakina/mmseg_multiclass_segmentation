import os
import json
import numpy as np
from PIL import Image
from pycocotools import mask
import cv2

def mmseg_to_coco(image_dir, mask_dir, output_json_path):
    """
    Конвертация данных из формата mmsegmentation в формат COCO.
    :param image_dir: Путь к папке с изображениями.
    :param mask_dir: Путь к папке с масками.
    :param output_json_path: Путь к выходному JSON файлу.
    """
    images = []
    annotations = []
    categories = [{'id': 1, 'name': 'class1'}, {'id': 2, 'name': 'class2'}, {'id': 0, 'name': 'background'}]  # Пример для двух классов

    image_id = 0
    annotation_id = 0

    for image_name in os.listdir(image_dir):
        if not image_name.endswith(".jpg"):
            continue

        image_path = os.path.join(image_dir, image_name)
        mask_path = os.path.join(mask_dir, image_name.replace(".jpg", ".png"))
        
        image = Image.open(image_path)
        width, height = image.size
        
        # Добавляем информацию о изображении
        image_info = {
            'id': image_id,
            'file_name': image_name,
            'width': width,
            'height': height
        }
        images.append(image_info)
        
        # Чтение маски и аннотирование
        mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        for class_id in np.unique(mask_image):
            if class_id == 0:  # Пропускаем фон
                continue

            # Получаем контуры для каждого класса
            binary_mask = np.uint8(mask_image == class_id)
            rle = mask.encode(np.asfortranarray(binary_mask))  # Используем pycocotools для RLE

            # Аннотация для текущего класса
            annotation = {
                'id': annotation_id,
                'image_id': image_id,
                'category_id': class_id,
                'segmentation': rle,
                'area': int(mask.area(rle)),
                'bbox': list(cv2.boundingRect(binary_mask)),  # Бокс для аннотации
                'iscrowd': 0
            }
            annotations.append(annotation)
            annotation_id += 1
        
        image_id += 1

    coco_format = {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }

    # Сохранение в JSON
    with open(output_json_path, 'w') as json_file:
        json.dump(coco_format, json_file)

if __name__ == "__main__":
    image_dir = "/Users/maria.chuvakinamail.ru/Desktop/mmseg_multiclass_project/data/raw/train_dataset_for_students/img/train"  # Путь к изображениям
    mask_dir =
    "/Users/maria.chuvakinamail.ru/Desktop/mmseg_multiclass_project/data/raw/train_dataset_for_students/labels/train"  # Путь     # Путь к маскам
    output_json_path = "/Users/maria.chuvakinamail.ru/Desktop/mmseg_multiclass_project/output/coco_annotations.json"  # Путь к файлу для сохранения COCO-формата
    mmseg_to_coco(image_dir, mask_dir, output_json_path)
