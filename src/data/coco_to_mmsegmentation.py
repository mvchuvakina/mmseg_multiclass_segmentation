import os
import json
import numpy as np
from pycocotools.coco import COCO
from PIL import Image
import cv2

def coco_to_mmseg(coco_json_path, output_image_dir, output_mask_dir):
    """
    Конвертация данных из формата COCO в формат mmsegmentation.
    :param coco_json_path: Путь к JSON файлу с аннотациями COCO.
    :param output_image_dir: Папка для сохранения изображений.
    :param output_mask_dir: Папка для сохранения масок.
    """
    # Загрузка COCO аннотаций
    coco = COCO(coco_json_path)
    
    # Создаем папки для масок, если их нет
    os.makedirs(output_mask_dir, exist_ok=True)
    
    # Получаем все изображения из аннотаций
    img_ids = coco.getImgIds()
    
    for img_id in img_ids:
        # Получаем информацию об изображении
        img_info = coco.loadImgs(img_id)[0]
        img_name = img_info['file_name']
        img_path = os.path.join(output_image_dir, img_name)
        
        # Получаем аннотации для этого изображения
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # Создаем пустую маску
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        
        for ann in anns:
            # Получаем RLE маску для аннотации
            rle = ann['segmentation']
            class_id = ann['category_id']
            mask_rle = coco.annToMask(ann)
            
            # Записываем маску
            mask = np.maximum(mask, mask_rle * class_id)  # Обновляем маску для каждого класса
        
        # Сохраняем маску
        mask_image = Image.fromarray(mask)
        mask_image.save(os.path.join(output_mask_dir, img_name.replace('.jpg', '.png')))

if __name__ == "__main__":
    coco_json_path = "/Users/maria.chuvakinamail.ru/Desktop/mmseg_multiclass_project/output/coco_annotations.json"  # Путь к COCO JSON файлу
    output_image_dir = "/Users/maria.chuvakinamail.ru/Desktop/mmseg_multiclass_project/data/clean/img"  # Папка для изображений
    output_mask_dir = "/Users/maria.chuvakinamail.ru/Desktop/mmseg_multiclass_project/data/clean/labels"  # Папка для масок

    coco_to_mmseg(coco_json_path, output_image_dir, output_mask_dir)
