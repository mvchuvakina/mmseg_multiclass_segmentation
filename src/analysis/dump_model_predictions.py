import torch
import os
import numpy as np
from PIL import Image

def save_predictions(model, dataloader, save_dir):
    """Сохранение предсказаний модели на тестовом наборе данных."""
    
    
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()  # Устанавливаем модель в режим инференса
    with torch.no_grad():  # Отключаем градиенты
        for idx, (images, masks) in enumerate(dataloader):
            # Перемещаем данные на устройство (GPU или CPU)
            images = images.to(device)
            output = model(images)  # Получаем предсказания модели
            _, preds = torch.max(output, 1)  # Получаем класс с максимальной вероятностью для каждого пикселя

            # Сохраняем каждое предсказание в папку
            for i in range(images.size(0)):
                pred = preds[i].cpu().numpy()  # Переводим в numpy массив
                pred_image = Image.fromarray(pred.astype(np.uint8))  # Преобразуем в изображение
                pred_image.save(os.path.join(save_dir, f"pred_{idx * len(images) + i}.png"))

            
            for i in range(images.size(0)):
                mask = masks[i].cpu().numpy()  # Преобразуем маску в numpy
                mask_image = Image.fromarray(mask.astype(np.uint8))  # Преобразуем в изображение
                mask_image.save(os.path.join(save_dir, f"mask_{idx * len(images) + i}.png"))
