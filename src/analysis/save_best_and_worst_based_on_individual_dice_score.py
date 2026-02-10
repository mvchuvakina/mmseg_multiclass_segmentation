import os
import numpy as np
from PIL import Image

def compute_mDice(pred, target, num_classes=3):
    """Вычисление mDice для одного изображения."""
    dice_scores = []
    for cls in range(num_classes):
        intersection = np.sum((pred == cls) & (target == cls))
        union = np.sum((pred == cls) | (target == cls))
        dice_score = 2. * intersection / (union + intersection + 1e-6)
        dice_scores.append(dice_score)
    return np.mean(dice_scores)

def save_best_and_worst_samples(predictions_dir, ground_truth_dir, save_best_dir='best_samples', save_worst_dir='worst_samples'):
    """Сохранение лучших и худших примеров на основе mDice."""
    
    os.makedirs(save_best_dir, exist_ok=True)
    os.makedirs(save_worst_dir, exist_ok=True)
    
    pred_files = sorted(os.listdir(predictions_dir))
    gt_files = sorted(os.listdir(ground_truth_dir))

    dice_scores = []
    for pred_file, gt_file in zip(pred_files, gt_files):
        pred = np.array(Image.open(os.path.join(predictions_dir, pred_file)))
        gt = np.array(Image.open(os.path.join(ground_truth_dir, gt_file)))
        
        # Вычисляем mDice для каждого изображения
        dice_score = compute_mDice(pred, gt)
        dice_scores.append((dice_score, pred_file, gt_file))

    # Сортируем по mDice
    dice_scores.sort(key=lambda x: x[0], reverse=True)

    # Сохраняем лучшие и худшие примеры
    for score, pred_file, gt_file in dice_scores[:5]:
        pred = np.array(Image.open(os.path.join(predictions_dir, pred_file)))
        gt = np.array(Image.open(os.path.join(ground_truth_dir, gt_file)))
        
        pred_image = Image.fromarray(pred.astype(np.uint8))
        gt_image = Image.fromarray(gt.astype(np.uint8))
        
        pred_image.save(os.path.join(save_best_dir, f"best_{pred_file}"))
        gt_image.save(os.path.join(save_best_dir, f"best_{gt_file}"))

    for score, pred_file, gt_file in dice_scores[-5:]:
        pred = np.array(Image.open(os.path.join(predictions_dir, pred_file)))
        gt = np.array(Image.open(os.path.join(ground_truth_dir, gt_file)))
        
        pred_image = Image.fromarray(pred.astype(np.uint8))
        gt_image = Image.fromarray(gt.astype(np.uint8))
        
        pred_image.save(os.path.join(save_worst_dir, f"worst_{pred_file}"))
        gt_image.save(os.path.join(save_worst_dir, f"worst_{gt_file}"))
