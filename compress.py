import torch
import os

# 1. Загружаем исходный файл
checkpoint = torch.load("animal_disease_model_2026.pth", map_location="cpu")

# 2. Проверяем, что внутри. 
# Если это словарь с 'model_state_dict', работаем с ним
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
    # Переводим веса в float16 (половинная точность)
    for key in state_dict:
        state_dict[key] = state_dict[key].to(torch.float16)
    checkpoint['model_state_dict'] = state_dict
else:
    # Если это просто веса без словаря
    for key in checkpoint:
        checkpoint[key] = checkpoint[key].to(torch.float16)

# 3. Сохраняем сжатую версию
output_path = "animal_disease_model_2026_compressed.pth"
torch.save(checkpoint, output_path, _use_new_zipfile_serialization=True)

size = os.path.getsize(output_path) / (1024 * 1024)
print(f"Готово! Новый размер: {size:.2f} МБ")
