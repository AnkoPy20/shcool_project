import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# 1. Подготовка данных
# Добавляем аугментацию (случайные повороты и отражения), это критично для медицинских фото
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), # Отражение (болезнь может быть с любой стороны)
    transforms.RandomRotation(15),     # Повороты
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. Загрузка данных
# Используем стандартный ImageFolder, он сам поймет названия классов по папкам
def prepare_data(data_dir):
    dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataset, loader

# 3. Создание модели
class DiseaseClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DiseaseClassifier, self).__init__()
        # Используем ResNet18 (актуально и в 2026 для быстрых тестов)
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        n_features = self.model.fc.in_features
        # Заменяем последний слой под количество наших болезней
        self.model.fc = nn.Linear(n_features, num_classes)

    def forward(self, x):
        return self.model(x)

# 4. Цикл обучения
def train_model():
    # Путь к папке, где лежат подпапки с болезнями
    data_dir = "C:\Users\Anton\Desktop\project\dataset"
    
    if not os.path.exists(data_dir):
        print(f"Ошибка: Создайте папку {data_dir} и положите в неё фото по категориям.")
        return

    dataset, dataloader = prepare_data(data_dir)
    num_classes = len(dataset.classes)
    print(f"Обнаружено классов: {num_classes} ({dataset.classes})")

    model = DiseaseClassifier(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # Снизил lr для дообучения

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Обучение запущено на {device}...")
    model.train()

    for epoch in range(15): # 15 эпох обычно хватает для начала
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # Считаем точность на ходу
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total
        print(f"Эпоха {epoch+1} | Ошибка: {running_loss/len(dataloader):.4f} | Точность: {accuracy:.2f}%")

    # Сохраняем модель и список классов
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': dataset.classes
    }, "animal_disease_model_2026.pth")
    print("Модель успешно обучена и сохранена!")

if __name__ == '__main__':
    train_model()
