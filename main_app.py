import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, scrolledtext
import joblib
import os
import re

# --- 1. Подключение базы знаний ---
try:
    from treatments import get_treatment_protocol
except ImportError:
    def get_treatment_protocol(n): 
        return "Error: Файл treatments.py не найден в папке проекта."   

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 2. Архитектуры нейросетей (ResNet18 для фото и MLP для текста) ---

class DiseaseClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18()
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, num_classes)
    def forward(self, x): return self.model(x)

class TextNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.model(x)

# --- 3. Функция загрузки всех ресурсов 2026 ---

def load_all_resources():
    print("Загрузка моделей и векторизатора...")
    res = {"v_model": None, "t_model": None, "vec": None, "v_classes": [], "t_classes": []}
    
    path_v = os.path.join(BASE_DIR, "animal_disease_model_2026.pth")
    path_t = os.path.join(BASE_DIR, "text_disease_model_2026.pth")
    path_vec = os.path.join(BASE_DIR, "vectorizer.pkl")

    # Загрузка фото-модели
    if os.path.exists(path_v):
        try:
            ckpt = torch.load(path_v, map_location=DEVICE)
            res["v_classes"] = ckpt.get('classes', [])
            res["v_model"] = DiseaseClassifier(len(res["v_classes"])).to(DEVICE)
            res["v_model"].load_state_dict(ckpt['model_state_dict'])
            res["v_model"].eval()
            print("Фото-модель: Готова.")
        except Exception as e: print(f"Ошибка фото-модели: {e}")

    # Загрузка текстовой модели
    if os.path.exists(path_t) and os.path.exists(path_vec):
        try:
            res["vec"] = joblib.load(path_vec)
            t_ckpt = torch.load(path_t, map_location=DEVICE)
            res["t_classes"] = t_ckpt.get('classes', [])
            input_dim = t_ckpt.get('input_dim', 0)
            res["t_model"] = TextNet(input_dim, len(res["t_classes"])).to(DEVICE)
            res["t_model"].load_state_dict(t_ckpt['model_state_dict'])
            res["t_model"].eval()
            print("Текстовая модель: Готова.")
        except Exception as e: print(f"Ошибка текстовой модели: {e}")
    
    return res

# --- 4. Основной графический интерфейс ---

class VeterinaryApp(tk.Tk):
    def __init__(self, res):
        super().__init__()
        self.res = res
        self.title("AI Ветеринарный Помощник 2026")
        self.geometry("900x800")
        self.configure(bg="#f4f7f6")
        self.create_ui()

    def create_ui(self):
        # Заголовок
        tk.Label(self, text="Предварительная ИИ-диагностика", font=("Arial", 16, "bold"), bg="#f4f7f6", fg="#2c3e50").pack(pady=15)

        # Панель управления
        ctrl = tk.Frame(self, bg="#f4f7f6")
        ctrl.pack(pady=10, fill=tk.X, padx=20) 
        
        tk.Button(ctrl, text="Загрузить Фото", command=self.do_photo, bg="#27ae60", fg="white", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=10)

        self.ent = tk.Entry(ctrl, width=45, font=("Arial", 12))
        self.ent.pack(side=tk.LEFT, padx=10, ipady=3)
        self.ent.insert(0, "Опишите симптомы текстом...")

        tk.Button(ctrl, text="🔍 Анализ текста", command=self.do_text, bg="#2980b9", fg="white", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=10)

        # Область для фото
        self.canvas = tk.Label(self, text="Изображение не загружено", bg="#dcdde1", width=60, height=12)
        self.canvas.pack(pady=15)

        # Окно вывода результатов
        self.output = scrolledtext.ScrolledText(self, width=100, height=25, font=("Arial", 11), bg="white")
        self.output.pack(pady=10, padx=20)

    def do_photo(self):
        path = filedialog.askopenfilename()
        if not path or not self.res["v_model"]: return
        
        img = Image.open(path).convert('RGB')
        self.photo_ref = ImageTk.PhotoImage(img.resize((320, 240)))
        self.canvas.config(image=self.photo_ref, text="")

        # Преобразование для нейросети
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        tensor = tf(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.res["v_model"](tensor)
            _, idx = torch.max(outputs, 1)
            raw_name = self.res["v_classes"][idx.item()]
            self.show_final_result(raw_name)

    def do_text(self):
        txt = self.ent.get().strip()
        if not txt or not self.res["t_model"] or "Опишите" in txt: return
        
        vec = self.res["vec"].transform([txt]).toarray()
        tensor = torch.FloatTensor(vec).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.res["t_model"](tensor)
            _, idx = torch.max(outputs, 1)
            raw_name = self.res["t_classes"][idx.item()]
            self.show_final_result(raw_name)

    def show_final_result(self, raw_name):
        self.output.delete(1.0, tk.END)
        
        # 1. Очистка технического имени для заголовка
        display_name = raw_name.replace("PetImage", "").replace("_", " ").strip()
        if len(display_name) < 2: display_name = "НЕОПРЕДЕЛЕННЫЙ ТИП"
        
        # 2. Вывод обязательного предупреждения в начало
        self.output.insert(tk.END, "⚠️ ВНИМАНИЕ: ЭТО ТОЛЬКО РЕКОМЕНДАЦИЯ! ОБРАТИТЕСЬ К ВРАЧУ.\n")
        self.output.insert(tk.END, f"ПРЕДПОЛАГАЕМАЯ КАТЕГОРИЯ: {display_name.upper()}\n")
        self.output.insert(tk.END, "="*60 + "\n\n")
        
        # 3. Получение расширенного протокола из treatments.py
        # (включая перевод fungal/hotspots и встроенный дисклеймер)
        protocol = get_treatment_protocol(raw_name)
        
        self.output.insert(tk.END, protocol)
        self.output.see(1.0) # Прокрутка к началу

# --- 5. Точка входа ---

if __name__ == "__main__":
    print("обновление системы 11.11.2026")
    resources = load_all_resources() # Имя исправлено, NameError исключен
    app = VeterinaryApp(resources)
    app.mainloop()
