import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import joblib
import os

#  Настройки страницы 2026 
st.set_page_config(page_title="AI Ветеринар", page_icon="🐾")

#  Архитектуры моделей (твои из проекта) 
class DiseaseClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18()
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, num_classes)
    def forward(self, x): 
        return self.model(x)

class TextNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, num_classes)
        )
    def forward(self, x): 
        return self.model(x)

#  Загрузка моделей с поддержкой сжатия (float16 -> float32) 
@st.cache_resource
def load_resources():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = {"v_model": None, "t_model": None, "vec": None, "v_classes": [], "t_classes": []}
    
    # Пути к файлам (используем сжатую модель)
    path_v = "animal_disease_model_2026_compressed.pth" 
    path_t = "text_disease_model_2026.pth"
    path_vec = "vectorizer.pkl"

    # Загрузка фото-модели
    if os.path.exists(path_v):
        ckpt = torch.load(path_v, map_location=DEVICE)
        res["v_classes"] = ckpt.get('classes', [])
        res["v_model"] = DiseaseClassifier(len(res["v_classes"])).to(DEVICE)
        
        # РАСПАКОВКА ВЕСОВ (важно для сжатого файла!)
        state_dict = ckpt['model_state_dict']
        for key in state_dict:
            state_dict[key] = state_dict[key].to(torch.float32)
            
        res["v_model"].load_state_dict(state_dict)
        res["v_model"].eval()

    # Загрузка текстовой модели
    if os.path.exists(path_t) and os.path.exists(path_vec):
        res["vec"] = joblib.load(path_vec)
        t_ckpt = torch.load(path_t, map_location=DEVICE)
        res["t_classes"] = t_ckpt.get('classes', [])
        res["t_model"] = TextNet(t_ckpt.get('input_dim', 0), len(res["t_classes"])).to(DEVICE)
        res["t_model"].load_state_dict(t_ckpt['model_state_dict'])
        res["t_model"].eval()
    
    return res, DEVICE

# --- Логика вывода результатов ---
def show_result(raw_name):
    try:
        from treatments import get_treatment_protocol
        protocol = get_treatment_protocol(raw_name)
    except:
        protocol = "⚠️ Протокол лечения не найден. Обратитесь к врачу!"

    display_name = raw_name.replace("PetImage", "").replace("_", " ").strip().upper()
    
    st.markdown("---")
    st.error("⚠️ ВНИМАНИЕ: ЭТО ТОЛЬКО РЕКОМЕНДАЦИЯ! ОБРАТИТЕСЬ К ВЕТЕРИНАРУ.")
    st.subheader(f"Результат: {display_name}")
    st.info(protocol)

# --- Интерфейс ---
def main():
    st.title("🐾 AI Ветеринарный Помощник 2026")
    res, DEVICE = load_resources()

    tab1, tab2 = st.tabs(["📸 По фото", "📝 По симптомам"])

    with tab1:
        file = st.file_uploader("Загрузите фото пораженного участка", type=["jpg", "png", "jpeg"])
        if file and res["v_model"]:
            img = Image.open(file).convert('RGB')
            st.image(img, width=300)
            if st.button("Анализ по фото"):
                tf = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                tensor = tf(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    out = res["v_model"](tensor)
                    _, idx = torch.max(out, 1)
                    show_result(res["v_classes"][idx.item()])

    with tab2:
        txt = st.text_area("Опишите, что беспокоит животное...")
        if st.button("Анализ текста") and txt:
            if res["t_model"] and res["vec"]:
                vec = res["vec"].transform([txt]).toarray()
                tensor = torch.FloatTensor(vec).to(DEVICE)
                with torch.no_grad():
                    out = res["t_model"](tensor)
                    _, idx = torch.max(out, 1)
                    show_result(res["t_classes"][idx.item()])

if __name__ == "__main__":
    main()
