import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, Dataset

DATA_DIR = r"C:\Users\MSI\Desktop\project\dataset2"
FILES = ["cleaned_animal_disease_prediction.csv", "pet-health-symptoms-dataset.csv", "veterinary_clinical_data.csv"]
MODEL_SAVE_PATH = "text_disease_model_2026.pth"
VECTORIZER_PATH = "vectorizer.pkl"

def load_and_fix_data():
    all_texts, all_labels = [], []
    for file_name in FILES:
        path = os.path.join(DATA_DIR, file_name)
        if not os.path.exists(path): continue
        df = pd.read_csv(path)
        t_col = next((c for c in df.columns if any(x in c.lower() for x in ['symptom', 'text', 'description'])), None)
        l_col = next((c for c in df.columns if any(x in c.lower() for x in ['disease', 'label', 'diagnosis'])), None)
        if t_col and l_col:
            temp_df = df[[t_col, l_col]].dropna()
            all_texts.extend(temp_df[t_col].astype(str).str.strip().tolist())
            all_labels.extend(temp_df[l_col].astype(str).str.strip().tolist())
    return pd.DataFrame({'text': all_texts, 'label': all_labels})

class VeterinaryDataset(Dataset):
    def __init__(self, df):
        self.texts = df['text'].values
        # ГАРАНТИРУЕМ ПОРЯДОК: Сначала Healthy, потом болезни по алфавиту
        self.classes = sorted(list(set(df['label'])))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.labels = [self.class_to_idx[label] for label in df['label']]
        self.vectorizer = TfidfVectorizer(max_features=2000, token_pattern=r"(?u)\b\w+\b")
        self.X = self.vectorizer.fit_transform(self.texts).toarray()

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

class TextNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU(), nn.Linear(512, num_classes))
    def forward(self, x): return self.net(x)

def train():
    df = load_and_fix_data()
    dataset = VeterinaryDataset(df)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = TextNet(dataset.X.shape[1], len(dataset.classes))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(30):
        for texts, labels in dataloader:
            optimizer.zero_grad()
            criterion(model(texts), labels).backward()
            optimizer.step()
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': dataset.classes,
        'input_dim': dataset.X.shape[1]
    }, MODEL_SAVE_PATH)
    joblib.dump(dataset.vectorizer, VECTORIZER_PATH)
    print(f"✅ Текстовая модель обучена. Классы: {dataset.classes}")

if __name__ == "__main__": train()

