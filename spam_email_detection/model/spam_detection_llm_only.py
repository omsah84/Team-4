import os
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# ------------------ Config ------------------
MODEL_NAME = "distilbert-base-uncased"
NUM_EPOCHS = 3
BATCH_SIZE = 16
MAX_LENGTH = 128
LEARNING_RATE = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ Load Enron Dataset ------------------
def load_enron_dataset(base_path):
    data = []
    for label in ['ham', 'spam']:
        folder = os.path.join(base_path, label)
        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            with open(path, encoding="latin-1", errors="ignore") as f:
                content = f.read()
                data.append({"label": 0 if label == "ham" else 1, "text": content})
    return pd.DataFrame(data)

# ------------------ Dataset Class ------------------
class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ------------------ Model Loader ------------------
def create_model():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    return model.to(DEVICE)

# ------------------ Train Function ------------------
def train_model(model, train_loader, val_loader):
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

    best_acc = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            model.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        val_acc, val_f1 = evaluate_model(model, val_loader)
        print(f"Epoch {epoch+1} | Train Loss: {total_loss:.4f} | Val Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            model.save_pretrained("best_email_spam_model")
            print("✅ New best model saved!\n")

# ------------------ Evaluation ------------------
def evaluate_model(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels_batch = batch["labels"].to(DEVICE)

            output = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = torch.argmax(output.logits, dim=1)

            preds.extend(pred.cpu().numpy())
            labels.extend(labels_batch.cpu().numpy())

    return accuracy_score(labels, preds), f1_score(labels, preds)

# ------------------ Prediction ------------------
class SpamEmailClassifier:
    def __init__(self, model_path="best_email_spam_model"):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(DEVICE)
        self.model.eval()

    def predict(self, text):
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True
        )
        with torch.no_grad():
            input_ids = encoding["input_ids"].to(DEVICE)
            attention_mask = encoding["attention_mask"].to(DEVICE)
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(output.logits, dim=1)
            spam_score = probs[0][1].item()
            return "spam" if spam_score > 0.5 else "ham", spam_score

# ------------------ Main ------------------
if __name__ == "__main__":
    print("📥 Loading Enron dataset...")
    df = load_enron_dataset("./enron1/enron1")
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle

    print(f"✅ Total Emails Loaded: {len(df)}")

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = EmailDataset(train_df["text"], train_df["label"], tokenizer, MAX_LENGTH)
    val_ds = EmailDataset(val_df["text"], val_df["label"], tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = create_model()
    print("🚀 Training model...")
    train_model(model, train_loader, val_loader)

    # Load & test
    classifier = SpamEmailClassifier()

    test_emails = [
        "URGENT: Your email account has been compromised. Please reset your password immediately by clicking this link.",
        "Hi John, just following up on the budget proposal draft. Let me know your feedback.",
        "You’ve been selected for a $1000 gift card! Click now to claim.",
        "Meeting moved to 3PM today. Please bring the slides."
    ]

    print("\n📨 Testing new emails...\n")
    for email in test_emails:
        label, prob = classifier.predict(email)
        print(f"Text: {email}\nPrediction: {label.upper()} | Spam Probability: {prob:.4f}\n")
