import pandas as pd
import numpy as np
import re
import joblib
from tqdm import tqdm  # <--- Import tqdm here
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import os

# --- 1. Feature Extractor Class ---
class SpamFeatureExtractor:
    def __init__(self):
        self.urgency_words = ['urgent', 'immediate', 'act now', 'warning']
        self.money_words = ['bank', 'credit', 'cash', 'lottery', 'inheritance', 'offer']
        
    def get_heuristics(self, text):
        # Handle empty or non-string text
        text = str(text) if pd.notnull(text) else ""
        
        uppercase_count = sum(1 for c in text if c.isupper())
        shouting_ratio = uppercase_count / max(len(text), 1)
        dollar_count = text.count('$') + text.count('€') + text.count('£')
        exclamation_count = text.count('!')
        # Regex for URLs
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        url_count = len(urls)
        
        # Simple urgency/money keywords (Binary)
        text_lower = text.lower()
        has_urgency = 1 if any(w in text_lower for w in self.urgency_words) else 0
        has_money = 1 if any(w in text_lower for w in self.money_words) else 0

        return [shouting_ratio, dollar_count, exclamation_count, url_count, len(text), has_urgency, has_money]

    def transform(self, text_series):
        # Applies logic to a pandas Series of text
        return np.array([self.get_heuristics(t) for t in text_series])

# --- 2. Load and Prepare Data ---
print("Loading train_data.csv...")
df = pd.read_csv('data/train_data.csv')
X_raw = df['text']
y = df['label']

# --- 3. Feature Extraction Pipeline ---
print("Extracting features...")

# A. Dense Heuristic Features
extractor = SpamFeatureExtractor()
X_dense = extractor.transform(X_raw)

# Scale dense features (Crucial for SVM)
scaler = StandardScaler()
X_dense_scaled = scaler.fit_transform(X_dense)

# B. Sparse TF-IDF Features
tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
X_tfidf = tfidf.fit_transform(X_raw.astype(str))

# C. Combine
X_final = hstack([X_dense_scaled, X_tfidf])

# --- 4. Create Held-Out Validation Set ---
# Split training data: 80% for Training, 20% for Validation (Hyperparameter Tuning)
X_train, X_val, y_train, y_val = train_test_split(X_final, y, test_size=0.2, random_state=42, stratify=y)

# --- 5. Hyperparameter Tuning (C parameter) ---
print("\n--- Tuning Hyperparameters (C) ---")
c_values = [0.1, 1, 10, 100]
best_score = 0
best_c = 1
best_model = None

validation_scores = []

for c in tqdm(c_values, desc="Grid Search Progress"):    
    clf = SVC(C=c, kernel='linear', random_state=42)
    clf.fit(X_train, y_train)
    
    val_preds = clf.predict(X_val)
    score = accuracy_score(y_val, val_preds)
    
    # Store the score
    validation_scores.append(score)
    
    tqdm.write(f"C={c}: Validation Accuracy = {score:.4f}")
    
    if score > best_score:
        best_score = score
        best_c = c
        best_model = clf

# 2. Create the Bar Graph
plt.figure(figsize=(8, 6))

# Convert C values to strings so they are treated as categories (equal spacing)
x_labels = [str(c) for c in c_values]

# Create bars
bars = plt.bar(x_labels, validation_scores, color='cornflowerblue', edgecolor='black', alpha=0.8)

# Add labels and title
plt.xlabel('C Parameter (Regularization)', fontsize=12)
plt.ylabel('Validation Accuracy', fontsize=12)
plt.title('SVM Hyperparameter Tuning Results', fontsize=14)
plt.ylim(min(validation_scores) - 0.05, max(validation_scores) + 0.05) # Zoom in to show differences

# Add text labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

# Save the plot
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/hyperparameter_tuning.png')
print("✅ Hyperparameter tuning plot saved to 'hyperparameter_tuning.png'")

print(f"\nBest C parameter: {best_c} with Accuracy: {best_score:.4f}")

# --- 6. Finalize and Save ---
# Optional: Refit best model on ALL data (Train + Val) for production
print("Refitting best model on full training set...")
final_model = SVC(C=best_c, kernel='linear', random_state=42)
final_model.fit(X_final, y)

print("Saving artifacts...")
os.makedirs('model', exist_ok=True)
joblib.dump(final_model, 'model/best_svc_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(tfidf, 'model/tfidf_vectorizer.pkl')
print("Done. Model and preprocessors saved.")