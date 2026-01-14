import os
import glob
import re
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Redefine Feature Extractor (Must match training logic) ---
class SpamFeatureExtractor:
    def __init__(self):
        self.urgency_words = ['urgent', 'immediate', 'act now', 'warning']
        self.money_words = ['bank', 'credit', 'cash', 'lottery', 'inheritance', 'offer']
        
    def get_heuristics(self, text):
        text = str(text) if pd.notnull(text) else ""
        uppercase_count = sum(1 for c in text if c.isupper())
        shouting_ratio = uppercase_count / max(len(text), 1)
        dollar_count = text.count('$') + text.count('€') + text.count('£')
        exclamation_count = text.count('!')
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        url_count = len(urls)
        text_lower = text.lower()
        has_urgency = 1 if any(w in text_lower for w in self.urgency_words) else 0
        has_money = 1 if any(w in text_lower for w in self.money_words) else 0
        return [shouting_ratio, dollar_count, exclamation_count, url_count, len(text), has_urgency, has_money]

    def transform(self, text_series):
        return np.array([self.get_heuristics(t) for t in text_series])

def run_test():
    # --- 2. Load Artifacts ---
    print("Loading model and preprocessors...")
    try:
        model = joblib.load('model/best_svc_model.pkl')
        scaler = joblib.load('model/scaler.pkl')
        tfidf = joblib.load('model/tfidf_vectorizer.pkl')
    except FileNotFoundError:
        print("Error: Model files not found. Run train_model.py first.")
        return

    # --- 3. Read Test Files ---
    folder_path = 'test'
    # Use glob to find files, sort them to ensure order matches logic if needed
    file_paths = glob.glob(os.path.join(folder_path, "email*.txt"))
    
    # Sort files naturally (email1, email2... instead of email1, email10)
    # This assumes the filename format is 'email{number}.txt'
    file_paths.sort(key=lambda f: int(re.search(r'email(\d+)', f).group(1)))

    print(f"Found {len(file_paths)} files in {folder_path}...")

    filenames = []
    raw_texts = []

    for fp in file_paths:
        filenames.append(os.path.basename(fp))
        with open(fp, 'r', encoding='utf-8') as f:
            raw_texts.append(f.read())
            
    # Convert to Series for easy handling
    X_series = pd.Series(raw_texts)

    # --- 4. Transform Features (DO NOT FIT - ONLY TRANSFORM) ---
    print("Transforming features...")
    
    # A. Dense Features
    extractor = SpamFeatureExtractor()
    X_dense = extractor.transform(X_series)
    X_dense_scaled = scaler.transform(X_dense) # Use saved scaler
    
    # B. TF-IDF Features
    X_tfidf = tfidf.transform(X_series)        # Use saved vectorizer
    
    # C. Combine
    X_test_final = hstack([X_dense_scaled, X_tfidf])

    # --- 5. Predict ---
    print("Predicting...")
    predictions = model.predict(X_test_final)

    # --- 6. Save Results ---
    results_df = pd.DataFrame({
        'Filename': filenames,
        'Prediction': predictions
    })
    
    results_df.to_csv('test_predictions.csv', index=False)
    print("Predictions saved to 'test_predictions.csv'.")

    # --- 7. Evaluate (Optional: Compare with Ground Truth if available) ---
    if os.path.exists(os.path.join(folder_path, 'test_labels.csv')):
        print("\n--- Evaluation ---")
        ground_truth = pd.read_csv(os.path.join(folder_path, 'test_labels.csv'))
        
        # Merge on Filename to ensure alignment
        comparison = pd.merge(results_df, ground_truth, on='Filename')
        
        # Calculate accuracy
        acc = (comparison['Prediction'] == comparison['Label']).mean()
        print(f"Test Set Accuracy: {acc:.4f}")
        
        # Show some mismatches
        mismatches = comparison[comparison['Prediction'] != comparison['Label']]
        if not mismatches.empty:
            print("\nSample Mismatches:")
            print(mismatches.head())

        # Plot confusion matrix
        cm = confusion_matrix(comparison["Label"], comparison["Prediction"])
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Predicted Ham', 'Predicted Spam'],
                    yticklabels=['Actual Ham', 'Actual Spam'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        if not os.path.exists('plots'):
            os.makedirs('plots')
        plt.savefig('plots/confusion_matrix.png')



if __name__ == "__main__":
    run_test()