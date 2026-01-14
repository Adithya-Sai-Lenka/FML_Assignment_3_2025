import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

def plot_svm_feature_importance():
    # --- 1. Load Artifacts ---
    print("Loading artifacts...")
    try:
        model = joblib.load('model/best_svc_model.pkl')
        tfidf = joblib.load('model/tfidf_vectorizer.pkl')
    except FileNotFoundError:
        print("❌ Error: Files not found. Make sure best_svc_model.pkl and tfidf_vectorizer.pkl exist.")
        return

    # --- 2. Reconstruct Feature Names ---
    # We must manually list the heuristic features in the EXACT order they were added in train_model.py
    # (Order: Shouting, Dollars, Exclamations, URLs, Length, Urgency_KW, Money_KW)
    heuristic_names = [
        'Heuristic: Shouting_Ratio', 
        'Heuristic: Money_Symbols', 
        'Heuristic: Exclamations', 
        'Heuristic: URL_Count', 
        'Heuristic: Text_Length', 
        'Heuristic: Urgency_KW', 
        'Heuristic: Money_KW'
    ]
    
    # Get the vocabulary from the TF-IDF vectorizer
    tfidf_names = tfidf.get_feature_names_out()
    
    # Combine them (Heuristics were stacked FIRST, then TF-IDF)
    all_feature_names = np.array(heuristic_names + list(tfidf_names))
    
    # --- 3. Extract Coefficients ---
    # SVM Coefs are usually shape (1, n_features) for binary classification
    # We flatten it to a 1D array
    if hasattr(model, 'coef_'):
        coefs = model.coef_.toarray().ravel() if hasattr(model.coef_, 'toarray') else model.coef_.ravel()
    else:
        print("Error: The loaded model does not have linear coefficients (did you use a non-linear kernel?).")
        return

    # Sanity Check: Lengths must match
    if len(coefs) != len(all_feature_names):
        print(f"Error: Model has {len(coefs)} coefficients but we found {len(all_feature_names)} feature names.")
        return

    # --- 4. Organize Data ---
    df_features = pd.DataFrame({
        'Feature': all_feature_names,
        'Coefficient': coefs
    })

    # Sort by magnitude
    # Top 10 predictors for SPAM (Highest Positive values)
    top_spam = df_features.nlargest(10, 'Coefficient')
    
    # Top 10 predictors for HAM (Lowest Negative values)
    top_ham = df_features.nsmallest(10, 'Coefficient').sort_values(by='Coefficient', ascending=False)
    
    # Combine for plotting
    plot_df = pd.concat([top_spam, top_ham])
    
    # --- 5. Plotting ---
    plt.figure(figsize=(10, 8))
    
    # Create color map: Red for Spam (Pos), Blue for Ham (Neg)
    colors = ['#d62728' if x > 0 else '#1f77b4' for x in plot_df['Coefficient']]
    
    sns.barplot(x='Coefficient', y='Feature', data=plot_df, palette=colors)
    
    plt.title('Top 20 Features: Determining Spam vs. Ham\n(Linear SVM Coefficients)')
    plt.xlabel('Coefficient Magnitude\n( <--- Ham Indicator | Spam Indicator ---> )')
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/feature_importance.png')
    print("✅ Feature importance plot saved to 'feature_importance.png'")

if __name__ == "__main__":
    plot_svm_feature_importance()