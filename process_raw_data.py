import pandas as pd
from sklearn.model_selection import train_test_split
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

enron = pd.read_csv('raw_data/enron_spam_data.csv')
sp_assassin = pd.read_csv('raw_data/spam_assassin.csv')

mask = np.random.rand(len(enron)) < 0.5
enron['text'] = np.where(mask, enron['Subject'] + " " + enron['Message'], enron['Message'])
enron['target'] = enron['Spam/Ham'].map({'ham': 0, 'spam': 1})

enron_clean = enron[['text', 'target']].copy()
sp_assassin_clean = sp_assassin[['text', 'target']].copy()

enron_clean['original_dataset'] = 'enron'
sp_assassin_clean['original_dataset'] = 'spam_assassin'


final_df = pd.concat([enron_clean, sp_assassin_clean], axis=0, ignore_index=True)
final_df['stratify_key'] = final_df['target'].astype(str) + "_" + final_df['original_dataset'].astype(str)

strat_counts = final_df['stratify_key'].value_counts()

plt.figure(figsize=(10, 8))
# Plotting
wedges, texts, autotexts = plt.pie(
    strat_counts, 
    labels=strat_counts.index.map({"0_enron": "Ham (Enron)", "1_enron": "Spam (Enron)", "0_spam_assassin": "Ham (SpamAssassin)", "1_spam_assassin": "Spam (SpamAssassin)"}),
    autopct='%1.1f%%', 
    startangle=140,
    colors=['#66b3ff','#99ff99','#ffcc99','#ff9999'] # Soft colors
)

plt.title('Distribution of Data Points by Stratification Key')
plt.setp(autotexts, size=10, weight="bold")

os.makedirs('plots', exist_ok=True)
# Save the plot
plt.savefig('plots/stratification_distribution.png')
print(f"✅ Pie chart saved to 'stratification_distribution.png'")
plt.close()
# -----------

X_train, X_test, y_train, y_test = train_test_split(
    final_df[['text', 'original_dataset']],  # Keeping origin in X to verify later if needed
    final_df['target'], 
    test_size=0.1, 
    stratify=final_df['stratify_key'], # <--- The Magic Step
    random_state=42
)



# Train Data Export
train_export = X_train.copy()
train_export['label'] = y_train

os.makedirs('data', exist_ok=True)
train_filename = 'data/train_data.csv'
train_export.to_csv(train_filename, index=False)
print(f"✅ Training data saved to: {train_filename} ({len(train_export)} rows)")



# Save Test Data for evaluation
folder_name = 'test'

# Clean/Create the directory
if os.path.exists(folder_name):
    shutil.rmtree(folder_name) # Clean up previous runs
os.makedirs(folder_name)

# Reset index so we can generate clean filenames (email1.txt, email2.txt...)
X_test_reset = X_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

label_mapping = []

print(f"📂 Writing {len(X_test_reset)} files to '{folder_name}/'...")

for idx, row in X_test_reset.iterrows():
    # Generate filename (email1.txt, email2.txt, etc.)
    # idx+1 makes it start at 1 instead of 0
    filename = f"email{idx+1}.txt"
    file_path = os.path.join(folder_name, filename)
    
    # Write the text content to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(str(row['text']))
    
    # Keep track of the label for the CSV
    label_mapping.append({
        'Filename': filename,
        'Label': y_test_reset[idx]
    })

# ---------------------------------------------------------
# 3. Save Test Labels CSV
# ---------------------------------------------------------
labels_df = pd.DataFrame(label_mapping)
labels_path = os.path.join(folder_name, 'test_labels.csv')
labels_df.to_csv(labels_path, index=False)

print(f"✅ Test labels saved to: {labels_path}")
print("Done.")