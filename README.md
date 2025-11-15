# AI Planet Hunter

A 1D-Convolutional Neural Network (Keras) built to detect exoplanets in NASA's Kepler time-series data. This project's primary focus is not just classification, but solving the extreme **0.7% class imbalance** and **high signal-to-noise ratio** inherent in the dataset.



## How to Run This Project

This project is designed to run in a cloud environment like **Google Colab** from a single script.

### Prerequisites

1.  **Python 3.x**
2.  **A Kaggle Account** (to download the dataset)

### Instructions

1.  **Get your Kaggle API Key:**
    * Log in to your Kaggle account.
    * Go to **Account** > **API**.
    * Click **"Create New API Token"**. This will download a file named `kaggle.json`.

2.  **Run the Code:**
    * Open a new Google Colab notebook.
    * Copy and paste the entire `planet_hunter_ai.py` script (below) into a single cell.
    * Run the cell.

3.  **Upload Your Key:**
    * The script will pause at **Step 2** and show an "Upload" button.
    * Upload the `kaggle.json` file you downloaded.

The script will then run from start to finish, downloading the data, preprocessing it, training the model, and generating the final "honest" evaluation.

---
## `planet_hunter_ai.py`

```python
#################################################################
# Project: AI Planet Hunter
# Step 1: Setup
#################################################################
print("Step 1: Installing dependencies...")
!pip install lightkurve --quiet
!pip install imbalanced-learn --quiet

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from google.colab import files

print("All libraries imported.")

#################################################################
# Step 2: Download and Load Data
#################################################################
print("\nStep 2: Downloading data...")

if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
    print("Please upload your kaggle.json file:")
    uploaded = files.upload()
    if 'kaggle.json' in uploaded:
        !mkdir -p ~/.kaggle
        !mv kaggle.json ~/.kaggle/
        !chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d keplersmachines/kepler-labelled-time-series-data -q --force
!unzip -o kepler-labelled-time-series-data.zip

train_df = pd.read_csv('exoTrain.csv')
test_df = pd.read_csv('exoTest.csv')

print(f"Loaded {len(train_df)} training stars and {len(test_df)} testing stars.")

#################################################################
# Step 3: Preprocessing and Signal Cleaning
#################################################################
print("\nStep 3: Preprocessing data...")

def preprocess_data(df):
    X = df.drop('LABEL', axis=1)
    y = df['LABEL'].replace(1, 0).replace(2, 1)
    
    # Detrending with Savitzky-Golay Filter
    X_flat_np = savgol_filter(X.values, 
                            window_length=201, 
                            polyorder=3, 
                            axis=1)
    
    X_detrended = X.values - X_flat_np
    
    return X_detrended, y

X_train_clean, y_train_clean = preprocess_data(train_df)
X_test_clean, y_test_clean = preprocess_data(test_df)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_clean)
X_test_scaled = scaler.transform(X_test_clean)

print("Data detrended and scaled.")

#################################################################
# Step 4: Fix Imbalance with SMOTE
#################################################################
print("\nStep 4: Using SMOTE to balance dataset...")

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train_clean)

print(f"Data balanced: {len(y_train_smote)} total samples.")

#################################################################
# Step 5: Build and Train the 1D-CNN Model
#################################################################
print("\nStep 5: Building and training the 1D-CNN model...")

X_train_final = X_train_smote.reshape(X_train_smote.shape[0], X_train_smote.shape[1], 1)
X_test_final = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

def create_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.4))
    
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

model = create_model((X_train_final.shape[1], 1))
model.summary()

history = model.fit(
    X_train_final,
    y_train_smote,
    epochs=15,
    batch_size=64,
    validation_split=0.2 
)

print("Model training complete.")

#################################################################
# Step 6: Final Evaluation
#################################################################
print("\nStep 6: Evaluating model on the unseen test set...")

y_pred_probs = model.predict(X_test_final)
y_pred = (y_pred_probs > 0.5).astype(int) 

print("\n--- FINAL Classification Report ---")
print(classification_report(y_test_clean, y_pred, target_names=['No Planet (0)', 'Planet (1)']))

print("\n--- FINAL Confusion Matrix ---")
cm = confusion_matrix(y_test_clean, y_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('FINAL MODEL on Unseen Imbalanced Test Data')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

print("\n\nProject complete.")

```





## Project Evolution & Methodology

The project evolved through three key models, demonstrating a process of iterative problem-solving.

### 1. Baseline Model: The Imbalance Trap

A standard 1D-CNN was trained on the raw, imbalanced data.

* **Result:** The model achieved **~99.3% accuracy** by succumbing to the class imbalance. It learned to be a "lazy" classifier, predicting "No Planet" (the majority class) for every sample.
* **Performance:** **0% Recall** for the "Planet" class. It found **0 out of 5** planets in the test set.



### 2. Weighted Loss Function: The Overcorrection

* **Technique:** To combat the "lazy" model, a `class_weight` was applied to the `binary_crossentropy` loss function. This heavily penalized the model for missing a planet.
* **Result:** The model overcorrected, creating a "panicked" classifier that predicted "Planet" for nearly every sample to avoid the penalty.
* **Performance:** **100% Planet Recall** (good), but **0% "No Planet" Recall** (terrible). This model was equally useless.



### 3. The Final Solution: A Hybrid Approach

The successful model required a two-stage preprocessing pipeline before training.

**Stage 1: Signal Detrending**
First, to handle the high signal-to-noise ratio, a **Savitzky-Golay Filter** was applied to all 5,000+ light curves. This "detrended" the data by removing the slow, long-term stellar variability (noise), forcing the model to focus on the sharp, periodic dips of a potential transit.



**Stage 2: Generative Oversampling (SMOTE)**
Second, to fix the 37-to-5050 class imbalance, **SMOTE (Synthetic Minority Over-sampling Technique)** was used. This algorithm generated new, synthetic "Planet" samples by interpolating between existing ones, creating a perfectly balanced 50/50 training set.

## Final Results & Evaluation

The final 1D-CNN was trained on this new, clean, and balanced dataset. It was then unleashed on the original, unseen, and imbalanced test set (`exoTest.csv`) to provide an honest, real-world performance metric.

### Final Model Performance (on unseen test data):

* **Planet Recall (Sensitivity):** **0.20** (1 out of 5 planets found)
* **No-Planet Recall (Specificity):** **0.996** (563 out of 565 non-planets correctly identified)
* **Planet Precision:** **0.33**

This demonstrates the model's viability as a "smart filter." It successfully identifies a portion of the rare positive class while maintaining near-perfect specificity, solving the core failures of the "lazy" and "panicked" models.
