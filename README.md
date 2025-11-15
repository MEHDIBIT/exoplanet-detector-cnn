# AI Planet Hunter

A 1D-Convolutional Neural Network (Keras) built to detect exoplanets in NASA's Kepler time-series data. This project's primary focus is not just classification, but solving the extreme **0.7% class imbalance** and **high signal-to-noise ratio** inherent in the dataset.



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
