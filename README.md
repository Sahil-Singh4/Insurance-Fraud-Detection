# Insurance-Fraud-Detection

# 🛡️ Insurance Fraud Detection Using Machine Learning

## 📌 Project Overview
Insurance fraud costs companies billions of dollars annually. The goal of this project is to build a Machine Learning model capable of identifying fraudulent insurance claims. 

The primary challenge addressed in this project is **Class Imbalance**. In the real world, the vast majority of claims are legitimate (around 95%), which causes standard predictive models to become heavily biased toward predicting "Not Fraud."

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn
* **Algorithm:** Random Forest Classifier

## 📊 Model Performance & Evaluation
Relying strictly on "Accuracy" is dangerous in fraud detection. This baseline model achieves a **96% overall accuracy**, but a deeper dive into the Confusion Matrix reveals the true performance:

**Confusion Matrix:**
* **1885** True Negatives (Legitimate claims correctly approved)
* **5** False Positives (Legitimate claims accidentally flagged as fraud)
* **30** True Positives (Fraudulent claims successfully caught)
* **80** False Negatives (Fraudulent claims that slipped through undetected)

**Key Metrics:**
* **Precision (Fraud): 0.86** - When the model flags a claim as fraud, it is correct 86% of the time.
* **Recall (Fraud): 0.27** - The model currently only catches 27% of all actual fraudulent claims. 

> **Next Steps:** Future iterations of this project will focus on increasing the Recall score using advanced data sampling techniques like SMOTE (Synthetic Minority Over-sampling Technique) so fewer fraudulent claims slip through.

## 🚀 How to Run Locally
1. Clone this repository to your local machine.
2. python fraud_detection.py
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
