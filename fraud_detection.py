import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def create_synthetic_data():
    """Generates an imbalanced dataset mimicking insurance claims."""
    print("Generating synthetic insurance dataset...")
    # 10,000 samples, 95% legitimate (0), 5% fraud (1)
    X, y = make_classification(n_samples=10000, n_features=8, n_informative=5, 
                               n_redundant=2, weights=[0.95, 0.05], 
                               random_state=42, class_sep=0.7)
    
    # Assigning realistic column names
    columns = ['customer_age', 'policy_deductable', 'incident_severity', 
               'past_claims_count', 'claim_amount', 'police_report_filed', 
               'vehicle_age', 'repair_estimate']
    
    df = pd.DataFrame(X, columns=columns)
    df['is_fraud'] = y
    return df

def main():
    # 1. Load Data
    df = create_synthetic_data()
    print(f"Dataset Shape: {df.shape}")
    print(f"Fraud vs Legitimate breakdown:\n{df['is_fraud'].value_counts(normalize=True)}\n")

    # 2. Split Features (X) and Target (y)
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']

    # 3. Train/Test Split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Initialize Model 
    # Note: class_weight='balanced' tells the model to pay extra attention to the rare fraud cases!
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

    # 5. Train Model
    model.fit(X_train, y_train)

    # 6. Make Predictions
    y_pred = model.predict(X_test)

    # 7. Evaluate the Model
    print("\n--- Model Evaluation ---")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()