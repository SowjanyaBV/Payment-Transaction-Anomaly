import pandas as pd
import joblib

# Load dataset
df = pd.read_csv("data/archive (1)/creditcard.csv")

# Load model and scaler
model = joblib.load("decision_tree_model.pkl")
scaler = joblib.load("scaler.pkl")

# Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Remove rows with missing target
valid = y.notna()
X = X[valid]
y = y[valid]

# Select actual fraudulent transactions
fraud_transactions = X[y == 1]

print("Number of fraud transactions:", len(fraud_transactions))

# Test first 10 fraud transactions
for i in range(min(10, len(fraud_transactions))):

    transaction = fraud_transactions.iloc[[i]]

    transaction_scaled = scaler.transform(transaction)

    prediction = model.predict(transaction_scaled)[0]

    print(
        f"Fraud transaction {i+1}: "
        f"Model prediction = {prediction}"
    )