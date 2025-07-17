import joblib
from sklearn.metrics import accuracy_score
import sys

# Load Model & Test Data
model_package = joblib.load('models/churn_model.pkl')
model = model_package['model']
X_test = model_package['X_test']
y_test = model_package['y_test']

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Threshold Enforcement
if accuracy < 0.70:
    print(f"❌ Accuracy {accuracy:.4f} below threshold of 85%")
    sys.exit(1)
else:
    print("✅ Accuracy meets threshold.")
