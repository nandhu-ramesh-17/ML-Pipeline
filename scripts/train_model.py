import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load Data
df = pd.read_csv('data/Telco-Customer-Churn.csv')

# Preprocessing
df.drop(['customerID'], axis=1, inplace=True)
df.replace(' ', pd.NA, inplace=True)
df.dropna(inplace=True)

label_enc = LabelEncoder()
df['Churn'] = label_enc.fit_transform(df['Churn'])

for col in df.select_dtypes(include=['object']).columns:
    df[col] = label_enc.fit_transform(df[col])

# Train/Test Split
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump({'model': model, 'X_test': X_test, 'y_test': y_test}, 'models/churn_model.pkl')

print("✅ Model trained and saved as churn_model.pkl")