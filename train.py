import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
df = pd.read_csv("students.csv")

X = df[["cgpa","internships"]]
y = df["placed"]

# Train
model = RandomForestClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("Model trained and saveddssgit as model.pkl")