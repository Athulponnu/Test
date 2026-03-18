import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("students.csv")

X = df[["cgpa","internships"]]
y = df["placed"]

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "model.pkl")

print("Training done done")