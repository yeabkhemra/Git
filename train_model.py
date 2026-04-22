import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("dataset_model.csv")

# -------------------------
# ENCODING
# -------------------------
le_gender = LabelEncoder()
le_activity = LabelEncoder()
le_weather = LabelEncoder()

df["Male/Female"] = le_gender.fit_transform(df["Male/Female"])
df["Physical Activity"] = le_activity.fit_transform(df["Physical Activity"])
df["Weather Condition"] = le_weather.fit_transform(df["Weather Condition"])

# Features & target
X = df[[
    "Your age",
    "Male/Female",
    "Body Weight",
    "Daily Water Intake",
    "Physical Activity",
    "Weather Condition"
]]

y = df["target"]  # <-- replace with your actual target column

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model + encoders
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(le_gender, open("gender.pkl", "wb"))
pickle.dump(le_activity, open("activity.pkl", "wb"))
pickle.dump(le_weather, open("weather.pkl", "wb"))
