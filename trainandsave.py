import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

# Load Dataset
stardata = pd.read_csv("stardataset.csv")
df = pd.DataFrame(stardata)

# Rename Columns
renamemap = {"Temperature (K)": "Temperature", "Luminosity(L/Lo)": "Luminosity", "Radius(R/Ro)": "Radius",
             "Absolute magnitude(Mv)": "AbsoluteMagnitude", "Star type": "StarType", "Star color": "StarColor"}
df.rename(columns=renamemap, inplace=True)
print("\nData Frame Head after Renaming")
print(df.head())
print("\nData Frame Columns after Renaming")
print(df.columns)

# Encode Starcolor
lecolor = LabelEncoder()
df["StarColorEncoded"] = lecolor.fit_transform(df["StarColor"])

# Define X (features) and Y (target star type)
features = ["Temperature", "Luminosity", "Radius", "AbsoluteMagnitude", "StarColorEncoded"]
x = df[features]
y = df["StarType"]

# Scale numerical features
scaler = StandardScaler()
xscaled = scaler.fit_transform(x)
xscaleddf = pd.DataFrame(xscaled, columns=features)

# Data Splitting
xtrain, xtest, ytrain, ytest = train_test_split(xscaleddf, y, test_size=0.3, random_state=42, stratify=y)

# Hyperparameter Tuning with GridSearchCV
paramgrid = {
    "n_estimators":[50, 100, 150],
    "max_features":["sqrt", "log2"],
    "max_depth":[10, 20, None],
    "min_samples_split":[2, 5],
    "min_samples_leaf":[1, 2]
}

gridsearch = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=paramgrid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=2
)
gridsearch.fit(xtrain, ytrain)

# Get the best model
model = gridsearch.best_estimator_

# Evaluate the model on the test set
ypred = model.predict(xtest)
print("--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(ytest, ypred):.2f}")

# Save the trained model, scaler, and label encoder
joblib.dump(model, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(lecolor, 'lecolor.pkl')
print("\nModel, scaler, and label encoder saved successfully.")
