import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load Dataset
stardata = pd.read_csv("stardataset.csv")
df = pd.DataFrame(stardata)
print("Original Data Head:")
print(df.head())
print("\nOriginal DataFrame columns:")
print(df.columns)

# Rename Columns
renamemap = {"Temperature (K)": "Temperature", "Luminosity(L/Lo)": "Luminosity", "Radius(R/Ro)": "Radius", 
             "Absolute magnitude(Mv)": "AbsoluteMagnitude", "Star type": "StarType", "Star color": "StarColor"}
df.rename(columns=renamemap, inplace=True)
print("\nData Frame Head after Renaming")
print(df.head())
print("\nData Frame Columns after Renaming")
print(df.columns)

# Encode Starcolor and Spectralclass
lecolor = LabelEncoder()
df["StarColorEncoded"] = lecolor.fit_transform(df["StarColor"])

# Define X (features) and Y (target star type)
# Use all physical properties along with encoded color and spectral class
features = ["Temperature", "Luminosity", "Radius", "AbsoluteMagnitude", "StarColorEncoded"]
X = df[features]
Y = df["StarType"]

# Scale numerical features
scaler = StandardScaler()
Xscaled = scaler.fit_transform(X)
Xscaleddf = pd.DataFrame(Xscaled, columns=features)
print(f"\nFeatures {Xscaleddf} head after scaling:")
print(Xscaleddf.head())
print(f"\nTarget {Y} head:")
print(Y.head())

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(Xscaleddf, Y, test_size=0.3, random_state=42, stratify=Y)
print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Model Training (Random Forest Classifier) and Evaluation
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Map the numerical Star Type back to original labels
# The unique values in 'Star Type' are 0 to 5.
# Based on common stellar classification:
# 0: Brown Dwarf
# 1: Red Dwarf
# 2: White Dwarf
# 3: Main Sequence
# 4: Supergiant
# 5: Hypergiant
startypelabels = {0: "Brown Dwarf", 1: "Red Dwarf", 2: "White Dwarf", 3: "Main Sequence", 4: "Supergiant", 5: "Hypergiant"}
targetnameslist = []
for i in sorted(startypelabels.keys()):
    targetnameslist.append(startypelabels[i])
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=targetnameslist, zero_division=0))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=targetnameslist,
            yticklabels=targetnameslist)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Star Type Classification')
plt.show()

# Importance Plot
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    featureimportancedf = pd.DataFrame({"feature":features, "importance":importances})
    featureimportancedf = featureimportancedf.sort_values(by="importance", ascending=False)

    print("\nFeature Importances:")
    print(featureimportancedf)
    sns.barplot(x="importance", y="feature", data=featureimportancedf, palette="magma")
    plt.title("Feature Importances from RF Classifier")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()

# Learning Curve
print("\nGenerating Learning Curve (might take a few moments)...")
trainsizes, trainscores, testscores = learning_curve(
    estimator=model,
    X=Xscaleddf,
    y=Y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42)

# Generate mean and standard deviation for training and test scores
trainscoresmean = np.mean(trainscores, axis=1)
trainscoresstd = np.std(trainscores, axis=1)
testscoresmean = np.mean(testscores, axis=1)
testscoresstd = np.std(testscores, axis=1)

plt.figure(figsize=(12, 7))
plt.plot(trainsizes, trainscoresmean, "o-", color="r", label="Training score")
plt.plot(trainsizes, testscoresmean, "o-", color="g", label="Cross-validation score")
plt.show()

# Classification Report Visualization
# Convert classification report to a DataFrame Heatmap
report = classification_report(y_test, y_pred, target_names=targetnameslist, output_dict=True, zero_division=0)
dfreport = pd.DataFrame(report).transpose()
# Select precision, recall, and f1-score for each class
dfmetrics = dfreport.iloc[:-3, :3]

plt.figure(figsize=(10,6))
sns.heatmap(dfmetrics.astype(float), annot=True, cmap="viridis", fmt=".2f", linewidths=0.5)
plt.title("Classification Report Metrics per Star Type")
plt.xlabel("Metrics")
plt.ylabel("Star Type")
plt.show()
