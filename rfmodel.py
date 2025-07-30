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

# Initial Data Distribution Plots

# Distribution of Star Types (target variable)
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x="StarType", palette="viridis")
plt.title("Distribution of Star Types in Original Dataser")
plt.xlabel("Star Type (numerical)")
plt.ylabel("Count")
plt.show()

startypelabels = {0:"Brown Dwarf", 1:"Red Dwarf", 2:"White Dwarf",
                  3:"Main Sequence", 4:"Supergiant", 5:"Hypergiant"}
df["StarTypeLabel"] = df["StarType"].map(startypelabels)

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x="StarTypeLabel", palette="viridis", order=[startypelabels[i] for i in sorted(startypelabels.keys())])
plt.title("Distribution of Star Types (Labeled)")
plt.xlabel("Star Type")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.show()

# Distribution of Star Colors (categorical feature)
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x="StarColor", palette="tab10", order=df["StarColor"].value_counts().index)
plt.title("Distribution of Star Colors")
plt.xlabel("Star Color")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.show()

# Distribution of Numerical Features (histograms)
numericalfeaturesplotting = ["Temperature", "Luminosity", "Radius", "AbsoluteMagnitude"]
plt.figure(figsize=(16, 10))
for i, col in enumerate(numericalfeaturesplotting):
    plt.subplot(2, 2, i + 1)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")

    if col in ["Luminosity", "Radius", "Temperature"] and (df[col] > 0).all():
        plt.xscale("log")
        plt.title(f"Distribution of {col} (Log Scale)")
    elif col == "AbsoluteMagnitude":
        pass

plt.tight_layout()
plt.show()

# Encode Starcolor and Spectralclass
lecolor = LabelEncoder()
df["StarColorEncoded"] = lecolor.fit_transform(df["StarColor"])

# Define X (features) and Y (target star type)
# Use all physical properties along with encoded color and spectral class
features = ["Temperature", "Luminosity", "Radius", "AbsoluteMagnitude", "StarColorEncoded"]
x = df[features]
y = df["StarType"]

# Scale numerical features
scaler = StandardScaler()
xscaled = scaler.fit_transform(x)
xscaleddf = pd.DataFrame(xscaled, columns=features)
print(f"\nFeatures {xscaleddf} head after scaling:")
print(xscaleddf.head())
print(f"\nTarget {y} head:")
print(y.head())

# Data Splitting
xtrain, xtest, ytrain, ytest = train_test_split(xscaleddf, y, test_size=0.3, random_state=42, stratify=y)
print(f"\nTraining set shape: {xtrain.shape}")
print(f"Testing set shape: {xtest.shape}")

# Model Training (Random Forest Classifier) and Evaluation
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(xtrain, ytrain)

ypred = model.predict(xtest)
print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(ytest, ypred):.2f}")

# Map the numerical Star Type back to original labels
# The unique values in 'Star Type' are 0 to 5.
# Based on common stellar classification:
# 0: Brown Dwarf
# 1: Red Dwarf
# 2: White Dwarf
# 3: Main Sequence
# 4: Supergiant
# 5: Hypergiant
targetnameslist = []
for i in sorted(startypelabels.keys()):
    targetnameslist.append(startypelabels[i])
print("\nClassification Report:")
print(classification_report(ytest, ypred, target_names=targetnameslist, zero_division=0))

# Confusion Matrix
cm = confusion_matrix(ytest, ypred)
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
    X=xscaleddf,
    y=y,
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
plt.fill_between(trainsizes, trainscoresmean - trainscoresstd, trainscoresmean + trainscoresstd, alpha=0.1, color="r")
plt.fill_between(trainsizes, testscoresmean - testscoresstd, testscoresmean + testscoresstd, alpha=0.1, color="g")
plt.title("Learning Curve for RF Classifier")
plt.xlabel("Training examples")
plt.ylabel("Accuracy Score")
plt.legend(loc="best")
plt.grid()
plt.show()

# Classification Report Visualization
# Convert classification report to a DataFrame Heatmap
report = classification_report(ytest, ypred, target_names=targetnameslist, output_dict=True, zero_division=0)
dfreport = pd.DataFrame(report).transpose()
# Select precision, recall, and f1-score for each class
dfmetrics = dfreport.iloc[:-3, :3]

plt.figure(figsize=(10,6))
sns.heatmap(dfmetrics.astype(float), annot=True, cmap="viridis", fmt=".2f", linewidths=0.5)
plt.title("Classification Report Metrics per Star Type")
plt.xlabel("Metrics")
plt.ylabel("Star Type")
plt.show()

# Showing Predictions on Sample Data

samplesnum = 10
sampleindices = np.random.choice(xtest.index, samplesnum, replace=False)

# Get the actual data points from the DataFram
# This is important to get the feature values for display
sampleoriginaldata = df.loc[sampleindices, ["Temperature", "Luminosity", "Radius", "AbsoluteMagnitude", "StarColor", "StarType"]]
sampleoriginaldata = sampleoriginaldata.copy()

sampleXtest = xtest.loc[sampleindices]
sampleYtrue = ytest.loc[sampleindices]
sampleypred = model.predict(sampleXtest)

sampleypredlabels = []
for val in sampleypred:
    sampleypredlabels.append(val)
sampleytruelabels = []
for val in sampleYtrue:
    sampleytruelabels.append(val)

sampleoriginaldata["TrueStarType"] = sampleytruelabels
sampleoriginaldata["PredictedStarType"] = sampleypredlabels
sampleoriginaldata["PredictionCorrect"] = (sampleoriginaldata["TrueStarType"] == sampleoriginaldata["PredictedStarType"])

print("Sample of Test Data with Predictions:")
print(sampleoriginaldata)
