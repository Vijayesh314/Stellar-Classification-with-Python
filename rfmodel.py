import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load Dataset
stardata = pd.read_csv("stardataset.csv")
df = pd.DataFrame(stardata)
print("Original Data Head:")
print(df.head())
print("\nOriginal DataFrame columns:")
print(df.columns)

# Rename Columns
renamemap = {"Temperature (K)": "Temperature", "Luminosity(L/Lo)": "Luminosity", "Radius(R/Ro)": "Radius", "Absolute magnitude(Mv)": "AbsoluteMagnitude", 
             "Star type": "StarType", "Star color": "StarColor", "Spectral Class": "SpectralClass"}
df.rename(columns=renamemap, inplace=True)
print("\nData Frame Head after Renaming")
print(df.head())
print("\nData Frame Columns after Renaming")
print(df.columns)

# Encode Starcolor and Spectralclass
lecolor = LabelEncoder()
df["StarColorEncoded"] = lecolor.fit_transform(df["StarColor"])
lespectralclass = LabelEncoder()
df["SpectralClassEncoded"] = lespectralclass.fit_transform(df["SpectralClass"])

# Define X (features) and Y (target star type)
# Use all physical properties along with encoded color and spectral class
features = ["Temperature", "Luminosity", "Radius", "AbsoluteMagnitude", "StarColorEncoded", "SpectralClassEncoded"]
X = df[features]
Y = df["StarType"]

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=features)
print(f"\nFeatures {X_scaled_df} head after scaling:")
print(X_scaled_df.head())
print(f"\nTarget {Y} head:")
print(Y.head())

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, Y, test_size=0.3, random_state=42, stratify=Y)
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
