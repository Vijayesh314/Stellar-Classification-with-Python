# Stellar Classification

## Introduction
This project aims to build a machine learning model to classify stars based on different astronomical parameters. Stellar classification is an important task in astronomy, helping us understand the evolution, properties, and distribution of stars in the universe. We employ a Random Forest Classifier due to its robustness and good performance on the data.

## Dataset
The dataset used in this project is from Kaggle: [Star Type Classification Dataset](https://www.kaggle.com/datasets/deepu1109/star-dataset)

Download the [stardataset.csv](https://github.com/Vijayesh314/Stellar-Classification-with-Python/blob/main/stardataset.csv) file and place it in the root directory of this project

## Features and Target Variable
The dataset includes the following features (renamed in my code for clarity purposes)
* Temperature: Surface temperature of the star in Kelvin
* Luminosity: Relative to the Sun's luminosity
* Radius: Relative to the Sun's radius
* Absolute Magnitude: Visible magnitude
* Star Color: Categorical feature of observed color (red, blue, yellow, white, etc.)
The target variable is Star Type: a numerical label representing the star's classification (0-5)
* 0: Brown Dwarf
* 1: Red Dwarf
* 2: White Dwarf
* 3: Main Sequence
* 4: Redgiant
* 5: Hypergiant

## Code Overview
The script rfmodel.py executes the following

### Data Loading and Initial Inspection
Loads the stardataset.csv file into a Pandas DataFrame. Column names are renamed for easier access and readability. Initial data head and column names are printed.

### Exploratory Data Analysis
Generates several plots to show the distribution of the key features
* **Star Type Distribution**: Count plots showing the frequency of each star type, both numerically and with readable labels
* **Star Color Distribution**: A count plot illustrating the distribution of different star colors
* **Numerical Features Histograms**: Histograms for Temperature, Luminosity, Radius, and Absolute Magnitude, with logarithmic scaling applied to highly skewed distributions (Luminosity, Radius, Temperature) for better visualization

### Data Preprocessing
* **Categorical Encoding**: The Star Color feature is encoded into numerical format using LabelEncoder
* **Feature Selection**: Defines the features (Temperature, Luminosity, Radius, AbsoluteMagnitude, StarColorEncoded) used for training
* **Feature Scaling**: Numerical features are scaled using StandardScaler to ensure they contribute equally to the model training process
* **Data Splitting**: The dataset is split into training (70%) and testing (30%) sets, maintains the proportion of star types in both sets

### Model Training and Evaluation
* **Random Forest Classifier**: An initial RandomForestClassifier model is trained with default parameters (n_estimators=100)
* **Cross-Validation for initial model**: The performance of the initial model is evaluated using 5-fold cross-validation on the entire scaled dataset
* **Classification Report**: A detailed report including Precision, Recall, F1-Score, and Support for each star type
* **Confusion Matrix**: A heatmap visualization of the confusion matrix, showing correct and incorrect classifications for each star type
* **Feature Importance Plot**: A bar plot displaying the importance of each feature in the Random Forest model's decision-making process
* **Learning Curve**: Plots training and cross-validation accuracy against the number of training examples, including error bands, to diagnose bias-variance trade-offs

### Hyperparameter Tuning
* **GridSearchCV**: The script performs a hyperparameter tuning process using GridSearchCV with 5-fold cross-validation. This explores a predefined grid of hyperparameters for the Random Forest Classifier to find the best combination that shows the best accuracy
* **Parameters tuned**: n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf
* The model is then updated to use the best_estimator_ found by GridSearchCV, and the following evaluations show the performance of this new updated model

### Sample Predictions
* The model show how to make predictions on a few random samples from the test set, showing the original features, true star type, predicted star type, and whether the prediction was correct
* It also includes an example of predicting the star type for a completely hypothetical star, demonstrating the full end-to-end prediction pipeline

### Results
The model achieves very high accuracy on this dataset, indicating a strong correlation between the physical properties and star type. The various plots and reports provide a deep dive into the model's performance and the characteristics of the data. The stellar classification is correlated to the Hertzsprung–Russell diagram, a key astrophysics relationship.
