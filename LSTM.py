import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import accuracy_score

data = pd.read_csv("Combined_dataset")

# Drop unnecessary columns
columns_to_drop = ["Src IP Addr", "Dst IP Addr", "Bytes", "Date first seen", "attackType", "attackID", "attackDescription"]
data = data.drop(columns=columns_to_drop)

# Drop non-numeric columns except the target 
non_numeric_columns = [col for col in data.columns if data[col].dtype == 'object' and col != 'label']
data = data.drop(columns=non_numeric_columns)

# Encoding target variable
label_encoder = LabelEncoder()
data["label"] = label_encoder.fit_transform(data["label"])

# Split features and target variable
X = data.drop(columns=["label"])
y = data["label"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Impute missing values using mean imputation
imputer = SimpleImputer(strategy="mean")
X_scaled_imputed = imputer.fit_transform(X_scaled)

# Reshape data into sequences for LSTM input
n_steps = 5  
n_features = X_scaled_imputed.shape[1]  # Number of features

X_reshaped = np.array([X_scaled_imputed[i:i + n_steps] for i in range(len(X_scaled_imputed) - n_steps)])
y_reshaped = y.values[n_steps:]

# Define LSTM model
def create_model():
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(n_steps, n_features)),
        Dropout(0.2),  # Add dropout regularization
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_index, test_index in kfold.split(X_reshaped, y_reshaped):
    X_train, X_test = X_reshaped[train_index], X_reshaped[test_index]
    y_train, y_test = y_reshaped[train_index], y_reshaped[test_index]

    model = create_model()
    model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=0)

    # Evaluate the model
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    cv_scores.append(accuracy)

# Calculate mean cross-validation accuracy
mean_cv_accuracy = np.mean(cv_scores)
print("Mean Cross-Validation Accuracy:", mean_cv_accuracy)

# Train the final model on the entire dataset
final_model = create_model()
final_model.fit(X_reshaped, y_reshaped, epochs=20, batch_size=64, verbose=1)

# Evaluate the final model on test data
y_pred = (final_model.predict(X_reshaped) > 0.5).astype("int32")
test_accuracy = accuracy_score(y_reshaped, y_pred)
print("Test Accuracy:", test_accuracy)

np.save("y_pred.npy", y_pred)
np.save("y_reshaped.npy", y_reshaped)

# Compute confusion matrix
final_conf_matrix = confusion_matrix(y_reshaped, y_pred)
np.save("final_conf_matrix.npy", final_conf_matrix)

y_pred = np.load("y_pred.npy")
y_reshaped = np.load("y_reshaped.npy")
final_conf_matrix = np.load("final_conf_matrix.npy")

# Generate classification report
print("Classification Report:")
print(classification_report(y_reshaped, y_pred))

# Generate confusion matrix
plt.figure(figsize=(5, 2.5))
sns.heatmap(final_conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
