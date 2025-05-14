import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import tensorflow as tf

# Load the data
try:
    data_df = pd.read_excel("Dataset")
except FileNotFoundError:
    print("Please check the path.")
    exit()

# Step 1: Prepare the features and target labels
X = data_df.drop(columns=['Samples'])  # Features
y = data_df['Samples'].apply(lambda x: x[:2])  # Target labels

# Ensure we use only numeric columns
X = X.select_dtypes(include=[np.number])

# Step 2: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Encode the labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Step 4: Load the best model
best_model = load_model('best_model.h5')

# Step 5: Generate scores (probabilities) for the data
y_probabilities = best_model.predict(X_scaled)

# Define the new class labels
new_classes = ['GC', 'DC', 'N']

# Step 6: Convert target labels to one-hot encoding for ROC curve calculation
y_onehot = np.zeros((y_encoded.shape[0], 3))

# Update one-hot encoding for N category
n_indices = [i for i, class_name in enumerate(le.classes_) if class_name.startswith('N')]
for i, encoded_label in enumerate(y_encoded):
    if encoded_label in n_indices:
        y_onehot[i, 2] = 1  # Assign to N category
    elif encoded_label == le.classes_.tolist().index('GC'):
        y_onehot[i, 0] = 1  # Assign to GC category
    elif encoded_label == le.classes_.tolist().index('DC'):
        y_onehot[i, 1] = 1  # Assign to DC category

# Calculate average probability for N category
n_probs = np.mean(y_probabilities[:, n_indices], axis=1, keepdims=True)
# Combine GC, DC, and N probabilities
combined_probabilities = np.hstack([y_probabilities[:, le.classes_.tolist().index('GC')].reshape(-1, 1),
                                    y_probabilities[:, le.classes_.tolist().index('DC')].reshape(-1, 1),
                                    n_probs])

# Step 7: Calculate ROC curve and AUC for each class
fpr = {}
tpr = {}
roc_auc = {}
precision = {}
recall = {}
average_precision = {}

for i, class_name in enumerate(new_classes):
    fpr[class_name], tpr[class_name], _ = roc_curve(y_onehot[:, i], combined_probabilities[:, i])
    roc_auc[class_name] = auc(fpr[class_name], tpr[class_name])
    precision[class_name], recall[class_name], _ = precision_recall_curve(y_onehot[:, i], combined_probabilities[:, i])
    average_precision[class_name] = average_precision_score(y_onehot[:, i], combined_probabilities[:, i])

# Step 8: Plot the ROC curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for class_name in new_classes:
    plt.plot(fpr[class_name], tpr[class_name], label=f'ROC curve for {class_name} (AUC = {roc_auc[class_name]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# Step 9: Plot the Precision-Recall curves
plt.subplot(1, 2, 2)
for class_name in new_classes:
    plt.plot(recall[class_name], precision[class_name], label=f'Precision-Recall curve for {class_name} (AP = {average_precision[class_name]:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")

plt.tight_layout()
plt.show()

# ====================== SHAP Interpretation ======================

# Step 1: SHAP Initialization - Use KernelExplainer for TensorFlow/Keras models
background = X_scaled[np.random.choice(X_scaled.shape[0], 50, replace=False)]  # Select representative background data
explainer = shap.KernelExplainer(best_model.predict, background)  # KernelExplainer for black-box models

# Step 2: Compute SHAP values
shap_values = explainer.shap_values(X_scaled, nsamples=100)  # Approximate with 100 samples

# Debugging - Print Shapes
print("SHAP values shape:", np.array(shap_values).shape)  # Expect (num_classes, num_samples, num_features)
print("Feature matrix shape:", X.shape)

# Step 3: Ensure proper SHAP values selection for multi-class models
if isinstance(shap_values, list):  
    class_idx = 0  # Choose a specific class (modify as needed)
    shap_values_class = shap_values[class_idx]  # Extract SHAP values for selected class
else:
    shap_values_class = shap_values  # Single-class case

# Additional shape check for unexpected cases
if len(shap_values_class.shape) == 3:
    class_idx = 0  
    shap_values_class = shap_values_class[:, :, class_idx]

    # Debugging - Check the adjusted shapes
    print("New SHAP values shape:", shap_values_class.shape)  # Should be (num_samples, num_features - 1)
    print("New feature matrix shape:", X_filtered.shape)  # Should match shap_values_class

else:
    X_filtered = X  # Keep X as is if "Age" is not found

# Step 4: Generate SHAP Summary Plot Without "Age"
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_class, X_filtered, feature_names=feature_names)

# Step 6: Generate SHAP Force Plot for a Single Prediction
shap.initjs()  # Enable JavaScript visualization
instance_idx = 0  # Select the first instance
expected_value = explainer.expected_value[class_idx] if isinstance(explainer.expected_value, list) else explainer.expected_value

shap.force_plot(expected_value, shap_values_class[instance_idx], X_filtered.iloc[instance_idx, :], matplotlib=True)

# Step 5: Generate SHAP Dependence Plots for Key Features
top_features = np.argsort(np.abs(shap_values_class).mean(axis=0))[-5:]  # Select top 5 important features
for feature_idx in top_features:
    shap.dependence_plot(feature_idx, shap_values_class, X_filtered, feature_names=feature_names)

print("SHAP analysis completed. Check the plots for feature impact insights.")
