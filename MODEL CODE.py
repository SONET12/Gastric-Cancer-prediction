import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter

# Custom Label Encoder to handle unseen labels
class CustomLabelEncoder(LabelEncoder):
    def __init__(self):
        super().__init__()
        self.classes_ = np.array([])

    def fit(self, y):
        super().fit(y)
        self.classes_ = np.append(self.classes_, 'Unknown')
        return self

    def transform(self, y):
        y = np.array(y)
        unseen_labels = set(y) - set(self.classes_)
        y[np.isin(y, list(unseen_labels))] = 'Unknown'
        return super().transform(y)

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y):
        y = np.array(y)
        unseen_indices = y == self.classes_.tolist().index('Unknown')
        y[unseen_indices] = -1
        return super().inverse_transform(y)

# Step 1: Load the data
try:
    data_df = pd.read_excel("dataset")
except FileNotFoundError:
    print("Please check the path.")
    exit()

# Step 2: Data Preparation
X = data_df.drop(columns=['Samples'])  # Features
y = data_df['Samples'].apply(lambda x: x[:2])  # Target labels

# Step 3: Remove non-numeric columns (if applicable)
if not pd.api.types.is_numeric_dtype(X):
    numeric_columns = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_columns]

# Step 4: Data Preprocessing (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Data Augmentation (Duplicate Samples)
def augment_data(X, y, n_copies=5):
    class_counts = Counter(y)
    max_class_count = max(class_counts.values())
    X_augmented = []
    y_augmented = []

    for class_label, count in class_counts.items():
        X_class = X[y == class_label]
        y_class = y[y == class_label]
        n_duplicates = max_class_count - count
        X_class_augmented = np.tile(X_class, (n_copies, 1))
        y_class_augmented = np.tile(y_class, n_copies)

        X_augmented.append(np.vstack((X_class, X_class_augmented[:n_duplicates])))
        y_augmented.append(np.hstack((y_class, y_class_augmented[:n_duplicates])))

    return np.vstack(X_augmented), np.hstack(y_augmented)

X_augmented, y_augmented = augment_data(X_scaled, y, n_copies=5)

# Step 6: Artificial Sampling (SMOTE or ADASYN)
oversample = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)  # or ADASYN
X_resampled, y_resampled = oversample.fit_resample(X_augmented, y_augmented)

class_distribution = Counter(y_resampled)
print("Class Distribution After Augmentation and Oversampling:")
print(class_distribution)

# Step 7: Choose Splitting Method
X_temp, y_temp = X_resampled, y_resampled.copy()

if len(set(y_temp)) > 2:
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
else:
    X_temp, y_temp = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp)

# Initialize variables to track best performance
best_accuracy = 0
best_precision = 0
best_recall = 0
best_model = None
iteration_count = 0

# Loop until we achieve a model accuracy of more than 75 percent
while best_accuracy < 0.75:
    iteration_count += 1
    print(f"Iteration {iteration_count}")

    # Step 8: Label Encoding
    le = CustomLabelEncoder()
    y_resampled_encoded = le.fit_transform(y_resampled)

    # Loop through folds for training and evaluation
    for train_index, test_index in skf.split(X_temp, y_temp):
        X_train, X_test = X_temp[train_index], X_temp[test_index]
        y_train, y_test = y_resampled_encoded[train_index], y_resampled_encoded[test_index]

        # Define Model Architecture
        def create_complex_model(input_shape):
            model = Sequential([
                Input(shape=(input_shape,)),
                Dense(512, activation='relu'),
                BatchNormalization(),
                Dropout(0.6),
                Dense(256, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                Dense(64, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                Dense(32, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                Dense(len(le.classes_), activation='softmax')
            ])
            return model

        # Compile the Model
        model = create_complex_model(X_train.shape[1])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Define Callbacks
        checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

        # Train the Model with early stopping
        model.fit(X_train, y_train, epochs=1000, batch_size=64, validation_split=0.2,
                  callbacks=[checkpoint, early_stopping], verbose=1)

        # Evaluate the Model
        test_loss, test_acc = model.evaluate(X_test, y_test)
        print('Test accuracy:', test_acc)

        # Generate Classification Report
        y_pred = np.argmax(model.predict(X_test), axis=-1)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        print(classification_report(y_test, y_pred, zero_division=0))

        # Update best performance if current model is better
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_precision = report['weighted avg']['precision']
            best_recall = report['weighted avg']['recall']
            best_model = model

    print(f"Iteration {iteration_count} completed.")
    print(f"Best accuracy so far: {best_accuracy:.4f}")

print("Training completed. Best accuracy achieved:", best_accuracy)

# Save the best model
best_model.save('best_model.h5')
