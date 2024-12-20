import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Paths to the dataset
train_path = r'H:\UAS_KB\Train'
test_path = r'H:\UAS_KB\Test'

# Function to load images and labels
def load_images_from_folder(folder):
    images = []
    labels = []
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                img_path = os.path.join(subfolder_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (64, 64))  # Resize images to 64x64
                    img = img.flatten()  # Flatten the image
                    images.append(img)
                    labels.append(subfolder)
    return np.array(images), np.array(labels)

# Ensure the dataset paths exist
if not os.path.exists(train_path):
    raise FileNotFoundError(f"Training path not found: {train_path}")
if not os.path.exists(test_path):
    raise FileNotFoundError(f"Testing path not found: {test_path}")

# Load training data
X_train, y_train = load_images_from_folder(train_path)
# Load testing data
X_test, y_test = load_images_from_folder(test_path)

# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Train the SVM model with RBF kernel
svm_model = SVC(kernel='rbf', gamma='scale', random_state=42)
svm_model.fit(X_train, y_train)

# Predict on the test set
y_pred = svm_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Ensure accuracy is above 70%
if accuracy >= 0.70:
    print("The model achieved the desired accuracy.")
else:
    print("The model did not achieve the desired accuracy.")
