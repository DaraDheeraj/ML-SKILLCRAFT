import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset path
dataset_path = "C:/Users/dheer/Downloads/Data_Set"
categories = ["cats", "dogs"]

# Preprocess images
data = []
labels = []
img_size = 64  # Resize images to 64x64

for category in categories:
    folder_path = os.path.join(dataset_path, category)
    label = categories.index(category)
    
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        
        # Open image using Pillow (PIL)
        with Image.open(img_path) as img:
            img = img.resize((img_size, img_size))  # Resize
            img = img.convert("RGB")  # Ensure 3 channels (R, G, B)
            
            # Convert image to list of pixel values
            img_data = list(img.getdata())  # Returns a list of (R, G, B) tuples
            img_data = [val / 255.0 for pixel in img_data for val in pixel]  # Normalize

            data.append(img_data)
            labels.append(label)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Predictions
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"SVM Accuracy: {accuracy:.2f}")