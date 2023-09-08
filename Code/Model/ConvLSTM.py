import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Input, ConvLSTM2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tqdm.keras import TqdmCallback
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Function to standardize the sequence of images to a given length
def standardize_sequence(sequence, desired_length=32):
    while len(sequence) < desired_length:
        sequence.append(sequence[-1])
    return sequence[:desired_length]

# Load and preprocess the image data from a given dataset path
def load_data(dataset_path, img_height, img_width):
    X, y = [], []
    labels = [0, 1]

    # Iterate over each label's directory
    for label in labels:
        label_folder_path = os.path.join(dataset_path, str(label))
        subfolders = [os.path.join(label_folder_path, subfolder) for subfolder in os.listdir(label_folder_path) if
                      os.path.isdir(os.path.join(label_folder_path, subfolder))]

        # Load images from each subfolder
        for subfolder in subfolders:
            sequence = []
            for frame_file in sorted(os.listdir(subfolder)):
                if frame_file.endswith(('.jpg', '.png')):
                    frame_path = os.path.join(subfolder, frame_file)
                    img = cv2.imread(frame_path)
                    img = cv2.resize(img, (img_height, img_width))
                    img = img.astype(np.float32) / 255.0
                    sequence.append(img)

            if not sequence:
                print(f"Warning: No images found in {subfolder}. Skipping...")
                continue

            sequence = standardize_sequence(sequence)
            X.append(sequence)
            y.append(label)

    return np.array(X, dtype=np.float32), np.array(y)

# Function to split the data into training and validation sets while keeping class distribution balanced
def split_balanced_data(X, y, test_size=0.2):
    # Split the data based on class labels
    X_zeros = X[y == 0]
    X_ones = X[y == 1]
    y_zeros = y[y == 0]
    y_ones = y[y == 1]

    # Use train_test_split to ensure balanced splitting
    X_train_zeros, X_val_zeros, y_train_zeros, y_val_zeros = train_test_split(X_zeros, y_zeros, test_size=test_size, random_state=42)
    X_train_ones, X_val_ones, y_train_ones, y_val_ones = train_test_split(X_ones, y_ones, test_size=test_size, random_state=42)

    # Concatenate results to get the final training and validation sets
    X_train = np.concatenate([X_train_zeros, X_train_ones], axis=0)
    X_val = np.concatenate([X_val_zeros, X_val_ones], axis=0)
    y_train = np.concatenate([y_train_zeros, y_train_ones], axis=0)
    y_val = np.concatenate([y_val_zeros, y_val_ones], axis=0)

    return X_train, X_val, y_train, y_val

# Define the ConvLSTM2D model for video classification
def define_model(input_shape, num_classes):
    inputs = Input(input_shape)
    x = ConvLSTM2D(64, kernel_size=(3, 3), padding='same', return_sequences=True)(inputs)
    x = ConvLSTM2D(64, kernel_size=(3, 3), padding='same', return_sequences=False)(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Custom data generator class for video sequences
class VideoDataGenerator:
    def __init__(self, datagen):
        self.datagen = datagen

    # Method to produce batches of augmented data
    def flow(self, X, y, batch_size):
        while True:
            idx = np.random.choice(len(X), batch_size)
            X_batch = X[idx]
            y_batch = y[idx]

            augmented_X = np.zeros_like(X_batch)
            for i in range(batch_size):
                for j in range(X_batch.shape[1]):
                    augmented_X[i, j] = self.datagen.random_transform(X_batch[i, j])

            yield augmented_X, y_batch

# Set up data augmentation configurations using ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)
video_datagen = VideoDataGenerator(datagen)

# Load the data and perform train-validation split
flow_dataset_path = 'Path_to_your_DL_flow_dataset'
flow_test_dataset_path = 'Path_to_your_DL_test_flow_dataset'
img_height, img_width = 64, 64
X, y = load_data(flow_dataset_path, img_height, img_width)
X_train, X_val, y_train, y_val = split_balanced_data(X, y, test_size=0.2)

# Load the test data and ensure it's balanced
X_test, y_test = load_data(flow_test_dataset_path, img_height, img_width)
if np.sum(y_test == 0) != np.sum(y_test == 1):
    print("Warning: Test data is not balanced!")

# Convert labels to one-hot encoding format
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

# Train the ConvLSTM2D model
batch_size = 16
epochs = 50
input_shape = (None, img_height, img_width, 3)
num_classes = 2

model = define_model(input_shape, num_classes)
steps_per_epoch = len(X_train) // batch_size
history = model.fit(video_datagen.flow(X_train, y_train, batch_size=batch_size),
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    steps_per_epoch=steps_per_epoch,
                    verbose=0,
                    callbacks=[TqdmCallback(verbose=1)])

# Evaluate the trained model using ROC curve
y_pred_val = model.predict(X_val)[:, 1]
fpr_val, tpr_val, thresholds_val = roc_curve(y_val[:, 1], y_pred_val)
auc_value_val = auc(fpr_val, tpr_val)

y_pred_test = model.predict(X_test)[:, 1]
fpr_test, tpr_test, thresholds_test = roc_curve(y_test[:, 1], y_pred_test)
auc_value_test = auc(fpr_test, tpr_test)

# Plot the ROC curves for validation and test data
plt.figure(figsize=(10, 8))
plt.plot(fpr_val, tpr_val, color='darkorange', lw=2, label=f'Validation ROC curve (AUC = {auc_value_val:.2f})')
plt.plot(fpr_test, tpr_test, color='green', lw=2, label=f'Test ROC curve (AUC = {auc_value_test:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
