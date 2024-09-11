import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Data directories
data_dir = "data/Chess/"
img_size = 128
batch_size = 32

# Data augmentation
datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    data_dir, 
    target_size=(img_size, img_size), 
    batch_size=batch_size, 
    class_mode='categorical', 
    subset='training'
)

val_gen = datagen.flow_from_directory(
    data_dir, 
    target_size=(img_size, img_size), 
    batch_size=batch_size, 
    class_mode='categorical', 
    subset='validation'
)

# Model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(6, activation='softmax')  # 6 categories for chess pieces
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen
)

# Save the model
model.save("model/best_chess_model.h5")
