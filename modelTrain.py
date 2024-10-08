import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import datetime

#paths
model_file_path = './BDmodel.keras'
train_dir = 'dataset/training_set'
test_dir = 'dataset/test_set'
batch_size = 32
image_size = (64, 64)

#CNN model with L2 Regularization and additional layers
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001), padding='same', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001), padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001), padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.001), padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', kernel_regularizer=l2(0.001), padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dense(2, activation='sigmoid')  #two outputs
])

#compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#load training and validation datasets using image_dataset_from_directory
training_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical',  #two outputs
    shuffle=True
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical',
    shuffle=True
)

#normalize images to [0, 1] range
normalization_layer = tf.keras.layers.Rescaling(1./255)

training_dataset = training_dataset.map(lambda x, y: (normalization_layer(x), y))
validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))

#prefetch to improve performance
training_dataset = training_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

#callbacks
checkpoint = ModelCheckpoint(
    model_file_path,
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min'
)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    mode='min',
    verbose=1
)

#load the saved model if it exists
if os.path.exists(model_file_path):
    model.load_weights(model_file_path)
    print("Loaded model from", model_file_path)
else:
    print("No existing model found. Training from scratch.")

#tensorFlow Profiler to monitor performance
tf.profiler.experimental.start(log_dir)

#train model
history = model.fit(
    training_dataset,
    epochs=20,
    validation_data=validation_dataset,
    callbacks=[checkpoint, tensorboard, early_stopping]
)

#stop profiler
tf.profiler.experimental.stop()

#tensorboard --logdir=logs/fit
