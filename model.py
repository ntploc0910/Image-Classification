import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Kích thước của ảnh sau khi được chuyển đổi
img_width, img_height = 300, 300

# Khởi tạo mô hình
def model_base():
    base_model = VGG16(include_top=False, input_shape=(img_width, img_height, 3))

# Freeze the pre-trained layers so they won't be updated during training
    for layer in base_model.layers:
        layer.trainable = False

# Create a new model on top of the VGG16 base
    model = Sequential([
        base_model,
        Flatten(),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(2, activation='softmax')  # Output layer with sigmoid activation (binary classification)
    ])

    # # Compile the model
    # model.compile(optimizer='adam',
    #             loss='binary_crossentropy',
    #             metrics=['accuracy'])
    
    return model

def processing():
    image_dir = "data"
    filenames = os.listdir(image_dir)
    labels = [x.split(".")[0] for x in filenames]
    data = pd.DataFrame({"filename":filenames, "label": labels})

    labels = data['label']
    X_train, X_temp = train_test_split(data, test_size=0.2, stratify=labels, random_state = 23)

    label_test_val = X_temp['label']
    X_test, X_val = train_test_split(X_temp, test_size=0.5, stratify=label_test_val, random_state = 23)
    return X_train, X_val, X_test


X_train, X_val, X_test = processing()

generator = ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg16.preprocess_input)

train_generator = generator.flow_from_dataframe(
    X_train, "data/",
    x_col='filename',
    y_col='label',
    target_size=(300, 300),
    class_mode='categorical',
    batch_size=32
)

val_generator = generator.flow_from_dataframe(
    X_val, "data/",
    x_col='filename',
    y_col='label',
    target_size=(300, 300),
    class_mode='categorical',
    batch_size=32,
    shuffle = False
)

test_generator = generator.flow_from_dataframe(
    X_test, "data/",
    x_col='filename',
    y_col='label',
    target_size=(300, 300),
    class_mode='categorical',
    batch_size=32,
    shuffle = False
)
# print(train_generator)
model = model_base()
    # Compile mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_ckpt = tf.keras.callbacks.ModelCheckpoint("DogCat",
                                                monitor="val_loss",
                                                save_best_only=True)
model.fit(train_generator,batch_size=32,epochs=8,validation_data=val_generator, callbacks=[model_ckpt])

model.save("DogCat.h5")
cat_dog_model = tf.keras.models.load_model("DogCat")
result = cat_dog_model.predict(test_generator)

result_argmax = np.argmax(result, axis=1)

y_true = test_generator.labels

y_pred = result_argmax

accuracy = (y_pred == y_true).mean()

print("Test Accuracy:", accuracy)
print("Finished Model")