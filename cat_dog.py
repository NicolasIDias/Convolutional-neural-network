from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
from PIL import Image

model = Sequential([
    Conv2D(64, (3, 3), input_shape=(64, 64, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    
    Dense(128, activation='relu'),
    Dropout(0.2),
    
    Dense(128, activation='relu'),
    Dropout(0.2),
    
    Dense(1, activation='sigmoid')
])
model.compile(optimizer=tensorflow.keras.optimizers.Adam(3e-4), loss='binary_crossentropy', metrics=['accuracy'])

train_gen = ImageDataGenerator(rescale=1./255,
                                     rotation_range=7,
                                     horizontal_flip=True,
                                     shear_range=0.2,
                                     height_shift_range=0.07,
                                     zoom_range=0.2)

test_gen = ImageDataGenerator(rescale=1./255,
                                     rotation_range=7,
                                     horizontal_flip=True,
                                     shear_range=0.2,
                                     height_shift_range=0.07,
                                     zoom_range=0.2)

train_ds = train_gen.flow_from_directory('dataset/training_set',
                                           target_size=(64,64),
                                           batch_size=64,
                                           class_mode='binary')

val_ds = test_gen.flow_from_directory('dataset/test_set',
                                         target_size=(64,64),
                                         batch_size=64,
                                         class_mode='binary')

model.fit(train_ds, steps_per_epoch=4000//32,
                         epochs=25, 
                         validation_data = val_ds, 
                         validation_steps = 1000//32)

image = ['dataset/test_set/cachorro/dog.3501.jpg', 'dataset/test_set/gato/cat.3501.jpg']

animal = 1

image_test = load_img(image[animal],
                        target_size=(64,64))
image[animal] = Image.open(image[animal])
image[animal].show()

image_test = img_to_array(image_test)
image_test /= 255
image_test = np.expand_dims(image_test, axis=0)

# Avaliar o modelo
loss, accuracy = model.evaluate(val_ds)
print(f'Validação - Loss: {loss}, Accuracy: {accuracy}')


predict = model.predict(image_test)

print(predict)


print(f'Prediction: {predict[0]}')
print(f'Class: {"Gato" if predict[0] > 0.5 else "Cachorro"}')

train_ds.class_indices







