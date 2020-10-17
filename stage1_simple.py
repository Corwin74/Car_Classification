#Скрипт - первая часть. Шаги 1-3

import numpy as np
import pandas as pd
import sys
import os

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.models as M
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
import tensorflow.keras.callbacks as C
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras import optimizers
import efficientnet.tfkeras as efn
from sklearn.model_selection import train_test_split

import PIL
from PIL import ImageOps, ImageFilter

print('Python       :', sys.version.split('\n')[0])
print('Numpy        :', np.__version__)
print('Tensorflow   :', tf.__version__)
print('Keras        :', keras.__version__)

# Ограничеваем использование памяти GPU TensorFlow
# в противном случае на первой же операции резервируется вся память
# и обучение не может быть выполнено при любом размере batch

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# Основные настройки

# Batch size
BATCH_SIZE           = 8 

# Epochs
EPOCHS_STEP1         = 1  
EPOCHS_STEP2         = 1  
EPOCHS_STEP3         = 1   

# Learning Rates
LR_STEP1             = 0.001
LR_STEP2             = 0.0001
LR_STEP3             = 0.00001

# Test-validation split

VAL_SPLIT            = 0.2 # сколько данных выделяем на тест = 20%
CLASS_NUM            = 10  # количество классов в нашей задаче
IMG_SIZE             = 250 # какого размера подаем изображения в сеть
IMG_CHANNELS         = 3   # у RGB 3 канала
input_shape          = (IMG_SIZE, IMG_SIZE, IMG_CHANNELS)
RANDOM_SEED = 1488
np.random.seed(RANDOM_SEED)  
PYTHONHASHSEED = 0

DATA_PATH = '/home/alex/Car_Classification/input/'
sample_submission = pd.read_csv(DATA_PATH + "sample-submission.csv")
MODEL_PATH = '/home/alex/Car_Classification/model/'

# Аугментация данных очень важна когда у нас не большой датасет (как в нашем случае)

train_datagen = ImageDataGenerator(
    rescale=1. / 255, 
    rotation_range = 30,
    shear_range=0.2,
    zoom_range=[0.75,1.25],
    brightness_range=[0.5, 1.5],
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=VAL_SPLIT,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# Завернем наши данные в генератор:

train_generator = train_datagen.flow_from_directory(
    DATA_PATH+'train/',      # директория где расположены папки с картинками 
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True, 
    seed=RANDOM_SEED,
    subset='training') # set as training data

test_generator = train_datagen.flow_from_directory(
    DATA_PATH+'train/',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True, 
    seed=RANDOM_SEED,
    subset='validation') # set as validation data

# # Построение модели

# За основу берем сеть EfficientB6 в реализации https://github.com/qubvel/efficientnet
# 
# 


base_model = efn.EfficientNetB6(
    weights='imagenet', # Подгружаем веса imagenet
    include_top=False,  # Выходной слой (голову) будем менять т.к. у нас другие классы
    input_shape=input_shape)


# Заморозим веса imagenet в базовой модели, чтобы она работала в качестве feature extractor 
# и наша голова обучалась делать классификацию на наши 10 классов

base_model.trainable = False

# Устанавливаем "голову" в минималистическо классической конфигурации

model=M.Sequential()
model.add(base_model)
model.add(L.GlobalAveragePooling2D(),) 
model.add(L.Dense(256, activation='relu'))
model.add(L.BatchNormalization())
model.add(L.Dropout(0.25))
model.add(L.Dense(CLASS_NUM, activation='softmax'))


# Обучение модели

# Step 1 - обучение "головы"

# Настройки

model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=LR_STEP1), metrics=["accuracy"])

# Добавим ModelCheckpoint чтоб сохранять прогресс обучения модели и можно было потом подгрузить и дообучить модель.

checkpoint = ModelCheckpoint(MODEL_PATH+'step1-{epoch:02d}-{val_loss:.4f}.hdf5' , monitor = ['val_accuracy'] , verbose = 1, mode = 'max')
earlystop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# Обучаем
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples//train_generator.batch_size,
    validation_data = test_generator, 
    validation_steps = test_generator.samples//test_generator.batch_size,
    epochs=EPOCHS_STEP1,
    callbacks=[checkpoint, earlystop],
    verbose=2
    )

scores = model.evaluate(test_generator, verbose=2)
print("Accuracy step1: %.2f%%" % (scores[1]*100))

# Сохраняем обученную модель на первом шаге

model.save('model/model_step1.hdf5')

# Запрашиваем какой файл с весами загрузить как лучший на первом шаге обучения

while not os.path.isfile(weights_name := input('Input name of weights file: ')):
    print("File not exist!")

model.load_weights(weights_name)


# Step 2 - FineTuning - обучение половины весов EfficientNetb6

# Разморозим базовую модель
base_model.trainable = True

# Установим количество слоев, которые будем переобучать
fine_tune_at = len(base_model.layers)//2

# Заморозим первую половину слоев
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Количество параметров
len(base_model.trainable_variables)

model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=LR_STEP2), metrics=["accuracy"])
checkpoint2 = ModelCheckpoint(MODEL_PATH+'step2-{epoch:02d}-{val_loss:.4f}.hdf5' , monitor = ['val_accuracy'] , verbose = 1, mode = 'max')

# Обучаем

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples//train_generator.batch_size,
    validation_data = test_generator, 
    validation_steps = test_generator.samples//test_generator.batch_size,
    epochs = EPOCHS_STEP2,
    callbacks = [checkpoint2, earlystop],
    verbose=2
)

scores = model.evaluate(test_generator, verbose=2)
print("Accuracy step2: %.2f%%" % (scores[1]*100))


model.save('model/model_step2.hdf5')

while not os.path.isfile(weights_name := input('Input name of weights file: ')):
    print("File not exist!")
model.load_weights(weights_name)


# Step 3 - FineTuning - разморозка всей сети EfficientNetB6 и дообучение

# Разморозим базовую модель
base_model.trainable = True

LR=0.00001
model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=LR_STEP3), metrics=["accuracy"])
checkpoint3 = ModelCheckpoint(MODEL_PATH + 'step3-{epoch:02d}-{val_loss:.4f}.hdf5' , monitor = ['val_accuracy'] , verbose = 1, mode = 'max')

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples//train_generator.batch_size,
    validation_data = test_generator, 
    validation_steps = test_generator.samples//test_generator.batch_size,
    epochs = EPOCHS_STEP3,
    callbacks = [checkpoint3, earlystop],
    verbose=2
)

model.save('model/model_step3.hdf5')

scores = model.evaluate(test_generator, verbose=2)
print("Accuracy step3: %.2f%%" % (scores[1]*100))
