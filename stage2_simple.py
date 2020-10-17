# # Установка и импорт необходимых библиотек

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import pickle
#import scipy.io
from skimage import io
import csv
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
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import optimizers
import efficientnet.tfkeras as efn

#from ImageDataAugmentor.image_data_augmentor import *
#import albumentations

from sklearn.model_selection import train_test_split

#import PIL
#from PIL import ImageOps, ImageFilter

# # Основные настройки

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

# Batch size
BATCH_SIZE           = 8 # уменьшаем batch если сеть большая, иначе не влезет в память на GPU
BATCH_SIZE_STEP4     = 2 # Размер batch на шаге с увеличенными изображениями

# Epochs
  
EPOCHS_STEP1         = 1  
EPOCHS_STEP2         = 1  
EPOCHS_STEP3         = 1    
EPOCHS_STEP4         = 1  

# Learning Rates
LR_STEP1             = 1e-3
LR_STEP2             = 1e-4
LR_STEP3             = 1e-5
LR_STEP4             = 1e-5


# Test-validation split
VAL_SPLIT            = 0.2 # сколько данных выделяем на тест = 20%

CLASS_NUM            = 10  # количество классов в нашей задаче

IMG_SIZE             = 250 # какого размера подаем изображения в сеть
IMG_SIZE_STEP4       = 512
IMG_CHANNELS         = 3   # у RGB 3 канала
input_shape          = (IMG_SIZE_STEP4, IMG_SIZE_STEP4, IMG_CHANNELS)


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)  
PYTHONHASHSEED = 0
DATA_PATH = '/home/alex/Car_Classification/input/'
MODEL_PATH = '/home/alex/Car_Classification/model/'
sample_submission = pd.read_csv(DATA_PATH+"sample-submission.csv")

# Step - увеличение размера изображения

# Увеличим размер изображения и уменьшим уровень аугментации

input_shape_step4 = (IMG_SIZE_STEP4, IMG_SIZE_STEP4, IMG_CHANNELS)

# Пересобираем модель под новый input_shape

base_model = efn.EfficientNetB6(
    weights=None, # Веса не подгружаем, т.к. будем загружать уже обученные
    include_top=False,  # Выходной слой (голову) будем менять т.к. у нас другие классы и их количество
    input_shape=input_shape_step4)

model=M.Sequential()
model.add(base_model)
model.add(L.GlobalAveragePooling2D(),) 
model.add(L.Dense(256, activation='relu'))
model.add(L.BatchNormalization())
model.add(L.Dropout(0.25))
model.add(L.Dense(CLASS_NUM, activation='softmax'))

while not os.path.isfile(model_name := input('Input name of model file: ')):
    print(model_name, "File not exist!")
model = keras.models.load_model(model_name)

train_datagen = ImageDataGenerator(
    rescale=1. / 255, 
    rotation_range = 30,
    #shear_range=0.2,
    #zoom_range=[0.75,1.25],
    #brightness_range=[0.5, 1.5],
    #width_shift_range=0.1,
    #height_shift_range=0.1,
    validation_split=VAL_SPLIT,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    DATA_PATH+'train/',      # директория где расположены папки с картинками 
    target_size = (IMG_SIZE_STEP4, IMG_SIZE_STEP4),
    batch_size = BATCH_SIZE_STEP4,
    class_mode = 'categorical',
    shuffle = True, 
    seed = RANDOM_SEED,
    subset = 'training') # set as training data

test_generator = train_datagen.flow_from_directory(
    DATA_PATH+'train/',
    target_size = (IMG_SIZE_STEP4, IMG_SIZE_STEP4),
    batch_size = BATCH_SIZE_STEP4,
    class_mode = 'categorical',
    shuffle = True, 
    seed = RANDOM_SEED,
    subset = 'validation') # set as validation data


# Заново создаем сеть с новым размером входных данных


# Настройки
batch_size = BATCH_SIZE_STEP4
epochs = EPOCHS_STEP4
base_lr = LR_STEP4
max_lr = base_lr*10


# Расчет количества итерация и шага изменения learning rate
iterations = round(train_generator.samples//train_generator.batch_size*epochs)
iterations = list(range(0,iterations+1))
model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=LR_STEP4), metrics=["accuracy"])  


# Добавим ModelCheckpoint чтоб сохранять прогресс обучения модели и можно было потом подгрузить и дообучить модель.    
checkpoint = ModelCheckpoint(MODEL_PATH+'step4-{epoch:02d}-{val_loss:.4f}.hdf5' , monitor = ['val_accuracy'] , verbose = 1  , mode = 'max')
earlystop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
callbacks_list = [checkpoint, earlystop]

# Обучаем
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples//train_generator.batch_size,
    validation_data = test_generator, 
    validation_steps = test_generator.samples//test_generator.batch_size,
    epochs = epochs,
    callbacks = callbacks_list
)



model.save(MODEL_PATH+'model_step4.hdf5')
while not os.path.isfile(weights_name := input('Input name of weights file: ')):
    print("File not exist!")

model.load_weights(weights_name)

scores = model.evaluate_generator(test_generator, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))


# # Предсказание на тестовых данных


from sklearn.metrics import accuracy_score



test_sub_generator = test_datagen.flow_from_dataframe( 
    dataframe=sample_submission,
    directory=DATA_PATH+'test_upload/',
    x_col="Id",
    y_col=None,
    shuffle=False,
    class_mode=None,
    seed=RANDOM_SEED,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,)


test_sub_generator.reset()
predictions = model.predict(test_sub_generator, steps=len(test_sub_generator), verbose=1) 
predictions = np.argmax(predictions, axis=-1) #multiple categories
label_map = (train_generator.class_indices)
label_map = dict((v,k) for k,v in label_map.items()) #flip k,v
predictions = [label_map[k] for k in predictions]



filenames_with_dir=test_sub_generator.filenames
submission = pd.DataFrame({'Id':filenames_with_dir, 'Category':predictions}, columns=['Id', 'Category'])
submission['Id'] = submission['Id'].replace('test_upload/','')
submission.to_csv('submission.csv', index=False)
print('Save submit')



submission.head()


# ## Test Time Augmentation
# https://towardsdatascience.com/test-time-augmentation-tta-and-how-to-perform-it-with-keras-4ac19b67fb4d
# 
# Аугментируем тестовые изображения и сделаем несколько предсказаний одной картинки в разном виде.
# Взяв среднее значение из нескольких предсказаний получим итоговое предсказание.


test_datagen = ImageDataGenerator( 
    rescale=1. / 255, 
    rotation_range = 30,
    shear_range=0.2,
    zoom_range=[0.75,1.25],
    brightness_range=[0.5, 1.5],
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=VAL_SPLIT,
    horizontal_flip=True)


test_sub_generator = test_datagen.flow_from_dataframe( 
    dataframe=sample_submission,
    directory=DATA_PATH+'test_upload/',
    x_col="Id",
    y_col=None,
    shuffle=False,
    class_mode=None,
    seed=RANDOM_SEED,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,)



tta_steps = 7 # берем среднее из 7 предсказаний
predictions = []

for i in range(tta_steps):
    preds = model.predict(test_sub_generator, steps=len(test_sub_generator), verbose=1) 
    predictions.append(preds)

pred = np.mean(predictions, axis=0)



predictions = np.argmax(pred, axis=-1) #multiple categories
label_map = (train_generator.class_indices)
label_map = dict((v,k) for k,v in label_map.items()) #flip k,v
predictions = [label_map[k] for k in predictions]



filenames_with_dir=test_sub_generator.filenames
submission = pd.DataFrame({'Id':filenames_with_dir, 'Category':predictions}, columns=['Id', 'Category'])
submission['Id'] = submission['Id'].replace('test_upload/','')
submission.to_csv('submission_TTA.csv', index=False)
print('Save submit')

