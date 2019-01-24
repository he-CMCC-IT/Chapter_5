# !/usr/bin/env python
# -*- coding:utf-8 -*-
from keras.preprocessing.image import ImageDataGenerator
from Chapter_5 import downloading_the_data

train_dir = downloading_the_data.train_dir
validation_dir = downloading_the_data.validation_dir

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

# 验证数据准确性
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break
"""
data batch shape: (20, 150, 150, 3)
labels batch shape: (20,)
"""
