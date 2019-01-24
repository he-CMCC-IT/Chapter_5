# !/usr/bin/env python
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
from Chapter_5 import building_our_network
from Chapter_5 import data_preprocessing

model = building_our_network.model
train_generator = data_preprocessing.train_generator
validation_generator = data_preprocessing.validation_generator

# 模型训练
history = model.fit_generator(train_generator,
                              steps_per_epoch=100,
                              epochs=30,
                              validation_data=validation_generator,
                              validation_steps=50)

# 保存模型
model.save('cats_and_dogs_small_1.h5')

# 训练验证的精度和损失绘制
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
