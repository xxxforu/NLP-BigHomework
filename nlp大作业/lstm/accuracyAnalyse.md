[//]: # (执行结果如下)
2024-12-12 11:50:02.093037: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable TF_ENABLE_ONEDNN_OPTS=0.
2024-12-12 11:50:02.931071: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable TF_ENABLE_ONEDNN_OPTS=0.
2024-12-12 11:50:04.694782: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
WARNING:tensorflow:From C:\Users\XXX\Anaconda3\envs\pytorch\Lib\site-packages\keras\src\backend\tensorflow\core.py:222: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.

Epoch 1/10
729/729 ━━━━━━━━━━━━━━━━━━━━ 355s 484ms/step - accuracy: 0.7444 - loss: 2.0274 - val_accuracy: 0.7872 - val_loss: 1.4200
Epoch 2/10
729/729 ━━━━━━━━━━━━━━━━━━━━ 349s 479ms/step - accuracy: 0.7947 - loss: 1.3410 - val_accuracy: 0.8105 - val_loss: 1.2429
Epoch 3/10
729/729 ━━━━━━━━━━━━━━━━━━━━ 337s 462ms/step - accuracy: 0.8183 - loss: 1.1294 - val_accuracy: 0.8223 - val_loss: 1.1468
Epoch 4/10
729/729 ━━━━━━━━━━━━━━━━━━━━ 331s 454ms/step - accuracy: 0.8309 - loss: 0.9856 - val_accuracy: 0.8300 - val_loss: 1.0766
Epoch 5/10
729/729 ━━━━━━━━━━━━━━━━━━━━ 334s 459ms/step - accuracy: 0.8427 - loss: 0.8519 - val_accuracy: 0.8366 - val_loss: 1.0287
Epoch 6/10
729/729 ━━━━━━━━━━━━━━━━━━━━ 333s 456ms/step - accuracy: 0.8555 - loss: 0.7339 - val_accuracy: 0.8424 - val_loss: 0.9937
Epoch 7/10
729/729 ━━━━━━━━━━━━━━━━━━━━ 332s 456ms/step - accuracy: 0.8707 - loss: 0.6204 - val_accuracy: 0.8470 - val_loss: 0.9764
Epoch 8/10
729/729 ━━━━━━━━━━━━━━━━━━━━ 337s 462ms/step - accuracy: 0.8862 - loss: 0.5253 - val_accuracy: 0.8509 - val_loss: 0.9626
Epoch 9/10
729/729 ━━━━━━━━━━━━━━━━━━━━ 339s 466ms/step - accuracy: 0.9029 - loss: 0.4354 - val_accuracy: 0.8528 - val_loss: 0.9608
Epoch 10/10
729/729 ━━━━━━━━━━━━━━━━━━━━ 330s 453ms/step - accuracy: 0.9181 - loss: 0.3629 - val_accuracy: 0.8549 - val_loss: 0.9651
WARNING:absl:You are saving your model as an HDF5 file via model.save() or keras.saving.save_model(model). This file format is considered legacy. We recommend using instead the native Keras format, e.g. model.save('my_model.keras') or keras.saving.save_model(model, 'my_model.keras'). 
Model training complete.

训练结果
准确率提升：训练集的准确率从 0.7444 增长到 0.9181，验证集准确率从 0.7872 增长到 0.8549。这表明模型在学习翻译任务上表现出色。
损失下降：训练集损失从 2.0274 降到 0.3629，验证集损失略有波动但整体下降，最终为 0.9651。验证集的损失略高于训练集，但差距在合理范围内，说明模型没有明显过拟合。
时间消耗：每个 epoch 约 5 分钟，整个训练耗时约 55 分钟（取决于硬件性能）。