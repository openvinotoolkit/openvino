Big Transfer Image Classification Model Quantization pipeline with NNCF
=======================================================================

This tutorial demonstrates the Quantization of the Big Transfer Image
Classification model, which is fine-tuned on the sub-set of ImageNet
dataset with 10 class labels with
`NNCF <https://github.com/openvinotoolkit/nncf>`__. It uses
`BiT-M-R50x1/1 <https://www.kaggle.com/models/google/bit/frameworks/tensorFlow2/variations/m-r50x1/versions/1?tfhub-redirect=true>`__
model, which is trained on ImageNet-21k. Big Transfer is a recipe for
pre-training image classification models on large supervised datasets
and efficiently fine-tuning them on any given target task. The recipe
achieves excellent performance on a wide variety of tasks, even when
using very few labeled examples from the target dataset. This tutorial
uses OpenVINO backend for performing model quantization in NNCF.

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Prepare Dataset <#prepare-dataset>`__
-  `Plotting data samples <#plotting-data-samples>`__
-  `Model Fine-tuning <#model-fine-tuning>`__
-  `Perform model optimization (IR)
   step <#perform-model-optimization-ir-step>`__
-  `Compute accuracy of the TF
   model <#compute-accuracy-of-the-tf-model>`__
-  `Compute accuracy of the OpenVINO
   model <#compute-accuracy-of-the-openvino-model>`__
-  `Quantize OpenVINO model using
   NNCF <#quantize-openvino-model-using-nncf>`__
-  `Compute accuracy of the quantized
   model <#compute-accuracy-of-the-quantized-model>`__
-  `Compare FP32 and INT8 accuracy <#compare-fp32-and-int8-accuracy>`__
-  `Compare inference results on one
   picture <#compare-inference-results-on-one-picture>`__

.. code:: ipython3

    %pip install -q "tensorflow-macos>=2.5; sys_platform == 'darwin' and platform_machine == 'arm64' and python_version > '3.8'" # macOS M1 and M2
    %pip install -q "tensorflow-macos>=2.5,<=2.12.0; sys_platform == 'darwin' and platform_machine == 'arm64' and python_version <= '3.8'" # macOS M1 and M2
    %pip install -q "tensorflow>=2.5; sys_platform == 'darwin' and platform_machine != 'arm64' and python_version > '3.8'" # macOS x86
    %pip install -q "tensorflow>=2.5,<=2.12.0; sys_platform == 'darwin' and platform_machine != 'arm64' and python_version <= '3.8'" # macOS x86
    %pip install -q "tensorflow>=2.5; sys_platform != 'darwin' and python_version > '3.8'"
    %pip install -q "tensorflow>=2.5,<=2.12.0; sys_platform != 'darwin' and python_version <= '3.8'"

    %pip install -q "openvino>=2024.0.0" "nncf>=2.7.0" "tensorflow-hub>=0.15.0" "tensorflow_datasets" tf_keras
    %pip install -q "scikit-learn>=1.3.2"


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. code:: ipython3

    import os
    import sys
    import numpy as np
    from pathlib import Path

    from openvino.runtime import Core
    import openvino as ov
    import nncf
    import logging

    sys.path.append("../utils")
    from nncf.common.logging.logger import set_log_level
    set_log_level(logging.ERROR)

    from sklearn.metrics import accuracy_score

    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    import tensorflow as tf
    import tensorflow_datasets as tfds
    import tensorflow_hub as hub

    tfds.core.utils.gcs_utils._is_gcs_disabled = True
    os.environ['NO_GCE_CHECK'] = 'true'


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


.. code:: ipython3

    core = Core()
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


    # For top 5 labels.
    MAX_PREDS = 1
    TRAINING_BATCH_SIZE = 128
    BATCH_SIZE = 1
    IMG_SIZE = (256, 256)  # Default Imagenet image size
    NUM_CLASSES = 10  # For Imagenette dataset
    FINE_TUNING_STEPS = 1
    LR = 1e-5

    MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)  # From Imagenet dataset
    STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)  # From Imagenet dataset


Prepare Dataset
~~~~~~~~~~~~~~~



.. code:: ipython3

    datasets, datasets_info = tfds.load('imagenette/160px', shuffle_files=True, as_supervised=True, with_info=True,
                                        read_config=tfds.ReadConfig(shuffle_seed=0))
    train_ds, validation_ds = datasets['train'], datasets['validation']



.. parsed-literal::

    2024-03-25 22:57:53.190464: E tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:266] failed call to cuInit: CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: forward compatibility was attempted on non supported HW
    2024-03-25 22:57:53.190690: E tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:312] kernel version 470.182.3 does not match DSO version 470.223.2 -- cannot find working devices in this configuration


.. code:: ipython3

    def preprocessing(image, label):
        image = tf.image.resize(image, IMG_SIZE)
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.one_hot(label, NUM_CLASSES)
        return image, label

    train_dataset = (train_ds.map(preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                     .batch(TRAINING_BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE))
    validation_dataset = (validation_ds.map(preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                          .batch(TRAINING_BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE))

.. code:: ipython3

    # Class labels dictionary with imagenette sample names and classes
    lbl_dict = dict(
        n01440764='tench',
        n02102040='English springer',
        n02979186='cassette player',
        n03000684='chain saw',
        n03028079='church',
        n03394916='French horn',
        n03417042='garbage truck',
        n03425413='gas pump',
        n03445777='golf ball',
        n03888257='parachute'
    )

    # Imagenette samples name index
    class_idx_dict = ['n01440764', 'n02102040', 'n02979186', 'n03000684',
                      'n03028079', 'n03394916', 'n03417042', 'n03425413',
                      'n03445777', 'n03888257']

    def label_func(key):
        return lbl_dict[key]

Plotting data samples
~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    import matplotlib.pyplot as plt

    # Get the class labels from the dataset info
    class_labels = datasets_info.features['label'].names

    # Display labels along with the examples
    num_examples_to_display = 4
    fig, axes = plt.subplots(nrows=1, ncols=num_examples_to_display, figsize=(10, 5))

    for i, (image, label_index) in enumerate(train_ds.take(num_examples_to_display)):
        label_name = class_labels[label_index.numpy()]

        axes[i].imshow(image.numpy())
        axes[i].set_title(f"{label_func(label_name)}")
        axes[i].axis('off')
        plt.tight_layout()
    plt.show()



.. image:: 127-tensorflow-bit-image-classification-nncf-quantization-with-output_files/127-tensorflow-bit-image-classification-nncf-quantization-with-output_9_0.png


.. code:: ipython3

    # Get the class labels from the dataset info
    class_labels = datasets_info.features['label'].names

    # Display labels along with the examples
    num_examples_to_display = 4
    fig, axes = plt.subplots(nrows=1, ncols=num_examples_to_display, figsize=(10, 5))

    for i, (image, label_index) in enumerate(validation_ds.take(num_examples_to_display)):
        label_name = class_labels[label_index.numpy()]

        axes[i].imshow(image.numpy())
        axes[i].set_title(f"{label_func(label_name)}")
        axes[i].axis('off')
        plt.tight_layout()
    plt.show()



.. image:: 127-tensorflow-bit-image-classification-nncf-quantization-with-output_files/127-tensorflow-bit-image-classification-nncf-quantization-with-output_10_0.png


Model Fine-tuning
~~~~~~~~~~~~~~~~~



.. code:: ipython3

    # Load the Big Transfer model
    bit_model_url = "https://www.kaggle.com/models/google/bit/frameworks/TensorFlow2/variations/m-r50x1/versions/1"
    bit_m = hub.KerasLayer(bit_model_url, trainable=True)

    # Customize the model for the new task
    model = tf.keras.Sequential([
        bit_m,
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Fine-tune the model
    model.fit(train_dataset.take(3000),
              epochs=FINE_TUNING_STEPS,
              validation_data=validation_dataset.take(1000))
    model.save("./bit_tf_model/", save_format='tf')


.. parsed-literal::


  1/101 [..............................] - ETA: 45:40 - loss: 4.2558 - accuracy: 0.0859

.. parsed-literal::

    
  2/101 [..............................] - ETA: 15:17 - loss: 4.1474 - accuracy: 0.1250

.. parsed-literal::

    
  3/101 [..............................] - ETA: 15:10 - loss: 3.8788 - accuracy: 0.1667

.. parsed-literal::

    
  4/101 [>.............................] - ETA: 14:59 - loss: 3.6036 - accuracy: 0.1973

.. parsed-literal::

    
  5/101 [>.............................] - ETA: 14:48 - loss: 3.3633 - accuracy: 0.2281

.. parsed-literal::

    
  6/101 [>.............................] - ETA: 14:39 - loss: 3.1307 - accuracy: 0.2682

.. parsed-literal::

    
  7/101 [=>............................] - ETA: 14:29 - loss: 2.8882 - accuracy: 0.3136

.. parsed-literal::

    
  8/101 [=>............................] - ETA: 14:19 - loss: 2.6904 - accuracy: 0.3516

.. parsed-literal::

    
  9/101 [=>............................] - ETA: 14:10 - loss: 2.4984 - accuracy: 0.3976

.. parsed-literal::

    
 10/101 [=>............................] - ETA: 14:01 - loss: 2.3419 - accuracy: 0.4352

.. parsed-literal::

    
 11/101 [==>...........................] - ETA: 13:52 - loss: 2.2112 - accuracy: 0.4645

.. parsed-literal::

    
 12/101 [==>...........................] - ETA: 13:43 - loss: 2.0797 - accuracy: 0.4915

.. parsed-literal::

    
 13/101 [==>...........................] - ETA: 13:34 - loss: 1.9630 - accuracy: 0.5198

.. parsed-literal::

    
 14/101 [===>..........................] - ETA: 13:25 - loss: 1.8636 - accuracy: 0.5435

.. parsed-literal::

    
 15/101 [===>..........................] - ETA: 13:15 - loss: 1.7697 - accuracy: 0.5641

.. parsed-literal::

    
 16/101 [===>..........................] - ETA: 13:06 - loss: 1.6770 - accuracy: 0.5889

.. parsed-literal::

    
 17/101 [====>.........................] - ETA: 12:57 - loss: 1.5973 - accuracy: 0.6071

.. parsed-literal::

    
 18/101 [====>.........................] - ETA: 12:48 - loss: 1.5206 - accuracy: 0.6254

.. parsed-literal::

    
 19/101 [====>.........................] - ETA: 12:39 - loss: 1.4620 - accuracy: 0.6386

.. parsed-literal::

    
 20/101 [====>.........................] - ETA: 12:29 - loss: 1.4064 - accuracy: 0.6523

.. parsed-literal::

    
 21/101 [=====>........................] - ETA: 12:20 - loss: 1.3516 - accuracy: 0.6652

.. parsed-literal::

    
 22/101 [=====>........................] - ETA: 12:11 - loss: 1.2978 - accuracy: 0.6783

.. parsed-literal::

    
 23/101 [=====>........................] - ETA: 12:01 - loss: 1.2578 - accuracy: 0.6878

.. parsed-literal::

    
 24/101 [======>.......................] - ETA: 11:52 - loss: 1.2103 - accuracy: 0.6989

.. parsed-literal::

    
 25/101 [======>.......................] - ETA: 11:43 - loss: 1.1785 - accuracy: 0.7072

.. parsed-literal::

    
 26/101 [======>.......................] - ETA: 11:33 - loss: 1.1457 - accuracy: 0.7148

.. parsed-literal::

    
 27/101 [=======>......................] - ETA: 11:24 - loss: 1.1094 - accuracy: 0.7228

.. parsed-literal::

    
 28/101 [=======>......................] - ETA: 11:15 - loss: 1.0773 - accuracy: 0.7307

.. parsed-literal::

    
 29/101 [=======>......................] - ETA: 11:06 - loss: 1.0499 - accuracy: 0.7368

.. parsed-literal::

    
 30/101 [=======>......................] - ETA: 10:57 - loss: 1.0230 - accuracy: 0.7437

.. parsed-literal::

    
 31/101 [========>.....................] - ETA: 10:47 - loss: 0.9949 - accuracy: 0.7508

.. parsed-literal::

    
 32/101 [========>.....................] - ETA: 10:38 - loss: 0.9745 - accuracy: 0.7559

.. parsed-literal::

    
 33/101 [========>.....................] - ETA: 10:29 - loss: 0.9514 - accuracy: 0.7618

.. parsed-literal::

    
 34/101 [=========>....................] - ETA: 10:20 - loss: 0.9337 - accuracy: 0.7661

.. parsed-literal::

    
 35/101 [=========>....................] - ETA: 10:10 - loss: 0.9149 - accuracy: 0.7708

.. parsed-literal::

    
 36/101 [=========>....................] - ETA: 10:01 - loss: 0.8927 - accuracy: 0.7754

.. parsed-literal::

    
 37/101 [=========>....................] - ETA: 9:52 - loss: 0.8722 - accuracy: 0.7802

.. parsed-literal::

    
 38/101 [==========>...................] - ETA: 9:43 - loss: 0.8541 - accuracy: 0.7845

.. parsed-literal::

    
 39/101 [==========>...................] - ETA: 9:33 - loss: 0.8330 - accuracy: 0.7899

.. parsed-literal::

    
 40/101 [==========>...................] - ETA: 9:24 - loss: 0.8181 - accuracy: 0.7936

.. parsed-literal::

    
 41/101 [===========>..................] - ETA: 9:15 - loss: 0.8015 - accuracy: 0.7974

.. parsed-literal::

    
 42/101 [===========>..................] - ETA: 9:06 - loss: 0.7839 - accuracy: 0.8017

.. parsed-literal::

    
 43/101 [===========>..................] - ETA: 8:56 - loss: 0.7674 - accuracy: 0.8056

.. parsed-literal::

    
 44/101 [============>.................] - ETA: 8:47 - loss: 0.7536 - accuracy: 0.8086

.. parsed-literal::

    
 45/101 [============>.................] - ETA: 8:38 - loss: 0.7412 - accuracy: 0.8118

.. parsed-literal::

    
 46/101 [============>.................] - ETA: 8:29 - loss: 0.7270 - accuracy: 0.8154

.. parsed-literal::

    
 47/101 [============>.................] - ETA: 8:20 - loss: 0.7134 - accuracy: 0.8185

.. parsed-literal::

    
 48/101 [=============>................] - ETA: 8:10 - loss: 0.7014 - accuracy: 0.8215

.. parsed-literal::

    
 49/101 [=============>................] - ETA: 8:01 - loss: 0.6902 - accuracy: 0.8241

.. parsed-literal::

    
 50/101 [=============>................] - ETA: 7:52 - loss: 0.6808 - accuracy: 0.8266

.. parsed-literal::

    
 51/101 [==============>...............] - ETA: 7:43 - loss: 0.6691 - accuracy: 0.8294

.. parsed-literal::

    
 52/101 [==============>...............] - ETA: 7:33 - loss: 0.6574 - accuracy: 0.8322

.. parsed-literal::

    
 53/101 [==============>...............] - ETA: 7:24 - loss: 0.6465 - accuracy: 0.8349

.. parsed-literal::

    
 54/101 [===============>..............] - ETA: 7:15 - loss: 0.6362 - accuracy: 0.8374

.. parsed-literal::

    
 55/101 [===============>..............] - ETA: 7:06 - loss: 0.6263 - accuracy: 0.8399

.. parsed-literal::

    
 56/101 [===============>..............] - ETA: 6:56 - loss: 0.6163 - accuracy: 0.8424

.. parsed-literal::

    
 57/101 [===============>..............] - ETA: 6:47 - loss: 0.6112 - accuracy: 0.8435

.. parsed-literal::

    
 58/101 [================>.............] - ETA: 6:38 - loss: 0.6034 - accuracy: 0.8454

.. parsed-literal::

    
 59/101 [================>.............] - ETA: 6:28 - loss: 0.5957 - accuracy: 0.8475

.. parsed-literal::

    
 60/101 [================>.............] - ETA: 6:19 - loss: 0.5872 - accuracy: 0.8495

.. parsed-literal::

    
 61/101 [=================>............] - ETA: 6:10 - loss: 0.5789 - accuracy: 0.8513

.. parsed-literal::

    
 62/101 [=================>............] - ETA: 6:01 - loss: 0.5706 - accuracy: 0.8533

.. parsed-literal::

    
 63/101 [=================>............] - ETA: 5:51 - loss: 0.5649 - accuracy: 0.8545

.. parsed-literal::

    
 64/101 [==================>...........] - ETA: 5:42 - loss: 0.5597 - accuracy: 0.8560

.. parsed-literal::

    
 65/101 [==================>...........] - ETA: 5:33 - loss: 0.5550 - accuracy: 0.8575

.. parsed-literal::

    
 66/101 [==================>...........] - ETA: 5:24 - loss: 0.5490 - accuracy: 0.8590

.. parsed-literal::

    
 67/101 [==================>...........] - ETA: 5:14 - loss: 0.5428 - accuracy: 0.8601

.. parsed-literal::

    
 68/101 [===================>..........] - ETA: 5:05 - loss: 0.5366 - accuracy: 0.8617

.. parsed-literal::

    
 69/101 [===================>..........] - ETA: 4:56 - loss: 0.5301 - accuracy: 0.8633

.. parsed-literal::

    
 70/101 [===================>..........] - ETA: 4:47 - loss: 0.5233 - accuracy: 0.8650

.. parsed-literal::

    
 71/101 [====================>.........] - ETA: 4:37 - loss: 0.5173 - accuracy: 0.8665

.. parsed-literal::

    
 72/101 [====================>.........] - ETA: 4:28 - loss: 0.5115 - accuracy: 0.8682

.. parsed-literal::

    
 73/101 [====================>.........] - ETA: 4:19 - loss: 0.5056 - accuracy: 0.8698

.. parsed-literal::

    
 74/101 [====================>.........] - ETA: 4:10 - loss: 0.5005 - accuracy: 0.8707

.. parsed-literal::

    
 75/101 [=====================>........] - ETA: 4:00 - loss: 0.4959 - accuracy: 0.8718

.. parsed-literal::

    
 76/101 [=====================>........] - ETA: 3:51 - loss: 0.4906 - accuracy: 0.8728

.. parsed-literal::

    
 77/101 [=====================>........] - ETA: 3:42 - loss: 0.4862 - accuracy: 0.8741

.. parsed-literal::

    
 78/101 [======================>.......] - ETA: 3:33 - loss: 0.4806 - accuracy: 0.8755

.. parsed-literal::

    
 79/101 [======================>.......] - ETA: 3:23 - loss: 0.4755 - accuracy: 0.8768

.. parsed-literal::

    
 80/101 [======================>.......] - ETA: 3:14 - loss: 0.4714 - accuracy: 0.8779

.. parsed-literal::

    
 81/101 [=======================>......] - ETA: 3:05 - loss: 0.4658 - accuracy: 0.8794

.. parsed-literal::

    
 82/101 [=======================>......] - ETA: 2:56 - loss: 0.4609 - accuracy: 0.8807

.. parsed-literal::

    
 83/101 [=======================>......] - ETA: 2:46 - loss: 0.4559 - accuracy: 0.8819

.. parsed-literal::

    
 84/101 [=======================>......] - ETA: 2:37 - loss: 0.4508 - accuracy: 0.8833

.. parsed-literal::

    
 85/101 [========================>.....] - ETA: 2:28 - loss: 0.4461 - accuracy: 0.8845

.. parsed-literal::

    
 86/101 [========================>.....] - ETA: 2:18 - loss: 0.4413 - accuracy: 0.8857

.. parsed-literal::

    
 87/101 [========================>.....] - ETA: 2:09 - loss: 0.4369 - accuracy: 0.8868

.. parsed-literal::

    
 88/101 [=========================>....] - ETA: 2:00 - loss: 0.4328 - accuracy: 0.8878

.. parsed-literal::

    
 89/101 [=========================>....] - ETA: 1:51 - loss: 0.4287 - accuracy: 0.8888

.. parsed-literal::

    
    90/101 [=========================>....] - ETA: 1:41 - loss: 0.4250 - accuracy: 0.8897

.. parsed-literal::

    
    91/101 [==========================>...] - ETA: 1:32 - loss: 0.4206 - accuracy: 0.8908

.. parsed-literal::

    
    92/101 [==========================>...] - ETA: 1:23 - loss: 0.4163 - accuracy: 0.8918

.. parsed-literal::

    
    93/101 [==========================>...] - ETA: 1:14 - loss: 0.4121 - accuracy: 0.8928

.. parsed-literal::

    
    94/101 [==========================>...] - ETA: 1:04 - loss: 0.4096 - accuracy: 0.8935

.. parsed-literal::

    
    95/101 [===========================>..] - ETA: 55s - loss: 0.4053 - accuracy: 0.8946

.. parsed-literal::

    
    96/101 [===========================>..] - ETA: 46s - loss: 0.4021 - accuracy: 0.8954

.. parsed-literal::

    
    97/101 [===========================>..] - ETA: 37s - loss: 0.3992 - accuracy: 0.8961

.. parsed-literal::

    
    98/101 [============================>.] - ETA: 27s - loss: 0.3957 - accuracy: 0.8971

.. parsed-literal::

    
    99/101 [============================>.] - ETA: 18s - loss: 0.3941 - accuracy: 0.8978

.. parsed-literal::

    
    100/101 [============================>.] - ETA: 9s - loss: 0.3918 - accuracy: 0.8984

.. parsed-literal::

    
    101/101 [==============================] - ETA: 0s - loss: 0.3892 - accuracy: 0.8991

.. parsed-literal::

    
    101/101 [==============================] - 967s 9s/step - loss: 0.3892 - accuracy: 0.8991 - val_loss: 0.0904 - val_accuracy: 0.9760


.. parsed-literal::

    WARNING:absl:Found untraced functions such as _update_step_xla while saving (showing 1 of 1). These functions will not be directly callable after loading.


Perform model optimization (IR) step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    ir_path = Path("./bit_ov_model/bit_m_r50x1_1.xml")
    if not ir_path.exists():
        print("Initiating model optimization..!!!")
        ov_model = ov.convert_model("./bit_tf_model")
        ov.save_model(ov_model, ir_path)
    else:
        print(f"IR model {ir_path} already exists.")


.. parsed-literal::

    Initiating model optimization..!!!


Compute accuracy of the TF model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    tf_model = tf.keras.models.load_model("./bit_tf_model/")

    tf_predictions = []
    gt_label = []

    for _, label in validation_dataset:
        for cls_label in label:
            l_list = cls_label.numpy().tolist()
            gt_label.append(l_list.index(1))

    for img_batch, label_batch in validation_dataset:
        tf_result_batch = tf_model.predict(img_batch, verbose=0)
        for i in range(len(img_batch)):
            tf_result = tf_result_batch[i]
            tf_result = tf.reshape(tf_result, [-1])
            top5_label_idx = np.argsort(tf_result)[-MAX_PREDS::][::-1]
            tf_predictions.append(top5_label_idx)

    # Convert the lists to NumPy arrays for accuracy calculation
    tf_predictions = np.array(tf_predictions)
    gt_label = np.array(gt_label)

    tf_acc_score = accuracy_score(tf_predictions, gt_label)


Compute accuracy of the OpenVINO model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Select device for inference:

.. code:: ipython3

    import ipywidgets as widgets

    core = ov.Core()

    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value='AUTO',
        description='Device:',
        disabled=False,
    )

    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



.. code:: ipython3

    ov_fp32_model = core.read_model("./bit_ov_model/bit_m_r50x1_1.xml")
    ov_fp32_model.reshape([1, IMG_SIZE[0], IMG_SIZE[1], 3])

    # Target device set to CPU (Other options Ex: AUTO/GPU/dGPU/)
    compiled_model = ov.compile_model(ov_fp32_model, device.value)
    output = compiled_model.outputs[0]

    ov_predictions = []
    for img_batch, _ in validation_dataset:
        for image in img_batch:
            image = tf.expand_dims(image, axis=0)
            pred = compiled_model(image)[output]
            ov_result = tf.reshape(pred, [-1])
            top_label_idx = np.argsort(ov_result)[-MAX_PREDS::][::-1]
            ov_predictions.append(top_label_idx)

    fp32_acc_score = accuracy_score(ov_predictions, gt_label)


Quantize OpenVINO model using NNCF
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Model Quantization using NNCF

1. Preprocessing and preparing validation samples for NNCF calibration
2. Perform NNCF Quantization on OpenVINO FP32 model
3. Serialize Quantized OpenVINO INT8 model

.. code:: ipython3

    def nncf_preprocessing(image, label):
        image = tf.image.resize(image, IMG_SIZE)
        image = image - MEAN_RGB
        image = image / STDDEV_RGB
        return image

    val_ds = (validation_ds.map(nncf_preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
              .batch(1)
              .prefetch(tf.data.experimental.AUTOTUNE))

    calibration_dataset = nncf.Dataset(val_ds)

    ov_fp32_model = core.read_model("./bit_ov_model/bit_m_r50x1_1.xml")

    ov_int8_model = nncf.quantize(ov_fp32_model, calibration_dataset, fast_bias_correction=False)

    ov.save_model(ov_int8_model, "./bit_ov_int8_model/bit_m_r50x1_1_ov_int8.xml")



.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>




.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>



Compute accuracy of the quantized model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    nncf_quantized_model = core.read_model("./bit_ov_int8_model/bit_m_r50x1_1_ov_int8.xml")
    nncf_quantized_model.reshape([1, IMG_SIZE[0], IMG_SIZE[1], 3])

    # Target device set to CPU by default
    compiled_model = ov.compile_model(nncf_quantized_model, device.value)
    output = compiled_model.outputs[0]

    ov_predictions = []
    inp_tensor = nncf_quantized_model.inputs[0]
    out_tensor = nncf_quantized_model.outputs[0]

    for img_batch, _ in validation_dataset:
        for image in img_batch:
            image = tf.expand_dims(image, axis=0)
            pred = compiled_model(image)[output]
            ov_result = tf.reshape(pred, [-1])
            top_label_idx = np.argsort(ov_result)[-MAX_PREDS::][::-1]
            ov_predictions.append(top_label_idx)

    int8_acc_score = accuracy_score(ov_predictions, gt_label)


Compare FP32 and INT8 accuracy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    print(f"Accuracy of the tensorflow model (fp32): {tf_acc_score * 100: .2f}%")
    print(f"Accuracy of the OpenVINO optimized model (fp32): {fp32_acc_score * 100: .2f}%")
    print(f"Accuracy of the OpenVINO quantized model (int8): {int8_acc_score * 100: .2f}%")
    accuracy_drop = fp32_acc_score - int8_acc_score
    print(f"Accuracy drop between OV FP32 and INT8 model: {accuracy_drop * 100:.1f}% ")


.. parsed-literal::

    Accuracy of the tensorflow model (fp32):  97.60%
    Accuracy of the OpenVINO optimized model (fp32):  97.60%
    Accuracy of the OpenVINO quantized model (int8):  97.60%
    Accuracy drop between OV FP32 and INT8 model: 0.0%


Compare inference results on one picture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3


    # Accessing validation sample
    sample_idx = 50
    vds = datasets['validation']

    if len(vds) > sample_idx:
        sample = vds.take(sample_idx + 1).skip(sample_idx).as_numpy_iterator().next()
    else:
        print("Dataset does not have enough samples...!!!")

    # Image data
    sample_data = sample[0]

    # Label info
    sample_label = sample[1]

    # Image data pre-processing
    image = tf.image.resize(sample_data, IMG_SIZE)
    image = tf.expand_dims(image, axis=0)
    image = tf.cast(image, tf.float32) / 255.0

    # OpenVINO inference
    def ov_inference(model: ov.Model, image) -> str:
        compiled_model = ov.compile_model(model, device.value)
        output = compiled_model.outputs[0]
        pred = compiled_model(image)[output]
        ov_result = tf.reshape(pred, [-1])
        pred_label = np.argsort(ov_result)[-MAX_PREDS::][::-1]
        return pred_label

    # OpenVINO FP32 model
    ov_fp32_model = core.read_model("./bit_ov_model/bit_m_r50x1_1.xml")
    ov_fp32_model.reshape([1, IMG_SIZE[0], IMG_SIZE[1], 3])

    # OpenVINO INT8 model
    ov_int8_model = core.read_model("./bit_ov_int8_model/bit_m_r50x1_1_ov_int8.xml")
    ov_int8_model.reshape([1, IMG_SIZE[0], IMG_SIZE[1], 3])

    # OpenVINO FP32 model inference
    ov_fp32_pred_label = ov_inference(ov_fp32_model, image)

    print(f"Predicted label for the sample picture by float (fp32) model: {label_func(class_idx_dict[int(ov_fp32_pred_label)])}\n")

    # OpenVINO FP32 model inference
    ov_int8_pred_label = ov_inference(ov_int8_model, image)
    print(f"Predicted label for the sample picture by qunatized (int8) model: {label_func(class_idx_dict[int(ov_int8_pred_label)])}\n")

    # Plotting the image sample with ground truth
    plt.figure()
    plt.imshow(sample_data)
    plt.title(f"Ground truth: {label_func(class_idx_dict[sample_label])}")
    plt.axis('off')
    plt.show()



.. parsed-literal::

    Predicted label for the sample picture by float (fp32) model: gas pump



.. parsed-literal::

    Predicted label for the sample picture by qunatized (int8) model: gas pump




.. image:: 127-tensorflow-bit-image-classification-nncf-quantization-with-output_files/127-tensorflow-bit-image-classification-nncf-quantization-with-output_27_2.png

