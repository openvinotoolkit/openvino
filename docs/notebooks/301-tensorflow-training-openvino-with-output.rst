From Training to Deployment with TensorFlow and OpenVINO™
=========================================================

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `TensorFlow Image Classification
   Training <#tensorflow-image-classification-training>`__
-  `Import TensorFlow and Other
   Libraries <#import-tensorflow-and-other-libraries>`__
-  `Download and Explore the
   Dataset <#download-and-explore-the-dataset>`__
-  `Load Using keras.preprocessing <#load-using-keras-preprocessing>`__
-  `Create a Dataset <#create-a-dataset>`__
-  `Visualize the Data <#visualize-the-data>`__
-  `Configure the Dataset for
   Performance <#configure-the-dataset-for-performance>`__
-  `Standardize the Data <#standardize-the-data>`__
-  `Create the Model <#create-the-model>`__
-  `Compile the Model <#compile-the-model>`__
-  `Model Summary <#model-summary>`__
-  `Train the Model <#train-the-model>`__
-  `Visualize Training Results <#visualize-training-results>`__
-  `Overfitting <#overfitting>`__
-  `Data Augmentation <#data-augmentation>`__
-  `Dropout <#dropout>`__
-  `Compile and Train the Model <#compile-and-train-the-model>`__
-  `Visualize Training Results <#visualize-training-results>`__
-  `Predict on New Data <#predict-on-new-data>`__
-  `Save the TensorFlow Model <#save-the-tensorflow-model>`__
-  `Convert the TensorFlow model with OpenVINO Model Conversion
   API <#convert-the-tensorflow-model-with-openvino-model-conversion-api>`__
-  `Preprocessing Image Function <#preprocessing-image-function>`__
-  `OpenVINO Runtime Setup <#openvino-runtime-setup>`__

   -  `Select inference device <#select-inference-device>`__

-  `Run the Inference Step <#run-the-inference-step>`__
-  `The Next Steps <#the-next-steps>`__

.. code:: ipython3

    # @title Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    # https://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.

    # Copyright 2018 The TensorFlow Authors
    #
    # Modified for OpenVINO Notebooks

This tutorial demonstrates how to train, convert, and deploy an image
classification model with TensorFlow and OpenVINO. This particular
notebook shows the process where we perform the inference step on the
freshly trained model that is converted to OpenVINO IR with model
conversion API. For faster inference speed on the model created in this
notebook, check out the `Post-Training Quantization with TensorFlow
Classification Model <301-tensorflow-training-openvino-nncf-with-output.html>`__
notebook.

This training code comprises the official `TensorFlow Image
Classification
Tutorial <https://www.tensorflow.org/tutorials/images/classification>`__
in its entirety.

The ``flower_ir.bin`` and ``flower_ir.xml`` (pre-trained models) can be
obtained by executing the code with ‘Runtime->Run All’ or the
``Ctrl+F9`` command.

.. code:: ipython3

    %pip install -q "openvino>=2023.1.0"


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


TensorFlow Image Classification Training
----------------------------------------



The first part of the tutorial shows how to classify images of flowers
(based on the TensorFlow’s official tutorial). It creates an image
classifier using a ``keras.Sequential`` model, and loads data using
``preprocessing.image_dataset_from_directory``. You will gain practical
experience with the following concepts:

-  Efficiently loading a dataset off disk.
-  Identifying overfitting and applying techniques to mitigate it,
   including data augmentation and Dropout.

This tutorial follows a basic machine learning workflow:

1. Examine and understand data
2. Build an input pipeline
3. Build the model
4. Train the model
5. Test the model

Import TensorFlow and Other Libraries
-------------------------------------



.. code:: ipython3

    import os
    import sys
    from pathlib import Path

    import PIL
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    from PIL import Image
    import openvino as ov
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.models import Sequential

    sys.path.append("../utils")
    from notebook_utils import download_file


.. parsed-literal::

    2024-03-13 01:02:24.497427: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-03-13 01:02:24.532546: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


.. parsed-literal::

    2024-03-13 01:02:25.044342: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


Download and Explore the Dataset
--------------------------------



This tutorial uses a dataset of about 3,700 photos of flowers. The
dataset contains 5 sub-directories, one per class:

::

   flower_photo/
     daisy/
     dandelion/
     roses/
     sunflowers/
     tulips/

.. code:: ipython3

    import pathlib
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)

After downloading, you should now have a copy of the dataset available.
There are 3,670 total images:

.. code:: ipython3

    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)


.. parsed-literal::

    3670


Here are some roses:

.. code:: ipython3

    roses = list(data_dir.glob('roses/*'))
    PIL.Image.open(str(roses[0]))




.. image:: 301-tensorflow-training-openvino-with-output_files/301-tensorflow-training-openvino-with-output_14_0.png



.. code:: ipython3

    PIL.Image.open(str(roses[1]))




.. image:: 301-tensorflow-training-openvino-with-output_files/301-tensorflow-training-openvino-with-output_15_0.png



And some tulips:

.. code:: ipython3

    tulips = list(data_dir.glob('tulips/*'))
    PIL.Image.open(str(tulips[0]))




.. image:: 301-tensorflow-training-openvino-with-output_files/301-tensorflow-training-openvino-with-output_17_0.png



.. code:: ipython3

    PIL.Image.open(str(tulips[1]))




.. image:: 301-tensorflow-training-openvino-with-output_files/301-tensorflow-training-openvino-with-output_18_0.png



Load Using keras.preprocessing
------------------------------



Let’s load these images off disk using the helpful
`image_dataset_from_directory <https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory>`__
utility. This will take you from a directory of images on disk to a
``tf.data.Dataset`` in just a couple lines of code. If you like, you can
also write your own data loading code from scratch by visiting the `load
images <https://www.tensorflow.org/tutorials/load_data/images>`__
tutorial.

Create a Dataset
----------------



Define some parameters for the loader:

.. code:: ipython3

    batch_size = 32
    img_height = 180
    img_width = 180

It’s good practice to use a validation split when developing your model.
Let’s use 80% of the images for training, and 20% for validation.

.. code:: ipython3

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      subset="training",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)


.. parsed-literal::

    Found 3670 files belonging to 5 classes.


.. parsed-literal::

    Using 2936 files for training.


.. parsed-literal::

    2024-03-13 01:02:28.106945: E tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:266] failed call to cuInit: CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: forward compatibility was attempted on non supported HW
    2024-03-13 01:02:28.106977: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:168] retrieving CUDA diagnostic information for host: iotg-dev-workstation-07
    2024-03-13 01:02:28.106982: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:175] hostname: iotg-dev-workstation-07
    2024-03-13 01:02:28.107105: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:199] libcuda reported version is: 470.223.2
    2024-03-13 01:02:28.107122: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:203] kernel reported version is: 470.182.3
    2024-03-13 01:02:28.107125: E tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:312] kernel version 470.182.3 does not match DSO version 470.223.2 -- cannot find working devices in this configuration


.. code:: ipython3

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      subset="validation",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)


.. parsed-literal::

    Found 3670 files belonging to 5 classes.


.. parsed-literal::

    Using 734 files for validation.


You can find the class names in the ``class_names`` attribute on these
datasets. These correspond to the directory names in alphabetical order.

.. code:: ipython3

    class_names = train_ds.class_names
    print(class_names)


.. parsed-literal::

    ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']


Visualize the Data
------------------



Here are the first 9 images from the training dataset.

.. code:: ipython3

    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")


.. parsed-literal::

    2024-03-13 01:02:28.449873: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-03-13 01:02:28.450244: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]



.. image:: 301-tensorflow-training-openvino-with-output_files/301-tensorflow-training-openvino-with-output_29_1.png


You will train a model using these datasets by passing them to
``model.fit`` in a moment. If you like, you can also manually iterate
over the dataset and retrieve batches of images:

.. code:: ipython3

    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break


.. parsed-literal::

    (32, 180, 180, 3)
    (32,)


.. parsed-literal::

    2024-03-13 01:02:29.296083: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-03-13 01:02:29.296450: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]


The ``image_batch`` is a tensor of the shape ``(32, 180, 180, 3)``. This
is a batch of 32 images of shape ``180x180x3`` (the last dimension
refers to color channels RGB). The ``label_batch`` is a tensor of the
shape ``(32,)``, these are corresponding labels to the 32 images.

You can call ``.numpy()`` on the ``image_batch`` and ``labels_batch``
tensors to convert them to a ``numpy.ndarray``.

Configure the Dataset for Performance
-------------------------------------



Let’s make sure to use buffered prefetching so you can yield data from
disk without having I/O become blocking. These are two important methods
you should use when loading data.

``Dataset.cache()`` keeps the images in memory after they’re loaded off
disk during the first epoch. This will ensure the dataset does not
become a bottleneck while training your model. If your dataset is too
large to fit into memory, you can also use this method to create a
performant on-disk cache.

``Dataset.prefetch()`` overlaps data preprocessing and model execution
while training.

Interested readers can learn more about both methods, as well as how to
cache data to disk in the `data performance
guide <https://www.tensorflow.org/guide/data_performance#prefetching>`__.

.. code:: ipython3

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

Standardize the Data
--------------------



The RGB channel values are in the ``[0, 255]`` range. This is not ideal
for a neural network; in general you should seek to make your input
values small. Here, you will standardize values to be in the ``[0, 1]``
range by using a Rescaling layer.

.. code:: ipython3

    normalization_layer = layers.Rescaling(1./255)

Note: The Keras Preprocessing utilities and layers introduced in this
section are currently experimental and may change.

There are two ways to use this layer. You can apply it to the dataset by
calling map:

.. code:: ipython3

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixels values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image))


.. parsed-literal::

    2024-03-13 01:02:29.513870: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]
    2024-03-13 01:02:29.514493: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]


.. parsed-literal::

    0.0 1.0


Or, you can include the layer inside your model definition, which can
simplify deployment. Let’s use the second approach here.

Note: you previously resized images using the ``image_size`` argument of
``image_dataset_from_directory``. If you want to include the resizing
logic in your model as well, you can use the
`Resizing <https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/Resizing>`__
layer.

Create the Model
----------------



The model consists of three convolution blocks with a max pool layer in
each of them. There’s a fully connected layer with 128 units on top of
it that is activated by a ``relu`` activation function. This model has
not been tuned for high accuracy, the goal of this tutorial is to show a
standard approach.

.. code:: ipython3

    num_classes = 5

    model = Sequential([
      layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(num_classes)
    ])

Compile the Model
-----------------



For this tutorial, choose the ``optimizers.Adam`` optimizer and
``losses.SparseCategoricalCrossentropy`` loss function. To view training
and validation accuracy for each training epoch, pass the ``metrics``
argument.

.. code:: ipython3

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

Model Summary
-------------



View all the layers of the network using the model’s ``summary`` method.

   **NOTE:** This section is commented out for performance reasons.
   Please feel free to uncomment these to compare the results.

.. code:: ipython3

    # model.summary()

Train the Model
---------------



.. code:: ipython3

    # epochs=10
    # history = model.fit(
    #   train_ds,
    #   validation_data=val_ds,
    #   epochs=epochs
    # )

Visualize Training Results
--------------------------



Create plots of loss and accuracy on the training and validation sets.

.. code:: ipython3

    # acc = history.history['accuracy']
    # val_acc = history.history['val_accuracy']

    # loss = history.history['loss']
    # val_loss = history.history['val_loss']

    # epochs_range = range(epochs)

    # plt.figure(figsize=(8, 8))
    # plt.subplot(1, 2, 1)
    # plt.plot(epochs_range, acc, label='Training Accuracy')
    # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    # plt.legend(loc='lower right')
    # plt.title('Training and Validation Accuracy')

    # plt.subplot(1, 2, 2)
    # plt.plot(epochs_range, loss, label='Training Loss')
    # plt.plot(epochs_range, val_loss, label='Validation Loss')
    # plt.legend(loc='upper right')
    # plt.title('Training and Validation Loss')
    # plt.show()

As you can see from the plots, training accuracy and validation accuracy
are off by large margin and the model has achieved only around 60%
accuracy on the validation set.

Let’s look at what went wrong and try to increase the overall
performance of the model.

Overfitting
-----------



In the plots above, the training accuracy is increasing linearly over
time, whereas validation accuracy stalls around 60% in the training
process. Also, the difference in accuracy between training and
validation accuracy is noticeable — a sign of
`overfitting <https://www.tensorflow.org/tutorials/keras/overfit_and_underfit>`__.

When there are a small number of training examples, the model sometimes
learns from noises or unwanted details from training examples—to an
extent that it negatively impacts the performance of the model on new
examples. This phenomenon is known as overfitting. It means that the
model will have a difficult time generalizing on a new dataset.

There are multiple ways to fight overfitting in the training process. In
this tutorial, you’ll use *data augmentation* and add *Dropout* to your
model.

Data Augmentation
-----------------



Overfitting generally occurs when there are a small number of training
examples. `Data
augmentation <https://www.tensorflow.org/tutorials/images/data_augmentation>`__
takes the approach of generating additional training data from your
existing examples by augmenting them using random transformations that
yield believable-looking images. This helps expose the model to more
aspects of the data and generalize better.

You will implement data augmentation using the layers from
``tf.keras.layers.experimental.preprocessing``. These can be included
inside your model like other layers, and run on the GPU.

.. code:: ipython3

    data_augmentation = keras.Sequential(
      [
        layers.RandomFlip("horizontal",
                          input_shape=(img_height,
                                       img_width,
                                       3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
      ]
    )

Let’s visualize what a few augmented examples look like by applying data
augmentation to the same image several times:

.. code:: ipython3

    plt.figure(figsize=(10, 10))
    for images, _ in train_ds.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_images[0].numpy().astype("uint8"))
            plt.axis("off")


.. parsed-literal::

    2024-03-13 01:02:30.494557: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]
    2024-03-13 01:02:30.495526: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]



.. image:: 301-tensorflow-training-openvino-with-output_files/301-tensorflow-training-openvino-with-output_57_1.png


You will use data augmentation to train a model in a moment.

Dropout
-------



Another technique to reduce overfitting is to introduce
`Dropout <https://developers.google.com/machine-learning/glossary#dropout_regularization>`__
to the network, a form of *regularization*.

When you apply Dropout to a layer it randomly drops out (by setting the
activation to zero) a number of output units from the layer during the
training process. Dropout takes a fractional number as its input value,
in the form such as 0.1, 0.2, 0.4, etc. This means dropping out 10%, 20%
or 40% of the output units randomly from the applied layer.

Let’s create a new neural network using ``layers.Dropout``, then train
it using augmented images.

.. code:: ipython3

    model = Sequential([
        data_augmentation,
        layers.Rescaling(1./255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, name="outputs")
    ])

Compile and Train the Model
---------------------------



.. code:: ipython3

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

.. code:: ipython3

    model.summary()


.. parsed-literal::

    Model: "sequential_2"


.. parsed-literal::

    _________________________________________________________________


.. parsed-literal::

     Layer (type)                Output Shape              Param #


.. parsed-literal::

    =================================================================


.. parsed-literal::

     sequential_1 (Sequential)   (None, 180, 180, 3)       0







.. parsed-literal::

     rescaling_2 (Rescaling)     (None, 180, 180, 3)       0







.. parsed-literal::

     conv2d_3 (Conv2D)           (None, 180, 180, 16)      448







.. parsed-literal::

     max_pooling2d_3 (MaxPooling  (None, 90, 90, 16)       0


.. parsed-literal::

     2D)







.. parsed-literal::

     conv2d_4 (Conv2D)           (None, 90, 90, 32)        4640







.. parsed-literal::

     max_pooling2d_4 (MaxPooling  (None, 45, 45, 32)       0


.. parsed-literal::

     2D)







.. parsed-literal::

     conv2d_5 (Conv2D)           (None, 45, 45, 64)        18496







.. parsed-literal::

     max_pooling2d_5 (MaxPooling  (None, 22, 22, 64)       0


.. parsed-literal::

     2D)







.. parsed-literal::

     dropout (Dropout)           (None, 22, 22, 64)        0







.. parsed-literal::

     flatten_1 (Flatten)         (None, 30976)             0







.. parsed-literal::

     dense_2 (Dense)             (None, 128)               3965056







.. parsed-literal::

     outputs (Dense)             (None, 5)                 645







.. parsed-literal::

    =================================================================


.. parsed-literal::

    Total params: 3,989,285


.. parsed-literal::

    Trainable params: 3,989,285


.. parsed-literal::

    Non-trainable params: 0


.. parsed-literal::

    _________________________________________________________________


.. code:: ipython3

    epochs = 15
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )


.. parsed-literal::

    Epoch 1/15


.. parsed-literal::

    2024-03-13 01:02:31.608332: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-03-13 01:02:31.608737: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]


.. parsed-literal::


 1/92 [..............................] - ETA: 1:24 - loss: 1.6184 - accuracy: 0.1875

.. parsed-literal::

    
 2/92 [..............................] - ETA: 6s - loss: 2.2743 - accuracy: 0.2344

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 2.2543 - accuracy: 0.2708

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 2.1636 - accuracy: 0.2344

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 2.0592 - accuracy: 0.2313

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 1.9847 - accuracy: 0.2188

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 5s - loss: 1.9289 - accuracy: 0.2232

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 5s - loss: 1.8900 - accuracy: 0.2109

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 5s - loss: 1.8577 - accuracy: 0.2118

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 5s - loss: 1.8316 - accuracy: 0.2219

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 1.8075 - accuracy: 0.2386

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 1.7888 - accuracy: 0.2396

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 1.7741 - accuracy: 0.2380

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 1.7589 - accuracy: 0.2388

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 1.7447 - accuracy: 0.2438

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 1.7372 - accuracy: 0.2363

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 1.7278 - accuracy: 0.2353

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 1.7193 - accuracy: 0.2344

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 1.7100 - accuracy: 0.2434

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 1.7013 - accuracy: 0.2500

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 1.6927 - accuracy: 0.2515

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 1.6837 - accuracy: 0.2571

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 1.6761 - accuracy: 0.2609

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 4s - loss: 1.6660 - accuracy: 0.2591

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 1.6576 - accuracy: 0.2600

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 1.6490 - accuracy: 0.2632

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 1.6399 - accuracy: 0.2650

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 1.6377 - accuracy: 0.2634

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 1.6316 - accuracy: 0.2694

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 1.6295 - accuracy: 0.2708

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 1.6243 - accuracy: 0.2762

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 1.6173 - accuracy: 0.2803

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 1.6127 - accuracy: 0.2812

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 1.6060 - accuracy: 0.2840

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 1.5981 - accuracy: 0.2929

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 1.5899 - accuracy: 0.2969

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 1.5830 - accuracy: 0.3015

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 1.5829 - accuracy: 0.3043

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 1.5791 - accuracy: 0.3061

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 1.5691 - accuracy: 0.3086

.. parsed-literal::

    
41/92 [============>.................] - ETA: 3s - loss: 1.5647 - accuracy: 0.3095

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 1.5574 - accuracy: 0.3110

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 1.5506 - accuracy: 0.3125

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 1.5488 - accuracy: 0.3146

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 1.5400 - accuracy: 0.3222

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 1.5359 - accuracy: 0.3247

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 1.5324 - accuracy: 0.3265

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 1.5247 - accuracy: 0.3307

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 1.5200 - accuracy: 0.3304

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 1.5206 - accuracy: 0.3300

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 1.5190 - accuracy: 0.3303

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 1.5134 - accuracy: 0.3341

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 1.5079 - accuracy: 0.3361

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 1.5048 - accuracy: 0.3374

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 1.4971 - accuracy: 0.3398

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 1.4886 - accuracy: 0.3415

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 1.4843 - accuracy: 0.3464

.. parsed-literal::

    
58/92 [=================>............] - ETA: 2s - loss: 1.4789 - accuracy: 0.3490

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 1.4769 - accuracy: 0.3521

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 1.4753 - accuracy: 0.3536

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 1.4759 - accuracy: 0.3529

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 1.4752 - accuracy: 0.3527

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 1.4726 - accuracy: 0.3531

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 1.4728 - accuracy: 0.3520

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 1.4683 - accuracy: 0.3533

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 1.4644 - accuracy: 0.3541

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 1.4629 - accuracy: 0.3553

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 1.4577 - accuracy: 0.3584

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 1.4533 - accuracy: 0.3618

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 1.4489 - accuracy: 0.3651

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 1.4463 - accuracy: 0.3684

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 1.4421 - accuracy: 0.3706

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 1.4395 - accuracy: 0.3716

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 1.4346 - accuracy: 0.3746

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 1.4299 - accuracy: 0.3779

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 1.4250 - accuracy: 0.3804

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 1.4247 - accuracy: 0.3803

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 1.4206 - accuracy: 0.3822

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 1.4185 - accuracy: 0.3829

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 1.4136 - accuracy: 0.3848

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 1.4083 - accuracy: 0.3870

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 1.4077 - accuracy: 0.3869

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 1.4029 - accuracy: 0.3901

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 1.3995 - accuracy: 0.3922

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 1.3979 - accuracy: 0.3945

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 1.3949 - accuracy: 0.3969

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 1.3903 - accuracy: 0.3999

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 1.3886 - accuracy: 0.4010

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 1.3860 - accuracy: 0.4004

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 1.3812 - accuracy: 0.4011

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 1.3783 - accuracy: 0.4029

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 1.3797 - accuracy: 0.4026

.. parsed-literal::

    2024-03-13 01:02:37.888562: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [734]
    	 [[{{node Placeholder/_4}}]]
    2024-03-13 01:02:37.888844: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [734]
    	 [[{{node Placeholder/_4}}]]


.. parsed-literal::

    
92/92 [==============================] - 7s 67ms/step - loss: 1.3797 - accuracy: 0.4026 - val_loss: 1.1118 - val_accuracy: 0.5763


.. parsed-literal::

    Epoch 2/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.9105 - accuracy: 0.5625

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.9593 - accuracy: 0.5938

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.9856 - accuracy: 0.5729

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 1.0403 - accuracy: 0.5859

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 1.0360 - accuracy: 0.5875

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 1.0193 - accuracy: 0.6094

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 5s - loss: 0.9867 - accuracy: 0.6295

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 1.0265 - accuracy: 0.6094

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 1.0582 - accuracy: 0.5903

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 1.0566 - accuracy: 0.5875

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 1.0390 - accuracy: 0.5938

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 1.0259 - accuracy: 0.6068

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 1.0263 - accuracy: 0.6010

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 1.0501 - accuracy: 0.5871

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 1.0419 - accuracy: 0.5938

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 1.0477 - accuracy: 0.5938

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 1.0513 - accuracy: 0.5882

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 1.0676 - accuracy: 0.5816

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 1.0657 - accuracy: 0.5839

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 1.0612 - accuracy: 0.5813

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 1.0648 - accuracy: 0.5833

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 1.0619 - accuracy: 0.5852

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 1.0571 - accuracy: 0.5856

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 1.0605 - accuracy: 0.5833

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 1.0545 - accuracy: 0.5813

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 1.0681 - accuracy: 0.5745

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 1.0650 - accuracy: 0.5718

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 1.0666 - accuracy: 0.5714

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 1.0674 - accuracy: 0.5711

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 1.0589 - accuracy: 0.5760

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 1.0596 - accuracy: 0.5726

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 1.0527 - accuracy: 0.5781

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 1.0541 - accuracy: 0.5795

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 1.0645 - accuracy: 0.5754

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 1.0628 - accuracy: 0.5759

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 1.0694 - accuracy: 0.5755

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 1.0661 - accuracy: 0.5769

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 1.0729 - accuracy: 0.5757

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 1.0714 - accuracy: 0.5777

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 1.0669 - accuracy: 0.5797

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 1.0680 - accuracy: 0.5777

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 1.0679 - accuracy: 0.5796

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 1.0672 - accuracy: 0.5799

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 1.0661 - accuracy: 0.5824

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 1.0683 - accuracy: 0.5806

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 1.0705 - accuracy: 0.5781

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 1.0696 - accuracy: 0.5785

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 1.0685 - accuracy: 0.5781

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 1.0680 - accuracy: 0.5784

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 1.0653 - accuracy: 0.5794

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 1.0619 - accuracy: 0.5803

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 1.0605 - accuracy: 0.5793

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 1.0614 - accuracy: 0.5784

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 1.0619 - accuracy: 0.5799

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 1.0602 - accuracy: 0.5818

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 1.0626 - accuracy: 0.5809

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 1.0606 - accuracy: 0.5811

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 1.0610 - accuracy: 0.5803

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 1.0651 - accuracy: 0.5768

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 1.0656 - accuracy: 0.5766

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 1.0637 - accuracy: 0.5763

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 1.0669 - accuracy: 0.5762

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 1.0693 - accuracy: 0.5755

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 1.0669 - accuracy: 0.5777

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 1.0697 - accuracy: 0.5770

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 1.0703 - accuracy: 0.5758

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 1.0701 - accuracy: 0.5756

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 1.0717 - accuracy: 0.5745

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 1.0695 - accuracy: 0.5757

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 1.0663 - accuracy: 0.5777

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 1.0643 - accuracy: 0.5784

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 1.0641 - accuracy: 0.5786

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 1.0667 - accuracy: 0.5775

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 1.0668 - accuracy: 0.5786

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 1.0638 - accuracy: 0.5813

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 1.0649 - accuracy: 0.5794

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 1.0628 - accuracy: 0.5808

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 1.0628 - accuracy: 0.5813

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 1.0607 - accuracy: 0.5835

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 1.0591 - accuracy: 0.5832

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 1.0591 - accuracy: 0.5837

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 1.0640 - accuracy: 0.5812

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 1.0634 - accuracy: 0.5825

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 1.0626 - accuracy: 0.5833

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 1.0595 - accuracy: 0.5853

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 1.0590 - accuracy: 0.5857

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 1.0568 - accuracy: 0.5873

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 1.0556 - accuracy: 0.5880

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 1.0545 - accuracy: 0.5874

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 1.0543 - accuracy: 0.5868

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 1.0518 - accuracy: 0.5882

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 1.0518 - accuracy: 0.5882 - val_loss: 0.9841 - val_accuracy: 0.5981


.. parsed-literal::

    Epoch 3/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 1.1521 - accuracy: 0.5312

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 1.0563 - accuracy: 0.5781

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 1.0220 - accuracy: 0.5938

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 1.0244 - accuracy: 0.5859

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 1.0256 - accuracy: 0.5813

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.9926 - accuracy: 0.5990

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 5s - loss: 0.9785 - accuracy: 0.6071

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.9672 - accuracy: 0.6094

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.9527 - accuracy: 0.6076

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.9405 - accuracy: 0.6156

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.9489 - accuracy: 0.6193

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.9709 - accuracy: 0.6094

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.9715 - accuracy: 0.6106

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.9619 - accuracy: 0.6138

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.9677 - accuracy: 0.6125

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.9741 - accuracy: 0.6074

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.9724 - accuracy: 0.6085

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.9591 - accuracy: 0.6146

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.9407 - accuracy: 0.6234

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.9417 - accuracy: 0.6281

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.9341 - accuracy: 0.6310

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.9311 - accuracy: 0.6335

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.9406 - accuracy: 0.6318

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.9421 - accuracy: 0.6341

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.9379 - accuracy: 0.6363

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.9524 - accuracy: 0.6274

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.9595 - accuracy: 0.6238

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.9559 - accuracy: 0.6261

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.9540 - accuracy: 0.6250

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.9505 - accuracy: 0.6250

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.9499 - accuracy: 0.6290

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.9488 - accuracy: 0.6309

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.9437 - accuracy: 0.6316

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.9398 - accuracy: 0.6324

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.9302 - accuracy: 0.6366

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.9310 - accuracy: 0.6328

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.9278 - accuracy: 0.6343

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.9292 - accuracy: 0.6324

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.9283 - accuracy: 0.6322

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.9275 - accuracy: 0.6336

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.9367 - accuracy: 0.6319

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.9357 - accuracy: 0.6317

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.9372 - accuracy: 0.6330

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.9420 - accuracy: 0.6286

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.9441 - accuracy: 0.6264

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.9472 - accuracy: 0.6264

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.9472 - accuracy: 0.6283

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.9445 - accuracy: 0.6289

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.9433 - accuracy: 0.6301

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.9481 - accuracy: 0.6300

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.9492 - accuracy: 0.6281

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.9519 - accuracy: 0.6268

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.9495 - accuracy: 0.6291

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.9574 - accuracy: 0.6238

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.9581 - accuracy: 0.6227

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.9565 - accuracy: 0.6228

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.9548 - accuracy: 0.6239

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.9557 - accuracy: 0.6245

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.9570 - accuracy: 0.6229

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.9561 - accuracy: 0.6245

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.9546 - accuracy: 0.6255

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.9585 - accuracy: 0.6225

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.9584 - accuracy: 0.6210

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.9548 - accuracy: 0.6230

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.9520 - accuracy: 0.6231

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.9485 - accuracy: 0.6255

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.9491 - accuracy: 0.6264

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.9464 - accuracy: 0.6273

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.9452 - accuracy: 0.6291

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.9460 - accuracy: 0.6299

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.9446 - accuracy: 0.6307

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.9407 - accuracy: 0.6324

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.9405 - accuracy: 0.6327

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.9419 - accuracy: 0.6322

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.9417 - accuracy: 0.6317

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.9425 - accuracy: 0.6332

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.9392 - accuracy: 0.6339

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.9406 - accuracy: 0.6341

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.9403 - accuracy: 0.6356

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.9384 - accuracy: 0.6354

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.9377 - accuracy: 0.6369

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.9376 - accuracy: 0.6371

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.9396 - accuracy: 0.6377

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.9436 - accuracy: 0.6350

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.9425 - accuracy: 0.6356

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.9419 - accuracy: 0.6358

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.9404 - accuracy: 0.6364

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.9380 - accuracy: 0.6380

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.9362 - accuracy: 0.6382

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.9391 - accuracy: 0.6371

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.9395 - accuracy: 0.6362

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.9395 - accuracy: 0.6362 - val_loss: 0.9104 - val_accuracy: 0.6226


.. parsed-literal::

    Epoch 4/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.8301 - accuracy: 0.6250

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.8889 - accuracy: 0.6719

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.8489 - accuracy: 0.6667

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.9063 - accuracy: 0.6562

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.9087 - accuracy: 0.6750

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.9049 - accuracy: 0.6667

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.9044 - accuracy: 0.6652

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.8891 - accuracy: 0.6680

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.8748 - accuracy: 0.6736

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.8838 - accuracy: 0.6750

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.8795 - accuracy: 0.6761

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.8821 - accuracy: 0.6719

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.8807 - accuracy: 0.6779

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.8823 - accuracy: 0.6808

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.8780 - accuracy: 0.6833

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.8646 - accuracy: 0.6875

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.8655 - accuracy: 0.6912

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.8688 - accuracy: 0.6875

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.8594 - accuracy: 0.6891

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.8546 - accuracy: 0.6891

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.8565 - accuracy: 0.6875

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.8586 - accuracy: 0.6832

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.8535 - accuracy: 0.6821

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 4s - loss: 0.8458 - accuracy: 0.6862

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.8491 - accuracy: 0.6837

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.8537 - accuracy: 0.6803

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.8588 - accuracy: 0.6771

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.8589 - accuracy: 0.6797

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.8469 - accuracy: 0.6832

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.8618 - accuracy: 0.6771

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.8629 - accuracy: 0.6754

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.8558 - accuracy: 0.6748

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.8503 - accuracy: 0.6771

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.8537 - accuracy: 0.6765

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.8730 - accuracy: 0.6661

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.8816 - accuracy: 0.6667

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.8813 - accuracy: 0.6647

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.8844 - accuracy: 0.6605

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.8790 - accuracy: 0.6619

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.8801 - accuracy: 0.6603

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.8767 - accuracy: 0.6617

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.8735 - accuracy: 0.6630

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.8766 - accuracy: 0.6621

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.8854 - accuracy: 0.6585

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.8815 - accuracy: 0.6612

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.8805 - accuracy: 0.6584

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.8752 - accuracy: 0.6597

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.8762 - accuracy: 0.6609

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.8765 - accuracy: 0.6608

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.8728 - accuracy: 0.6638

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.8726 - accuracy: 0.6636

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.8774 - accuracy: 0.6611

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.8785 - accuracy: 0.6605

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.8763 - accuracy: 0.6610

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.8768 - accuracy: 0.6614

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.8795 - accuracy: 0.6591

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.8808 - accuracy: 0.6580

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.8781 - accuracy: 0.6585

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.8796 - accuracy: 0.6574

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.8845 - accuracy: 0.6574

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.8840 - accuracy: 0.6574

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.8821 - accuracy: 0.6584

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.8823 - accuracy: 0.6569

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.8862 - accuracy: 0.6549

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.8851 - accuracy: 0.6549

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.8861 - accuracy: 0.6540

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.8845 - accuracy: 0.6545

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.8839 - accuracy: 0.6536

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.8839 - accuracy: 0.6532

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.8829 - accuracy: 0.6546

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.8809 - accuracy: 0.6555

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.8842 - accuracy: 0.6538

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.8871 - accuracy: 0.6525

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.8866 - accuracy: 0.6534

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.8853 - accuracy: 0.6543

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.8851 - accuracy: 0.6551

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.8832 - accuracy: 0.6555

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.8857 - accuracy: 0.6548

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.8858 - accuracy: 0.6552

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.8850 - accuracy: 0.6560

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.8846 - accuracy: 0.6560

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.8838 - accuracy: 0.6556

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.8853 - accuracy: 0.6556

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.8843 - accuracy: 0.6556

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.8818 - accuracy: 0.6563

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.8787 - accuracy: 0.6585

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.8786 - accuracy: 0.6578

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.8795 - accuracy: 0.6574

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.8813 - accuracy: 0.6563

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.8799 - accuracy: 0.6570

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.8800 - accuracy: 0.6570

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.8800 - accuracy: 0.6570 - val_loss: 0.9390 - val_accuracy: 0.6553


.. parsed-literal::

    Epoch 5/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.8812 - accuracy: 0.5938

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.9164 - accuracy: 0.6094

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.8794 - accuracy: 0.6250

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.9695 - accuracy: 0.5938

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.9142 - accuracy: 0.6187

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.9198 - accuracy: 0.6406

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 5s - loss: 0.9138 - accuracy: 0.6562

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.9124 - accuracy: 0.6406

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.8889 - accuracy: 0.6562

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.8911 - accuracy: 0.6625

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.9110 - accuracy: 0.6506

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.9076 - accuracy: 0.6510

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.9137 - accuracy: 0.6490

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.9154 - accuracy: 0.6496

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.9019 - accuracy: 0.6562

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.9025 - accuracy: 0.6562

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.8931 - accuracy: 0.6581

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.8880 - accuracy: 0.6615

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.8759 - accuracy: 0.6645

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.8755 - accuracy: 0.6609

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.8677 - accuracy: 0.6622

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.8679 - accuracy: 0.6619

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.8670 - accuracy: 0.6590

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.8763 - accuracy: 0.6549

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.8764 - accuracy: 0.6550

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.8705 - accuracy: 0.6587

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.8718 - accuracy: 0.6609

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.8648 - accuracy: 0.6641

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.8630 - accuracy: 0.6670

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.8663 - accuracy: 0.6635

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.8637 - accuracy: 0.6623

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.8571 - accuracy: 0.6650

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.8529 - accuracy: 0.6657

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.8525 - accuracy: 0.6645

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.8567 - accuracy: 0.6625

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.8536 - accuracy: 0.6623

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.8573 - accuracy: 0.6605

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.8572 - accuracy: 0.6612

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.8635 - accuracy: 0.6587

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.8663 - accuracy: 0.6570

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.8664 - accuracy: 0.6578

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.8603 - accuracy: 0.6615

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.8584 - accuracy: 0.6606

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.8597 - accuracy: 0.6619

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.8649 - accuracy: 0.6597

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.8618 - accuracy: 0.6624

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.8613 - accuracy: 0.6642

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.8609 - accuracy: 0.6654

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.8622 - accuracy: 0.6633

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.8586 - accuracy: 0.6644

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.8583 - accuracy: 0.6642

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.8594 - accuracy: 0.6647

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.8609 - accuracy: 0.6645

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.8660 - accuracy: 0.6632

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.8657 - accuracy: 0.6614

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.8618 - accuracy: 0.6618

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.8631 - accuracy: 0.6612

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.8668 - accuracy: 0.6579

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.8626 - accuracy: 0.6589

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.8650 - accuracy: 0.6568

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.8623 - accuracy: 0.6593

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.8625 - accuracy: 0.6583

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.8627 - accuracy: 0.6582

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.8592 - accuracy: 0.6602

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.8576 - accuracy: 0.6611

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.8557 - accuracy: 0.6610

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.8520 - accuracy: 0.6637

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.8513 - accuracy: 0.6650

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.8522 - accuracy: 0.6644

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.8504 - accuracy: 0.6661

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.8502 - accuracy: 0.6668

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.8468 - accuracy: 0.6684

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.8451 - accuracy: 0.6691

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.8438 - accuracy: 0.6706

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.8427 - accuracy: 0.6704

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.8413 - accuracy: 0.6718

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.8387 - accuracy: 0.6728

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.8374 - accuracy: 0.6730

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.8359 - accuracy: 0.6728

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.8352 - accuracy: 0.6738

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.8345 - accuracy: 0.6743

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.8320 - accuracy: 0.6748

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.8285 - accuracy: 0.6757

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.8268 - accuracy: 0.6763

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.8247 - accuracy: 0.6764

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.8244 - accuracy: 0.6765

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.8236 - accuracy: 0.6763

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.8236 - accuracy: 0.6764

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.8291 - accuracy: 0.6748

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.8322 - accuracy: 0.6739

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.8332 - accuracy: 0.6737

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.8332 - accuracy: 0.6737 - val_loss: 0.8496 - val_accuracy: 0.6744


.. parsed-literal::

    Epoch 6/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.6199 - accuracy: 0.7812

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.7833 - accuracy: 0.7344

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.8298 - accuracy: 0.7500

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.8400 - accuracy: 0.7188

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.8397 - accuracy: 0.7125

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.8399 - accuracy: 0.7083

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.8296 - accuracy: 0.7098

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.8467 - accuracy: 0.7031

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.8815 - accuracy: 0.6806

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.8850 - accuracy: 0.6719

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.8874 - accuracy: 0.6705

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.8762 - accuracy: 0.6771

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.8703 - accuracy: 0.6803

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.8811 - accuracy: 0.6808

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.8734 - accuracy: 0.6875

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.8746 - accuracy: 0.6816

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.8690 - accuracy: 0.6838

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.8626 - accuracy: 0.6892

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.8588 - accuracy: 0.6891

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.8482 - accuracy: 0.6906

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.8316 - accuracy: 0.6979

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.8284 - accuracy: 0.6974

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.8323 - accuracy: 0.6929

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.8270 - accuracy: 0.6953

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.8250 - accuracy: 0.6963

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.8211 - accuracy: 0.6959

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.8161 - accuracy: 0.6991

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.8185 - accuracy: 0.6975

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.8167 - accuracy: 0.6972

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.8212 - accuracy: 0.6958

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.8256 - accuracy: 0.6935

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.8234 - accuracy: 0.6943

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.8263 - accuracy: 0.6922

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.8224 - accuracy: 0.6921

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.8256 - accuracy: 0.6938

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.8271 - accuracy: 0.6918

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.8301 - accuracy: 0.6934

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.8221 - accuracy: 0.6974

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.8311 - accuracy: 0.6939

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.8315 - accuracy: 0.6930

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.8302 - accuracy: 0.6921

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.8271 - accuracy: 0.6935

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.8218 - accuracy: 0.6955

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.8249 - accuracy: 0.6939

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.8202 - accuracy: 0.6944

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.8188 - accuracy: 0.6943

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.8185 - accuracy: 0.6948

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.8134 - accuracy: 0.6960

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.8133 - accuracy: 0.6945

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.8111 - accuracy: 0.6956

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.8098 - accuracy: 0.6961

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.8141 - accuracy: 0.6929

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.8125 - accuracy: 0.6940

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.8133 - accuracy: 0.6939

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.8159 - accuracy: 0.6926

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.8124 - accuracy: 0.6942

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.8129 - accuracy: 0.6946

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.8087 - accuracy: 0.6967

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.8073 - accuracy: 0.6970

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.8078 - accuracy: 0.6953

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.8053 - accuracy: 0.6967

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.8042 - accuracy: 0.6981

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.8060 - accuracy: 0.6980

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.8099 - accuracy: 0.6969

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.8084 - accuracy: 0.6977

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.8075 - accuracy: 0.6980

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.8117 - accuracy: 0.6956

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.8098 - accuracy: 0.6964

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.8070 - accuracy: 0.6971

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.8074 - accuracy: 0.6961

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.8085 - accuracy: 0.6947

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.8054 - accuracy: 0.6954

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.8072 - accuracy: 0.6936

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.8059 - accuracy: 0.6948

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.8074 - accuracy: 0.6931

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.8065 - accuracy: 0.6938

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.8042 - accuracy: 0.6953

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.8038 - accuracy: 0.6960

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.8034 - accuracy: 0.6947

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.8033 - accuracy: 0.6943

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.8084 - accuracy: 0.6919

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.8063 - accuracy: 0.6926

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.8063 - accuracy: 0.6925

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.8048 - accuracy: 0.6936

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.8068 - accuracy: 0.6931

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.8051 - accuracy: 0.6945

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.8063 - accuracy: 0.6934

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.8043 - accuracy: 0.6944

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.8045 - accuracy: 0.6939

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.8011 - accuracy: 0.6952

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.8013 - accuracy: 0.6952

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.8013 - accuracy: 0.6952 - val_loss: 0.7885 - val_accuracy: 0.6921


.. parsed-literal::

    Epoch 7/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.7625 - accuracy: 0.5938

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.8142 - accuracy: 0.5938

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.7949 - accuracy: 0.6771

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.7718 - accuracy: 0.6719

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.7636 - accuracy: 0.6687

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.8231 - accuracy: 0.6458

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.8185 - accuracy: 0.6518

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.8045 - accuracy: 0.6680

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.7917 - accuracy: 0.6736

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.7723 - accuracy: 0.6750

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.7690 - accuracy: 0.6818

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.7872 - accuracy: 0.6797

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.7886 - accuracy: 0.6755

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.7963 - accuracy: 0.6741

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.7875 - accuracy: 0.6771

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.7746 - accuracy: 0.6777

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.7709 - accuracy: 0.6783

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.7652 - accuracy: 0.6858

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.7664 - accuracy: 0.6859

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.7756 - accuracy: 0.6828

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.7639 - accuracy: 0.6890

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.7706 - accuracy: 0.6875

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.7612 - accuracy: 0.6902

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.7561 - accuracy: 0.6927

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.7639 - accuracy: 0.6888

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.7809 - accuracy: 0.6791

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.7862 - accuracy: 0.6771

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.7785 - accuracy: 0.6793

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.7737 - accuracy: 0.6817

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.7615 - accuracy: 0.6890

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.7587 - accuracy: 0.6900

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.7596 - accuracy: 0.6880

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.7565 - accuracy: 0.6880

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.7545 - accuracy: 0.6888

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.7500 - accuracy: 0.6914

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.7499 - accuracy: 0.6922

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.7522 - accuracy: 0.6929

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.7490 - accuracy: 0.6944

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.7522 - accuracy: 0.6950

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.7488 - accuracy: 0.6933

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.7466 - accuracy: 0.6961

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.7524 - accuracy: 0.6930

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.7599 - accuracy: 0.6893

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.7605 - accuracy: 0.6899

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.7626 - accuracy: 0.6872

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.7634 - accuracy: 0.6872

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.7714 - accuracy: 0.6846

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.7746 - accuracy: 0.6827

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.7786 - accuracy: 0.6834

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.7776 - accuracy: 0.6841

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.7777 - accuracy: 0.6842

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.7754 - accuracy: 0.6842

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.7761 - accuracy: 0.6837

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.7774 - accuracy: 0.6832

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.7812 - accuracy: 0.6811

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.7819 - accuracy: 0.6817

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.7844 - accuracy: 0.6807

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.7844 - accuracy: 0.6809

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.7820 - accuracy: 0.6825

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.7807 - accuracy: 0.6831

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.7782 - accuracy: 0.6832

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.7790 - accuracy: 0.6843

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.7773 - accuracy: 0.6858

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.7760 - accuracy: 0.6873

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.7741 - accuracy: 0.6882

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.7741 - accuracy: 0.6868

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.7709 - accuracy: 0.6877

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.7695 - accuracy: 0.6895

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.7678 - accuracy: 0.6918

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.7654 - accuracy: 0.6930

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.7645 - accuracy: 0.6925

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.7624 - accuracy: 0.6937

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.7617 - accuracy: 0.6945

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.7631 - accuracy: 0.6940

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.7600 - accuracy: 0.6947

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.7557 - accuracy: 0.6963

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.7526 - accuracy: 0.6982

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.7559 - accuracy: 0.6976

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.7533 - accuracy: 0.6995

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.7518 - accuracy: 0.7009

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.7533 - accuracy: 0.7007

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.7527 - accuracy: 0.7005

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.7522 - accuracy: 0.7007

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.7542 - accuracy: 0.7006

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.7532 - accuracy: 0.7008

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.7522 - accuracy: 0.7014

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.7515 - accuracy: 0.7016

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.7499 - accuracy: 0.7021

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.7476 - accuracy: 0.7026

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.7485 - accuracy: 0.7025

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.7487 - accuracy: 0.7027

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.7487 - accuracy: 0.7027 - val_loss: 0.7632 - val_accuracy: 0.7180


.. parsed-literal::

    Epoch 8/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.7172 - accuracy: 0.7188

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.6769 - accuracy: 0.7344

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6460 - accuracy: 0.7396

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6215 - accuracy: 0.7344

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.6422 - accuracy: 0.7312

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.6388 - accuracy: 0.7448

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6498 - accuracy: 0.7411

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6860 - accuracy: 0.7148

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.7119 - accuracy: 0.7083

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.7098 - accuracy: 0.7188

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.7116 - accuracy: 0.7216

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.7132 - accuracy: 0.7266

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.7007 - accuracy: 0.7260

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.7044 - accuracy: 0.7254

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.7288 - accuracy: 0.7188

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.7369 - accuracy: 0.7188

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.7261 - accuracy: 0.7261

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.7367 - accuracy: 0.7240

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.7338 - accuracy: 0.7231

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.7249 - accuracy: 0.7274

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.7204 - accuracy: 0.7299

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.7158 - accuracy: 0.7335

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.7139 - accuracy: 0.7368

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.7096 - accuracy: 0.7361

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.7115 - accuracy: 0.7354

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.7129 - accuracy: 0.7360

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.7124 - accuracy: 0.7365

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.7017 - accuracy: 0.7380

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.7011 - accuracy: 0.7374

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.7043 - accuracy: 0.7388

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.7160 - accuracy: 0.7352

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.7132 - accuracy: 0.7347

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.7126 - accuracy: 0.7352

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.7083 - accuracy: 0.7374

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.7061 - accuracy: 0.7378

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.7082 - accuracy: 0.7347

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.7058 - accuracy: 0.7334

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6990 - accuracy: 0.7363

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.7005 - accuracy: 0.7366

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.7000 - accuracy: 0.7362

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.7049 - accuracy: 0.7320

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.7077 - accuracy: 0.7295

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.7090 - accuracy: 0.7293

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.7103 - accuracy: 0.7304

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.7073 - accuracy: 0.7309

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.7056 - accuracy: 0.7326

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.7036 - accuracy: 0.7336

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.7025 - accuracy: 0.7340

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.7066 - accuracy: 0.7324

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.7055 - accuracy: 0.7334

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.7037 - accuracy: 0.7337

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.7035 - accuracy: 0.7340

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6989 - accuracy: 0.7355

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6984 - accuracy: 0.7357

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6962 - accuracy: 0.7365

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6983 - accuracy: 0.7357

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.7028 - accuracy: 0.7316

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.7032 - accuracy: 0.7298

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.7020 - accuracy: 0.7306

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6995 - accuracy: 0.7320

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6997 - accuracy: 0.7308

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6983 - accuracy: 0.7311

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6967 - accuracy: 0.7324

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6963 - accuracy: 0.7331

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6936 - accuracy: 0.7338

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6913 - accuracy: 0.7350

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6887 - accuracy: 0.7357

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6888 - accuracy: 0.7359

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6861 - accuracy: 0.7370

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6876 - accuracy: 0.7372

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6856 - accuracy: 0.7382

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6849 - accuracy: 0.7384

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6884 - accuracy: 0.7377

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6878 - accuracy: 0.7383

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6862 - accuracy: 0.7389

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6864 - accuracy: 0.7386

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6870 - accuracy: 0.7371

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6894 - accuracy: 0.7365

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6936 - accuracy: 0.7359

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6963 - accuracy: 0.7349

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6976 - accuracy: 0.7339

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6980 - accuracy: 0.7334

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6979 - accuracy: 0.7332

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6987 - accuracy: 0.7327

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6967 - accuracy: 0.7340

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6976 - accuracy: 0.7334

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6974 - accuracy: 0.7343

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6991 - accuracy: 0.7338

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.7016 - accuracy: 0.7319

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.7019 - accuracy: 0.7311

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.7012 - accuracy: 0.7316

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.7012 - accuracy: 0.7316 - val_loss: 0.7702 - val_accuracy: 0.7153


.. parsed-literal::

    Epoch 9/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.7327 - accuracy: 0.7812

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.6516 - accuracy: 0.7812

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.7418 - accuracy: 0.7396

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.7133 - accuracy: 0.7422

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.6932 - accuracy: 0.7563

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.7019 - accuracy: 0.7292

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6774 - accuracy: 0.7366

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6611 - accuracy: 0.7422

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6545 - accuracy: 0.7569

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6741 - accuracy: 0.7563

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.7102 - accuracy: 0.7443

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.7125 - accuracy: 0.7396

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.7043 - accuracy: 0.7428

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.7054 - accuracy: 0.7455

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.7136 - accuracy: 0.7417

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6973 - accuracy: 0.7480

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.7043 - accuracy: 0.7482

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6905 - accuracy: 0.7535

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6920 - accuracy: 0.7516

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6807 - accuracy: 0.7531

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6729 - accuracy: 0.7545

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6833 - accuracy: 0.7514

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.6871 - accuracy: 0.7514

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6880 - accuracy: 0.7539

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6813 - accuracy: 0.7563

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6815 - accuracy: 0.7560

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6813 - accuracy: 0.7558

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6902 - accuracy: 0.7511

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6940 - accuracy: 0.7500

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6877 - accuracy: 0.7510

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6805 - accuracy: 0.7540

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6852 - accuracy: 0.7500

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6852 - accuracy: 0.7491

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6803 - accuracy: 0.7491

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6842 - accuracy: 0.7473

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6855 - accuracy: 0.7465

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6893 - accuracy: 0.7432

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6896 - accuracy: 0.7410

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6872 - accuracy: 0.7412

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.6829 - accuracy: 0.7437

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6822 - accuracy: 0.7454

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6821 - accuracy: 0.7455

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6765 - accuracy: 0.7485

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6732 - accuracy: 0.7514

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6690 - accuracy: 0.7535

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6715 - accuracy: 0.7514

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6701 - accuracy: 0.7520

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6688 - accuracy: 0.7552

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6664 - accuracy: 0.7551

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6705 - accuracy: 0.7531

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6677 - accuracy: 0.7537

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6683 - accuracy: 0.7512

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6632 - accuracy: 0.7535

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6671 - accuracy: 0.7512

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6669 - accuracy: 0.7517

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6650 - accuracy: 0.7522

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6592 - accuracy: 0.7549

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6616 - accuracy: 0.7522

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6631 - accuracy: 0.7526

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6615 - accuracy: 0.7547

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6653 - accuracy: 0.7536

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6664 - accuracy: 0.7525

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6658 - accuracy: 0.7520

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6643 - accuracy: 0.7520

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6618 - accuracy: 0.7543

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6601 - accuracy: 0.7547

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6585 - accuracy: 0.7551

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6609 - accuracy: 0.7523

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6676 - accuracy: 0.7486

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6645 - accuracy: 0.7509

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6621 - accuracy: 0.7522

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6610 - accuracy: 0.7526

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6592 - accuracy: 0.7530

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6584 - accuracy: 0.7533

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6593 - accuracy: 0.7541

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6574 - accuracy: 0.7545

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6558 - accuracy: 0.7552

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6547 - accuracy: 0.7552

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6563 - accuracy: 0.7551

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6566 - accuracy: 0.7550

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6598 - accuracy: 0.7550

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6627 - accuracy: 0.7538

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6615 - accuracy: 0.7537

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6648 - accuracy: 0.7515

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6672 - accuracy: 0.7489

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6659 - accuracy: 0.7493

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6646 - accuracy: 0.7500

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6653 - accuracy: 0.7489

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6656 - accuracy: 0.7479

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6670 - accuracy: 0.7483

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6711 - accuracy: 0.7480

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.6711 - accuracy: 0.7480 - val_loss: 0.8020 - val_accuracy: 0.6962


.. parsed-literal::

    Epoch 10/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.9496 - accuracy: 0.6250

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.9003 - accuracy: 0.5938

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.7666 - accuracy: 0.6771

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.7763 - accuracy: 0.6875

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.7471 - accuracy: 0.7000

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.7708 - accuracy: 0.6771

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.7444 - accuracy: 0.7009

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.7261 - accuracy: 0.7109

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.7338 - accuracy: 0.7014

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.7229 - accuracy: 0.7125

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.7093 - accuracy: 0.7244

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.7040 - accuracy: 0.7292

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6978 - accuracy: 0.7284

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6941 - accuracy: 0.7277

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6810 - accuracy: 0.7354

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6905 - accuracy: 0.7305

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6876 - accuracy: 0.7261

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6831 - accuracy: 0.7326

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6771 - accuracy: 0.7368

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6791 - accuracy: 0.7375

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6655 - accuracy: 0.7470

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6608 - accuracy: 0.7486

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.6494 - accuracy: 0.7514

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6466 - accuracy: 0.7552

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6500 - accuracy: 0.7538

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6396 - accuracy: 0.7572

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6376 - accuracy: 0.7558

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6443 - accuracy: 0.7533

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6406 - accuracy: 0.7563

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6387 - accuracy: 0.7561

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6517 - accuracy: 0.7510

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6472 - accuracy: 0.7529

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6590 - accuracy: 0.7472

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6533 - accuracy: 0.7500

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6531 - accuracy: 0.7500

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6553 - accuracy: 0.7509

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6546 - accuracy: 0.7517

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6583 - accuracy: 0.7500

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.6534 - accuracy: 0.7516

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6557 - accuracy: 0.7492

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6575 - accuracy: 0.7485

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6557 - accuracy: 0.7493

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6548 - accuracy: 0.7500

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6536 - accuracy: 0.7521

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6548 - accuracy: 0.7514

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6491 - accuracy: 0.7540

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6510 - accuracy: 0.7533

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6537 - accuracy: 0.7526

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6507 - accuracy: 0.7525

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6462 - accuracy: 0.7549

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6456 - accuracy: 0.7554

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6444 - accuracy: 0.7536

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6474 - accuracy: 0.7535

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6433 - accuracy: 0.7551

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6406 - accuracy: 0.7556

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6381 - accuracy: 0.7577

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6361 - accuracy: 0.7581

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6355 - accuracy: 0.7601

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6337 - accuracy: 0.7605

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6314 - accuracy: 0.7603

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6352 - accuracy: 0.7591

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6361 - accuracy: 0.7575

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6343 - accuracy: 0.7583

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6303 - accuracy: 0.7597

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6301 - accuracy: 0.7595

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6309 - accuracy: 0.7589

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6335 - accuracy: 0.7574

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6331 - accuracy: 0.7573

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6354 - accuracy: 0.7563

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6353 - accuracy: 0.7562

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6337 - accuracy: 0.7574

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6341 - accuracy: 0.7569

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6345 - accuracy: 0.7564

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6398 - accuracy: 0.7529

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6416 - accuracy: 0.7525

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6396 - accuracy: 0.7541

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6385 - accuracy: 0.7548

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6400 - accuracy: 0.7544

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6401 - accuracy: 0.7539

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6398 - accuracy: 0.7539

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6389 - accuracy: 0.7538

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6372 - accuracy: 0.7545

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6382 - accuracy: 0.7552

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6419 - accuracy: 0.7541

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6430 - accuracy: 0.7533

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6418 - accuracy: 0.7536

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6407 - accuracy: 0.7539

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6416 - accuracy: 0.7546

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6424 - accuracy: 0.7552

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6426 - accuracy: 0.7555

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6408 - accuracy: 0.7554

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.6408 - accuracy: 0.7554 - val_loss: 0.7525 - val_accuracy: 0.7112


.. parsed-literal::

    Epoch 11/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.4288 - accuracy: 0.8125

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5379 - accuracy: 0.7656

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6520 - accuracy: 0.7083

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6468 - accuracy: 0.7266

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5931 - accuracy: 0.7625

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.6002 - accuracy: 0.7708

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.5982 - accuracy: 0.7768

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6006 - accuracy: 0.7852

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5806 - accuracy: 0.7951

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5907 - accuracy: 0.7844

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5741 - accuracy: 0.7898

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5914 - accuracy: 0.7786

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5997 - accuracy: 0.7764

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5910 - accuracy: 0.7746

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5887 - accuracy: 0.7812

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5754 - accuracy: 0.7891

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5765 - accuracy: 0.7868

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5621 - accuracy: 0.7917

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5564 - accuracy: 0.7944

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5581 - accuracy: 0.7906

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5506 - accuracy: 0.7946

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5530 - accuracy: 0.7940

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.5548 - accuracy: 0.7935

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5654 - accuracy: 0.7852

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5796 - accuracy: 0.7800

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5784 - accuracy: 0.7776

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5750 - accuracy: 0.7778

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5857 - accuracy: 0.7723

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5865 - accuracy: 0.7737

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5927 - accuracy: 0.7688

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5894 - accuracy: 0.7692

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5873 - accuracy: 0.7695

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5872 - accuracy: 0.7689

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5840 - accuracy: 0.7702

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5864 - accuracy: 0.7688

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5859 - accuracy: 0.7682

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5904 - accuracy: 0.7652

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5938 - accuracy: 0.7640

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5931 - accuracy: 0.7636

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5935 - accuracy: 0.7630

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5924 - accuracy: 0.7657

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5920 - accuracy: 0.7675

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5865 - accuracy: 0.7707

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5830 - accuracy: 0.7737

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5833 - accuracy: 0.7753

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5803 - accuracy: 0.7754

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5783 - accuracy: 0.7762

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5784 - accuracy: 0.7763

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5813 - accuracy: 0.7758

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5816 - accuracy: 0.7746

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5823 - accuracy: 0.7748

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5830 - accuracy: 0.7749

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5879 - accuracy: 0.7733

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5928 - accuracy: 0.7728

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5927 - accuracy: 0.7724

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5908 - accuracy: 0.7731

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5892 - accuracy: 0.7727

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5875 - accuracy: 0.7734

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5894 - accuracy: 0.7730

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5912 - accuracy: 0.7716

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5880 - accuracy: 0.7723

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5862 - accuracy: 0.7719

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5880 - accuracy: 0.7711

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5874 - accuracy: 0.7722

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5865 - accuracy: 0.7723

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5869 - accuracy: 0.7720

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5846 - accuracy: 0.7726

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5852 - accuracy: 0.7723

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5853 - accuracy: 0.7720

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5856 - accuracy: 0.7725

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5948 - accuracy: 0.7687

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5983 - accuracy: 0.7676

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5987 - accuracy: 0.7669

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5982 - accuracy: 0.7671

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5964 - accuracy: 0.7677

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5961 - accuracy: 0.7687

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5969 - accuracy: 0.7681

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6019 - accuracy: 0.7675

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6002 - accuracy: 0.7684

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5994 - accuracy: 0.7697

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6019 - accuracy: 0.7695

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6030 - accuracy: 0.7685

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6032 - accuracy: 0.7679

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6072 - accuracy: 0.7662

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6082 - accuracy: 0.7657

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6130 - accuracy: 0.7630

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6125 - accuracy: 0.7632

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6117 - accuracy: 0.7641

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6107 - accuracy: 0.7643

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6086 - accuracy: 0.7645

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6071 - accuracy: 0.7657

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.6071 - accuracy: 0.7657 - val_loss: 0.6969 - val_accuracy: 0.7316


.. parsed-literal::

    Epoch 12/15


.. parsed-literal::


 1/92 [..............................] - ETA: 6s - loss: 0.4407 - accuracy: 0.8125

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.4792 - accuracy: 0.8281

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.4696 - accuracy: 0.8229

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.4918 - accuracy: 0.8047

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5115 - accuracy: 0.8000

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.5140 - accuracy: 0.8021

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.5059 - accuracy: 0.8036

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5174 - accuracy: 0.8086

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.4993 - accuracy: 0.8160

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5029 - accuracy: 0.8188

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5025 - accuracy: 0.8210

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5059 - accuracy: 0.8125

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5044 - accuracy: 0.8101

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5114 - accuracy: 0.8080

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5230 - accuracy: 0.8021

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5226 - accuracy: 0.7988

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5257 - accuracy: 0.7960

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5255 - accuracy: 0.7951

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5267 - accuracy: 0.7944

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5281 - accuracy: 0.7937

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5360 - accuracy: 0.7917

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5355 - accuracy: 0.7940

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.5374 - accuracy: 0.7908

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5475 - accuracy: 0.7878

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5563 - accuracy: 0.7875

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5524 - accuracy: 0.7897

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5504 - accuracy: 0.7905

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5477 - accuracy: 0.7924

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5441 - accuracy: 0.7931

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5505 - accuracy: 0.7885

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5585 - accuracy: 0.7853

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5641 - accuracy: 0.7832

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5631 - accuracy: 0.7831

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5656 - accuracy: 0.7831

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5645 - accuracy: 0.7830

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5611 - accuracy: 0.7830

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5690 - accuracy: 0.7796

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5690 - accuracy: 0.7804

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5674 - accuracy: 0.7788

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.5766 - accuracy: 0.7750

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5767 - accuracy: 0.7721

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5762 - accuracy: 0.7731

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5790 - accuracy: 0.7703

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5790 - accuracy: 0.7713

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5772 - accuracy: 0.7722

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5747 - accuracy: 0.7745

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5772 - accuracy: 0.7733

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5753 - accuracy: 0.7747

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5834 - accuracy: 0.7710

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5847 - accuracy: 0.7700

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5822 - accuracy: 0.7727

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5826 - accuracy: 0.7728

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5850 - accuracy: 0.7712

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5859 - accuracy: 0.7703

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5846 - accuracy: 0.7710

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5853 - accuracy: 0.7695

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5860 - accuracy: 0.7692

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5864 - accuracy: 0.7683

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5843 - accuracy: 0.7691

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5904 - accuracy: 0.7672

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5940 - accuracy: 0.7659

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5946 - accuracy: 0.7661

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5941 - accuracy: 0.7662

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5945 - accuracy: 0.7669

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5928 - accuracy: 0.7671

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5914 - accuracy: 0.7669

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5956 - accuracy: 0.7648

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5948 - accuracy: 0.7655

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5943 - accuracy: 0.7657

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5981 - accuracy: 0.7650

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5964 - accuracy: 0.7652

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5972 - accuracy: 0.7655

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5965 - accuracy: 0.7653

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5957 - accuracy: 0.7663

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5946 - accuracy: 0.7673

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5946 - accuracy: 0.7683

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5945 - accuracy: 0.7681

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5910 - accuracy: 0.7702

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5894 - accuracy: 0.7708

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5895 - accuracy: 0.7721

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5885 - accuracy: 0.7722

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5864 - accuracy: 0.7730

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5844 - accuracy: 0.7743

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5862 - accuracy: 0.7729

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5881 - accuracy: 0.7715

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5890 - accuracy: 0.7705

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5906 - accuracy: 0.7699

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5901 - accuracy: 0.7708

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5904 - accuracy: 0.7702

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5886 - accuracy: 0.7710

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5875 - accuracy: 0.7721

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.5875 - accuracy: 0.7721 - val_loss: 0.7032 - val_accuracy: 0.7384


.. parsed-literal::

    Epoch 13/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.3616 - accuracy: 0.9062

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.4409 - accuracy: 0.8750

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.4413 - accuracy: 0.8750

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5168 - accuracy: 0.8438

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5532 - accuracy: 0.8125

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.5588 - accuracy: 0.7969

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.5653 - accuracy: 0.7946

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5709 - accuracy: 0.7969

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6174 - accuracy: 0.7743

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5993 - accuracy: 0.7875

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5979 - accuracy: 0.7812

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5813 - accuracy: 0.7891

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5770 - accuracy: 0.7933

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5728 - accuracy: 0.7924

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5796 - accuracy: 0.7833

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5670 - accuracy: 0.7852

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5622 - accuracy: 0.7849

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5590 - accuracy: 0.7847

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5540 - accuracy: 0.7878

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5541 - accuracy: 0.7875

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5493 - accuracy: 0.7902

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5467 - accuracy: 0.7869

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.5461 - accuracy: 0.7894

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5429 - accuracy: 0.7930

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5391 - accuracy: 0.7962

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5366 - accuracy: 0.7945

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5330 - accuracy: 0.7975

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5354 - accuracy: 0.7958

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5403 - accuracy: 0.7920

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5424 - accuracy: 0.7917

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5444 - accuracy: 0.7923

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5420 - accuracy: 0.7930

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5413 - accuracy: 0.7936

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5392 - accuracy: 0.7941

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5409 - accuracy: 0.7937

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5409 - accuracy: 0.7943

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5421 - accuracy: 0.7956

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5464 - accuracy: 0.7944

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5520 - accuracy: 0.7909

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.5568 - accuracy: 0.7867

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5623 - accuracy: 0.7828

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5603 - accuracy: 0.7850

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5618 - accuracy: 0.7842

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5662 - accuracy: 0.7827

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5643 - accuracy: 0.7840

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5687 - accuracy: 0.7826

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5722 - accuracy: 0.7840

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5726 - accuracy: 0.7840

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5729 - accuracy: 0.7839

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5721 - accuracy: 0.7833

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5763 - accuracy: 0.7820

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5782 - accuracy: 0.7814

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5750 - accuracy: 0.7831

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5739 - accuracy: 0.7837

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5775 - accuracy: 0.7831

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5768 - accuracy: 0.7830

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5750 - accuracy: 0.7841

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5743 - accuracy: 0.7840

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5729 - accuracy: 0.7856

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5686 - accuracy: 0.7876

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5710 - accuracy: 0.7874

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5717 - accuracy: 0.7869

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5701 - accuracy: 0.7877

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5698 - accuracy: 0.7876

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5658 - accuracy: 0.7899

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5665 - accuracy: 0.7898

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5672 - accuracy: 0.7897

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5672 - accuracy: 0.7900

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5653 - accuracy: 0.7903

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5679 - accuracy: 0.7893

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5667 - accuracy: 0.7892

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5643 - accuracy: 0.7908

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5644 - accuracy: 0.7907

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5694 - accuracy: 0.7880

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5691 - accuracy: 0.7875

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5663 - accuracy: 0.7887

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5676 - accuracy: 0.7866

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5672 - accuracy: 0.7869

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5638 - accuracy: 0.7884

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5651 - accuracy: 0.7872

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5663 - accuracy: 0.7871

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5657 - accuracy: 0.7874

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5643 - accuracy: 0.7881

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5631 - accuracy: 0.7883

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5607 - accuracy: 0.7894

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5604 - accuracy: 0.7893

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5636 - accuracy: 0.7892

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5628 - accuracy: 0.7901

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5636 - accuracy: 0.7907

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5645 - accuracy: 0.7906

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5650 - accuracy: 0.7899

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.5650 - accuracy: 0.7899 - val_loss: 0.6865 - val_accuracy: 0.7384


.. parsed-literal::

    Epoch 14/15


.. parsed-literal::


 1/92 [..............................] - ETA: 6s - loss: 0.5964 - accuracy: 0.8125

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5740 - accuracy: 0.7656

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5757 - accuracy: 0.7604

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5654 - accuracy: 0.7812

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5495 - accuracy: 0.8000

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.5494 - accuracy: 0.8021

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.5713 - accuracy: 0.7857

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5753 - accuracy: 0.7930

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5722 - accuracy: 0.7882

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5584 - accuracy: 0.8000

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5542 - accuracy: 0.8040

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5490 - accuracy: 0.8073

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5517 - accuracy: 0.8101

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5408 - accuracy: 0.8125

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5433 - accuracy: 0.8125

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5399 - accuracy: 0.8125

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5283 - accuracy: 0.8143

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5167 - accuracy: 0.8194

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5163 - accuracy: 0.8174

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5131 - accuracy: 0.8203

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5171 - accuracy: 0.8170

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5165 - accuracy: 0.8168

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.5095 - accuracy: 0.8166

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5024 - accuracy: 0.8164

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5024 - accuracy: 0.8175

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5048 - accuracy: 0.8173

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.4967 - accuracy: 0.8218

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.4993 - accuracy: 0.8203

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5085 - accuracy: 0.8179

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5115 - accuracy: 0.8167

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5066 - accuracy: 0.8196

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5124 - accuracy: 0.8174

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5149 - accuracy: 0.8134

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5102 - accuracy: 0.8143

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5071 - accuracy: 0.8161

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5083 - accuracy: 0.8168

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5078 - accuracy: 0.8159

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5065 - accuracy: 0.8150

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5074 - accuracy: 0.8133

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.5130 - accuracy: 0.8102

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5113 - accuracy: 0.8110

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5091 - accuracy: 0.8118

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5025 - accuracy: 0.8154

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.4998 - accuracy: 0.8168

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5032 - accuracy: 0.8160

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5006 - accuracy: 0.8173

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.4978 - accuracy: 0.8178

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5000 - accuracy: 0.8151

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5077 - accuracy: 0.8119

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5054 - accuracy: 0.8119

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5055 - accuracy: 0.8119

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5013 - accuracy: 0.8131

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5074 - accuracy: 0.8096

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5076 - accuracy: 0.8108

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5025 - accuracy: 0.8131

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5015 - accuracy: 0.8142

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5071 - accuracy: 0.8114

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5062 - accuracy: 0.8109

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5082 - accuracy: 0.8099

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5128 - accuracy: 0.8083

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5145 - accuracy: 0.8079

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5156 - accuracy: 0.8075

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5146 - accuracy: 0.8085

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5137 - accuracy: 0.8081

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5141 - accuracy: 0.8072

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5159 - accuracy: 0.8068

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5163 - accuracy: 0.8060

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5170 - accuracy: 0.8033

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5140 - accuracy: 0.8043

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5126 - accuracy: 0.8049

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5156 - accuracy: 0.8024

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5149 - accuracy: 0.8021

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5143 - accuracy: 0.8027

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5171 - accuracy: 0.8007

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5207 - accuracy: 0.7987

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5188 - accuracy: 0.7998

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5171 - accuracy: 0.8007

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5175 - accuracy: 0.8005

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5181 - accuracy: 0.8002

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5183 - accuracy: 0.8000

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5224 - accuracy: 0.7986

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5184 - accuracy: 0.8006

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5171 - accuracy: 0.8015

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5180 - accuracy: 0.8020

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5204 - accuracy: 0.8014

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5185 - accuracy: 0.8022

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5177 - accuracy: 0.8027

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5209 - accuracy: 0.8018

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5180 - accuracy: 0.8029

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5165 - accuracy: 0.8034

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5172 - accuracy: 0.8031

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.5172 - accuracy: 0.8031 - val_loss: 0.7685 - val_accuracy: 0.7207


.. parsed-literal::

    Epoch 15/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.3400 - accuracy: 0.8438

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5200 - accuracy: 0.7812

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.4460 - accuracy: 0.8229

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.4709 - accuracy: 0.8281

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5184 - accuracy: 0.7937

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.5586 - accuracy: 0.7760

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.5329 - accuracy: 0.7902

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5370 - accuracy: 0.7930

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5265 - accuracy: 0.7951

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5387 - accuracy: 0.7844

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5586 - accuracy: 0.7784

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5551 - accuracy: 0.7839

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5399 - accuracy: 0.7909

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5232 - accuracy: 0.7969

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5354 - accuracy: 0.7917

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5393 - accuracy: 0.7852

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5327 - accuracy: 0.7904

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5186 - accuracy: 0.7986

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5108 - accuracy: 0.8026

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5105 - accuracy: 0.8047

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5022 - accuracy: 0.8110

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5011 - accuracy: 0.8111

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.4977 - accuracy: 0.8118

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5022 - accuracy: 0.8093

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.4976 - accuracy: 0.8107

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5001 - accuracy: 0.8119

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.4987 - accuracy: 0.8119

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.4996 - accuracy: 0.8098

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5062 - accuracy: 0.8088

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5003 - accuracy: 0.8110

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.4968 - accuracy: 0.8120

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.4972 - accuracy: 0.8120

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5130 - accuracy: 0.8065

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5042 - accuracy: 0.8103

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5037 - accuracy: 0.8086

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5029 - accuracy: 0.8087

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5110 - accuracy: 0.8063

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5150 - accuracy: 0.8032

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.5086 - accuracy: 0.8050

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5060 - accuracy: 0.8052

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5078 - accuracy: 0.8046

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5087 - accuracy: 0.8041

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5077 - accuracy: 0.8043

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5100 - accuracy: 0.8038

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5142 - accuracy: 0.8012

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5095 - accuracy: 0.8021

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5086 - accuracy: 0.8037

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5117 - accuracy: 0.8019

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5146 - accuracy: 0.8003

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5168 - accuracy: 0.8005

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5155 - accuracy: 0.8001

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5158 - accuracy: 0.8004

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5139 - accuracy: 0.8012

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5207 - accuracy: 0.7997

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5199 - accuracy: 0.8016

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5180 - accuracy: 0.8034

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5204 - accuracy: 0.8014

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5194 - accuracy: 0.8016

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5176 - accuracy: 0.8023

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5157 - accuracy: 0.8030

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5148 - accuracy: 0.8036

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5116 - accuracy: 0.8058

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5100 - accuracy: 0.8069

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5105 - accuracy: 0.8060

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5124 - accuracy: 0.8023

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5127 - accuracy: 0.8034

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5081 - accuracy: 0.8044

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5059 - accuracy: 0.8055

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5087 - accuracy: 0.8024

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5087 - accuracy: 0.8030

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5055 - accuracy: 0.8049

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5050 - accuracy: 0.8050

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5085 - accuracy: 0.8025

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5105 - accuracy: 0.8014

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5090 - accuracy: 0.8020

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5121 - accuracy: 0.8013

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5111 - accuracy: 0.8010

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5103 - accuracy: 0.8016

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5117 - accuracy: 0.7998

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5111 - accuracy: 0.7995

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5085 - accuracy: 0.8012

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5101 - accuracy: 0.8002

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5108 - accuracy: 0.8004

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5135 - accuracy: 0.7994

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5154 - accuracy: 0.7992

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5170 - accuracy: 0.7986

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5180 - accuracy: 0.7988

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5165 - accuracy: 0.7996

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5162 - accuracy: 0.8005

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5184 - accuracy: 0.7992

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5179 - accuracy: 0.7994

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.5179 - accuracy: 0.7994 - val_loss: 0.7059 - val_accuracy: 0.7357


Visualize Training Results
--------------------------



After applying data augmentation and Dropout, there is less overfitting
than before, and training and validation accuracy are closer aligned.

.. code:: ipython3

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()



.. image:: 301-tensorflow-training-openvino-with-output_files/301-tensorflow-training-openvino-with-output_66_0.png


Predict on New Data
-------------------



Finally, let us use the model to classify an image that was not included
in the training or validation sets.

   **Note**: Data augmentation and Dropout layers are inactive at
   inference time.

.. code:: ipython3

    sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
    sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

    img = keras.preprocessing.image.load_img(
        sunflower_path, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )


.. parsed-literal::


   1/1 [==============================] - ETA: 0s

.. parsed-literal::

    
1/1 [==============================] - 0s 99ms/step


.. parsed-literal::

    This image most likely belongs to sunflowers with a 96.93 percent confidence.


Save the TensorFlow Model
-------------------------



.. code:: ipython3

    #save the trained model - a new folder flower will be created
    #and the file "saved_model.pb" is the pre-trained model
    model_dir = "model"
    saved_model_dir = f"{model_dir}/flower/saved_model"
    model.save(saved_model_dir)


.. parsed-literal::

    2024-03-13 01:04:01.596956: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'random_flip_input' with dtype float and shape [?,180,180,3]
    	 [[{{node random_flip_input}}]]
    2024-03-13 01:04:01.681918: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-13 01:04:01.691876: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'random_flip_input' with dtype float and shape [?,180,180,3]
    	 [[{{node random_flip_input}}]]
    2024-03-13 01:04:01.702720: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-13 01:04:01.709631: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-13 01:04:01.716729: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-13 01:04:01.728191: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-13 01:04:01.769744: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'sequential_1_input' with dtype float and shape [?,180,180,3]
    	 [[{{node sequential_1_input}}]]


.. parsed-literal::

    2024-03-13 01:04:01.859999: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-13 01:04:01.880443: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'sequential_1_input' with dtype float and shape [?,180,180,3]
    	 [[{{node sequential_1_input}}]]
    2024-03-13 01:04:01.919461: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,22,22,64]
    	 [[{{node inputs}}]]
    2024-03-13 01:04:01.944149: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-13 01:04:02.016302: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]


.. parsed-literal::

    2024-03-13 01:04:02.156624: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-13 01:04:02.294259: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,22,22,64]
    	 [[{{node inputs}}]]
    2024-03-13 01:04:02.328020: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-13 01:04:02.355628: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]


.. parsed-literal::

    2024-03-13 01:04:02.402015: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    WARNING:absl:Found untraced functions such as _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op, _update_step_xla while saving (showing 4 of 4). These functions will not be directly callable after loading.


.. parsed-literal::

    INFO:tensorflow:Assets written to: model/flower/saved_model/assets


.. parsed-literal::

    INFO:tensorflow:Assets written to: model/flower/saved_model/assets


Convert the TensorFlow model with OpenVINO Model Conversion API
---------------------------------------------------------------

To convert the model to
OpenVINO IR with ``FP16`` precision, use model conversion Python API.

.. code:: ipython3

    # Convert the model to ir model format and save it.
    ir_model_path = Path("model/flower")
    ir_model_path.mkdir(parents=True, exist_ok=True)
    ir_model = ov.convert_model(saved_model_dir, input=[1,180,180,3])
    ov.save_model(ir_model, ir_model_path / "flower_ir.xml")

Preprocessing Image Function
----------------------------



.. code:: ipython3

    def pre_process_image(imagePath, img_height=180):
        # Model input format
        n, h, w, c = [1, img_height, img_height, 3]
        image = Image.open(imagePath)
        image = image.resize((h, w), resample=Image.BILINEAR)

        # Convert to array and change data layout from HWC to CHW
        image = np.array(image)
        input_image = image.reshape((n, h, w, c))

        return input_image

OpenVINO Runtime Setup
----------------------



Select inference device
~~~~~~~~~~~~~~~~~~~~~~~



select device from dropdown list for running inference using OpenVINO

.. code:: ipython3

    import ipywidgets as widgets

    # Initialize OpenVINO runtime
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

    class_names=["daisy", "dandelion", "roses", "sunflowers", "tulips"]

    compiled_model = core.compile_model(model=ir_model, device_name=device.value)

    del ir_model

    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

Run the Inference Step
----------------------



.. code:: ipython3

    # Run inference on the input image...
    inp_img_url = "https://upload.wikimedia.org/wikipedia/commons/4/48/A_Close_Up_Photo_of_a_Dandelion.jpg"
    OUTPUT_DIR = "output"
    inp_file_name = f"A_Close_Up_Photo_of_a_Dandelion.jpg"
    file_path = Path(OUTPUT_DIR)/Path(inp_file_name)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Download the image
    download_file(inp_img_url, inp_file_name, directory=OUTPUT_DIR)

    # Pre-process the image and get it ready for inference.
    input_image = pre_process_image(file_path)

    print(input_image.shape)
    print(input_layer.shape)
    res = compiled_model([input_image])[output_layer]

    score = tf.nn.softmax(res[0])

    # Show the results
    image = Image.open(file_path)
    plt.imshow(image)
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )


.. parsed-literal::

    'output/A_Close_Up_Photo_of_a_Dandelion.jpg' already exists.
    (1, 180, 180, 3)
    [1,180,180,3]
    This image most likely belongs to dandelion with a 95.08 percent confidence.



.. image:: 301-tensorflow-training-openvino-with-output_files/301-tensorflow-training-openvino-with-output_79_1.png


The Next Steps
--------------



This tutorial showed how to train a TensorFlow model, how to convert
that model to OpenVINO’s IR format, and how to do inference on the
converted model. For faster inference speed, you can quantize the IR
model. To see how to quantize this model with OpenVINO’s `Post-training
Quantization with NNCF
Tool <https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.html>`__,
check out the `Post-Training Quantization with TensorFlow Classification
Model <301-tensorflow-training-openvino-nncf-with-output.html>`__ notebook.
