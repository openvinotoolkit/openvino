From Training to Deployment with TensorFlow and OpenVINO™
=========================================================

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `TensorFlow Image Classification
   Training <#TensorFlow-Image-Classification-Training>`__
-  `Import TensorFlow and Other
   Libraries <#Import-TensorFlow-and-Other-Libraries>`__
-  `Download and Explore the
   Dataset <#Download-and-Explore-the-Dataset>`__
-  `Load Using keras.preprocessing <#Load-Using-keras.preprocessing>`__
-  `Create a Dataset <#Create-a-Dataset>`__
-  `Visualize the Data <#Visualize-the-Data>`__
-  `Configure the Dataset for
   Performance <#Configure-the-Dataset-for-Performance>`__
-  `Standardize the Data <#Standardize-the-Data>`__
-  `Create the Model <#Create-the-Model>`__
-  `Compile the Model <#Compile-the-Model>`__
-  `Model Summary <#Model-Summary>`__
-  `Train the Model <#Train-the-Model>`__
-  `Visualize Training Results <#Visualize-Training-Results>`__
-  `Overfitting <#Overfitting>`__
-  `Data Augmentation <#Data-Augmentation>`__
-  `Dropout <#Dropout>`__
-  `Compile and Train the Model <#Compile-and-Train-the-Model>`__
-  `Visualize Training Results <#Visualize-Training-Results>`__
-  `Predict on New Data <#Predict-on-New-Data>`__
-  `Save the TensorFlow Model <#Save-the-TensorFlow-Model>`__
-  `Convert the TensorFlow model with OpenVINO Model Conversion
   API <#Convert-the-TensorFlow-model-with-OpenVINO-Model-Conversion-API>`__
-  `Preprocessing Image Function <#Preprocessing-Image-Function>`__
-  `OpenVINO Runtime Setup <#OpenVINO-Runtime-Setup>`__

   -  `Select inference device <#Select-inference-device>`__

-  `Run the Inference Step <#Run-the-Inference-Step>`__
-  `The Next Steps <#The-Next-Steps>`__

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
Classification Model <./301-tensorflow-training-openvino-nncf.ipynb>`__
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

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.0 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


TensorFlow Image Classification Training
----------------------------------------

`back to top ⬆️ <#Table-of-contents:>`__

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

`back to top ⬆️ <#Table-of-contents:>`__

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

    2024-01-26 00:42:00.332239: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-01-26 00:42:00.367395: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


.. parsed-literal::

    2024-01-26 00:42:00.882008: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


Download and Explore the Dataset
--------------------------------

`back to top ⬆️ <#Table-of-contents:>`__

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

`back to top ⬆️ <#Table-of-contents:>`__

Let’s load these images off disk using the helpful
`image_dataset_from_directory <https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory>`__
utility. This will take you from a directory of images on disk to a
``tf.data.Dataset`` in just a couple lines of code. If you like, you can
also write your own data loading code from scratch by visiting the `load
images <https://www.tensorflow.org/tutorials/load_data/images>`__
tutorial.

Create a Dataset
----------------

`back to top ⬆️ <#Table-of-contents:>`__

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

    2024-01-26 00:42:03.998215: E tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:266] failed call to cuInit: CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: forward compatibility was attempted on non supported HW
    2024-01-26 00:42:03.998247: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:168] retrieving CUDA diagnostic information for host: iotg-dev-workstation-07
    2024-01-26 00:42:03.998252: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:175] hostname: iotg-dev-workstation-07
    2024-01-26 00:42:03.998380: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:199] libcuda reported version is: 470.223.2
    2024-01-26 00:42:03.998394: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:203] kernel reported version is: 470.182.3
    2024-01-26 00:42:03.998398: E tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:312] kernel version 470.182.3 does not match DSO version 470.223.2 -- cannot find working devices in this configuration


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

`back to top ⬆️ <#Table-of-contents:>`__

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

    2024-01-26 00:42:04.349495: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-01-26 00:42:04.350184: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
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

    2024-01-26 00:42:05.179899: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-01-26 00:42:05.180559: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]


The ``image_batch`` is a tensor of the shape ``(32, 180, 180, 3)``. This
is a batch of 32 images of shape ``180x180x3`` (the last dimension
refers to color channels RGB). The ``label_batch`` is a tensor of the
shape ``(32,)``, these are corresponding labels to the 32 images.

You can call ``.numpy()`` on the ``image_batch`` and ``labels_batch``
tensors to convert them to a ``numpy.ndarray``.

Configure the Dataset for Performance
-------------------------------------

`back to top ⬆️ <#Table-of-contents:>`__

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

`back to top ⬆️ <#Table-of-contents:>`__

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

    2024-01-26 00:42:05.337060: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]
    2024-01-26 00:42:05.337348: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]


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

`back to top ⬆️ <#Table-of-contents:>`__

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

`back to top ⬆️ <#Table-of-contents:>`__

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

`back to top ⬆️ <#Table-of-contents:>`__

View all the layers of the network using the model’s ``summary`` method.

   **NOTE:** This section is commented out for performance reasons.
   Please feel free to uncomment these to compare the results.

.. code:: ipython3

    # model.summary()

Train the Model
---------------

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    # epochs=10
    # history = model.fit(
    #   train_ds,
    #   validation_data=val_ds,
    #   epochs=epochs
    # )

Visualize Training Results
--------------------------

`back to top ⬆️ <#Table-of-contents:>`__

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

`back to top ⬆️ <#Table-of-contents:>`__

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

`back to top ⬆️ <#Table-of-contents:>`__

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

    2024-01-26 00:42:06.253764: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-01-26 00:42:06.254075: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]



.. image:: 301-tensorflow-training-openvino-with-output_files/301-tensorflow-training-openvino-with-output_57_1.png


You will use data augmentation to train a model in a moment.

Dropout
-------

`back to top ⬆️ <#Table-of-contents:>`__

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

`back to top ⬆️ <#Table-of-contents:>`__

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

                                                                     


.. parsed-literal::

     rescaling_2 (Rescaling)     (None, 180, 180, 3)       0         


.. parsed-literal::

                                                                     


.. parsed-literal::

     conv2d_3 (Conv2D)           (None, 180, 180, 16)      448       


.. parsed-literal::

                                                                     


.. parsed-literal::

     max_pooling2d_3 (MaxPooling  (None, 90, 90, 16)       0         


.. parsed-literal::

     2D)                                                             


.. parsed-literal::

                                                                     


.. parsed-literal::

     conv2d_4 (Conv2D)           (None, 90, 90, 32)        4640      


.. parsed-literal::

                                                                     


.. parsed-literal::

     max_pooling2d_4 (MaxPooling  (None, 45, 45, 32)       0         


.. parsed-literal::

     2D)                                                             


.. parsed-literal::

                                                                     


.. parsed-literal::

     conv2d_5 (Conv2D)           (None, 45, 45, 64)        18496     


.. parsed-literal::

                                                                     


.. parsed-literal::

     max_pooling2d_5 (MaxPooling  (None, 22, 22, 64)       0         


.. parsed-literal::

     2D)                                                             


.. parsed-literal::

                                                                     


.. parsed-literal::

     dropout (Dropout)           (None, 22, 22, 64)        0         


.. parsed-literal::

                                                                     


.. parsed-literal::

     flatten_1 (Flatten)         (None, 30976)             0         


.. parsed-literal::

                                                                     


.. parsed-literal::

     dense_2 (Dense)             (None, 128)               3965056   


.. parsed-literal::

                                                                     


.. parsed-literal::

     outputs (Dense)             (None, 5)                 645       


.. parsed-literal::

                                                                     


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

    2024-01-26 00:42:07.329333: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-01-26 00:42:07.329961: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]


.. parsed-literal::

     1/92 [..............................] - ETA: 1:25 - loss: 1.6203 - accuracy: 0.1875

.. parsed-literal::

     2/92 [..............................] - ETA: 6s - loss: 1.9754 - accuracy: 0.2031  

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 1.9186 - accuracy: 0.1875

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 1.8590 - accuracy: 0.2031

.. parsed-literal::

     5/92 [>.............................] - ETA: 5s - loss: 1.8031 - accuracy: 0.2125

.. parsed-literal::

     6/92 [>.............................] - ETA: 5s - loss: 1.7529 - accuracy: 0.2188

.. parsed-literal::

     7/92 [=>............................] - ETA: 5s - loss: 1.7289 - accuracy: 0.2188

.. parsed-literal::

     8/92 [=>............................] - ETA: 5s - loss: 1.7038 - accuracy: 0.2344

.. parsed-literal::

     9/92 [=>............................] - ETA: 5s - loss: 1.6795 - accuracy: 0.2604

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 1.6676 - accuracy: 0.2562

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 1.6419 - accuracy: 0.2756

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 1.6254 - accuracy: 0.2708

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 1.6172 - accuracy: 0.2740

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 1.5975 - accuracy: 0.2790

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 1.5838 - accuracy: 0.2833

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 1.5660 - accuracy: 0.2969

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 1.5509 - accuracy: 0.3033

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 1.5461 - accuracy: 0.3003

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 1.5413 - accuracy: 0.3026

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 1.5437 - accuracy: 0.3031

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 1.5320 - accuracy: 0.3110

.. parsed-literal::

    22/92 [======>.......................] - ETA: 4s - loss: 1.5235 - accuracy: 0.3168

.. parsed-literal::

    23/92 [======>.......................] - ETA: 4s - loss: 1.5163 - accuracy: 0.3234

.. parsed-literal::

    24/92 [======>.......................] - ETA: 4s - loss: 1.5035 - accuracy: 0.3242

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 1.4887 - accuracy: 0.3350

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 1.4841 - accuracy: 0.3377

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 1.4836 - accuracy: 0.3345

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 1.4765 - accuracy: 0.3348

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 1.4688 - accuracy: 0.3373

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 1.4625 - accuracy: 0.3417

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 1.4595 - accuracy: 0.3448

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 1.4511 - accuracy: 0.3467

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 1.4499 - accuracy: 0.3466

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 1.4439 - accuracy: 0.3493

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 1.4377 - accuracy: 0.3545

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 1.4344 - accuracy: 0.3533

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 1.4311 - accuracy: 0.3590

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 1.4285 - accuracy: 0.3602

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 1.4242 - accuracy: 0.3614

.. parsed-literal::

    40/92 [============>.................] - ETA: 3s - loss: 1.4229 - accuracy: 0.3594

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 1.4200 - accuracy: 0.3620

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 1.4185 - accuracy: 0.3638

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 1.4174 - accuracy: 0.3670

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 1.4132 - accuracy: 0.3686

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 1.4114 - accuracy: 0.3694

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 1.4081 - accuracy: 0.3709

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 1.4015 - accuracy: 0.3730

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 1.3994 - accuracy: 0.3737

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 1.3961 - accuracy: 0.3750

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 1.3938 - accuracy: 0.3756

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 1.3888 - accuracy: 0.3787

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 1.3848 - accuracy: 0.3780

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 1.3801 - accuracy: 0.3809

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 1.3771 - accuracy: 0.3831

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 1.3750 - accuracy: 0.3852

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 1.3746 - accuracy: 0.3867

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 1.3701 - accuracy: 0.3914

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 1.3630 - accuracy: 0.3939

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 1.3625 - accuracy: 0.3957

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 1.3603 - accuracy: 0.3958

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 1.3590 - accuracy: 0.3955

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 1.3592 - accuracy: 0.3952

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 1.3561 - accuracy: 0.3968

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 1.3516 - accuracy: 0.3965

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 1.3487 - accuracy: 0.3990

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 1.3489 - accuracy: 0.3987

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 1.3444 - accuracy: 0.4011

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 1.3413 - accuracy: 0.4036

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 1.3384 - accuracy: 0.4055

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 1.3371 - accuracy: 0.4046

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 1.3366 - accuracy: 0.4059

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 1.3337 - accuracy: 0.4055

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 1.3321 - accuracy: 0.4064

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 1.3283 - accuracy: 0.4081

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 1.3256 - accuracy: 0.4084

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 1.3243 - accuracy: 0.4088

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 1.3221 - accuracy: 0.4104

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 1.3225 - accuracy: 0.4112

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 1.3212 - accuracy: 0.4115

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 1.3216 - accuracy: 0.4103

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 1.3196 - accuracy: 0.4114

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 1.3148 - accuracy: 0.4140

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 1.3129 - accuracy: 0.4158

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 1.3104 - accuracy: 0.4172

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 1.3077 - accuracy: 0.4192

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 1.3082 - accuracy: 0.4191

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 1.3039 - accuracy: 0.4218

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 1.3012 - accuracy: 0.4224

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 1.2993 - accuracy: 0.4239

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 1.2963 - accuracy: 0.4244

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 1.2937 - accuracy: 0.4249

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 1.2896 - accuracy: 0.4264

.. parsed-literal::

    2024-01-26 00:42:13.569078: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [734]
    	 [[{{node Placeholder/_4}}]]
    2024-01-26 00:42:13.569352: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [734]
    	 [[{{node Placeholder/_4}}]]


.. parsed-literal::

    92/92 [==============================] - 7s 66ms/step - loss: 1.2896 - accuracy: 0.4264 - val_loss: 1.1305 - val_accuracy: 0.5313


.. parsed-literal::

    Epoch 2/15


.. parsed-literal::

     1/92 [..............................] - ETA: 8s - loss: 1.0067 - accuracy: 0.5625

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 1.0781 - accuracy: 0.5469

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 1.0575 - accuracy: 0.5625

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 1.0389 - accuracy: 0.5703

.. parsed-literal::

     5/92 [>.............................] - ETA: 5s - loss: 1.0066 - accuracy: 0.5813

.. parsed-literal::

     6/92 [>.............................] - ETA: 5s - loss: 1.0164 - accuracy: 0.5677

.. parsed-literal::

     7/92 [=>............................] - ETA: 4s - loss: 1.0422 - accuracy: 0.5714

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 1.0701 - accuracy: 0.5781

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 1.0450 - accuracy: 0.5938

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 1.0396 - accuracy: 0.5906

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 1.0486 - accuracy: 0.5710

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 1.0365 - accuracy: 0.5729

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 1.0414 - accuracy: 0.5673

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 1.0664 - accuracy: 0.5603

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 1.0629 - accuracy: 0.5583

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 1.0657 - accuracy: 0.5586

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 1.0727 - accuracy: 0.5607

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 1.0701 - accuracy: 0.5590

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 1.0686 - accuracy: 0.5592

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 1.0909 - accuracy: 0.5516

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 1.0968 - accuracy: 0.5476

.. parsed-literal::

    22/92 [======>.......................] - ETA: 4s - loss: 1.0948 - accuracy: 0.5483

.. parsed-literal::

    23/92 [======>.......................] - ETA: 4s - loss: 1.1035 - accuracy: 0.5489

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 1.1073 - accuracy: 0.5469

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 1.1018 - accuracy: 0.5500

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 1.1041 - accuracy: 0.5493

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 1.1050 - accuracy: 0.5486

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 1.1117 - accuracy: 0.5424

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 1.1145 - accuracy: 0.5377

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 1.1058 - accuracy: 0.5406

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 1.1066 - accuracy: 0.5403

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 1.1098 - accuracy: 0.5391

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 1.1188 - accuracy: 0.5360

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 1.1182 - accuracy: 0.5386

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 1.1158 - accuracy: 0.5402

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 1.1131 - accuracy: 0.5425

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 1.1148 - accuracy: 0.5405

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 1.1095 - accuracy: 0.5461

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 1.1034 - accuracy: 0.5473

.. parsed-literal::

    40/92 [============>.................] - ETA: 3s - loss: 1.0962 - accuracy: 0.5500

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 1.0945 - accuracy: 0.5518

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 1.0917 - accuracy: 0.5528

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 1.0885 - accuracy: 0.5531

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 1.0893 - accuracy: 0.5526

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 1.0825 - accuracy: 0.5556

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 1.0843 - accuracy: 0.5550

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 1.0807 - accuracy: 0.5572

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 1.0801 - accuracy: 0.5579

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 1.0763 - accuracy: 0.5599

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 1.0758 - accuracy: 0.5594

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 1.0737 - accuracy: 0.5613

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 1.0737 - accuracy: 0.5619

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 1.0737 - accuracy: 0.5654

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 1.0708 - accuracy: 0.5671

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 1.0734 - accuracy: 0.5659

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 1.0724 - accuracy: 0.5675

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 1.0737 - accuracy: 0.5669

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 1.0798 - accuracy: 0.5647

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 1.0763 - accuracy: 0.5651

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 1.0758 - accuracy: 0.5651

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 1.0763 - accuracy: 0.5656

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 1.0766 - accuracy: 0.5665

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 1.0791 - accuracy: 0.5660

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 1.0769 - accuracy: 0.5674

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 1.0760 - accuracy: 0.5683

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 1.0759 - accuracy: 0.5663

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 1.0759 - accuracy: 0.5662

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 1.0751 - accuracy: 0.5671

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 1.0733 - accuracy: 0.5693

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 1.0717 - accuracy: 0.5688

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 1.0715 - accuracy: 0.5691

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 1.0706 - accuracy: 0.5699

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 1.0694 - accuracy: 0.5689

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 1.0668 - accuracy: 0.5714

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 1.0649 - accuracy: 0.5729

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 1.0649 - accuracy: 0.5732

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 1.0664 - accuracy: 0.5722

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 1.0630 - accuracy: 0.5753

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 1.0597 - accuracy: 0.5760

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 1.0591 - accuracy: 0.5755

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 1.0585 - accuracy: 0.5757

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 1.0612 - accuracy: 0.5752

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 1.0596 - accuracy: 0.5746

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 1.0604 - accuracy: 0.5749

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 1.0574 - accuracy: 0.5765

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 1.0576 - accuracy: 0.5760

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 1.0567 - accuracy: 0.5759

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 1.0575 - accuracy: 0.5750

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 1.0566 - accuracy: 0.5745

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 1.0576 - accuracy: 0.5747

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 1.0573 - accuracy: 0.5753

.. parsed-literal::

    92/92 [==============================] - 6s 64ms/step - loss: 1.0573 - accuracy: 0.5753 - val_loss: 1.0449 - val_accuracy: 0.6035


.. parsed-literal::

    Epoch 3/15


.. parsed-literal::

     1/92 [..............................] - ETA: 7s - loss: 1.1557 - accuracy: 0.5000

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 1.0650 - accuracy: 0.5781

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 1.0306 - accuracy: 0.5729

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 1.0342 - accuracy: 0.5703

.. parsed-literal::

     5/92 [>.............................] - ETA: 5s - loss: 1.0044 - accuracy: 0.5750

.. parsed-literal::

     6/92 [>.............................] - ETA: 4s - loss: 0.9928 - accuracy: 0.5885

.. parsed-literal::

     7/92 [=>............................] - ETA: 4s - loss: 0.9934 - accuracy: 0.5982

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 1.0011 - accuracy: 0.5859

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 1.0114 - accuracy: 0.5764

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 1.0015 - accuracy: 0.5844

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 0.9888 - accuracy: 0.5824

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 0.9908 - accuracy: 0.5781

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 0.9801 - accuracy: 0.5865

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 0.9834 - accuracy: 0.5759

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 0.9809 - accuracy: 0.5729

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 0.9973 - accuracy: 0.5645

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 0.9889 - accuracy: 0.5662

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 0.9866 - accuracy: 0.5677

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 0.9672 - accuracy: 0.5822

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 0.9611 - accuracy: 0.5844

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 0.9658 - accuracy: 0.5878

.. parsed-literal::

    22/92 [======>.......................] - ETA: 4s - loss: 0.9588 - accuracy: 0.5923

.. parsed-literal::

    23/92 [======>.......................] - ETA: 3s - loss: 0.9503 - accuracy: 0.5978

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 0.9453 - accuracy: 0.6042

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 0.9493 - accuracy: 0.6112

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 0.9528 - accuracy: 0.6130

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 0.9477 - accuracy: 0.6157

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 0.9615 - accuracy: 0.6105

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 0.9635 - accuracy: 0.6145

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 0.9791 - accuracy: 0.6087

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 0.9716 - accuracy: 0.6122

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 0.9729 - accuracy: 0.6107

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 0.9754 - accuracy: 0.6083

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 0.9711 - accuracy: 0.6106

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 0.9697 - accuracy: 0.6110

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 0.9723 - accuracy: 0.6097

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 0.9652 - accuracy: 0.6126

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 0.9645 - accuracy: 0.6121

.. parsed-literal::

    40/92 [============>.................] - ETA: 2s - loss: 0.9650 - accuracy: 0.6108

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 0.9560 - accuracy: 0.6143

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 0.9581 - accuracy: 0.6130

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 0.9563 - accuracy: 0.6133

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 0.9547 - accuracy: 0.6136

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 0.9558 - accuracy: 0.6138

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 0.9564 - accuracy: 0.6141

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 0.9545 - accuracy: 0.6136

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 0.9599 - accuracy: 0.6132

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 0.9561 - accuracy: 0.6154

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 0.9533 - accuracy: 0.6181

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 0.9518 - accuracy: 0.6195

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 0.9506 - accuracy: 0.6214

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 0.9505 - accuracy: 0.6232

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 0.9523 - accuracy: 0.6221

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 0.9475 - accuracy: 0.6250

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 0.9474 - accuracy: 0.6233

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 0.9472 - accuracy: 0.6233

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 0.9449 - accuracy: 0.6245

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 0.9449 - accuracy: 0.6255

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 0.9461 - accuracy: 0.6245

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 0.9460 - accuracy: 0.6245

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 0.9456 - accuracy: 0.6240

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 0.9464 - accuracy: 0.6250

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 0.9448 - accuracy: 0.6265

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 0.9452 - accuracy: 0.6264

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 0.9424 - accuracy: 0.6264

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 0.9435 - accuracy: 0.6250

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 0.9419 - accuracy: 0.6264

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 0.9440 - accuracy: 0.6255

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 0.9466 - accuracy: 0.6254

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 0.9440 - accuracy: 0.6263

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 0.9442 - accuracy: 0.6259

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 0.9454 - accuracy: 0.6254

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 0.9437 - accuracy: 0.6254

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 0.9421 - accuracy: 0.6267

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 0.9441 - accuracy: 0.6258

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 0.9430 - accuracy: 0.6274

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 0.9427 - accuracy: 0.6258

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 0.9396 - accuracy: 0.6278

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 0.9403 - accuracy: 0.6262

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 0.9417 - accuracy: 0.6258

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 0.9408 - accuracy: 0.6261

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 0.9423 - accuracy: 0.6265

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 0.9408 - accuracy: 0.6272

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 0.9420 - accuracy: 0.6268

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 0.9406 - accuracy: 0.6279

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 0.9429 - accuracy: 0.6272

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 0.9432 - accuracy: 0.6264

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 0.9444 - accuracy: 0.6257

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 0.9439 - accuracy: 0.6260

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 0.9425 - accuracy: 0.6271

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 0.9424 - accuracy: 0.6270

.. parsed-literal::

    92/92 [==============================] - 6s 63ms/step - loss: 0.9424 - accuracy: 0.6270 - val_loss: 0.9305 - val_accuracy: 0.6240


.. parsed-literal::

    Epoch 4/15


.. parsed-literal::

     1/92 [..............................] - ETA: 7s - loss: 0.8790 - accuracy: 0.6875

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 0.9336 - accuracy: 0.6562

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 0.8945 - accuracy: 0.6771

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 0.9265 - accuracy: 0.6484

.. parsed-literal::

     5/92 [>.............................] - ETA: 4s - loss: 0.9851 - accuracy: 0.6125

.. parsed-literal::

     6/92 [>.............................] - ETA: 4s - loss: 0.9562 - accuracy: 0.6146

.. parsed-literal::

     7/92 [=>............................] - ETA: 4s - loss: 0.9706 - accuracy: 0.6161

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 0.9789 - accuracy: 0.6133

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 0.9524 - accuracy: 0.6285

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 0.9482 - accuracy: 0.6250

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 0.9476 - accuracy: 0.6335

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 0.9545 - accuracy: 0.6302

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 0.9310 - accuracy: 0.6418

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 0.9195 - accuracy: 0.6429

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 0.9373 - accuracy: 0.6292

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 0.9522 - accuracy: 0.6230

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 0.9512 - accuracy: 0.6232

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 0.9260 - accuracy: 0.6333

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 0.9390 - accuracy: 0.6297

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 0.9430 - accuracy: 0.6310

.. parsed-literal::

    22/92 [======>.......................] - ETA: 3s - loss: 0.9366 - accuracy: 0.6322

.. parsed-literal::

    23/92 [======>.......................] - ETA: 3s - loss: 0.9332 - accuracy: 0.6332

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 0.9337 - accuracy: 0.6316

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 0.9276 - accuracy: 0.6338

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 0.9203 - accuracy: 0.6359

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 0.9213 - accuracy: 0.6367

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 0.9148 - accuracy: 0.6441

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 0.9116 - accuracy: 0.6478

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 0.9109 - accuracy: 0.6481

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 0.9093 - accuracy: 0.6484

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 0.9057 - accuracy: 0.6467

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 0.8989 - accuracy: 0.6498

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 0.8952 - accuracy: 0.6481

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 0.8964 - accuracy: 0.6493

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 0.8883 - accuracy: 0.6521

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 0.8936 - accuracy: 0.6514

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 0.8956 - accuracy: 0.6498

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 0.9078 - accuracy: 0.6419

.. parsed-literal::

    40/92 [============>.................] - ETA: 2s - loss: 0.9082 - accuracy: 0.6415

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 0.9035 - accuracy: 0.6442

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 0.8994 - accuracy: 0.6460

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 0.9011 - accuracy: 0.6455

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 0.8990 - accuracy: 0.6464

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 0.8995 - accuracy: 0.6473

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 0.8976 - accuracy: 0.6475

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 0.8958 - accuracy: 0.6477

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 0.8925 - accuracy: 0.6499

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 0.8949 - accuracy: 0.6506

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 0.8920 - accuracy: 0.6514

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 0.8881 - accuracy: 0.6539

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 0.8854 - accuracy: 0.6564

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 0.8923 - accuracy: 0.6540

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 0.8901 - accuracy: 0.6552

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 0.8850 - accuracy: 0.6564

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 0.8850 - accuracy: 0.6570

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 0.8864 - accuracy: 0.6553

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 0.8836 - accuracy: 0.6558

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 0.8804 - accuracy: 0.6559

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 0.8785 - accuracy: 0.6559

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 0.8780 - accuracy: 0.6559

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 0.8755 - accuracy: 0.6569

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 0.8713 - accuracy: 0.6589

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 0.8718 - accuracy: 0.6583

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 0.8698 - accuracy: 0.6588

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 0.8666 - accuracy: 0.6602

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 0.8643 - accuracy: 0.6610

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 0.8638 - accuracy: 0.6601

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 0.8662 - accuracy: 0.6595

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 0.8658 - accuracy: 0.6604

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 0.8642 - accuracy: 0.6608

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 0.8649 - accuracy: 0.6598

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 0.8689 - accuracy: 0.6589

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 0.8701 - accuracy: 0.6576

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 0.8725 - accuracy: 0.6589

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 0.8726 - accuracy: 0.6584

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 0.8721 - accuracy: 0.6584

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 0.8708 - accuracy: 0.6584

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 0.8676 - accuracy: 0.6603

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 0.8675 - accuracy: 0.6595

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 0.8662 - accuracy: 0.6594

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 0.8669 - accuracy: 0.6586

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 0.8659 - accuracy: 0.6590

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 0.8695 - accuracy: 0.6567

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 0.8712 - accuracy: 0.6560

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 0.8703 - accuracy: 0.6563

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 0.8696 - accuracy: 0.6563

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 0.8673 - accuracy: 0.6581

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 0.8684 - accuracy: 0.6585

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 0.8684 - accuracy: 0.6577

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 0.8664 - accuracy: 0.6587

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 0.8674 - accuracy: 0.6584

.. parsed-literal::

    92/92 [==============================] - 6s 63ms/step - loss: 0.8674 - accuracy: 0.6584 - val_loss: 0.8974 - val_accuracy: 0.6594


.. parsed-literal::

    Epoch 5/15


.. parsed-literal::

     1/92 [..............................] - ETA: 7s - loss: 0.6094 - accuracy: 0.8438

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 0.7067 - accuracy: 0.7812

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 0.6755 - accuracy: 0.7917

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 0.6745 - accuracy: 0.7891

.. parsed-literal::

     5/92 [>.............................] - ETA: 5s - loss: 0.7159 - accuracy: 0.7500

.. parsed-literal::

     6/92 [>.............................] - ETA: 5s - loss: 0.7425 - accuracy: 0.7500

.. parsed-literal::

     7/92 [=>............................] - ETA: 4s - loss: 0.7390 - accuracy: 0.7589

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 0.7454 - accuracy: 0.7461

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 0.7451 - accuracy: 0.7361

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 0.7596 - accuracy: 0.7219

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 0.7364 - accuracy: 0.7330

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 0.7473 - accuracy: 0.7206

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 0.7571 - accuracy: 0.7136

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 0.7698 - accuracy: 0.7076

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 0.7877 - accuracy: 0.7004

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 0.7878 - accuracy: 0.6959

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 0.8037 - accuracy: 0.6866

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 0.8011 - accuracy: 0.6850

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 0.8051 - accuracy: 0.6820

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 0.8114 - accuracy: 0.6792

.. parsed-literal::

    22/92 [======>.......................] - ETA: 3s - loss: 0.8062 - accuracy: 0.6825

.. parsed-literal::

    23/92 [======>.......................] - ETA: 3s - loss: 0.8055 - accuracy: 0.6841

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 0.8056 - accuracy: 0.6829

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 0.8034 - accuracy: 0.6843

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 0.8004 - accuracy: 0.6881

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 0.8049 - accuracy: 0.6869

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 0.8011 - accuracy: 0.6892

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 0.7966 - accuracy: 0.6935

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 0.8069 - accuracy: 0.6933

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 0.8075 - accuracy: 0.6951

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 0.8047 - accuracy: 0.6949

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 0.8064 - accuracy: 0.6937

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 0.8016 - accuracy: 0.6972

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 0.7963 - accuracy: 0.6996

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 0.7950 - accuracy: 0.6976

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 0.7951 - accuracy: 0.6990

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 0.7964 - accuracy: 0.6987

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 0.8014 - accuracy: 0.6976

.. parsed-literal::

    40/92 [============>.................] - ETA: 2s - loss: 0.7988 - accuracy: 0.6989

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 0.7969 - accuracy: 0.6979

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 0.8011 - accuracy: 0.6969

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 0.8063 - accuracy: 0.6952

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 0.8123 - accuracy: 0.6936

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 0.8116 - accuracy: 0.6948

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 0.8077 - accuracy: 0.6974

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 0.8018 - accuracy: 0.6999

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 0.7984 - accuracy: 0.7022

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 0.8017 - accuracy: 0.7013

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 0.8002 - accuracy: 0.7016

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 0.7966 - accuracy: 0.7038

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 0.7944 - accuracy: 0.7047

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 0.7958 - accuracy: 0.7032

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 0.7994 - accuracy: 0.7035

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 0.7985 - accuracy: 0.7026

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 0.7991 - accuracy: 0.7012

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 0.7999 - accuracy: 0.7015

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 0.8023 - accuracy: 0.7008

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 0.8013 - accuracy: 0.7021

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 0.7999 - accuracy: 0.7029

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 0.7979 - accuracy: 0.7032

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 0.8003 - accuracy: 0.7024

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 0.8003 - accuracy: 0.7032

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 0.7988 - accuracy: 0.7034

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 0.8002 - accuracy: 0.7022

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 0.8026 - accuracy: 0.7001

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 0.8021 - accuracy: 0.6990

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 0.8058 - accuracy: 0.6974

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 0.8066 - accuracy: 0.6964

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 0.8068 - accuracy: 0.6958

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 0.8076 - accuracy: 0.6957

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 0.8057 - accuracy: 0.6969

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 0.8034 - accuracy: 0.6980

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 0.8010 - accuracy: 0.6983

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 0.8011 - accuracy: 0.6982

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 0.8012 - accuracy: 0.6984

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 0.8039 - accuracy: 0.6975

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 0.8099 - accuracy: 0.6945

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 0.8089 - accuracy: 0.6940

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 0.8066 - accuracy: 0.6951

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 0.8062 - accuracy: 0.6943

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 0.8053 - accuracy: 0.6950

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 0.8045 - accuracy: 0.6964

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 0.8067 - accuracy: 0.6959

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 0.8066 - accuracy: 0.6954

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 0.8071 - accuracy: 0.6950

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 0.8074 - accuracy: 0.6934

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 0.8056 - accuracy: 0.6941

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 0.8061 - accuracy: 0.6944

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 0.8052 - accuracy: 0.6943

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 0.8042 - accuracy: 0.6942

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 0.8057 - accuracy: 0.6935

.. parsed-literal::

    92/92 [==============================] - 6s 63ms/step - loss: 0.8057 - accuracy: 0.6935 - val_loss: 0.8567 - val_accuracy: 0.6662


.. parsed-literal::

    Epoch 6/15


.. parsed-literal::

     1/92 [..............................] - ETA: 7s - loss: 0.6646 - accuracy: 0.6875

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 0.7891 - accuracy: 0.6875

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 0.7486 - accuracy: 0.6979

.. parsed-literal::

     4/92 [>.............................] - ETA: 4s - loss: 0.7449 - accuracy: 0.7031

.. parsed-literal::

     5/92 [>.............................] - ETA: 4s - loss: 0.7592 - accuracy: 0.7250

.. parsed-literal::

     6/92 [>.............................] - ETA: 4s - loss: 0.7296 - accuracy: 0.7188

.. parsed-literal::

     7/92 [=>............................] - ETA: 4s - loss: 0.8006 - accuracy: 0.6830

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 0.7905 - accuracy: 0.6992

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 0.7754 - accuracy: 0.7083

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 0.7620 - accuracy: 0.7125

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 0.7567 - accuracy: 0.7074

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 0.7540 - accuracy: 0.7005

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 0.7360 - accuracy: 0.6995

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 0.7477 - accuracy: 0.6897

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 0.7623 - accuracy: 0.6854

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 0.7513 - accuracy: 0.6914

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 0.7484 - accuracy: 0.6949

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 0.7497 - accuracy: 0.6979

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 0.7361 - accuracy: 0.7089

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 0.7319 - accuracy: 0.7109

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 0.7328 - accuracy: 0.7113

.. parsed-literal::

    22/92 [======>.......................] - ETA: 4s - loss: 0.7595 - accuracy: 0.6989

.. parsed-literal::

    23/92 [======>.......................] - ETA: 3s - loss: 0.7554 - accuracy: 0.7011

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 0.7617 - accuracy: 0.7018

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 0.7729 - accuracy: 0.6988

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 0.7751 - accuracy: 0.6959

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 0.7807 - accuracy: 0.6968

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 0.7808 - accuracy: 0.6987

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 0.7772 - accuracy: 0.7026

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 0.7707 - accuracy: 0.7042

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 0.7753 - accuracy: 0.7036

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 0.7762 - accuracy: 0.6992

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 0.7814 - accuracy: 0.6989

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 0.7758 - accuracy: 0.7004

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 0.7725 - accuracy: 0.7018

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 0.7748 - accuracy: 0.7005

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 0.7738 - accuracy: 0.7010

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 0.7691 - accuracy: 0.7056

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 0.7687 - accuracy: 0.7059

.. parsed-literal::

    40/92 [============>.................] - ETA: 3s - loss: 0.7676 - accuracy: 0.7070

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 0.7675 - accuracy: 0.7066

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 0.7692 - accuracy: 0.7046

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 0.7653 - accuracy: 0.7071

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 0.7679 - accuracy: 0.7031

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 0.7696 - accuracy: 0.7014

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 0.7640 - accuracy: 0.7045

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 0.7636 - accuracy: 0.7028

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 0.7647 - accuracy: 0.7025

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 0.7679 - accuracy: 0.6985

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 0.7692 - accuracy: 0.6977

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 0.7697 - accuracy: 0.6969

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 0.7739 - accuracy: 0.6979

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 0.7702 - accuracy: 0.6994

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 0.7700 - accuracy: 0.6986

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 0.7676 - accuracy: 0.6990

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 0.7679 - accuracy: 0.6977

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 0.7653 - accuracy: 0.6981

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 0.7646 - accuracy: 0.6973

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 0.7609 - accuracy: 0.7003

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 0.7620 - accuracy: 0.6980

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 0.7582 - accuracy: 0.6989

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 0.7609 - accuracy: 0.6967

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 0.7592 - accuracy: 0.6985

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 0.7584 - accuracy: 0.6984

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 0.7568 - accuracy: 0.6977

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 0.7528 - accuracy: 0.6999

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 0.7546 - accuracy: 0.7011

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 0.7545 - accuracy: 0.6995

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 0.7547 - accuracy: 0.7003

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 0.7597 - accuracy: 0.7001

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 0.7607 - accuracy: 0.6990

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 0.7627 - accuracy: 0.6989

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 0.7640 - accuracy: 0.6970

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 0.7651 - accuracy: 0.6965

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 0.7657 - accuracy: 0.6947

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 0.7662 - accuracy: 0.6946

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 0.7687 - accuracy: 0.6949

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 0.7697 - accuracy: 0.6948

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 0.7707 - accuracy: 0.6951

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 0.7703 - accuracy: 0.6950

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 0.7707 - accuracy: 0.6946

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 0.7693 - accuracy: 0.6956

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 0.7709 - accuracy: 0.6948

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 0.7702 - accuracy: 0.6943

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 0.7713 - accuracy: 0.6942

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 0.7719 - accuracy: 0.6931

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 0.7720 - accuracy: 0.6934

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 0.7702 - accuracy: 0.6947

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 0.7688 - accuracy: 0.6957

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 0.7706 - accuracy: 0.6949

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 0.7720 - accuracy: 0.6948

.. parsed-literal::

    92/92 [==============================] - 6s 64ms/step - loss: 0.7720 - accuracy: 0.6948 - val_loss: 0.8453 - val_accuracy: 0.6744


.. parsed-literal::

    Epoch 7/15


.. parsed-literal::

     1/92 [..............................] - ETA: 7s - loss: 0.7669 - accuracy: 0.6562

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 0.7193 - accuracy: 0.6875

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 0.7496 - accuracy: 0.6979

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 0.7947 - accuracy: 0.6641

.. parsed-literal::

     5/92 [>.............................] - ETA: 5s - loss: 0.7650 - accuracy: 0.6750

.. parsed-literal::

     6/92 [>.............................] - ETA: 4s - loss: 0.7397 - accuracy: 0.6823

.. parsed-literal::

     7/92 [=>............................] - ETA: 4s - loss: 0.7535 - accuracy: 0.6696

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 0.7552 - accuracy: 0.6719

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 0.7539 - accuracy: 0.6736

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 0.7581 - accuracy: 0.6781

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 0.7646 - accuracy: 0.6790

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 0.7760 - accuracy: 0.6771

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 0.7873 - accuracy: 0.6755

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 0.7841 - accuracy: 0.6808

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 0.7791 - accuracy: 0.6875

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 0.7720 - accuracy: 0.6875

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 0.7866 - accuracy: 0.6820

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 0.7791 - accuracy: 0.6892

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 0.7801 - accuracy: 0.6957

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 0.7821 - accuracy: 0.6938

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 0.7791 - accuracy: 0.6964

.. parsed-literal::

    22/92 [======>.......................] - ETA: 4s - loss: 0.7940 - accuracy: 0.6889

.. parsed-literal::

    23/92 [======>.......................] - ETA: 3s - loss: 0.7884 - accuracy: 0.6943

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 0.7883 - accuracy: 0.6914

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 0.7830 - accuracy: 0.6925

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 0.7771 - accuracy: 0.6971

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 0.7746 - accuracy: 0.6968

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 0.7714 - accuracy: 0.6987

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 0.7725 - accuracy: 0.6972

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 0.7714 - accuracy: 0.6969

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 0.7666 - accuracy: 0.7016

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 0.7767 - accuracy: 0.6963

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 0.7865 - accuracy: 0.6903

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 0.7868 - accuracy: 0.6912

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 0.7820 - accuracy: 0.6938

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 0.7848 - accuracy: 0.6927

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 0.7785 - accuracy: 0.6951

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 0.7793 - accuracy: 0.6974

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 0.7771 - accuracy: 0.6987

.. parsed-literal::

    40/92 [============>.................] - ETA: 3s - loss: 0.7768 - accuracy: 0.7000

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 0.7755 - accuracy: 0.6997

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 0.7726 - accuracy: 0.7016

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 0.7698 - accuracy: 0.7020

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 0.7680 - accuracy: 0.7024

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 0.7613 - accuracy: 0.7063

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 0.7570 - accuracy: 0.7079

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 0.7548 - accuracy: 0.7101

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 0.7558 - accuracy: 0.7109

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 0.7526 - accuracy: 0.7124

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 0.7507 - accuracy: 0.7150

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 0.7525 - accuracy: 0.7145

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 0.7513 - accuracy: 0.7151

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 0.7500 - accuracy: 0.7170

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 0.7435 - accuracy: 0.7199

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 0.7411 - accuracy: 0.7210

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 0.7386 - accuracy: 0.7210

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 0.7352 - accuracy: 0.7226

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 0.7344 - accuracy: 0.7236

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 0.7358 - accuracy: 0.7235

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 0.7322 - accuracy: 0.7245

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 0.7298 - accuracy: 0.7254

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 0.7312 - accuracy: 0.7258

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 0.7335 - accuracy: 0.7237

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 0.7316 - accuracy: 0.7230

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 0.7285 - accuracy: 0.7239

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 0.7259 - accuracy: 0.7247

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 0.7256 - accuracy: 0.7246

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 0.7260 - accuracy: 0.7241

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 0.7276 - accuracy: 0.7227

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 0.7256 - accuracy: 0.7235

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 0.7290 - accuracy: 0.7230

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 0.7297 - accuracy: 0.7238

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 0.7291 - accuracy: 0.7242

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 0.7352 - accuracy: 0.7224

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 0.7315 - accuracy: 0.7240

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 0.7301 - accuracy: 0.7235

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 0.7287 - accuracy: 0.7247

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 0.7292 - accuracy: 0.7250

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 0.7283 - accuracy: 0.7253

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 0.7284 - accuracy: 0.7241

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 0.7277 - accuracy: 0.7240

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 0.7286 - accuracy: 0.7228

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 0.7288 - accuracy: 0.7228

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 0.7284 - accuracy: 0.7238

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 0.7283 - accuracy: 0.7245

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 0.7294 - accuracy: 0.7244

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 0.7292 - accuracy: 0.7251

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 0.7322 - accuracy: 0.7236

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 0.7339 - accuracy: 0.7218

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 0.7330 - accuracy: 0.7221

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 0.7323 - accuracy: 0.7228

.. parsed-literal::

    92/92 [==============================] - 6s 63ms/step - loss: 0.7323 - accuracy: 0.7228 - val_loss: 0.7490 - val_accuracy: 0.7071


.. parsed-literal::

    Epoch 8/15


.. parsed-literal::

     1/92 [..............................] - ETA: 7s - loss: 0.8459 - accuracy: 0.6875

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 0.7695 - accuracy: 0.7031

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 0.7538 - accuracy: 0.7083

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 0.7365 - accuracy: 0.7109

.. parsed-literal::

     5/92 [>.............................] - ETA: 5s - loss: 0.7053 - accuracy: 0.7312

.. parsed-literal::

     6/92 [>.............................] - ETA: 4s - loss: 0.6969 - accuracy: 0.7396

.. parsed-literal::

     7/92 [=>............................] - ETA: 4s - loss: 0.6882 - accuracy: 0.7366

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 0.7021 - accuracy: 0.7188

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 0.6909 - accuracy: 0.7292

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 0.6855 - accuracy: 0.7250

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 0.6844 - accuracy: 0.7273

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 0.6779 - accuracy: 0.7266

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 0.6842 - accuracy: 0.7260

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 0.6818 - accuracy: 0.7232

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 0.6826 - accuracy: 0.7271

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 0.6913 - accuracy: 0.7266

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 0.6892 - accuracy: 0.7261

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 0.6767 - accuracy: 0.7326

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 0.6794 - accuracy: 0.7319

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 0.6740 - accuracy: 0.7391

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 0.6798 - accuracy: 0.7411

.. parsed-literal::

    22/92 [======>.......................] - ETA: 4s - loss: 0.6855 - accuracy: 0.7386

.. parsed-literal::

    23/92 [======>.......................] - ETA: 4s - loss: 0.6879 - accuracy: 0.7364

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 0.6867 - accuracy: 0.7383

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 0.6964 - accuracy: 0.7312

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 0.6912 - accuracy: 0.7332

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 0.6927 - accuracy: 0.7338

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 0.6987 - accuracy: 0.7310

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 0.7016 - accuracy: 0.7328

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 0.6953 - accuracy: 0.7365

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 0.6871 - accuracy: 0.7379

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 0.6865 - accuracy: 0.7383

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 0.6805 - accuracy: 0.7405

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 0.6828 - accuracy: 0.7399

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 0.6883 - accuracy: 0.7384

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 0.6963 - accuracy: 0.7361

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 0.6933 - accuracy: 0.7390

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 0.6943 - accuracy: 0.7401

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 0.6995 - accuracy: 0.7372

.. parsed-literal::

    40/92 [============>.................] - ETA: 3s - loss: 0.7004 - accuracy: 0.7367

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 0.6993 - accuracy: 0.7363

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 0.6936 - accuracy: 0.7403

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 0.6869 - accuracy: 0.7442

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 0.6946 - accuracy: 0.7422

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 0.6975 - accuracy: 0.7403

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 0.6949 - accuracy: 0.7412

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 0.6913 - accuracy: 0.7420

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 0.6918 - accuracy: 0.7428

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 0.6916 - accuracy: 0.7423

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 0.6917 - accuracy: 0.7425

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 0.6917 - accuracy: 0.7414

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 0.6882 - accuracy: 0.7422

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 0.6940 - accuracy: 0.7390

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 0.6965 - accuracy: 0.7386

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 0.7000 - accuracy: 0.7371

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 0.6985 - accuracy: 0.7379

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 0.6970 - accuracy: 0.7376

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 0.6945 - accuracy: 0.7383

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 0.6985 - accuracy: 0.7348

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 0.7007 - accuracy: 0.7320

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 0.7032 - accuracy: 0.7303

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 0.7018 - accuracy: 0.7301

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 0.6983 - accuracy: 0.7314

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 0.6980 - accuracy: 0.7317

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 0.6965 - accuracy: 0.7334

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 0.6994 - accuracy: 0.7331

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 0.7010 - accuracy: 0.7329

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 0.6990 - accuracy: 0.7336

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 0.6972 - accuracy: 0.7352

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 0.6974 - accuracy: 0.7359

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 0.6971 - accuracy: 0.7352

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 0.7007 - accuracy: 0.7324

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 0.6982 - accuracy: 0.7347

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 0.6962 - accuracy: 0.7358

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 0.6976 - accuracy: 0.7351

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 0.6977 - accuracy: 0.7349

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 0.7002 - accuracy: 0.7343

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 0.6993 - accuracy: 0.7353

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 0.6976 - accuracy: 0.7359

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 0.6969 - accuracy: 0.7365

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 0.6962 - accuracy: 0.7370

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 0.6948 - accuracy: 0.7375

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 0.6951 - accuracy: 0.7377

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 0.6957 - accuracy: 0.7378

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 0.7002 - accuracy: 0.7354

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 0.6992 - accuracy: 0.7360

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 0.7038 - accuracy: 0.7336

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 0.7037 - accuracy: 0.7338

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 0.7043 - accuracy: 0.7336

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 0.7027 - accuracy: 0.7348

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 0.7030 - accuracy: 0.7343

.. parsed-literal::

    92/92 [==============================] - 6s 63ms/step - loss: 0.7030 - accuracy: 0.7343 - val_loss: 0.7453 - val_accuracy: 0.7071


.. parsed-literal::

    Epoch 9/15


.. parsed-literal::

     1/92 [..............................] - ETA: 6s - loss: 0.7925 - accuracy: 0.8125

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 0.7804 - accuracy: 0.7969

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 0.7230 - accuracy: 0.7708

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 0.7711 - accuracy: 0.7422

.. parsed-literal::

     5/92 [>.............................] - ETA: 5s - loss: 0.8044 - accuracy: 0.7063

.. parsed-literal::

     6/92 [>.............................] - ETA: 4s - loss: 0.8262 - accuracy: 0.6875

.. parsed-literal::

     7/92 [=>............................] - ETA: 4s - loss: 0.7973 - accuracy: 0.6964

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 0.7794 - accuracy: 0.7070

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 0.7521 - accuracy: 0.7118

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 0.7432 - accuracy: 0.7125

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 0.7201 - accuracy: 0.7188

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 0.7189 - accuracy: 0.7188

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 0.7244 - accuracy: 0.7188

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 0.7154 - accuracy: 0.7232

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 0.7102 - accuracy: 0.7271

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 0.7019 - accuracy: 0.7305

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 0.6936 - accuracy: 0.7335

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 0.6915 - accuracy: 0.7326

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 0.6837 - accuracy: 0.7352

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 0.6932 - accuracy: 0.7297

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 0.6994 - accuracy: 0.7292

.. parsed-literal::

    22/92 [======>.......................] - ETA: 4s - loss: 0.6958 - accuracy: 0.7301

.. parsed-literal::

    23/92 [======>.......................] - ETA: 3s - loss: 0.6944 - accuracy: 0.7296

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 0.6933 - accuracy: 0.7305

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 0.6905 - accuracy: 0.7337

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 0.6838 - accuracy: 0.7368

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 0.6805 - accuracy: 0.7396

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 0.6738 - accuracy: 0.7433

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 0.6779 - accuracy: 0.7403

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 0.6692 - accuracy: 0.7448

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 0.6665 - accuracy: 0.7429

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 0.6627 - accuracy: 0.7461

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 0.6586 - accuracy: 0.7491

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 0.6553 - accuracy: 0.7509

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 0.6549 - accuracy: 0.7500

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 0.6543 - accuracy: 0.7517

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 0.6558 - accuracy: 0.7500

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 0.6557 - accuracy: 0.7508

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 0.6540 - accuracy: 0.7508

.. parsed-literal::

    40/92 [============>.................] - ETA: 2s - loss: 0.6494 - accuracy: 0.7531

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 0.6452 - accuracy: 0.7553

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 0.6458 - accuracy: 0.7552

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 0.6420 - accuracy: 0.7558

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 0.6388 - accuracy: 0.7571

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 0.6386 - accuracy: 0.7583

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 0.6401 - accuracy: 0.7561

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 0.6399 - accuracy: 0.7566

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 0.6405 - accuracy: 0.7552

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 0.6386 - accuracy: 0.7551

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 0.6385 - accuracy: 0.7544

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 0.6417 - accuracy: 0.7549

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 0.6373 - accuracy: 0.7578

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 0.6351 - accuracy: 0.7588

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 0.6423 - accuracy: 0.7593

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 0.6442 - accuracy: 0.7585

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 0.6439 - accuracy: 0.7573

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 0.6435 - accuracy: 0.7560

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 0.6443 - accuracy: 0.7564

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 0.6435 - accuracy: 0.7584

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 0.6450 - accuracy: 0.7572

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 0.6483 - accuracy: 0.7561

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 0.6468 - accuracy: 0.7565

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 0.6482 - accuracy: 0.7554

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 0.6472 - accuracy: 0.7568

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 0.6475 - accuracy: 0.7571

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 0.6487 - accuracy: 0.7566

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 0.6487 - accuracy: 0.7569

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 0.6474 - accuracy: 0.7568

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 0.6477 - accuracy: 0.7563

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 0.6498 - accuracy: 0.7544

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 0.6508 - accuracy: 0.7539

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 0.6523 - accuracy: 0.7543

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 0.6537 - accuracy: 0.7530

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 0.6603 - accuracy: 0.7496

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 0.6593 - accuracy: 0.7500

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 0.6612 - accuracy: 0.7488

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 0.6604 - accuracy: 0.7496

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 0.6608 - accuracy: 0.7492

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 0.6606 - accuracy: 0.7496

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 0.6640 - accuracy: 0.7496

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 0.6654 - accuracy: 0.7481

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 0.6631 - accuracy: 0.7492

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 0.6624 - accuracy: 0.7500

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 0.6666 - accuracy: 0.7482

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 0.6677 - accuracy: 0.7474

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 0.6682 - accuracy: 0.7464

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 0.6713 - accuracy: 0.7439

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 0.6720 - accuracy: 0.7433

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 0.6702 - accuracy: 0.7441

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 0.6692 - accuracy: 0.7445

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 0.6697 - accuracy: 0.7435

.. parsed-literal::

    92/92 [==============================] - 6s 63ms/step - loss: 0.6697 - accuracy: 0.7435 - val_loss: 0.7617 - val_accuracy: 0.6812


.. parsed-literal::

    Epoch 10/15


.. parsed-literal::

     1/92 [..............................] - ETA: 7s - loss: 0.7222 - accuracy: 0.7500

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 0.7072 - accuracy: 0.7500

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 0.6980 - accuracy: 0.7292

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 0.6872 - accuracy: 0.7344

.. parsed-literal::

     6/92 [>.............................] - ETA: 4s - loss: 0.6213 - accuracy: 0.7609

.. parsed-literal::

     7/92 [=>............................] - ETA: 4s - loss: 0.6025 - accuracy: 0.7593

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 0.6285 - accuracy: 0.7540

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 0.6155 - accuracy: 0.7643

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 0.5926 - accuracy: 0.7724

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 0.5928 - accuracy: 0.7762

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 0.5926 - accuracy: 0.7739

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 0.5787 - accuracy: 0.7770

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 0.5849 - accuracy: 0.7773

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 0.5866 - accuracy: 0.7775

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 0.5933 - accuracy: 0.7679

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 0.5854 - accuracy: 0.7705

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 0.5866 - accuracy: 0.7694

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 0.5853 - accuracy: 0.7683

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 0.6080 - accuracy: 0.7563

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 0.6081 - accuracy: 0.7575

.. parsed-literal::

    22/92 [======>.......................] - ETA: 3s - loss: 0.6118 - accuracy: 0.7572

.. parsed-literal::

    23/92 [======>.......................] - ETA: 3s - loss: 0.6100 - accuracy: 0.7610

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 0.6306 - accuracy: 0.7566

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 0.6392 - accuracy: 0.7525

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 0.6382 - accuracy: 0.7536

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 0.6359 - accuracy: 0.7558

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 0.6364 - accuracy: 0.7568

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 0.6406 - accuracy: 0.7543

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 0.6472 - accuracy: 0.7542

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 0.6425 - accuracy: 0.7561

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 0.6385 - accuracy: 0.7579

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 0.6375 - accuracy: 0.7567

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 0.6345 - accuracy: 0.7583

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 0.6298 - accuracy: 0.7608

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 0.6286 - accuracy: 0.7622

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 0.6254 - accuracy: 0.7628

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 0.6335 - accuracy: 0.7583

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 0.6414 - accuracy: 0.7524

.. parsed-literal::

    40/92 [============>.................] - ETA: 2s - loss: 0.6414 - accuracy: 0.7508

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 0.6416 - accuracy: 0.7500

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 0.6484 - accuracy: 0.7463

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 0.6465 - accuracy: 0.7478

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 0.6418 - accuracy: 0.7486

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 0.6420 - accuracy: 0.7486

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 0.6367 - accuracy: 0.7500

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 0.6389 - accuracy: 0.7487

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 0.6340 - accuracy: 0.7520

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 0.6360 - accuracy: 0.7506

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 0.6343 - accuracy: 0.7506

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 0.6336 - accuracy: 0.7512

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 0.6344 - accuracy: 0.7518

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 0.6343 - accuracy: 0.7518

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 0.6354 - accuracy: 0.7517

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 0.6344 - accuracy: 0.7534

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 0.6353 - accuracy: 0.7545

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 0.6316 - accuracy: 0.7566

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 0.6294 - accuracy: 0.7576

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 0.6301 - accuracy: 0.7569

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 0.6339 - accuracy: 0.7563

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 0.6358 - accuracy: 0.7536

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 0.6368 - accuracy: 0.7525

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 0.6422 - accuracy: 0.7515

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 0.6413 - accuracy: 0.7520

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 0.6391 - accuracy: 0.7543

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 0.6371 - accuracy: 0.7548

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 0.6401 - accuracy: 0.7537

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 0.6403 - accuracy: 0.7532

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 0.6422 - accuracy: 0.7536

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 0.6414 - accuracy: 0.7540

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 0.6428 - accuracy: 0.7540

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 0.6408 - accuracy: 0.7561

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 0.6416 - accuracy: 0.7543

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 0.6407 - accuracy: 0.7547

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 0.6435 - accuracy: 0.7533

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 0.6461 - accuracy: 0.7521

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 0.6451 - accuracy: 0.7520

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 0.6437 - accuracy: 0.7524

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 0.6447 - accuracy: 0.7520

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 0.6449 - accuracy: 0.7516

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 0.6462 - accuracy: 0.7519

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 0.6453 - accuracy: 0.7511

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 0.6471 - accuracy: 0.7500

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 0.6463 - accuracy: 0.7504

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 0.6447 - accuracy: 0.7515

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 0.6437 - accuracy: 0.7526

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 0.6413 - accuracy: 0.7536

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 0.6436 - accuracy: 0.7536

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 0.6429 - accuracy: 0.7535

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 0.6442 - accuracy: 0.7528

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 0.6423 - accuracy: 0.7531

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 0.6435 - accuracy: 0.7534

.. parsed-literal::

    92/92 [==============================] - 6s 63ms/step - loss: 0.6435 - accuracy: 0.7534 - val_loss: 0.7413 - val_accuracy: 0.7153


.. parsed-literal::

    Epoch 11/15


.. parsed-literal::

     1/92 [..............................] - ETA: 7s - loss: 0.7133 - accuracy: 0.8125

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 0.6647 - accuracy: 0.7969

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 0.6096 - accuracy: 0.8125

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 0.5962 - accuracy: 0.8047

.. parsed-literal::

     5/92 [>.............................] - ETA: 5s - loss: 0.5831 - accuracy: 0.8188

.. parsed-literal::

     6/92 [>.............................] - ETA: 4s - loss: 0.5814 - accuracy: 0.8073

.. parsed-literal::

     7/92 [=>............................] - ETA: 4s - loss: 0.6287 - accuracy: 0.7812

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 0.6704 - accuracy: 0.7500

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 0.6516 - accuracy: 0.7569

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 0.6491 - accuracy: 0.7594

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 0.6339 - accuracy: 0.7670

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 0.6335 - accuracy: 0.7630

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 0.6365 - accuracy: 0.7620

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 0.6253 - accuracy: 0.7723

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 0.6229 - accuracy: 0.7729

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 0.6209 - accuracy: 0.7695

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 0.6167 - accuracy: 0.7739

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 0.6155 - accuracy: 0.7674

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 0.6206 - accuracy: 0.7648

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 0.6196 - accuracy: 0.7641

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 0.6178 - accuracy: 0.7649

.. parsed-literal::

    22/92 [======>.......................] - ETA: 4s - loss: 0.6328 - accuracy: 0.7599

.. parsed-literal::

    23/92 [======>.......................] - ETA: 3s - loss: 0.6299 - accuracy: 0.7609

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 0.6304 - accuracy: 0.7630

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 0.6361 - accuracy: 0.7625

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 0.6328 - accuracy: 0.7620

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 0.6369 - accuracy: 0.7604

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 0.6296 - accuracy: 0.7634

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 0.6318 - accuracy: 0.7629

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 0.6262 - accuracy: 0.7656

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 0.6260 - accuracy: 0.7661

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 0.6294 - accuracy: 0.7617

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 0.6239 - accuracy: 0.7652

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 0.6244 - accuracy: 0.7656

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 0.6160 - accuracy: 0.7679

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 0.6167 - accuracy: 0.7665

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 0.6156 - accuracy: 0.7652

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 0.6140 - accuracy: 0.7656

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 0.6140 - accuracy: 0.7652

.. parsed-literal::

    40/92 [============>.................] - ETA: 3s - loss: 0.6161 - accuracy: 0.7641

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 0.6160 - accuracy: 0.7630

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 0.6129 - accuracy: 0.7649

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 0.6084 - accuracy: 0.7674

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 0.6088 - accuracy: 0.7678

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 0.6031 - accuracy: 0.7708

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 0.6009 - accuracy: 0.7717

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 0.6027 - accuracy: 0.7699

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 0.6026 - accuracy: 0.7682

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 0.5976 - accuracy: 0.7698

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 0.5976 - accuracy: 0.7688

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 0.5955 - accuracy: 0.7702

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 0.5928 - accuracy: 0.7716

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 0.5908 - accuracy: 0.7724

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 0.5944 - accuracy: 0.7720

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 0.5941 - accuracy: 0.7710

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 0.5905 - accuracy: 0.7734

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 0.5973 - accuracy: 0.7708

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 0.5961 - accuracy: 0.7721

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 0.5988 - accuracy: 0.7707

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 0.5961 - accuracy: 0.7724

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 0.5953 - accuracy: 0.7736

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 0.5932 - accuracy: 0.7737

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 0.5910 - accuracy: 0.7748

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 0.5908 - accuracy: 0.7739

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 0.5940 - accuracy: 0.7712

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 0.5948 - accuracy: 0.7704

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 0.5933 - accuracy: 0.7715

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 0.5912 - accuracy: 0.7734

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 0.5903 - accuracy: 0.7736

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 0.5927 - accuracy: 0.7719

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 0.5927 - accuracy: 0.7724

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 0.5908 - accuracy: 0.7734

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 0.5903 - accuracy: 0.7744

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 0.5903 - accuracy: 0.7745

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 0.5921 - accuracy: 0.7742

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 0.5964 - accuracy: 0.7726

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 0.5939 - accuracy: 0.7727

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 0.5946 - accuracy: 0.7720

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 0.5970 - accuracy: 0.7718

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 0.5973 - accuracy: 0.7711

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 0.5981 - accuracy: 0.7697

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 0.5985 - accuracy: 0.7698

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 0.5997 - accuracy: 0.7690

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 0.5978 - accuracy: 0.7706

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 0.5953 - accuracy: 0.7719

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 0.5933 - accuracy: 0.7727

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 0.5930 - accuracy: 0.7735

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 0.5936 - accuracy: 0.7725

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 0.5958 - accuracy: 0.7723

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 0.5983 - accuracy: 0.7713

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 0.5991 - accuracy: 0.7708

.. parsed-literal::

    92/92 [==============================] - 6s 63ms/step - loss: 0.5991 - accuracy: 0.7708 - val_loss: 0.8277 - val_accuracy: 0.6866


.. parsed-literal::

    Epoch 12/15


.. parsed-literal::

     1/92 [..............................] - ETA: 7s - loss: 0.4354 - accuracy: 0.7812

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 0.4969 - accuracy: 0.7969

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 0.5232 - accuracy: 0.7812

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 0.5187 - accuracy: 0.7812

.. parsed-literal::

     5/92 [>.............................] - ETA: 5s - loss: 0.6045 - accuracy: 0.7375

.. parsed-literal::

     6/92 [>.............................] - ETA: 5s - loss: 0.5651 - accuracy: 0.7552

.. parsed-literal::

     7/92 [=>............................] - ETA: 4s - loss: 0.5464 - accuracy: 0.7679

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 0.5520 - accuracy: 0.7734

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 0.5991 - accuracy: 0.7535

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 0.5828 - accuracy: 0.7656

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 0.5897 - accuracy: 0.7670

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 0.5979 - accuracy: 0.7630

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 0.6037 - accuracy: 0.7596

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 0.6092 - accuracy: 0.7478

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 0.6051 - accuracy: 0.7479

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 0.6208 - accuracy: 0.7422

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 0.6264 - accuracy: 0.7390

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 0.6289 - accuracy: 0.7431

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 0.6152 - accuracy: 0.7533

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 0.6083 - accuracy: 0.7578

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 0.5986 - accuracy: 0.7619

.. parsed-literal::

    22/92 [======>.......................] - ETA: 4s - loss: 0.5917 - accuracy: 0.7642

.. parsed-literal::

    23/92 [======>.......................] - ETA: 4s - loss: 0.5912 - accuracy: 0.7649

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 0.5934 - accuracy: 0.7656

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 0.5850 - accuracy: 0.7700

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 0.5864 - accuracy: 0.7692

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 0.5923 - accuracy: 0.7650

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 0.5962 - accuracy: 0.7656

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 0.5917 - accuracy: 0.7683

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 0.5915 - accuracy: 0.7677

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 0.5931 - accuracy: 0.7671

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 0.5898 - accuracy: 0.7676

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 0.5919 - accuracy: 0.7680

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 0.5869 - accuracy: 0.7693

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 0.5852 - accuracy: 0.7696

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 0.5842 - accuracy: 0.7700

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 0.5922 - accuracy: 0.7694

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 0.5868 - accuracy: 0.7730

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 0.5871 - accuracy: 0.7748

.. parsed-literal::

    40/92 [============>.................] - ETA: 3s - loss: 0.5907 - accuracy: 0.7742

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 0.5921 - accuracy: 0.7744

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 0.5917 - accuracy: 0.7753

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 0.5961 - accuracy: 0.7718

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 0.5960 - accuracy: 0.7720

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 0.5996 - accuracy: 0.7701

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 0.5968 - accuracy: 0.7690

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 0.5944 - accuracy: 0.7693

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 0.5879 - accuracy: 0.7728

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 0.5869 - accuracy: 0.7730

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 0.5930 - accuracy: 0.7688

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 0.5938 - accuracy: 0.7678

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 0.5947 - accuracy: 0.7680

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 0.5944 - accuracy: 0.7671

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 0.5940 - accuracy: 0.7679

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 0.5922 - accuracy: 0.7676

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 0.5889 - accuracy: 0.7690

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 0.5896 - accuracy: 0.7697

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 0.5900 - accuracy: 0.7705

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 0.5882 - accuracy: 0.7728

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 0.5874 - accuracy: 0.7729

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 0.5850 - accuracy: 0.7741

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 0.5862 - accuracy: 0.7737

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 0.5836 - accuracy: 0.7748

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 0.5853 - accuracy: 0.7749

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 0.5867 - accuracy: 0.7745

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 0.5865 - accuracy: 0.7741

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 0.5841 - accuracy: 0.7752

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 0.5844 - accuracy: 0.7753

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 0.5848 - accuracy: 0.7749

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 0.5847 - accuracy: 0.7759

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 0.5846 - accuracy: 0.7746

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 0.5830 - accuracy: 0.7760

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 0.5840 - accuracy: 0.7748

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 0.5806 - accuracy: 0.7758

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 0.5782 - accuracy: 0.7767

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 0.5807 - accuracy: 0.7759

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 0.5824 - accuracy: 0.7748

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 0.5803 - accuracy: 0.7756

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 0.5800 - accuracy: 0.7761

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 0.5790 - accuracy: 0.7762

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 0.5779 - accuracy: 0.7762

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 0.5790 - accuracy: 0.7755

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 0.5818 - accuracy: 0.7745

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 0.5812 - accuracy: 0.7742

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 0.5817 - accuracy: 0.7735

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 0.5847 - accuracy: 0.7725

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 0.5849 - accuracy: 0.7726

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 0.5841 - accuracy: 0.7729

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 0.5850 - accuracy: 0.7726

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 0.5835 - accuracy: 0.7731

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 0.5839 - accuracy: 0.7732

.. parsed-literal::

    92/92 [==============================] - 6s 63ms/step - loss: 0.5839 - accuracy: 0.7732 - val_loss: 0.7337 - val_accuracy: 0.7166


.. parsed-literal::

    Epoch 13/15


.. parsed-literal::

     1/92 [..............................] - ETA: 7s - loss: 0.6519 - accuracy: 0.6562

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 0.6340 - accuracy: 0.6719

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 0.6107 - accuracy: 0.7083

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 0.6287 - accuracy: 0.7266

.. parsed-literal::

     5/92 [>.............................] - ETA: 5s - loss: 0.6135 - accuracy: 0.7312

.. parsed-literal::

     6/92 [>.............................] - ETA: 5s - loss: 0.5991 - accuracy: 0.7396

.. parsed-literal::

     7/92 [=>............................] - ETA: 4s - loss: 0.5764 - accuracy: 0.7455

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 0.5902 - accuracy: 0.7461

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 0.5901 - accuracy: 0.7465

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 0.5898 - accuracy: 0.7500

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 0.5906 - accuracy: 0.7500

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 0.5824 - accuracy: 0.7526

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 0.5730 - accuracy: 0.7620

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 0.5869 - accuracy: 0.7567

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 0.5934 - accuracy: 0.7542

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 0.5831 - accuracy: 0.7617

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 0.5717 - accuracy: 0.7665

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 0.5672 - accuracy: 0.7674

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 0.5529 - accuracy: 0.7763

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 0.5385 - accuracy: 0.7859

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 0.5396 - accuracy: 0.7827

.. parsed-literal::

    22/92 [======>.......................] - ETA: 4s - loss: 0.5346 - accuracy: 0.7855

.. parsed-literal::

    23/92 [======>.......................] - ETA: 3s - loss: 0.5340 - accuracy: 0.7853

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 0.5329 - accuracy: 0.7852

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 0.5340 - accuracy: 0.7875

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 0.5460 - accuracy: 0.7812

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 0.5410 - accuracy: 0.7836

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 0.5370 - accuracy: 0.7868

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 0.5391 - accuracy: 0.7856

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 0.5395 - accuracy: 0.7885

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 0.5363 - accuracy: 0.7893

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 0.5329 - accuracy: 0.7900

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 0.5356 - accuracy: 0.7879

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 0.5400 - accuracy: 0.7849

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 0.5468 - accuracy: 0.7857

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 0.5464 - accuracy: 0.7839

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 0.5396 - accuracy: 0.7872

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 0.5511 - accuracy: 0.7829

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 0.5514 - accuracy: 0.7837

.. parsed-literal::

    40/92 [============>.................] - ETA: 2s - loss: 0.5519 - accuracy: 0.7828

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 0.5500 - accuracy: 0.7835

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 0.5489 - accuracy: 0.7850

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 0.5479 - accuracy: 0.7849

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 0.5497 - accuracy: 0.7855

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 0.5485 - accuracy: 0.7868

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 0.5503 - accuracy: 0.7867

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 0.5466 - accuracy: 0.7886

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 0.5533 - accuracy: 0.7859

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 0.5510 - accuracy: 0.7864

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 0.5487 - accuracy: 0.7863

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 0.5475 - accuracy: 0.7874

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 0.5462 - accuracy: 0.7891

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 0.5449 - accuracy: 0.7895

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 0.5477 - accuracy: 0.7877

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 0.5496 - accuracy: 0.7870

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 0.5493 - accuracy: 0.7863

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 0.5511 - accuracy: 0.7852

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 0.5477 - accuracy: 0.7867

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 0.5478 - accuracy: 0.7866

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 0.5461 - accuracy: 0.7881

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 0.5453 - accuracy: 0.7885

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 0.5444 - accuracy: 0.7883

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 0.5475 - accuracy: 0.7877

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 0.5503 - accuracy: 0.7857

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 0.5544 - accuracy: 0.7828

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 0.5553 - accuracy: 0.7809

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 0.5556 - accuracy: 0.7818

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 0.5571 - accuracy: 0.7805

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 0.5535 - accuracy: 0.7827

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 0.5572 - accuracy: 0.7814

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 0.5550 - accuracy: 0.7822

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 0.5571 - accuracy: 0.7814

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 0.5556 - accuracy: 0.7831

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 0.5547 - accuracy: 0.7830

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 0.5544 - accuracy: 0.7834

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 0.5547 - accuracy: 0.7842

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 0.5551 - accuracy: 0.7842

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 0.5564 - accuracy: 0.7841

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 0.5540 - accuracy: 0.7857

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 0.5538 - accuracy: 0.7856

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 0.5568 - accuracy: 0.7856

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 0.5564 - accuracy: 0.7863

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 0.5580 - accuracy: 0.7862

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 0.5571 - accuracy: 0.7865

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 0.5556 - accuracy: 0.7872

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 0.5567 - accuracy: 0.7867

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 0.5556 - accuracy: 0.7874

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 0.5552 - accuracy: 0.7873

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 0.5522 - accuracy: 0.7886

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 0.5550 - accuracy: 0.7868

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 0.5548 - accuracy: 0.7871

.. parsed-literal::

    92/92 [==============================] - 6s 63ms/step - loss: 0.5548 - accuracy: 0.7871 - val_loss: 0.7029 - val_accuracy: 0.7302


.. parsed-literal::

    Epoch 14/15


.. parsed-literal::

     1/92 [..............................] - ETA: 6s - loss: 0.3551 - accuracy: 0.9062

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 0.4279 - accuracy: 0.8906

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 0.5521 - accuracy: 0.7917

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 0.5379 - accuracy: 0.8047

.. parsed-literal::

     5/92 [>.............................] - ETA: 5s - loss: 0.5586 - accuracy: 0.7812

.. parsed-literal::

     6/92 [>.............................] - ETA: 4s - loss: 0.5062 - accuracy: 0.8073

.. parsed-literal::

     7/92 [=>............................] - ETA: 4s - loss: 0.4904 - accuracy: 0.8080

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 0.4652 - accuracy: 0.8164

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 0.4739 - accuracy: 0.8160

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 0.4901 - accuracy: 0.8094

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 0.4914 - accuracy: 0.8097

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 0.5313 - accuracy: 0.7943

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 0.5191 - accuracy: 0.7981

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 0.5216 - accuracy: 0.7902

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 0.5071 - accuracy: 0.7979

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 0.5084 - accuracy: 0.7988

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 0.4927 - accuracy: 0.8046

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 0.4881 - accuracy: 0.8050

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 0.5039 - accuracy: 0.8054

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 0.4955 - accuracy: 0.8102

.. parsed-literal::

    22/92 [======>.......................] - ETA: 3s - loss: 0.4989 - accuracy: 0.8103

.. parsed-literal::

    23/92 [======>.......................] - ETA: 3s - loss: 0.4909 - accuracy: 0.8159

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 0.4926 - accuracy: 0.8184

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 0.4881 - accuracy: 0.8207

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 0.4840 - accuracy: 0.8228

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 0.4849 - accuracy: 0.8201

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 0.4837 - accuracy: 0.8209

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 0.4799 - accuracy: 0.8207

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 0.4800 - accuracy: 0.8193

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 0.4830 - accuracy: 0.8191

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 0.4886 - accuracy: 0.8169

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 0.4851 - accuracy: 0.8206

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 0.4874 - accuracy: 0.8176

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 0.4825 - accuracy: 0.8192

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 0.4786 - accuracy: 0.8208

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 0.4769 - accuracy: 0.8214

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 0.4839 - accuracy: 0.8187

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 0.4829 - accuracy: 0.8194

.. parsed-literal::

    40/92 [============>.................] - ETA: 2s - loss: 0.4789 - accuracy: 0.8208

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 0.4791 - accuracy: 0.8198

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 0.4766 - accuracy: 0.8211

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 0.4824 - accuracy: 0.8209

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 0.4857 - accuracy: 0.8193

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 0.4873 - accuracy: 0.8163

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 0.4842 - accuracy: 0.8176

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 0.4855 - accuracy: 0.8155

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 0.4833 - accuracy: 0.8154

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 0.4843 - accuracy: 0.8147

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 0.4818 - accuracy: 0.8160

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 0.4850 - accuracy: 0.8159

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 0.4850 - accuracy: 0.8164

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 0.4844 - accuracy: 0.8169

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 0.4826 - accuracy: 0.8169

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 0.4846 - accuracy: 0.8168

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 0.4880 - accuracy: 0.8156

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 0.4861 - accuracy: 0.8161

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 0.4897 - accuracy: 0.8139

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 0.4866 - accuracy: 0.8154

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 0.4879 - accuracy: 0.8149

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 0.4861 - accuracy: 0.8158

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 0.4941 - accuracy: 0.8122

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 0.4918 - accuracy: 0.8132

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 0.4925 - accuracy: 0.8127

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 0.4970 - accuracy: 0.8113

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 0.4988 - accuracy: 0.8094

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 0.5017 - accuracy: 0.8066

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 0.5028 - accuracy: 0.8058

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 0.5053 - accuracy: 0.8055

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 0.5053 - accuracy: 0.8051

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 0.5047 - accuracy: 0.8061

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 0.5076 - accuracy: 0.8044

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 0.5081 - accuracy: 0.8054

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 0.5112 - accuracy: 0.8042

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 0.5147 - accuracy: 0.8031

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 0.5146 - accuracy: 0.8040

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 0.5169 - accuracy: 0.8037

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 0.5145 - accuracy: 0.8047

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 0.5139 - accuracy: 0.8044

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 0.5193 - accuracy: 0.8013

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 0.5198 - accuracy: 0.8007

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 0.5197 - accuracy: 0.8024

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 0.5208 - accuracy: 0.8014

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 0.5185 - accuracy: 0.8022

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 0.5217 - accuracy: 0.8009

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 0.5217 - accuracy: 0.8007

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 0.5239 - accuracy: 0.8001

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 0.5255 - accuracy: 0.7988

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 0.5264 - accuracy: 0.7986

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 0.5268 - accuracy: 0.7981

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 0.5279 - accuracy: 0.7965

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 0.5279 - accuracy: 0.7967

.. parsed-literal::

    92/92 [==============================] - 6s 63ms/step - loss: 0.5279 - accuracy: 0.7967 - val_loss: 0.6787 - val_accuracy: 0.7302


.. parsed-literal::

    Epoch 15/15


.. parsed-literal::

     1/92 [..............................] - ETA: 7s - loss: 0.4064 - accuracy: 0.8125

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 0.3927 - accuracy: 0.8438

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 0.3869 - accuracy: 0.8333

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 0.3914 - accuracy: 0.8516

.. parsed-literal::

     5/92 [>.............................] - ETA: 5s - loss: 0.3752 - accuracy: 0.8562

.. parsed-literal::

     6/92 [>.............................] - ETA: 5s - loss: 0.4151 - accuracy: 0.8438

.. parsed-literal::

     7/92 [=>............................] - ETA: 4s - loss: 0.4284 - accuracy: 0.8348

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 0.4371 - accuracy: 0.8398

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 0.4250 - accuracy: 0.8368

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 0.4417 - accuracy: 0.8281

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 0.4399 - accuracy: 0.8324

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 0.4635 - accuracy: 0.8203

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 0.4745 - accuracy: 0.8149

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 0.4967 - accuracy: 0.8103

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 0.5027 - accuracy: 0.8062

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 0.5096 - accuracy: 0.8066

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 0.5084 - accuracy: 0.8070

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 0.5030 - accuracy: 0.8108

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 0.5024 - accuracy: 0.8109

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 0.4945 - accuracy: 0.8156

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 0.5018 - accuracy: 0.8155

.. parsed-literal::

    22/92 [======>.......................] - ETA: 4s - loss: 0.5075 - accuracy: 0.8168

.. parsed-literal::

    23/92 [======>.......................] - ETA: 3s - loss: 0.5105 - accuracy: 0.8166

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 0.5015 - accuracy: 0.8216

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 0.4969 - accuracy: 0.8225

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 0.4904 - accuracy: 0.8245

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 0.4932 - accuracy: 0.8229

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 0.4887 - accuracy: 0.8248

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 0.4833 - accuracy: 0.8276

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 0.4800 - accuracy: 0.8281

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 0.4789 - accuracy: 0.8276

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 0.4765 - accuracy: 0.8291

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 0.4785 - accuracy: 0.8258

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 0.4769 - accuracy: 0.8254

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 0.4802 - accuracy: 0.8250

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 0.4816 - accuracy: 0.8220

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 0.4815 - accuracy: 0.8201

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 0.4800 - accuracy: 0.8207

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 0.4814 - accuracy: 0.8205

.. parsed-literal::

    40/92 [============>.................] - ETA: 2s - loss: 0.4851 - accuracy: 0.8195

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 0.4976 - accuracy: 0.8163

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 0.4946 - accuracy: 0.8162

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 0.4922 - accuracy: 0.8176

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 0.4878 - accuracy: 0.8196

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 0.4879 - accuracy: 0.8188

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 0.4849 - accuracy: 0.8200

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 0.4871 - accuracy: 0.8205

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 0.4858 - accuracy: 0.8203

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 0.4880 - accuracy: 0.8202

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 0.4874 - accuracy: 0.8194

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 0.4886 - accuracy: 0.8199

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 0.4848 - accuracy: 0.8227

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 0.4889 - accuracy: 0.8208

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 0.4899 - accuracy: 0.8200

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 0.4896 - accuracy: 0.8193

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 0.4914 - accuracy: 0.8186

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 0.4907 - accuracy: 0.8185

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 0.4912 - accuracy: 0.8206

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 0.4901 - accuracy: 0.8204

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 0.4882 - accuracy: 0.8210

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 0.4885 - accuracy: 0.8203

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 0.4883 - accuracy: 0.8202

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 0.4859 - accuracy: 0.8216

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 0.4874 - accuracy: 0.8205

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 0.4885 - accuracy: 0.8203

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 0.4864 - accuracy: 0.8207

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 0.4873 - accuracy: 0.8210

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 0.4857 - accuracy: 0.8218

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 0.4841 - accuracy: 0.8226

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 0.4878 - accuracy: 0.8211

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 0.4914 - accuracy: 0.8197

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 0.4896 - accuracy: 0.8209

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 0.4938 - accuracy: 0.8178

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 0.4927 - accuracy: 0.8173

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 0.4936 - accuracy: 0.8164

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 0.4961 - accuracy: 0.8156

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 0.4973 - accuracy: 0.8139

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 0.4977 - accuracy: 0.8139

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 0.4978 - accuracy: 0.8143

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 0.4992 - accuracy: 0.8139

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 0.4991 - accuracy: 0.8146

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 0.4993 - accuracy: 0.8138

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 0.4985 - accuracy: 0.8142

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 0.4998 - accuracy: 0.8131

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 0.5021 - accuracy: 0.8116

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 0.5024 - accuracy: 0.8112

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 0.4999 - accuracy: 0.8123

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 0.5015 - accuracy: 0.8123

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 0.5018 - accuracy: 0.8113

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 0.5015 - accuracy: 0.8113

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 0.5024 - accuracy: 0.8110

.. parsed-literal::

    92/92 [==============================] - 6s 63ms/step - loss: 0.5024 - accuracy: 0.8110 - val_loss: 0.6752 - val_accuracy: 0.7371


Visualize Training Results
--------------------------

`back to top ⬆️ <#Table-of-contents:>`__

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

`back to top ⬆️ <#Table-of-contents:>`__

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

    1/1 [==============================] - 0s 75ms/step


.. parsed-literal::

    This image most likely belongs to sunflowers with a 99.01 percent confidence.


Save the TensorFlow Model
-------------------------

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    #save the trained model - a new folder flower will be created
    #and the file "saved_model.pb" is the pre-trained model
    model_dir = "model"
    saved_model_dir = f"{model_dir}/flower/saved_model"
    model.save(saved_model_dir)


.. parsed-literal::

    2024-01-26 00:43:36.610343: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'random_flip_input' with dtype float and shape [?,180,180,3]
    	 [[{{node random_flip_input}}]]
    2024-01-26 00:43:36.696808: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-01-26 00:43:36.707243: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'random_flip_input' with dtype float and shape [?,180,180,3]
    	 [[{{node random_flip_input}}]]
    2024-01-26 00:43:36.718442: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-01-26 00:43:36.725616: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-01-26 00:43:36.732498: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-01-26 00:43:36.743332: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-01-26 00:43:36.782893: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'sequential_1_input' with dtype float and shape [?,180,180,3]
    	 [[{{node sequential_1_input}}]]


.. parsed-literal::

    2024-01-26 00:43:36.850251: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-01-26 00:43:36.870734: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'sequential_1_input' with dtype float and shape [?,180,180,3]
    	 [[{{node sequential_1_input}}]]
    2024-01-26 00:43:36.909410: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,22,22,64]
    	 [[{{node inputs}}]]
    2024-01-26 00:43:36.935532: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-01-26 00:43:37.008478: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]


.. parsed-literal::

    2024-01-26 00:43:37.153120: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-01-26 00:43:37.291315: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,22,22,64]
    	 [[{{node inputs}}]]
    2024-01-26 00:43:37.326120: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-01-26 00:43:37.353891: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]


.. parsed-literal::

    2024-01-26 00:43:37.400911: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    WARNING:absl:Found untraced functions such as _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op, _update_step_xla while saving (showing 4 of 4). These functions will not be directly callable after loading.


.. parsed-literal::

    INFO:tensorflow:Assets written to: model/flower/saved_model/assets


.. parsed-literal::

    INFO:tensorflow:Assets written to: model/flower/saved_model/assets


Convert the TensorFlow model with OpenVINO Model Conversion API
---------------------------------------------------------------

`back to top ⬆️ <#Table-of-contents:>`__ To convert the model to
OpenVINO IR with ``FP16`` precision, use model conversion Python API.

.. code:: ipython3

    # Convert the model to ir model format and save it.
    ir_model_path = Path("model/flower")
    ir_model_path.mkdir(parents=True, exist_ok=True)
    ir_model = ov.convert_model(saved_model_dir, input=[1,180,180,3])
    ov.save_model(ir_model, ir_model_path / "flower_ir.xml")

Preprocessing Image Function
----------------------------

`back to top ⬆️ <#Table-of-contents:>`__

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

`back to top ⬆️ <#Table-of-contents:>`__

Select inference device
~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

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

`back to top ⬆️ <#Table-of-contents:>`__

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
    This image most likely belongs to dandelion with a 99.78 percent confidence.



.. image:: 301-tensorflow-training-openvino-with-output_files/301-tensorflow-training-openvino-with-output_79_1.png


The Next Steps
--------------

`back to top ⬆️ <#Table-of-contents:>`__

This tutorial showed how to train a TensorFlow model, how to convert
that model to OpenVINO’s IR format, and how to do inference on the
converted model. For faster inference speed, you can quantize the IR
model. To see how to quantize this model with OpenVINO’s `Post-training
Quantization with NNCF
Tool <https://docs.openvino.ai/nightly/basic_quantization_flow.html>`__,
check out the `Post-Training Quantization with TensorFlow Classification
Model <./301-tensorflow-training-openvino-nncf.ipynb>`__ notebook.
