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

    2024-02-10 01:12:04.614496: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-02-10 01:12:04.649325: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


.. parsed-literal::

    2024-02-10 01:12:05.161353: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


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

    2024-02-10 01:12:08.217732: E tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:266] failed call to cuInit: CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: forward compatibility was attempted on non supported HW
    2024-02-10 01:12:08.217763: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:168] retrieving CUDA diagnostic information for host: iotg-dev-workstation-07
    2024-02-10 01:12:08.217767: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:175] hostname: iotg-dev-workstation-07
    2024-02-10 01:12:08.217894: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:199] libcuda reported version is: 470.223.2
    2024-02-10 01:12:08.217909: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:203] kernel reported version is: 470.182.3
    2024-02-10 01:12:08.217913: E tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:312] kernel version 470.182.3 does not match DSO version 470.223.2 -- cannot find working devices in this configuration


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

    2024-02-10 01:12:08.550492: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]
    2024-02-10 01:12:08.550818: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
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

    2024-02-10 01:12:09.380029: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-02-10 01:12:09.380404: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
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

    2024-02-10 01:12:09.568220: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-02-10 01:12:09.568598: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
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

    2024-02-10 01:12:10.342151: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]
    2024-02-10 01:12:10.342455: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]



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

    2024-02-10 01:12:11.537227: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]
    2024-02-10 01:12:11.537529: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]


.. parsed-literal::


 1/92 [..............................] - ETA: 1:26 - loss: 1.6124 - accuracy: 0.2500

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 2.0113 - accuracy: 0.2344

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 2.0181 - accuracy: 0.1979

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 1.9673 - accuracy: 0.1953

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 1.8983 - accuracy: 0.2125

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 1.8645 - accuracy: 0.2240

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 5s - loss: 1.8276 - accuracy: 0.2188

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 5s - loss: 1.8073 - accuracy: 0.2070

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 1.7923 - accuracy: 0.1979

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 1.7763 - accuracy: 0.1969

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 1.7599 - accuracy: 0.1989

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 1.7465 - accuracy: 0.1953

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 1.7323 - accuracy: 0.1995

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 1.7228 - accuracy: 0.1964

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 1.7165 - accuracy: 0.1937

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 1.7071 - accuracy: 0.2012

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 1.6996 - accuracy: 0.2022

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 1.6922 - accuracy: 0.2031

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 1.6856 - accuracy: 0.2039

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 1.6816 - accuracy: 0.2109

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 1.6764 - accuracy: 0.2158

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 1.6696 - accuracy: 0.2188

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 1.6659 - accuracy: 0.2188

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 4s - loss: 1.6625 - accuracy: 0.2279

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 1.6613 - accuracy: 0.2250

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 1.6581 - accuracy: 0.2236

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 1.6524 - accuracy: 0.2280

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 1.6476 - accuracy: 0.2288

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 1.6450 - accuracy: 0.2284

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 1.6415 - accuracy: 0.2292

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 1.6390 - accuracy: 0.2288

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 1.6343 - accuracy: 0.2344

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 1.6313 - accuracy: 0.2358

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 1.6276 - accuracy: 0.2371

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 1.6229 - accuracy: 0.2384

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 1.6208 - accuracy: 0.2378

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 1.6166 - accuracy: 0.2432

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 1.6133 - accuracy: 0.2442

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 1.6112 - accuracy: 0.2420

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 1.6055 - accuracy: 0.2453

.. parsed-literal::

    
41/92 [============>.................] - ETA: 3s - loss: 1.6035 - accuracy: 0.2462

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 1.6018 - accuracy: 0.2448

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 1.5969 - accuracy: 0.2464

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 1.5921 - accuracy: 0.2507

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 1.5882 - accuracy: 0.2569

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 1.5821 - accuracy: 0.2615

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 1.5754 - accuracy: 0.2620

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 1.5700 - accuracy: 0.2637

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 1.5665 - accuracy: 0.2659

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 1.5562 - accuracy: 0.2725

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 1.5479 - accuracy: 0.2757

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 1.5424 - accuracy: 0.2800

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 1.5411 - accuracy: 0.2789

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 1.5413 - accuracy: 0.2807

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 1.5364 - accuracy: 0.2847

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 1.5344 - accuracy: 0.2868

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 1.5273 - accuracy: 0.2906

.. parsed-literal::

    
58/92 [=================>............] - ETA: 2s - loss: 1.5258 - accuracy: 0.2909

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 1.5204 - accuracy: 0.2966

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 1.5154 - accuracy: 0.3016

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 1.5101 - accuracy: 0.3053

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 1.5026 - accuracy: 0.3115

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 1.4993 - accuracy: 0.3125

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 1.4941 - accuracy: 0.3159

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 1.4912 - accuracy: 0.3178

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 1.4849 - accuracy: 0.3205

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 1.4824 - accuracy: 0.3223

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 1.4789 - accuracy: 0.3235

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 1.4753 - accuracy: 0.3252

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 1.4727 - accuracy: 0.3272

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 1.4665 - accuracy: 0.3288

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 1.4654 - accuracy: 0.3307

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 1.4620 - accuracy: 0.3313

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 1.4587 - accuracy: 0.3336

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 1s - loss: 1.4537 - accuracy: 0.3363

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 1.4491 - accuracy: 0.3392

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 1.4422 - accuracy: 0.3421

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 1.4408 - accuracy: 0.3438

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 1.4371 - accuracy: 0.3465

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 1.4337 - accuracy: 0.3484

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 1.4300 - accuracy: 0.3499

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 1.4264 - accuracy: 0.3518

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 1.4237 - accuracy: 0.3524

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 1.4204 - accuracy: 0.3534

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 1.4163 - accuracy: 0.3555

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 1.4132 - accuracy: 0.3586

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 1.4104 - accuracy: 0.3602

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 1.4067 - accuracy: 0.3629

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 1.4013 - accuracy: 0.3655

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 1.3969 - accuracy: 0.3677

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 1.3957 - accuracy: 0.3678

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 1.3942 - accuracy: 0.3682

.. parsed-literal::

    2024-02-10 01:12:17.857731: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [734]
    	 [[{{node Placeholder/_0}}]]
    2024-02-10 01:12:17.858066: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [734]
    	 [[{{node Placeholder/_0}}]]


.. parsed-literal::

    
92/92 [==============================] - 7s 67ms/step - loss: 1.3942 - accuracy: 0.3682 - val_loss: 1.2278 - val_accuracy: 0.4864


.. parsed-literal::

    Epoch 2/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 1.1103 - accuracy: 0.5000

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 1.1211 - accuracy: 0.5781

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 1.1221 - accuracy: 0.5521

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 1.1369 - accuracy: 0.5469

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 1.1210 - accuracy: 0.5312

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 1.1376 - accuracy: 0.5260

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 5s - loss: 1.0852 - accuracy: 0.5402

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 1.0779 - accuracy: 0.5312

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 1.1305 - accuracy: 0.5208

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 1.1464 - accuracy: 0.5094

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 1.1565 - accuracy: 0.5028

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 1.1657 - accuracy: 0.5052

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 1.1414 - accuracy: 0.5114

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 1.1527 - accuracy: 0.5064

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 1.1726 - accuracy: 0.4980

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 1.1768 - accuracy: 0.4907

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 1.1829 - accuracy: 0.4894

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 1.1829 - accuracy: 0.4917

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 1.1813 - accuracy: 0.4968

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 1.1744 - accuracy: 0.5030

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 1.1793 - accuracy: 0.5000

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 1.1772 - accuracy: 0.5014

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 1.1748 - accuracy: 0.5039

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 1.1731 - accuracy: 0.5038

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 1.1667 - accuracy: 0.5109

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 1.1709 - accuracy: 0.5093

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 1.1625 - accuracy: 0.5146

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 1.1613 - accuracy: 0.5152

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 1.1667 - accuracy: 0.5147

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 1.1620 - accuracy: 0.5193

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 1.1589 - accuracy: 0.5177

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 1.1593 - accuracy: 0.5181

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 1.1560 - accuracy: 0.5167

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 1.1544 - accuracy: 0.5162

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 1.1475 - accuracy: 0.5210

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 1.1431 - accuracy: 0.5213

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 1.1386 - accuracy: 0.5265

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 1.1383 - accuracy: 0.5282

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 1.1376 - accuracy: 0.5275

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 1.1386 - accuracy: 0.5268

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 1.1374 - accuracy: 0.5262

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 1.1333 - accuracy: 0.5278

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 1.1271 - accuracy: 0.5293

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 1.1306 - accuracy: 0.5258

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 1.1306 - accuracy: 0.5246

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 1.1274 - accuracy: 0.5261

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 1.1235 - accuracy: 0.5268

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 1.1230 - accuracy: 0.5282

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 1.1240 - accuracy: 0.5283

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 1.1228 - accuracy: 0.5302

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 1.1179 - accuracy: 0.5326

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 1.1113 - accuracy: 0.5367

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 1.1151 - accuracy: 0.5378

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 1.1139 - accuracy: 0.5388

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 1.1117 - accuracy: 0.5392

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 1.1080 - accuracy: 0.5413

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 1.1064 - accuracy: 0.5438

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 1.1016 - accuracy: 0.5452

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 1.1005 - accuracy: 0.5455

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 1.1002 - accuracy: 0.5448

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 1.0986 - accuracy: 0.5461

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 1.0959 - accuracy: 0.5483

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 1.0937 - accuracy: 0.5505

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 1.0917 - accuracy: 0.5502

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 1.0901 - accuracy: 0.5490

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 1.0901 - accuracy: 0.5501

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 1.0897 - accuracy: 0.5517

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 1.0857 - accuracy: 0.5523

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 1.0831 - accuracy: 0.5538

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 1.0809 - accuracy: 0.5548

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 1.0771 - accuracy: 0.5575

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 1.0745 - accuracy: 0.5597

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 1.0762 - accuracy: 0.5593

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 1.0742 - accuracy: 0.5594

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 1.0738 - accuracy: 0.5586

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 1.0748 - accuracy: 0.5574

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 1.0728 - accuracy: 0.5579

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 1.0742 - accuracy: 0.5595

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 1.0765 - accuracy: 0.5588

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 1.0769 - accuracy: 0.5600

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 1.0754 - accuracy: 0.5596

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 1.0743 - accuracy: 0.5597

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 1.0725 - accuracy: 0.5619

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 1.0725 - accuracy: 0.5608

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 1.0737 - accuracy: 0.5601

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 1.0731 - accuracy: 0.5602

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 1.0762 - accuracy: 0.5580

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 1.0749 - accuracy: 0.5588

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 1.0765 - accuracy: 0.5578

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 1.0753 - accuracy: 0.5579

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 1.0742 - accuracy: 0.5589

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 1.0742 - accuracy: 0.5589 - val_loss: 1.0685 - val_accuracy: 0.5627


.. parsed-literal::

    Epoch 3/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.9481 - accuracy: 0.6250

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 1.0400 - accuracy: 0.5312

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 1.0083 - accuracy: 0.5833

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.9945 - accuracy: 0.6016

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.9999 - accuracy: 0.5875

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.9974 - accuracy: 0.5833

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.9819 - accuracy: 0.5982

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.9862 - accuracy: 0.5938

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.9919 - accuracy: 0.6007

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.9875 - accuracy: 0.6062

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.9882 - accuracy: 0.6080

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.9726 - accuracy: 0.6120

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.9736 - accuracy: 0.6034

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.9760 - accuracy: 0.5982

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.9785 - accuracy: 0.5979

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.9527 - accuracy: 0.6172

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.9492 - accuracy: 0.6213

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.9475 - accuracy: 0.6215

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.9402 - accuracy: 0.6234

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.9313 - accuracy: 0.6281

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.9270 - accuracy: 0.6324

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.9411 - accuracy: 0.6264

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.9370 - accuracy: 0.6264

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.9346 - accuracy: 0.6276

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.9300 - accuracy: 0.6300

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.9249 - accuracy: 0.6298

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.9217 - accuracy: 0.6319

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.9224 - accuracy: 0.6283

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.9219 - accuracy: 0.6261

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.9185 - accuracy: 0.6250

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.9241 - accuracy: 0.6220

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.9325 - accuracy: 0.6182

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.9336 - accuracy: 0.6231

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.9345 - accuracy: 0.6241

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.9392 - accuracy: 0.6232

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.9411 - accuracy: 0.6224

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.9436 - accuracy: 0.6225

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.9506 - accuracy: 0.6201

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.9491 - accuracy: 0.6234

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.9481 - accuracy: 0.6227

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.9429 - accuracy: 0.6250

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.9385 - accuracy: 0.6272

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.9361 - accuracy: 0.6279

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.9339 - accuracy: 0.6271

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.9319 - accuracy: 0.6278

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.9336 - accuracy: 0.6277

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.9366 - accuracy: 0.6270

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.9346 - accuracy: 0.6263

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.9313 - accuracy: 0.6276

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.9317 - accuracy: 0.6263

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.9299 - accuracy: 0.6256

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.9252 - accuracy: 0.6274

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.9252 - accuracy: 0.6268

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.9259 - accuracy: 0.6279

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.9261 - accuracy: 0.6273

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.9282 - accuracy: 0.6278

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.9275 - accuracy: 0.6283

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.9368 - accuracy: 0.6245

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.9373 - accuracy: 0.6229

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.9358 - accuracy: 0.6234

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.9381 - accuracy: 0.6224

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.9375 - accuracy: 0.6235

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.9379 - accuracy: 0.6240

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.9418 - accuracy: 0.6230

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.9404 - accuracy: 0.6221

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.9375 - accuracy: 0.6245

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.9395 - accuracy: 0.6227

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.9382 - accuracy: 0.6232

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.9388 - accuracy: 0.6236

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.9366 - accuracy: 0.6246

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.9352 - accuracy: 0.6250

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.9365 - accuracy: 0.6233

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.9340 - accuracy: 0.6250

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.9335 - accuracy: 0.6258

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.9347 - accuracy: 0.6254

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.9330 - accuracy: 0.6262

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.9301 - accuracy: 0.6291

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.9306 - accuracy: 0.6290

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.9272 - accuracy: 0.6313

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.9265 - accuracy: 0.6316

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.9277 - accuracy: 0.6307

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.9297 - accuracy: 0.6295

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.9302 - accuracy: 0.6299

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.9299 - accuracy: 0.6309

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.9320 - accuracy: 0.6301

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.9358 - accuracy: 0.6279

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.9352 - accuracy: 0.6289

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.9329 - accuracy: 0.6292

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.9313 - accuracy: 0.6295

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.9285 - accuracy: 0.6309

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.9316 - accuracy: 0.6291

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.9316 - accuracy: 0.6291 - val_loss: 0.8948 - val_accuracy: 0.6540


.. parsed-literal::

    Epoch 4/15


.. parsed-literal::


 1/92 [..............................] - ETA: 6s - loss: 0.9437 - accuracy: 0.6250

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.9400 - accuracy: 0.6406

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 1.0213 - accuracy: 0.6146

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 1.0144 - accuracy: 0.6016

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.9660 - accuracy: 0.6250

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.9476 - accuracy: 0.6510

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.9413 - accuracy: 0.6518

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.9381 - accuracy: 0.6484

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.9303 - accuracy: 0.6389

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.9133 - accuracy: 0.6469

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.9262 - accuracy: 0.6420

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.9131 - accuracy: 0.6458

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.9015 - accuracy: 0.6538

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.8883 - accuracy: 0.6585

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.8822 - accuracy: 0.6562

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.8838 - accuracy: 0.6602

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.9114 - accuracy: 0.6489

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.9011 - accuracy: 0.6510

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.8960 - accuracy: 0.6530

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.8969 - accuracy: 0.6516

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.8903 - accuracy: 0.6503

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.8889 - accuracy: 0.6477

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.8916 - accuracy: 0.6467

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.8925 - accuracy: 0.6471

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.8842 - accuracy: 0.6513

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.8855 - accuracy: 0.6502

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.8771 - accuracy: 0.6539

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.8767 - accuracy: 0.6518

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.8698 - accuracy: 0.6541

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.8710 - accuracy: 0.6542

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.8748 - accuracy: 0.6532

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.8705 - accuracy: 0.6543

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.8692 - accuracy: 0.6562

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.8665 - accuracy: 0.6599

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.8706 - accuracy: 0.6617

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.8676 - accuracy: 0.6624

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.8652 - accuracy: 0.6623

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.8623 - accuracy: 0.6629

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.8594 - accuracy: 0.6651

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.8566 - accuracy: 0.6672

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.8507 - accuracy: 0.6707

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.8462 - accuracy: 0.6718

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.8465 - accuracy: 0.6693

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.8456 - accuracy: 0.6690

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.8501 - accuracy: 0.6653

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.8534 - accuracy: 0.6651

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.8511 - accuracy: 0.6656

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.8530 - accuracy: 0.6654

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.8499 - accuracy: 0.6658

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.8518 - accuracy: 0.6644

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.8521 - accuracy: 0.6636

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.8538 - accuracy: 0.6629

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.8603 - accuracy: 0.6605

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.8678 - accuracy: 0.6592

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.8674 - accuracy: 0.6592

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.8660 - accuracy: 0.6591

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.8648 - accuracy: 0.6607

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.8631 - accuracy: 0.6617

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.8645 - accuracy: 0.6595

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.8637 - accuracy: 0.6600

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.8635 - accuracy: 0.6594

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.8610 - accuracy: 0.6599

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.8574 - accuracy: 0.6623

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.8597 - accuracy: 0.6612

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.8600 - accuracy: 0.6602

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.8582 - accuracy: 0.6606

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.8573 - accuracy: 0.6601

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.8575 - accuracy: 0.6600

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.8576 - accuracy: 0.6604

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.8571 - accuracy: 0.6617

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.8587 - accuracy: 0.6598

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.8593 - accuracy: 0.6598

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.8549 - accuracy: 0.6610

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.8569 - accuracy: 0.6610

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.8579 - accuracy: 0.6605

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.8619 - accuracy: 0.6584

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.8620 - accuracy: 0.6580

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.8613 - accuracy: 0.6583

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.8616 - accuracy: 0.6591

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.8579 - accuracy: 0.6602

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.8597 - accuracy: 0.6594

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.8589 - accuracy: 0.6597

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.8584 - accuracy: 0.6601

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.8609 - accuracy: 0.6578

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.8626 - accuracy: 0.6567

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.8617 - accuracy: 0.6574

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.8618 - accuracy: 0.6571

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.8622 - accuracy: 0.6553

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.8624 - accuracy: 0.6542

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.8621 - accuracy: 0.6553

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.8626 - accuracy: 0.6550

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.8626 - accuracy: 0.6550 - val_loss: 0.8452 - val_accuracy: 0.6540


.. parsed-literal::

    Epoch 5/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.6427 - accuracy: 0.7188

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.6912 - accuracy: 0.7188

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.7137 - accuracy: 0.6875

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.7362 - accuracy: 0.6875

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.7649 - accuracy: 0.6812

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.8160 - accuracy: 0.6562

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.8130 - accuracy: 0.6562

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.7964 - accuracy: 0.6680

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.8136 - accuracy: 0.6632

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.8277 - accuracy: 0.6562

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.8489 - accuracy: 0.6477

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.8382 - accuracy: 0.6510

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.8242 - accuracy: 0.6611

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.8200 - accuracy: 0.6607

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.8231 - accuracy: 0.6562

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.8219 - accuracy: 0.6602

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.8155 - accuracy: 0.6654

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.8175 - accuracy: 0.6667

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.8136 - accuracy: 0.6727

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.8088 - accuracy: 0.6719

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.7999 - accuracy: 0.6771

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.8036 - accuracy: 0.6747

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.8034 - accuracy: 0.6780

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.8009 - accuracy: 0.6797

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.7954 - accuracy: 0.6837

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.7901 - accuracy: 0.6875

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.7908 - accuracy: 0.6852

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.7907 - accuracy: 0.6864

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.7913 - accuracy: 0.6864

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.7988 - accuracy: 0.6823

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.8004 - accuracy: 0.6804

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.8030 - accuracy: 0.6797

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.7989 - accuracy: 0.6809

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.8025 - accuracy: 0.6792

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.7999 - accuracy: 0.6786

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.7912 - accuracy: 0.6832

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.7929 - accuracy: 0.6833

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.7941 - accuracy: 0.6826

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.7943 - accuracy: 0.6843

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.8040 - accuracy: 0.6828

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.8066 - accuracy: 0.6822

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.8043 - accuracy: 0.6838

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.8004 - accuracy: 0.6860

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.7954 - accuracy: 0.6875

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.7981 - accuracy: 0.6875

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.7944 - accuracy: 0.6889

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.7967 - accuracy: 0.6875

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.7933 - accuracy: 0.6888

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.7877 - accuracy: 0.6913

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.7844 - accuracy: 0.6919

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.7897 - accuracy: 0.6893

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.7869 - accuracy: 0.6902

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.7861 - accuracy: 0.6913

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.7849 - accuracy: 0.6912

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.7852 - accuracy: 0.6917

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.7868 - accuracy: 0.6916

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.7890 - accuracy: 0.6916

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.7928 - accuracy: 0.6910

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.7903 - accuracy: 0.6930

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.7904 - accuracy: 0.6939

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.7863 - accuracy: 0.6969

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.7856 - accuracy: 0.6977

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.7854 - accuracy: 0.6971

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.7943 - accuracy: 0.6940

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.7963 - accuracy: 0.6939

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.8017 - accuracy: 0.6910

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.8011 - accuracy: 0.6923

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.8030 - accuracy: 0.6918

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.8016 - accuracy: 0.6913

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.8007 - accuracy: 0.6917

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.8018 - accuracy: 0.6908

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.8020 - accuracy: 0.6920

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.8055 - accuracy: 0.6903

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.8092 - accuracy: 0.6894

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.8083 - accuracy: 0.6914

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.8063 - accuracy: 0.6930

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.8076 - accuracy: 0.6929

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.8080 - accuracy: 0.6925

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.8081 - accuracy: 0.6932

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.8083 - accuracy: 0.6916

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.8071 - accuracy: 0.6919

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.8083 - accuracy: 0.6915

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.8077 - accuracy: 0.6914

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.8083 - accuracy: 0.6910

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.8087 - accuracy: 0.6902

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.8088 - accuracy: 0.6891

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.8067 - accuracy: 0.6898

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.8064 - accuracy: 0.6887

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.8070 - accuracy: 0.6891

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.8046 - accuracy: 0.6901

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.8027 - accuracy: 0.6911

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.8027 - accuracy: 0.6911 - val_loss: 0.9385 - val_accuracy: 0.6540


.. parsed-literal::

    Epoch 6/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.8309 - accuracy: 0.6875

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.8032 - accuracy: 0.7031

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.8544 - accuracy: 0.6979

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.7712 - accuracy: 0.7422

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.7464 - accuracy: 0.7375

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.7709 - accuracy: 0.7396

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.7587 - accuracy: 0.7411

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.8016 - accuracy: 0.7227

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.7806 - accuracy: 0.7153

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.7750 - accuracy: 0.7188

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.7794 - accuracy: 0.7159

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.7689 - accuracy: 0.7161

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.7657 - accuracy: 0.7115

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.7456 - accuracy: 0.7165

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.7359 - accuracy: 0.7188

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.7275 - accuracy: 0.7207

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.7406 - accuracy: 0.7151

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.7406 - accuracy: 0.7135

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.7385 - accuracy: 0.7138

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.7366 - accuracy: 0.7141

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.7355 - accuracy: 0.7143

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.7315 - accuracy: 0.7145

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.7283 - accuracy: 0.7160

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.7425 - accuracy: 0.7070

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.7453 - accuracy: 0.7088

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.7423 - accuracy: 0.7151

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.7448 - accuracy: 0.7118

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.7458 - accuracy: 0.7121

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.7452 - accuracy: 0.7144

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.7466 - accuracy: 0.7135

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.7409 - accuracy: 0.7157

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.7407 - accuracy: 0.7168

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.7351 - accuracy: 0.7197

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.7291 - accuracy: 0.7215

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.7285 - accuracy: 0.7223

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.7361 - accuracy: 0.7188

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.7333 - accuracy: 0.7204

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.7315 - accuracy: 0.7204

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.7309 - accuracy: 0.7204

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.7319 - accuracy: 0.7211

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.7360 - accuracy: 0.7180

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.7382 - accuracy: 0.7173

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.7401 - accuracy: 0.7151

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.7342 - accuracy: 0.7180

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.7370 - accuracy: 0.7174

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.7370 - accuracy: 0.7174

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.7349 - accuracy: 0.7201

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.7335 - accuracy: 0.7181

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.7344 - accuracy: 0.7162

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.7317 - accuracy: 0.7163

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.7332 - accuracy: 0.7157

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.7387 - accuracy: 0.7133

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.7450 - accuracy: 0.7117

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.7455 - accuracy: 0.7124

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.7468 - accuracy: 0.7114

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.7489 - accuracy: 0.7121

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.7542 - accuracy: 0.7094

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.7542 - accuracy: 0.7090

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.7528 - accuracy: 0.7113

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.7565 - accuracy: 0.7088

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.7541 - accuracy: 0.7110

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.7551 - accuracy: 0.7097

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.7536 - accuracy: 0.7108

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.7494 - accuracy: 0.7128

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.7460 - accuracy: 0.7144

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.7464 - accuracy: 0.7140

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.7455 - accuracy: 0.7145

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.7467 - accuracy: 0.7136

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.7478 - accuracy: 0.7142

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.7466 - accuracy: 0.7155

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.7430 - accuracy: 0.7160

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.7404 - accuracy: 0.7169

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.7433 - accuracy: 0.7148

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.7427 - accuracy: 0.7153

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.7442 - accuracy: 0.7149

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.7437 - accuracy: 0.7146

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.7435 - accuracy: 0.7150

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.7461 - accuracy: 0.7151

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.7502 - accuracy: 0.7143

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.7494 - accuracy: 0.7148

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.7490 - accuracy: 0.7156

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.7489 - accuracy: 0.7156

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.7506 - accuracy: 0.7134

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.7490 - accuracy: 0.7146

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.7500 - accuracy: 0.7143

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.7500 - accuracy: 0.7143

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.7505 - accuracy: 0.7140

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.7555 - accuracy: 0.7116

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.7560 - accuracy: 0.7117

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.7572 - accuracy: 0.7111

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.7571 - accuracy: 0.7115

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.7571 - accuracy: 0.7115 - val_loss: 0.7898 - val_accuracy: 0.6757


.. parsed-literal::

    Epoch 7/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 1.0297 - accuracy: 0.5312

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.9824 - accuracy: 0.6250

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.9507 - accuracy: 0.6250

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.8653 - accuracy: 0.6797

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.8340 - accuracy: 0.6875

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.7951 - accuracy: 0.7135

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 5s - loss: 0.7668 - accuracy: 0.7321

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.7473 - accuracy: 0.7383

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.7561 - accuracy: 0.7257

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.7352 - accuracy: 0.7344

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.7213 - accuracy: 0.7443

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.7234 - accuracy: 0.7396

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.7168 - accuracy: 0.7380

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.7381 - accuracy: 0.7344

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.7442 - accuracy: 0.7292

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.7399 - accuracy: 0.7305

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.7363 - accuracy: 0.7316

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.7337 - accuracy: 0.7292

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.7476 - accuracy: 0.7286

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.7442 - accuracy: 0.7297

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.7557 - accuracy: 0.7217

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.7469 - accuracy: 0.7244

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.7375 - accuracy: 0.7296

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.7313 - accuracy: 0.7318

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.7247 - accuracy: 0.7337

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.7243 - accuracy: 0.7344

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.7181 - accuracy: 0.7350

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.7166 - accuracy: 0.7333

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.7185 - accuracy: 0.7328

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.7122 - accuracy: 0.7354

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.7137 - accuracy: 0.7349

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.7101 - accuracy: 0.7363

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.7042 - accuracy: 0.7377

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.7058 - accuracy: 0.7362

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.7068 - accuracy: 0.7339

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.7114 - accuracy: 0.7318

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.7073 - accuracy: 0.7348

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.7022 - accuracy: 0.7360

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.7105 - accuracy: 0.7340

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.7085 - accuracy: 0.7344

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.7062 - accuracy: 0.7348

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.7022 - accuracy: 0.7359

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.7012 - accuracy: 0.7355

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.7048 - accuracy: 0.7358

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.7104 - accuracy: 0.7312

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.7085 - accuracy: 0.7303

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.7083 - accuracy: 0.7307

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.7098 - accuracy: 0.7298

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.7082 - accuracy: 0.7315

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.7100 - accuracy: 0.7319

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.7118 - accuracy: 0.7292

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.7132 - accuracy: 0.7290

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.7155 - accuracy: 0.7270

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.7165 - accuracy: 0.7274

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.7120 - accuracy: 0.7312

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.7100 - accuracy: 0.7321

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.7080 - accuracy: 0.7330

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.7078 - accuracy: 0.7322

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.7053 - accuracy: 0.7325

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.7040 - accuracy: 0.7323

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.7034 - accuracy: 0.7336

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.7052 - accuracy: 0.7329

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.7045 - accuracy: 0.7331

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.7029 - accuracy: 0.7329

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.7010 - accuracy: 0.7327

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.7028 - accuracy: 0.7320

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6986 - accuracy: 0.7341

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.7025 - accuracy: 0.7335

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.7025 - accuracy: 0.7337

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.7004 - accuracy: 0.7339

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.7004 - accuracy: 0.7337

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.7039 - accuracy: 0.7326

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.7038 - accuracy: 0.7324

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.7075 - accuracy: 0.7314

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.7079 - accuracy: 0.7308

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.7064 - accuracy: 0.7315

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.7045 - accuracy: 0.7321

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.7033 - accuracy: 0.7324

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.7059 - accuracy: 0.7318

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.7084 - accuracy: 0.7305

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.7065 - accuracy: 0.7307

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.7112 - accuracy: 0.7287

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.7109 - accuracy: 0.7282

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.7080 - accuracy: 0.7301

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.7118 - accuracy: 0.7281

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.7095 - accuracy: 0.7287

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.7078 - accuracy: 0.7297

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.7082 - accuracy: 0.7299

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.7092 - accuracy: 0.7295

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.7118 - accuracy: 0.7276

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.7101 - accuracy: 0.7279

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.7101 - accuracy: 0.7279 - val_loss: 0.7492 - val_accuracy: 0.7125


.. parsed-literal::

    Epoch 8/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.4827 - accuracy: 0.8438

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5721 - accuracy: 0.7969

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5558 - accuracy: 0.7812

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5700 - accuracy: 0.7578

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5991 - accuracy: 0.7625

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.6347 - accuracy: 0.7552

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6193 - accuracy: 0.7634

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6489 - accuracy: 0.7461

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6611 - accuracy: 0.7361

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6498 - accuracy: 0.7375

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6517 - accuracy: 0.7330

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6420 - accuracy: 0.7448

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6429 - accuracy: 0.7476

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6445 - accuracy: 0.7455

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6421 - accuracy: 0.7437

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6356 - accuracy: 0.7461

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6272 - accuracy: 0.7500

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6430 - accuracy: 0.7396

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6390 - accuracy: 0.7434

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6434 - accuracy: 0.7422

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6388 - accuracy: 0.7470

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6338 - accuracy: 0.7472

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.6420 - accuracy: 0.7418

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6480 - accuracy: 0.7383

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6484 - accuracy: 0.7425

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6535 - accuracy: 0.7440

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6577 - accuracy: 0.7407

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6681 - accuracy: 0.7388

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6671 - accuracy: 0.7414

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6700 - accuracy: 0.7396

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6742 - accuracy: 0.7389

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6706 - accuracy: 0.7393

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6776 - accuracy: 0.7367

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6775 - accuracy: 0.7362

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6801 - accuracy: 0.7366

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6821 - accuracy: 0.7361

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6873 - accuracy: 0.7356

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6916 - accuracy: 0.7336

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6910 - accuracy: 0.7324

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.6938 - accuracy: 0.7305

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6958 - accuracy: 0.7294

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6957 - accuracy: 0.7307

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6969 - accuracy: 0.7289

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6972 - accuracy: 0.7301

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6984 - accuracy: 0.7292

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6993 - accuracy: 0.7283

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6983 - accuracy: 0.7287

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6976 - accuracy: 0.7298

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6959 - accuracy: 0.7321

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6940 - accuracy: 0.7325

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6960 - accuracy: 0.7298

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6936 - accuracy: 0.7320

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6952 - accuracy: 0.7317

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6960 - accuracy: 0.7321

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6993 - accuracy: 0.7301

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6994 - accuracy: 0.7299

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.7015 - accuracy: 0.7303

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.7040 - accuracy: 0.7279

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.7009 - accuracy: 0.7293

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6984 - accuracy: 0.7318

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6969 - accuracy: 0.7321

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.7041 - accuracy: 0.7308

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.7038 - accuracy: 0.7326

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.7015 - accuracy: 0.7349

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.7055 - accuracy: 0.7327

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.7060 - accuracy: 0.7311

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.7073 - accuracy: 0.7304

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.7051 - accuracy: 0.7316

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.7052 - accuracy: 0.7310

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.7040 - accuracy: 0.7312

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.7014 - accuracy: 0.7315

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6997 - accuracy: 0.7313

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.7002 - accuracy: 0.7320

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.7024 - accuracy: 0.7306

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.7007 - accuracy: 0.7304

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6981 - accuracy: 0.7315

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6973 - accuracy: 0.7309

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6992 - accuracy: 0.7304

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.7035 - accuracy: 0.7282

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.7038 - accuracy: 0.7277

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.7013 - accuracy: 0.7296

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.7029 - accuracy: 0.7287

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.7047 - accuracy: 0.7282

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.7022 - accuracy: 0.7295

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.7017 - accuracy: 0.7298

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.7005 - accuracy: 0.7298

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.7001 - accuracy: 0.7293

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6992 - accuracy: 0.7292

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.7045 - accuracy: 0.7270

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.7020 - accuracy: 0.7280

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.7008 - accuracy: 0.7285

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.7008 - accuracy: 0.7285 - val_loss: 0.7422 - val_accuracy: 0.7139


.. parsed-literal::

    Epoch 9/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.5952 - accuracy: 0.8125

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.6358 - accuracy: 0.7656

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6093 - accuracy: 0.7812

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6300 - accuracy: 0.7812

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.6096 - accuracy: 0.7875

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.6280 - accuracy: 0.7812

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6085 - accuracy: 0.7902

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5959 - accuracy: 0.7930

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6384 - accuracy: 0.7674

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6578 - accuracy: 0.7594

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6381 - accuracy: 0.7557

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6323 - accuracy: 0.7526

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6182 - accuracy: 0.7620

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6125 - accuracy: 0.7679

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6359 - accuracy: 0.7604

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6226 - accuracy: 0.7676

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6165 - accuracy: 0.7665

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6314 - accuracy: 0.7622

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6526 - accuracy: 0.7549

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6501 - accuracy: 0.7578

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6442 - accuracy: 0.7574

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6517 - accuracy: 0.7557

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.6544 - accuracy: 0.7554

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6612 - accuracy: 0.7526

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6606 - accuracy: 0.7538

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6734 - accuracy: 0.7476

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6781 - accuracy: 0.7465

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6773 - accuracy: 0.7478

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6764 - accuracy: 0.7468

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6786 - accuracy: 0.7448

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6748 - accuracy: 0.7470

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6695 - accuracy: 0.7500

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6737 - accuracy: 0.7500

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6742 - accuracy: 0.7528

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6742 - accuracy: 0.7527

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6792 - accuracy: 0.7491

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6797 - accuracy: 0.7492

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6795 - accuracy: 0.7508

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6798 - accuracy: 0.7524

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.6784 - accuracy: 0.7531

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6777 - accuracy: 0.7546

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6793 - accuracy: 0.7537

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6765 - accuracy: 0.7551

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6809 - accuracy: 0.7521

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6783 - accuracy: 0.7528

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6778 - accuracy: 0.7520

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6757 - accuracy: 0.7527

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6733 - accuracy: 0.7552

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6682 - accuracy: 0.7570

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6681 - accuracy: 0.7563

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6729 - accuracy: 0.7543

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6719 - accuracy: 0.7554

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6718 - accuracy: 0.7559

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6663 - accuracy: 0.7581

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6645 - accuracy: 0.7574

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6582 - accuracy: 0.7600

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6585 - accuracy: 0.7599

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6641 - accuracy: 0.7586

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6662 - accuracy: 0.7590

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6648 - accuracy: 0.7599

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6666 - accuracy: 0.7592

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6684 - accuracy: 0.7576

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6665 - accuracy: 0.7594

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6657 - accuracy: 0.7598

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6704 - accuracy: 0.7582

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6705 - accuracy: 0.7576

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6678 - accuracy: 0.7579

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6693 - accuracy: 0.7574

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6673 - accuracy: 0.7586

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6658 - accuracy: 0.7588

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6684 - accuracy: 0.7583

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6677 - accuracy: 0.7582

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6661 - accuracy: 0.7581

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6632 - accuracy: 0.7588

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6618 - accuracy: 0.7591

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6600 - accuracy: 0.7602

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6619 - accuracy: 0.7592

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6630 - accuracy: 0.7583

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6628 - accuracy: 0.7586

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6634 - accuracy: 0.7593

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6638 - accuracy: 0.7592

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6610 - accuracy: 0.7606

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6593 - accuracy: 0.7616

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6583 - accuracy: 0.7622

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6555 - accuracy: 0.7635

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6543 - accuracy: 0.7637

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6534 - accuracy: 0.7639

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6534 - accuracy: 0.7637

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6510 - accuracy: 0.7646

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6516 - accuracy: 0.7634

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6513 - accuracy: 0.7636

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.6513 - accuracy: 0.7636 - val_loss: 0.7100 - val_accuracy: 0.7166


.. parsed-literal::

    Epoch 10/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.6052 - accuracy: 0.7812

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.7084 - accuracy: 0.7344

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.7363 - accuracy: 0.7292

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.7406 - accuracy: 0.7031

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.7162 - accuracy: 0.7125

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.6760 - accuracy: 0.7344

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 5s - loss: 0.6778 - accuracy: 0.7455

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6985 - accuracy: 0.7422

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6704 - accuracy: 0.7500

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6536 - accuracy: 0.7531

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6671 - accuracy: 0.7415

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6608 - accuracy: 0.7500

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6551 - accuracy: 0.7500

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6409 - accuracy: 0.7545

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6365 - accuracy: 0.7563

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6387 - accuracy: 0.7539

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6240 - accuracy: 0.7592

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6206 - accuracy: 0.7604

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6151 - accuracy: 0.7615

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6012 - accuracy: 0.7688

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6084 - accuracy: 0.7604

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6143 - accuracy: 0.7585

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.6111 - accuracy: 0.7609

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6047 - accuracy: 0.7643

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6062 - accuracy: 0.7638

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6128 - accuracy: 0.7620

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6038 - accuracy: 0.7650

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5998 - accuracy: 0.7667

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6027 - accuracy: 0.7672

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5977 - accuracy: 0.7698

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5898 - accuracy: 0.7742

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5843 - accuracy: 0.7764

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5808 - accuracy: 0.7784

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5807 - accuracy: 0.7785

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5837 - accuracy: 0.7777

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5817 - accuracy: 0.7769

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5841 - accuracy: 0.7762

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5874 - accuracy: 0.7747

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5920 - accuracy: 0.7740

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.5974 - accuracy: 0.7711

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5961 - accuracy: 0.7713

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5957 - accuracy: 0.7708

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5937 - accuracy: 0.7718

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5961 - accuracy: 0.7727

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5961 - accuracy: 0.7715

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5962 - accuracy: 0.7724

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5937 - accuracy: 0.7733

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5960 - accuracy: 0.7715

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5972 - accuracy: 0.7723

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5950 - accuracy: 0.7719

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5917 - accuracy: 0.7729

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5941 - accuracy: 0.7731

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5972 - accuracy: 0.7721

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5983 - accuracy: 0.7705

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5951 - accuracy: 0.7724

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5926 - accuracy: 0.7737

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5937 - accuracy: 0.7727

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5921 - accuracy: 0.7729

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5895 - accuracy: 0.7741

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5880 - accuracy: 0.7747

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5898 - accuracy: 0.7743

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5955 - accuracy: 0.7734

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5929 - accuracy: 0.7755

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5930 - accuracy: 0.7761

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5938 - accuracy: 0.7757

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5978 - accuracy: 0.7734

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5989 - accuracy: 0.7721

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5998 - accuracy: 0.7714

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6006 - accuracy: 0.7715

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6049 - accuracy: 0.7694

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6058 - accuracy: 0.7696

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6072 - accuracy: 0.7685

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6073 - accuracy: 0.7686

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6071 - accuracy: 0.7684

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6093 - accuracy: 0.7661

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6097 - accuracy: 0.7667

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6090 - accuracy: 0.7673

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6091 - accuracy: 0.7679

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6084 - accuracy: 0.7680

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6114 - accuracy: 0.7659

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6117 - accuracy: 0.7653

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6108 - accuracy: 0.7655

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6127 - accuracy: 0.7631

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6124 - accuracy: 0.7644

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6118 - accuracy: 0.7638

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6130 - accuracy: 0.7633

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6134 - accuracy: 0.7625

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6146 - accuracy: 0.7616

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6157 - accuracy: 0.7608

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6157 - accuracy: 0.7617

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6144 - accuracy: 0.7616

.. parsed-literal::

    
92/92 [==============================] - 6s 65ms/step - loss: 0.6144 - accuracy: 0.7616 - val_loss: 0.7003 - val_accuracy: 0.7180


.. parsed-literal::

    Epoch 11/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.5609 - accuracy: 0.8438

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.6191 - accuracy: 0.8125

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6127 - accuracy: 0.7917

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5744 - accuracy: 0.8047

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5683 - accuracy: 0.8188

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.5502 - accuracy: 0.8125

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.5450 - accuracy: 0.8170

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5383 - accuracy: 0.8164

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5373 - accuracy: 0.8125

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5662 - accuracy: 0.8000

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5610 - accuracy: 0.8011

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5628 - accuracy: 0.7969

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5678 - accuracy: 0.7885

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5793 - accuracy: 0.7857

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5903 - accuracy: 0.7812

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5735 - accuracy: 0.7871

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5678 - accuracy: 0.7941

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5694 - accuracy: 0.7951

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5696 - accuracy: 0.7911

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5765 - accuracy: 0.7875

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5767 - accuracy: 0.7887

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5683 - accuracy: 0.7912

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.5697 - accuracy: 0.7894

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5762 - accuracy: 0.7852

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5817 - accuracy: 0.7812

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5912 - accuracy: 0.7800

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5881 - accuracy: 0.7836

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5812 - accuracy: 0.7859

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5884 - accuracy: 0.7847

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5870 - accuracy: 0.7856

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5873 - accuracy: 0.7844

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5851 - accuracy: 0.7853

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5834 - accuracy: 0.7833

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5811 - accuracy: 0.7851

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5825 - accuracy: 0.7841

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5817 - accuracy: 0.7840

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5770 - accuracy: 0.7864

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5761 - accuracy: 0.7879

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.5797 - accuracy: 0.7862

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5812 - accuracy: 0.7860

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5845 - accuracy: 0.7837

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5830 - accuracy: 0.7844

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5943 - accuracy: 0.7800

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5970 - accuracy: 0.7800

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6007 - accuracy: 0.7773

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6019 - accuracy: 0.7781

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5988 - accuracy: 0.7801

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6033 - accuracy: 0.7776

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6090 - accuracy: 0.7770

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6071 - accuracy: 0.7777

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6072 - accuracy: 0.7760

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6075 - accuracy: 0.7749

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6071 - accuracy: 0.7750

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6111 - accuracy: 0.7728

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6106 - accuracy: 0.7735

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6089 - accuracy: 0.7742

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6079 - accuracy: 0.7744

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6050 - accuracy: 0.7766

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6046 - accuracy: 0.7762

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6085 - accuracy: 0.7752

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6054 - accuracy: 0.7763

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6062 - accuracy: 0.7754

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6037 - accuracy: 0.7775

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6036 - accuracy: 0.7770

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6023 - accuracy: 0.7776

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6014 - accuracy: 0.7776

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5986 - accuracy: 0.7781

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5991 - accuracy: 0.7773

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5972 - accuracy: 0.7778

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5990 - accuracy: 0.7761

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5993 - accuracy: 0.7761

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5993 - accuracy: 0.7758

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5950 - accuracy: 0.7767

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5950 - accuracy: 0.7755

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5929 - accuracy: 0.7764

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5914 - accuracy: 0.7769

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5930 - accuracy: 0.7765

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5929 - accuracy: 0.7754

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5919 - accuracy: 0.7759

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5910 - accuracy: 0.7759

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5926 - accuracy: 0.7752

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5900 - accuracy: 0.7764

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5929 - accuracy: 0.7754

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5931 - accuracy: 0.7758

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5959 - accuracy: 0.7751

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5965 - accuracy: 0.7749

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5951 - accuracy: 0.7760

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5932 - accuracy: 0.7768

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5922 - accuracy: 0.7775

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5940 - accuracy: 0.7769

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5958 - accuracy: 0.7755

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.5958 - accuracy: 0.7755 - val_loss: 0.6917 - val_accuracy: 0.7343


.. parsed-literal::

    Epoch 12/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.5158 - accuracy: 0.8438

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.4748 - accuracy: 0.8594

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.4990 - accuracy: 0.8542

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5068 - accuracy: 0.8438

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.4829 - accuracy: 0.8500

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.4823 - accuracy: 0.8438

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.4679 - accuracy: 0.8482

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.4979 - accuracy: 0.8320

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5087 - accuracy: 0.8299

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5058 - accuracy: 0.8313

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5191 - accuracy: 0.8210

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5316 - accuracy: 0.8125

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5217 - accuracy: 0.8125

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5414 - accuracy: 0.8036

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5347 - accuracy: 0.8083

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5313 - accuracy: 0.8086

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5244 - accuracy: 0.8088

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5229 - accuracy: 0.8090

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5354 - accuracy: 0.7993

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5348 - accuracy: 0.8000

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5396 - accuracy: 0.7946

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5375 - accuracy: 0.7926

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.5355 - accuracy: 0.7962

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5312 - accuracy: 0.7982

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5287 - accuracy: 0.7975

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5257 - accuracy: 0.7981

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5265 - accuracy: 0.7986

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5345 - accuracy: 0.7980

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5298 - accuracy: 0.7973

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5337 - accuracy: 0.7978

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5332 - accuracy: 0.7963

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5406 - accuracy: 0.7920

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5398 - accuracy: 0.7944

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5319 - accuracy: 0.7986

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5308 - accuracy: 0.7990

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5460 - accuracy: 0.7976

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5438 - accuracy: 0.7972

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5420 - accuracy: 0.7984

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.5416 - accuracy: 0.7980

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5411 - accuracy: 0.7983

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5440 - accuracy: 0.7964

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5429 - accuracy: 0.7982

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5435 - accuracy: 0.7993

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5461 - accuracy: 0.7982

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5474 - accuracy: 0.7992

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5516 - accuracy: 0.7981

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5529 - accuracy: 0.7971

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5537 - accuracy: 0.7962

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5525 - accuracy: 0.7971

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5512 - accuracy: 0.7962

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5524 - accuracy: 0.7947

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5529 - accuracy: 0.7956

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5568 - accuracy: 0.7924

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5531 - accuracy: 0.7945

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5524 - accuracy: 0.7943

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5604 - accuracy: 0.7902

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5582 - accuracy: 0.7911

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5608 - accuracy: 0.7915

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5603 - accuracy: 0.7924

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5640 - accuracy: 0.7912

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5657 - accuracy: 0.7900

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5675 - accuracy: 0.7893

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5684 - accuracy: 0.7892

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5670 - accuracy: 0.7891

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5671 - accuracy: 0.7880

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5679 - accuracy: 0.7870

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5679 - accuracy: 0.7874

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5674 - accuracy: 0.7868

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5682 - accuracy: 0.7872

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5684 - accuracy: 0.7862

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5679 - accuracy: 0.7861

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5695 - accuracy: 0.7848

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5699 - accuracy: 0.7852

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5709 - accuracy: 0.7834

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5713 - accuracy: 0.7826

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5702 - accuracy: 0.7834

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5696 - accuracy: 0.7838

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5722 - accuracy: 0.7833

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5749 - accuracy: 0.7817

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5762 - accuracy: 0.7817

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5755 - accuracy: 0.7829

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5726 - accuracy: 0.7847

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5699 - accuracy: 0.7854

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5690 - accuracy: 0.7850

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5712 - accuracy: 0.7839

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5746 - accuracy: 0.7828

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5740 - accuracy: 0.7828

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5743 - accuracy: 0.7827

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5750 - accuracy: 0.7824

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5774 - accuracy: 0.7820

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5758 - accuracy: 0.7827

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.5758 - accuracy: 0.7827 - val_loss: 0.6737 - val_accuracy: 0.7425


.. parsed-literal::

    Epoch 13/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.4396 - accuracy: 0.8125

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.4462 - accuracy: 0.8125

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.4262 - accuracy: 0.8229

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.4356 - accuracy: 0.8203

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.4686 - accuracy: 0.8062

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.4781 - accuracy: 0.8021

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.4812 - accuracy: 0.8036

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.4844 - accuracy: 0.8047

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5125 - accuracy: 0.7917

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5056 - accuracy: 0.7906

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5237 - accuracy: 0.7869

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5073 - accuracy: 0.7995

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5007 - accuracy: 0.8029

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5219 - accuracy: 0.7902

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5231 - accuracy: 0.7958

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5311 - accuracy: 0.7988

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5357 - accuracy: 0.7960

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5235 - accuracy: 0.8003

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5218 - accuracy: 0.7991

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5102 - accuracy: 0.8042

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5218 - accuracy: 0.8003

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.5211 - accuracy: 0.8022

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5188 - accuracy: 0.8039

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5143 - accuracy: 0.8068

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5156 - accuracy: 0.8070

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5277 - accuracy: 0.8002

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5307 - accuracy: 0.7984

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5394 - accuracy: 0.7957

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5341 - accuracy: 0.7983

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5317 - accuracy: 0.7978

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5310 - accuracy: 0.7972

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5332 - accuracy: 0.7968

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5486 - accuracy: 0.7880

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5522 - accuracy: 0.7896

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5504 - accuracy: 0.7920

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5485 - accuracy: 0.7951

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5474 - accuracy: 0.7955

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5452 - accuracy: 0.7976

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.5448 - accuracy: 0.7980

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5430 - accuracy: 0.7991

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5441 - accuracy: 0.7957

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5452 - accuracy: 0.7939

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5462 - accuracy: 0.7936

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5492 - accuracy: 0.7919

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5471 - accuracy: 0.7923

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5492 - accuracy: 0.7914

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5489 - accuracy: 0.7906

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5501 - accuracy: 0.7904

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5509 - accuracy: 0.7883

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5542 - accuracy: 0.7869

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5537 - accuracy: 0.7874

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5512 - accuracy: 0.7885

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5559 - accuracy: 0.7878

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5574 - accuracy: 0.7882

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5546 - accuracy: 0.7898

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5586 - accuracy: 0.7880

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5582 - accuracy: 0.7873

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5607 - accuracy: 0.7862

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5620 - accuracy: 0.7861

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5641 - accuracy: 0.7845

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5626 - accuracy: 0.7854

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5606 - accuracy: 0.7874

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5580 - accuracy: 0.7877

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5549 - accuracy: 0.7891

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5561 - accuracy: 0.7866

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5561 - accuracy: 0.7865

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5565 - accuracy: 0.7864

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5549 - accuracy: 0.7882

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5530 - accuracy: 0.7890

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5503 - accuracy: 0.7902

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5483 - accuracy: 0.7914

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5492 - accuracy: 0.7912

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5528 - accuracy: 0.7898

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5583 - accuracy: 0.7897

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5578 - accuracy: 0.7896

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5585 - accuracy: 0.7895

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5557 - accuracy: 0.7910

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5546 - accuracy: 0.7913

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5538 - accuracy: 0.7915

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5527 - accuracy: 0.7918

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5529 - accuracy: 0.7913

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5508 - accuracy: 0.7923

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5517 - accuracy: 0.7918

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5550 - accuracy: 0.7917

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5556 - accuracy: 0.7912

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5538 - accuracy: 0.7921

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5527 - accuracy: 0.7931

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5515 - accuracy: 0.7937

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5509 - accuracy: 0.7942

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5499 - accuracy: 0.7948

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5491 - accuracy: 0.7939

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.5491 - accuracy: 0.7939 - val_loss: 0.6629 - val_accuracy: 0.7493


.. parsed-literal::

    Epoch 14/15


.. parsed-literal::


 1/92 [..............................] - ETA: 8s - loss: 0.4716 - accuracy: 0.7812

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5226 - accuracy: 0.8125

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5847 - accuracy: 0.7604

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5652 - accuracy: 0.7578

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5379 - accuracy: 0.7688

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.5409 - accuracy: 0.7708

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.5271 - accuracy: 0.7902

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5225 - accuracy: 0.7930

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.4996 - accuracy: 0.8056

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5121 - accuracy: 0.7969

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5245 - accuracy: 0.7926

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5377 - accuracy: 0.7812

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5415 - accuracy: 0.7788

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5262 - accuracy: 0.7835

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5100 - accuracy: 0.7917

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5001 - accuracy: 0.7988

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.4880 - accuracy: 0.8088

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.4818 - accuracy: 0.8108

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5005 - accuracy: 0.8043

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5034 - accuracy: 0.8000

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5092 - accuracy: 0.7976

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5218 - accuracy: 0.7940

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.5222 - accuracy: 0.7908

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5202 - accuracy: 0.7930

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5256 - accuracy: 0.7925

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5329 - accuracy: 0.7921

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5374 - accuracy: 0.7917

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5278 - accuracy: 0.7969

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5416 - accuracy: 0.7931

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5388 - accuracy: 0.7948

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5455 - accuracy: 0.7913

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5467 - accuracy: 0.7900

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5513 - accuracy: 0.7888

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5546 - accuracy: 0.7886

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5517 - accuracy: 0.7902

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5480 - accuracy: 0.7908

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5459 - accuracy: 0.7922

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5450 - accuracy: 0.7936

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5391 - accuracy: 0.7965

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.5383 - accuracy: 0.7961

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5348 - accuracy: 0.7973

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5337 - accuracy: 0.7976

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5316 - accuracy: 0.7994

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5287 - accuracy: 0.7997

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5271 - accuracy: 0.8028

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5250 - accuracy: 0.8030

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5255 - accuracy: 0.8039

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5265 - accuracy: 0.8034

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5259 - accuracy: 0.8029

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5263 - accuracy: 0.8031

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5260 - accuracy: 0.8027

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5245 - accuracy: 0.8041

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5264 - accuracy: 0.8042

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5256 - accuracy: 0.8050

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5227 - accuracy: 0.8057

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5253 - accuracy: 0.8041

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5265 - accuracy: 0.8026

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5269 - accuracy: 0.8028

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5235 - accuracy: 0.8040

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5222 - accuracy: 0.8031

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5280 - accuracy: 0.8007

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5321 - accuracy: 0.7984

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5292 - accuracy: 0.7991

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5326 - accuracy: 0.7988

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5359 - accuracy: 0.7971

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5367 - accuracy: 0.7968

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5342 - accuracy: 0.7980

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5342 - accuracy: 0.7977

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5368 - accuracy: 0.7975

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5370 - accuracy: 0.7977

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5361 - accuracy: 0.7979

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5350 - accuracy: 0.7990

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5390 - accuracy: 0.7979

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5402 - accuracy: 0.7972

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5415 - accuracy: 0.7966

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5416 - accuracy: 0.7964

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5412 - accuracy: 0.7962

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5435 - accuracy: 0.7948

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5452 - accuracy: 0.7935

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5493 - accuracy: 0.7918

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5472 - accuracy: 0.7928

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5474 - accuracy: 0.7923

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5458 - accuracy: 0.7933

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5452 - accuracy: 0.7931

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5448 - accuracy: 0.7930

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5434 - accuracy: 0.7939

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5425 - accuracy: 0.7945

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5434 - accuracy: 0.7937

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5470 - accuracy: 0.7925

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5459 - accuracy: 0.7941

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5434 - accuracy: 0.7950

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.5434 - accuracy: 0.7950 - val_loss: 0.7003 - val_accuracy: 0.7357


.. parsed-literal::

    Epoch 15/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.3981 - accuracy: 0.8750

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.4149 - accuracy: 0.8438

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5013 - accuracy: 0.8125

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.4675 - accuracy: 0.8281

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.4681 - accuracy: 0.8188

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.4709 - accuracy: 0.8229

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.4701 - accuracy: 0.8214

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.4571 - accuracy: 0.8281

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.4488 - accuracy: 0.8368

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.4419 - accuracy: 0.8406

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.4333 - accuracy: 0.8438

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.4277 - accuracy: 0.8464

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.4366 - accuracy: 0.8462

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.4465 - accuracy: 0.8393

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.4502 - accuracy: 0.8354

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.4574 - accuracy: 0.8301

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.4492 - accuracy: 0.8364

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.4515 - accuracy: 0.8299

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.4427 - accuracy: 0.8306

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.4537 - accuracy: 0.8266

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.4501 - accuracy: 0.8274

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.4552 - accuracy: 0.8253

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.4613 - accuracy: 0.8207

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.4597 - accuracy: 0.8229

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.4571 - accuracy: 0.8250

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.4539 - accuracy: 0.8269

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.4631 - accuracy: 0.8252

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.4597 - accuracy: 0.8270

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.4645 - accuracy: 0.8244

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.4748 - accuracy: 0.8229

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.4794 - accuracy: 0.8206

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.4802 - accuracy: 0.8206

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.4806 - accuracy: 0.8204

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.4783 - accuracy: 0.8210

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.4816 - accuracy: 0.8199

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.4810 - accuracy: 0.8189

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.4834 - accuracy: 0.8179

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.4872 - accuracy: 0.8177

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.4897 - accuracy: 0.8160

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.4889 - accuracy: 0.8160

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.4955 - accuracy: 0.8129

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5023 - accuracy: 0.8099

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5021 - accuracy: 0.8107

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5024 - accuracy: 0.8101

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5017 - accuracy: 0.8101

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5044 - accuracy: 0.8102

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5106 - accuracy: 0.8069

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5094 - accuracy: 0.8064

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5061 - accuracy: 0.8090

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5033 - accuracy: 0.8103

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5022 - accuracy: 0.8116

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5020 - accuracy: 0.8116

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5025 - accuracy: 0.8105

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5012 - accuracy: 0.8122

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5013 - accuracy: 0.8122

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5014 - accuracy: 0.8128

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.4985 - accuracy: 0.8144

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.4989 - accuracy: 0.8138

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5007 - accuracy: 0.8117

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5011 - accuracy: 0.8107

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.4986 - accuracy: 0.8122

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.4991 - accuracy: 0.8123

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.4975 - accuracy: 0.8137

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.4949 - accuracy: 0.8156

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.4949 - accuracy: 0.8156

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.4917 - accuracy: 0.8165

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.4898 - accuracy: 0.8178

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.4893 - accuracy: 0.8177

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.4878 - accuracy: 0.8185

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.4858 - accuracy: 0.8193

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.4848 - accuracy: 0.8193

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.4851 - accuracy: 0.8183

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.4869 - accuracy: 0.8174

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.4863 - accuracy: 0.8165

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.4874 - accuracy: 0.8156

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.4895 - accuracy: 0.8156

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.4899 - accuracy: 0.8159

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.4896 - accuracy: 0.8159

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.4900 - accuracy: 0.8154

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.4929 - accuracy: 0.8135

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.4921 - accuracy: 0.8131

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.4925 - accuracy: 0.8134

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.4912 - accuracy: 0.8131

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.4906 - accuracy: 0.8142

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.4939 - accuracy: 0.8120

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.4967 - accuracy: 0.8102

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.4958 - accuracy: 0.8109

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.4973 - accuracy: 0.8095

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5005 - accuracy: 0.8085

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5006 - accuracy: 0.8089

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5029 - accuracy: 0.8079

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.5029 - accuracy: 0.8079 - val_loss: 0.6990 - val_accuracy: 0.7411


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

   **NOTE**: Data augmentation and Dropout layers are inactive at
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
    1/1 [==============================] - 0s 83ms/step


.. parsed-literal::

    This image most likely belongs to sunflowers with a 99.80 percent confidence.


Save the TensorFlow Model
-------------------------



.. code:: ipython3

    #save the trained model - a new folder flower will be created
    #and the file "saved_model.pb" is the pre-trained model
    model_dir = "model"
    saved_model_dir = f"{model_dir}/flower/saved_model"
    model.save(saved_model_dir)


.. parsed-literal::

    2024-02-10 01:13:41.810110: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'random_flip_input' with dtype float and shape [?,180,180,3]
    	 [[{{node random_flip_input}}]]
    2024-02-10 01:13:41.894772: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-02-10 01:13:41.904859: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'random_flip_input' with dtype float and shape [?,180,180,3]
    	 [[{{node random_flip_input}}]]
    2024-02-10 01:13:41.916513: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-02-10 01:13:41.923535: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-02-10 01:13:41.930525: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-02-10 01:13:41.941500: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-02-10 01:13:41.979801: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'sequential_1_input' with dtype float and shape [?,180,180,3]
    	 [[{{node sequential_1_input}}]]


.. parsed-literal::

    2024-02-10 01:13:42.047000: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-02-10 01:13:42.067363: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'sequential_1_input' with dtype float and shape [?,180,180,3]
    	 [[{{node sequential_1_input}}]]
    2024-02-10 01:13:42.106717: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,22,22,64]
    	 [[{{node inputs}}]]
    2024-02-10 01:13:42.133121: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-02-10 01:13:42.206690: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]


.. parsed-literal::

    2024-02-10 01:13:42.348773: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-02-10 01:13:42.485746: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,22,22,64]
    	 [[{{node inputs}}]]
    2024-02-10 01:13:42.519497: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-02-10 01:13:42.547013: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]


.. parsed-literal::

    2024-02-10 01:13:42.593572: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
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
    This image most likely belongs to dandelion with a 98.49 percent confidence.



.. image:: 301-tensorflow-training-openvino-with-output_files/301-tensorflow-training-openvino-with-output_79_1.png


The Next Steps
--------------



This tutorial showed how to train a TensorFlow model, how to convert
that model to OpenVINO’s IR format, and how to do inference on the
converted model. For faster inference speed, you can quantize the IR
model. To see how to quantize this model with OpenVINO’s `Post-training
Quantization with NNCF
Tool <https://docs.openvino.ai/nightly/basic_quantization_flow.html>`__,
check out the `Post-Training Quantization with TensorFlow Classification
Model <301-tensorflow-training-openvino-nncf-with-output.html>`__ notebook.
