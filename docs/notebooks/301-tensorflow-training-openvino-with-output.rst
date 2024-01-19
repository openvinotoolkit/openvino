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
-  `Load Using keras.preprocessing <#load-using-keraspreprocessing>`__
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

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.0 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063


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

    2024-01-19 00:31:25.474458: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-01-19 00:31:25.508619: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


.. parsed-literal::

    2024-01-19 00:31:26.021174: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


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

    2024-01-19 00:31:29.094309: E tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:266] failed call to cuInit: CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: forward compatibility was attempted on non supported HW
    2024-01-19 00:31:29.094344: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:168] retrieving CUDA diagnostic information for host: iotg-dev-workstation-07
    2024-01-19 00:31:29.094349: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:175] hostname: iotg-dev-workstation-07
    2024-01-19 00:31:29.094479: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:199] libcuda reported version is: 470.223.2
    2024-01-19 00:31:29.094495: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:203] kernel reported version is: 470.182.3
    2024-01-19 00:31:29.094499: E tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:312] kernel version 470.182.3 does not match DSO version 470.223.2 -- cannot find working devices in this configuration


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

    2024-01-19 00:31:29.421091: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-01-19 00:31:29.421622: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
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

    2024-01-19 00:31:30.247779: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-01-19 00:31:30.248148: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
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

    2024-01-19 00:31:30.455505: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-01-19 00:31:30.455822: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]


.. parsed-literal::

    0.02886732 1.0


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

    2024-01-19 00:31:31.326804: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-01-19 00:31:31.327120: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
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

    2024-01-19 00:31:32.297423: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]
    2024-01-19 00:31:32.297728: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]


.. parsed-literal::


 1/92 [..............................] - ETA: 1:26 - loss: 1.5970 - accuracy: 0.2188

.. parsed-literal::

    
 2/92 [..............................] - ETA: 6s - loss: 2.3163 - accuracy: 0.2344

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 2.4379 - accuracy: 0.2188

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 2.3180 - accuracy: 0.1953

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 2.2005 - accuracy: 0.1875

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 5s - loss: 2.0541 - accuracy: 0.1991

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 1.9960 - accuracy: 0.2056

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 1.9499 - accuracy: 0.2214

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 1.9181 - accuracy: 0.2212

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 1.8864 - accuracy: 0.2297

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 1.8611 - accuracy: 0.2261

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 1.8390 - accuracy: 0.2255

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 1.8206 - accuracy: 0.2182

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 1.8073 - accuracy: 0.2076

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 1.7837 - accuracy: 0.2282

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 1.7701 - accuracy: 0.2276

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 1.7552 - accuracy: 0.2342

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 1.7367 - accuracy: 0.2367

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 1.7223 - accuracy: 0.2373

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 1.7125 - accuracy: 0.2395

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 1.7010 - accuracy: 0.2414

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 1.6882 - accuracy: 0.2514

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 1.6802 - accuracy: 0.2513

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 1.6683 - accuracy: 0.2576

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 1.6512 - accuracy: 0.2658

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 1.6388 - accuracy: 0.2699

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 1.6304 - accuracy: 0.2770

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 1.6163 - accuracy: 0.2902

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 1.6073 - accuracy: 0.2994

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 1.5956 - accuracy: 0.3100

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 1.5826 - accuracy: 0.3159

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 1.5711 - accuracy: 0.3177

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 1.5667 - accuracy: 0.3185

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 1.5577 - accuracy: 0.3255

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 1.5473 - accuracy: 0.3313

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 1.5382 - accuracy: 0.3359

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 1.5378 - accuracy: 0.3394

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 1.5288 - accuracy: 0.3419

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 1.5189 - accuracy: 0.3428

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 1.5134 - accuracy: 0.3459

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 1.5092 - accuracy: 0.3473

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 1.5071 - accuracy: 0.3516

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 1.5047 - accuracy: 0.3536

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 1.5045 - accuracy: 0.3527

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 1.4962 - accuracy: 0.3552

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 1.4918 - accuracy: 0.3583

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 1.4870 - accuracy: 0.3599

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 1.4799 - accuracy: 0.3603

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 1.4739 - accuracy: 0.3612

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 1.4641 - accuracy: 0.3688

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 1.4618 - accuracy: 0.3690

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 1.4536 - accuracy: 0.3720

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 1.4484 - accuracy: 0.3750

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 1.4489 - accuracy: 0.3767

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 1.4437 - accuracy: 0.3784

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 1.4386 - accuracy: 0.3789

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 1.4394 - accuracy: 0.3804

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 1.4311 - accuracy: 0.3840

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 1.4262 - accuracy: 0.3865

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 1.4228 - accuracy: 0.3909

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 1.4170 - accuracy: 0.3932

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 1.4154 - accuracy: 0.3949

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 1.4109 - accuracy: 0.3961

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 1.4127 - accuracy: 0.3972

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 1.4077 - accuracy: 0.4002

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 1.4049 - accuracy: 0.4003

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 1.4034 - accuracy: 0.4004

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 1.4022 - accuracy: 0.4009

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 1.3982 - accuracy: 0.4005

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 1.3929 - accuracy: 0.4042

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 1.3885 - accuracy: 0.4051

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 1.3876 - accuracy: 0.4046

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 1.3868 - accuracy: 0.4038

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 1.3818 - accuracy: 0.4076

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 1.3777 - accuracy: 0.4088

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 1.3754 - accuracy: 0.4104

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 1.3728 - accuracy: 0.4124

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 1.3711 - accuracy: 0.4131

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 1.3671 - accuracy: 0.4146

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 1.3637 - accuracy: 0.4160

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 1.3599 - accuracy: 0.4182

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 1.3606 - accuracy: 0.4192

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 1.3587 - accuracy: 0.4201

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 1.3553 - accuracy: 0.4229

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 1.3533 - accuracy: 0.4242

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 1.3499 - accuracy: 0.4251

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 1.3476 - accuracy: 0.4256

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 1.3466 - accuracy: 0.4264

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 1.3463 - accuracy: 0.4269

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 1.3429 - accuracy: 0.4291

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 1.3392 - accuracy: 0.4302

.. parsed-literal::

    2024-01-19 00:31:38.507601: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [734]
    	 [[{{node Placeholder/_0}}]]
    2024-01-19 00:31:38.507874: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [734]
    	 [[{{node Placeholder/_4}}]]


.. parsed-literal::

    
92/92 [==============================] - 7s 65ms/step - loss: 1.3392 - accuracy: 0.4302 - val_loss: 1.1492 - val_accuracy: 0.5368


.. parsed-literal::

    Epoch 2/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 1.1790 - accuracy: 0.5000

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 1.0406 - accuracy: 0.5781

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 1.0271 - accuracy: 0.5625

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 1.0462 - accuracy: 0.5469

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 1.0797 - accuracy: 0.5125

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 1.0577 - accuracy: 0.5260

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 5s - loss: 1.0270 - accuracy: 0.5446

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 1.0287 - accuracy: 0.5469

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 1.0379 - accuracy: 0.5521

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 1.0595 - accuracy: 0.5500

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 1.0498 - accuracy: 0.5511

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 1.0365 - accuracy: 0.5599

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 1.0331 - accuracy: 0.5625

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 1.0414 - accuracy: 0.5603

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 1.0560 - accuracy: 0.5604

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 1.0519 - accuracy: 0.5547

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 1.0475 - accuracy: 0.5588

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 1.0466 - accuracy: 0.5590

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 1.0604 - accuracy: 0.5559

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 1.0708 - accuracy: 0.5500

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 1.0685 - accuracy: 0.5565

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 1.0630 - accuracy: 0.5611

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 1.0608 - accuracy: 0.5598

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 1.0606 - accuracy: 0.5599

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 1.0622 - accuracy: 0.5600

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 1.0566 - accuracy: 0.5625

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 1.0575 - accuracy: 0.5625

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 1.0552 - accuracy: 0.5658

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 1.0547 - accuracy: 0.5636

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 1.0592 - accuracy: 0.5625

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 1.0592 - accuracy: 0.5605

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 1.0608 - accuracy: 0.5625

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 1.0731 - accuracy: 0.5568

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 1.0740 - accuracy: 0.5579

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 1.0706 - accuracy: 0.5607

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 1.0672 - accuracy: 0.5651

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 1.0699 - accuracy: 0.5642

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 1.0661 - accuracy: 0.5677

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 1.0691 - accuracy: 0.5660

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 1.0688 - accuracy: 0.5660

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 1.0648 - accuracy: 0.5674

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 1.0632 - accuracy: 0.5665

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 1.0580 - accuracy: 0.5707

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 1.0537 - accuracy: 0.5726

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 1.0518 - accuracy: 0.5745

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 1.0509 - accuracy: 0.5735

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 1.0493 - accuracy: 0.5746

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 1.0532 - accuracy: 0.5731

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 1.0595 - accuracy: 0.5741

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 1.0622 - accuracy: 0.5720

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 1.0594 - accuracy: 0.5743

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 1.0584 - accuracy: 0.5741

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 1.0603 - accuracy: 0.5721

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 1.0631 - accuracy: 0.5696

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 1.0644 - accuracy: 0.5678

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 1.0647 - accuracy: 0.5688

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 1.0699 - accuracy: 0.5644

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 1.0714 - accuracy: 0.5644

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 1.0737 - accuracy: 0.5638

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 1.0751 - accuracy: 0.5628

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 1.0734 - accuracy: 0.5648

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 1.0739 - accuracy: 0.5627

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 1.0743 - accuracy: 0.5642

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 1.0731 - accuracy: 0.5666

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 1.0709 - accuracy: 0.5675

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 1.0661 - accuracy: 0.5707

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 1.0626 - accuracy: 0.5729

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 1.0588 - accuracy: 0.5750

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 1.0604 - accuracy: 0.5739

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 1.0640 - accuracy: 0.5742

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 1.0616 - accuracy: 0.5736

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 1.0601 - accuracy: 0.5739

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 1.0588 - accuracy: 0.5742

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 1.0606 - accuracy: 0.5748

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 1.0621 - accuracy: 0.5747

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 1.0622 - accuracy: 0.5737

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 1.0654 - accuracy: 0.5715

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 1.0666 - accuracy: 0.5722

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 1.0638 - accuracy: 0.5737

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 1.0627 - accuracy: 0.5747

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 1.0608 - accuracy: 0.5749

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 1.0583 - accuracy: 0.5755

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 1.0609 - accuracy: 0.5739

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 1.0625 - accuracy: 0.5726

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 1.0623 - accuracy: 0.5729

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 1.0595 - accuracy: 0.5742

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 1.0577 - accuracy: 0.5744

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 1.0553 - accuracy: 0.5768

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 1.0556 - accuracy: 0.5773

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 1.0552 - accuracy: 0.5771

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 1.0561 - accuracy: 0.5780

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 1.0561 - accuracy: 0.5780 - val_loss: 1.0646 - val_accuracy: 0.5804


.. parsed-literal::

    Epoch 3/15


.. parsed-literal::


 1/92 [..............................] - ETA: 6s - loss: 1.2304 - accuracy: 0.4062

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 1.1172 - accuracy: 0.5156

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 1.0113 - accuracy: 0.5417

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 4s - loss: 0.9614 - accuracy: 0.5938

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 4s - loss: 0.9190 - accuracy: 0.6250

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.8853 - accuracy: 0.6406

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.8784 - accuracy: 0.6339

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.9027 - accuracy: 0.6250

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.9354 - accuracy: 0.6076

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.9185 - accuracy: 0.6250

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.9371 - accuracy: 0.6165

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.9418 - accuracy: 0.6120

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.9317 - accuracy: 0.6130

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.9458 - accuracy: 0.6027

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.9358 - accuracy: 0.6125

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.9439 - accuracy: 0.6133

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.9458 - accuracy: 0.6140

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.9360 - accuracy: 0.6198

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.9313 - accuracy: 0.6234

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.9296 - accuracy: 0.6234

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.9251 - accuracy: 0.6250

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 3s - loss: 0.9250 - accuracy: 0.6293

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.9261 - accuracy: 0.6332

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.9374 - accuracy: 0.6289

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.9380 - accuracy: 0.6300

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.9342 - accuracy: 0.6320

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.9297 - accuracy: 0.6351

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.9416 - accuracy: 0.6304

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.9458 - accuracy: 0.6292

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.9370 - accuracy: 0.6341

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.9329 - accuracy: 0.6378

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.9363 - accuracy: 0.6365

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.9401 - accuracy: 0.6361

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.9435 - accuracy: 0.6322

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.9367 - accuracy: 0.6381

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.9375 - accuracy: 0.6378

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.9463 - accuracy: 0.6341

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 2s - loss: 0.9507 - accuracy: 0.6331

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.9528 - accuracy: 0.6313

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.9537 - accuracy: 0.6311

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.9555 - accuracy: 0.6287

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.9473 - accuracy: 0.6323

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.9436 - accuracy: 0.6314

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.9470 - accuracy: 0.6292

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.9516 - accuracy: 0.6250

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.9505 - accuracy: 0.6243

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.9520 - accuracy: 0.6250

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.9481 - accuracy: 0.6282

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.9556 - accuracy: 0.6250

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.9543 - accuracy: 0.6262

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.9577 - accuracy: 0.6244

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.9570 - accuracy: 0.6256

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.9532 - accuracy: 0.6273

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.9524 - accuracy: 0.6279

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.9492 - accuracy: 0.6295

.. parsed-literal::

    
57/92 [=================>............] - ETA: 1s - loss: 0.9459 - accuracy: 0.6311

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.9478 - accuracy: 0.6320

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.9457 - accuracy: 0.6330

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.9499 - accuracy: 0.6334

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.9497 - accuracy: 0.6322

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.9468 - accuracy: 0.6351

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.9452 - accuracy: 0.6365

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.9438 - accuracy: 0.6368

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.9447 - accuracy: 0.6351

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.9432 - accuracy: 0.6369

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.9420 - accuracy: 0.6362

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.9368 - accuracy: 0.6379

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.9340 - accuracy: 0.6382

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.9314 - accuracy: 0.6393

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.9317 - accuracy: 0.6374

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.9327 - accuracy: 0.6372

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.9303 - accuracy: 0.6387

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.9286 - accuracy: 0.6390

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.9280 - accuracy: 0.6396

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.9269 - accuracy: 0.6386

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.9268 - accuracy: 0.6388

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.9275 - accuracy: 0.6395

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.9295 - accuracy: 0.6397

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.9328 - accuracy: 0.6383

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.9321 - accuracy: 0.6393

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.9318 - accuracy: 0.6391

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.9307 - accuracy: 0.6390

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.9302 - accuracy: 0.6392

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.9287 - accuracy: 0.6390

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.9323 - accuracy: 0.6370

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.9348 - accuracy: 0.6347

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.9344 - accuracy: 0.6350

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.9351 - accuracy: 0.6345

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.9347 - accuracy: 0.6351

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.9388 - accuracy: 0.6340

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.9399 - accuracy: 0.6328

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.9399 - accuracy: 0.6328 - val_loss: 1.0007 - val_accuracy: 0.5940


.. parsed-literal::

    Epoch 4/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.9920 - accuracy: 0.6562

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.9589 - accuracy: 0.6094

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.8941 - accuracy: 0.6458

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.8847 - accuracy: 0.6406

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.8530 - accuracy: 0.6625

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.8984 - accuracy: 0.6354

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.8741 - accuracy: 0.6339

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.8435 - accuracy: 0.6523

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.8345 - accuracy: 0.6632

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.8324 - accuracy: 0.6625

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.8258 - accuracy: 0.6648

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.8292 - accuracy: 0.6745

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.8313 - accuracy: 0.6731

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.8395 - accuracy: 0.6629

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.8590 - accuracy: 0.6607

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.8605 - accuracy: 0.6567

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.8668 - accuracy: 0.6549

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.8596 - accuracy: 0.6583

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.8555 - accuracy: 0.6582

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.8555 - accuracy: 0.6566

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 3s - loss: 0.8601 - accuracy: 0.6552

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.8749 - accuracy: 0.6538

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.8740 - accuracy: 0.6566

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.8755 - accuracy: 0.6578

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.8832 - accuracy: 0.6566

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.8806 - accuracy: 0.6589

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.8805 - accuracy: 0.6588

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.8690 - accuracy: 0.6652

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.8663 - accuracy: 0.6660

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.8652 - accuracy: 0.6646

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.8622 - accuracy: 0.6654

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.8645 - accuracy: 0.6632

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.8598 - accuracy: 0.6648

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.8604 - accuracy: 0.6655

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.8581 - accuracy: 0.6635

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.8624 - accuracy: 0.6633

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.8658 - accuracy: 0.6606

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.8682 - accuracy: 0.6605

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.8645 - accuracy: 0.6627

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.8639 - accuracy: 0.6618

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.8615 - accuracy: 0.6639

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.8596 - accuracy: 0.6659

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.8645 - accuracy: 0.6636

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.8648 - accuracy: 0.6613

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.8711 - accuracy: 0.6592

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.8675 - accuracy: 0.6598

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.8673 - accuracy: 0.6597

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.8650 - accuracy: 0.6609

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.8646 - accuracy: 0.6608

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.8675 - accuracy: 0.6583

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.8771 - accuracy: 0.6540

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.8818 - accuracy: 0.6505

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.8818 - accuracy: 0.6500

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.8802 - accuracy: 0.6513

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.8783 - accuracy: 0.6508

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.8789 - accuracy: 0.6509

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.8808 - accuracy: 0.6499

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.8819 - accuracy: 0.6516

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.8827 - accuracy: 0.6538

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.8835 - accuracy: 0.6528

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.8845 - accuracy: 0.6533

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.8856 - accuracy: 0.6524

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.8843 - accuracy: 0.6534

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.8860 - accuracy: 0.6525

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.8888 - accuracy: 0.6507

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.8881 - accuracy: 0.6512

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.8849 - accuracy: 0.6541

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.8842 - accuracy: 0.6536

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.8863 - accuracy: 0.6528

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.8876 - accuracy: 0.6524

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.8854 - accuracy: 0.6542

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.8869 - accuracy: 0.6529

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.8877 - accuracy: 0.6534

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.8853 - accuracy: 0.6555

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.8825 - accuracy: 0.6564

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.8828 - accuracy: 0.6568

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.8832 - accuracy: 0.6568

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.8823 - accuracy: 0.6571

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.8876 - accuracy: 0.6552

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.8872 - accuracy: 0.6560

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.8859 - accuracy: 0.6563

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.8849 - accuracy: 0.6567

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.8818 - accuracy: 0.6575

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.8843 - accuracy: 0.6560

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.8839 - accuracy: 0.6563

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.8842 - accuracy: 0.6574

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.8822 - accuracy: 0.6585

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.8796 - accuracy: 0.6599

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.8780 - accuracy: 0.6602

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.8778 - accuracy: 0.6601

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.8753 - accuracy: 0.6618

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.8753 - accuracy: 0.6618 - val_loss: 0.9517 - val_accuracy: 0.6281


.. parsed-literal::

    Epoch 5/15


.. parsed-literal::


 1/92 [..............................] - ETA: 8s - loss: 0.8451 - accuracy: 0.5312

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.9136 - accuracy: 0.5625

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.8712 - accuracy: 0.6146

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.8188 - accuracy: 0.6562

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.7953 - accuracy: 0.6625

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.8285 - accuracy: 0.6615

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.8488 - accuracy: 0.6652

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.8358 - accuracy: 0.6797

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.8435 - accuracy: 0.6736

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.8447 - accuracy: 0.6719

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.8653 - accuracy: 0.6648

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.8961 - accuracy: 0.6510

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.8837 - accuracy: 0.6538

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.8675 - accuracy: 0.6629

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.8627 - accuracy: 0.6646

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.8706 - accuracy: 0.6602

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.8520 - accuracy: 0.6710

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.8466 - accuracy: 0.6736

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.8419 - accuracy: 0.6760

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.8411 - accuracy: 0.6766

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.8367 - accuracy: 0.6786

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.8345 - accuracy: 0.6804

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.8301 - accuracy: 0.6848

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.8269 - accuracy: 0.6875

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.8243 - accuracy: 0.6888

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.8270 - accuracy: 0.6863

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.8263 - accuracy: 0.6887

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.8322 - accuracy: 0.6842

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.8345 - accuracy: 0.6821

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.8468 - accuracy: 0.6771

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.8432 - accuracy: 0.6774

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.8431 - accuracy: 0.6758

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.8482 - accuracy: 0.6695

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.8446 - accuracy: 0.6719

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.8485 - accuracy: 0.6679

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.8453 - accuracy: 0.6701

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.8404 - accuracy: 0.6740

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.8394 - accuracy: 0.6727

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.8364 - accuracy: 0.6739

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.8342 - accuracy: 0.6750

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.8337 - accuracy: 0.6776

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.8413 - accuracy: 0.6734

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.8466 - accuracy: 0.6722

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.8462 - accuracy: 0.6719

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.8418 - accuracy: 0.6743

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.8382 - accuracy: 0.6766

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.8369 - accuracy: 0.6782

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.8368 - accuracy: 0.6777

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.8352 - accuracy: 0.6773

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.8319 - accuracy: 0.6794

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.8358 - accuracy: 0.6765

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.8366 - accuracy: 0.6773

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.8343 - accuracy: 0.6781

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.8301 - accuracy: 0.6806

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.8324 - accuracy: 0.6824

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.8284 - accuracy: 0.6836

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.8294 - accuracy: 0.6831

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.8284 - accuracy: 0.6837

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.8230 - accuracy: 0.6843

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.8228 - accuracy: 0.6849

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.8267 - accuracy: 0.6829

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.8279 - accuracy: 0.6815

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.8244 - accuracy: 0.6820

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.8274 - accuracy: 0.6816

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.8285 - accuracy: 0.6803

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.8273 - accuracy: 0.6804

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.8267 - accuracy: 0.6810

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.8288 - accuracy: 0.6792

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.8281 - accuracy: 0.6789

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.8298 - accuracy: 0.6786

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.8314 - accuracy: 0.6769

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.8325 - accuracy: 0.6762

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.8325 - accuracy: 0.6759

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.8292 - accuracy: 0.6765

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.8269 - accuracy: 0.6771

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.8243 - accuracy: 0.6776

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.8220 - accuracy: 0.6778

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.8196 - accuracy: 0.6779

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.8189 - accuracy: 0.6776

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.8163 - accuracy: 0.6784

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.8143 - accuracy: 0.6785

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.8156 - accuracy: 0.6790

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.8147 - accuracy: 0.6795

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.8132 - accuracy: 0.6803

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.8129 - accuracy: 0.6808

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.8137 - accuracy: 0.6801

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.8121 - accuracy: 0.6809

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.8121 - accuracy: 0.6824

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.8143 - accuracy: 0.6807

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.8151 - accuracy: 0.6804

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.8177 - accuracy: 0.6798

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.8177 - accuracy: 0.6798 - val_loss: 0.8919 - val_accuracy: 0.6594


.. parsed-literal::

    Epoch 6/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.7215 - accuracy: 0.7812

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.6720 - accuracy: 0.7500

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6412 - accuracy: 0.7500

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6503 - accuracy: 0.7422

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.6992 - accuracy: 0.7312

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.7268 - accuracy: 0.7135

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.7328 - accuracy: 0.7098

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.7341 - accuracy: 0.7070

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.7782 - accuracy: 0.6910

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.7772 - accuracy: 0.6906

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.7993 - accuracy: 0.6790

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.7833 - accuracy: 0.6901

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.7947 - accuracy: 0.6851

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.7912 - accuracy: 0.6875

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.8069 - accuracy: 0.6833

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.8130 - accuracy: 0.6816

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.8120 - accuracy: 0.6820

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.8100 - accuracy: 0.6840

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.8332 - accuracy: 0.6776

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.8463 - accuracy: 0.6734

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.8511 - accuracy: 0.6711

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.8618 - accuracy: 0.6634

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.8480 - accuracy: 0.6712

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.8450 - accuracy: 0.6719

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.8424 - accuracy: 0.6712

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.8359 - accuracy: 0.6731

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.8304 - accuracy: 0.6771

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.8189 - accuracy: 0.6786

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.8173 - accuracy: 0.6810

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.8145 - accuracy: 0.6823

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.8086 - accuracy: 0.6845

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.8071 - accuracy: 0.6855

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.7974 - accuracy: 0.6884

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.7998 - accuracy: 0.6884

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.7957 - accuracy: 0.6884

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.7988 - accuracy: 0.6901

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.7963 - accuracy: 0.6917

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.7952 - accuracy: 0.6924

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.7908 - accuracy: 0.6963

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.7827 - accuracy: 0.6986

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.7876 - accuracy: 0.6969

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.7918 - accuracy: 0.6959

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.7872 - accuracy: 0.6979

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.7906 - accuracy: 0.6969

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.7878 - accuracy: 0.6981

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.7878 - accuracy: 0.6979

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.7855 - accuracy: 0.7009

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.7850 - accuracy: 0.7013

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.7840 - accuracy: 0.7023

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.7823 - accuracy: 0.7014

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.7772 - accuracy: 0.7035

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.7784 - accuracy: 0.7020

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.7803 - accuracy: 0.7000

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.7794 - accuracy: 0.7009

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.7748 - accuracy: 0.7029

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.7802 - accuracy: 0.7026

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.7844 - accuracy: 0.7013

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.7801 - accuracy: 0.7037

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.7783 - accuracy: 0.7050

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.7749 - accuracy: 0.7058

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.7738 - accuracy: 0.7060

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.7809 - accuracy: 0.7042

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.7821 - accuracy: 0.7029

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.7819 - accuracy: 0.7037

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.7787 - accuracy: 0.7044

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.7817 - accuracy: 0.7037

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.7804 - accuracy: 0.7043

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.7823 - accuracy: 0.7032

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.7819 - accuracy: 0.7039

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.7873 - accuracy: 0.7010

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.7892 - accuracy: 0.7017

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.7905 - accuracy: 0.7015

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.7894 - accuracy: 0.7013

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.7888 - accuracy: 0.7019

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.7890 - accuracy: 0.7017

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.7866 - accuracy: 0.7024

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.7898 - accuracy: 0.7014

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.7879 - accuracy: 0.7024

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.7887 - accuracy: 0.7018

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.7870 - accuracy: 0.7024

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.7870 - accuracy: 0.7022

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.7869 - accuracy: 0.7024

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.7858 - accuracy: 0.7034

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.7862 - accuracy: 0.7028

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.7848 - accuracy: 0.7019

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.7852 - accuracy: 0.7024

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.7849 - accuracy: 0.7030

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.7826 - accuracy: 0.7035

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.7811 - accuracy: 0.7044

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.7792 - accuracy: 0.7049

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.7774 - accuracy: 0.7057

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.7774 - accuracy: 0.7057 - val_loss: 0.7700 - val_accuracy: 0.7071


.. parsed-literal::

    Epoch 7/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.6305 - accuracy: 0.7500

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.7776 - accuracy: 0.7656

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.7032 - accuracy: 0.7812

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6797 - accuracy: 0.7656

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.7095 - accuracy: 0.7437

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.7180 - accuracy: 0.7396

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6759 - accuracy: 0.7634

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6776 - accuracy: 0.7617

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6705 - accuracy: 0.7604

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6851 - accuracy: 0.7563

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6824 - accuracy: 0.7580

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6693 - accuracy: 0.7623

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6573 - accuracy: 0.7636

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6536 - accuracy: 0.7669

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6476 - accuracy: 0.7718

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6622 - accuracy: 0.7668

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6779 - accuracy: 0.7641

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6824 - accuracy: 0.7583

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6853 - accuracy: 0.7595

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6912 - accuracy: 0.7530

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 3s - loss: 0.6920 - accuracy: 0.7514

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.6944 - accuracy: 0.7473

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6974 - accuracy: 0.7474

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6935 - accuracy: 0.7513

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6974 - accuracy: 0.7488

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6950 - accuracy: 0.7488

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6980 - accuracy: 0.7477

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6945 - accuracy: 0.7478

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6903 - accuracy: 0.7511

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6891 - accuracy: 0.7520

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6879 - accuracy: 0.7510

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6926 - accuracy: 0.7500

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6932 - accuracy: 0.7519

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6992 - accuracy: 0.7473

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.7044 - accuracy: 0.7439

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.7034 - accuracy: 0.7440

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.7030 - accuracy: 0.7467

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.7009 - accuracy: 0.7476

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.7000 - accuracy: 0.7476

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.7047 - accuracy: 0.7446

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.7080 - accuracy: 0.7410

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.7066 - accuracy: 0.7405

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.7065 - accuracy: 0.7407

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.7174 - accuracy: 0.7353

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.7228 - accuracy: 0.7309

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.7230 - accuracy: 0.7313

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.7244 - accuracy: 0.7310

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.7259 - accuracy: 0.7308

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.7225 - accuracy: 0.7330

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.7226 - accuracy: 0.7346

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.7295 - accuracy: 0.7331

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.7325 - accuracy: 0.7322

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.7311 - accuracy: 0.7326

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.7265 - accuracy: 0.7340

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.7264 - accuracy: 0.7332

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.7247 - accuracy: 0.7340

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.7230 - accuracy: 0.7343

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.7234 - accuracy: 0.7340

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.7206 - accuracy: 0.7359

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.7174 - accuracy: 0.7377

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.7190 - accuracy: 0.7373

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.7191 - accuracy: 0.7361

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.7168 - accuracy: 0.7368

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.7166 - accuracy: 0.7365

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.7241 - accuracy: 0.7334

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.7215 - accuracy: 0.7346

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.7241 - accuracy: 0.7325

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.7253 - accuracy: 0.7318

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.7226 - accuracy: 0.7334

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.7209 - accuracy: 0.7350

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.7196 - accuracy: 0.7365

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.7207 - accuracy: 0.7363

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.7222 - accuracy: 0.7343

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.7229 - accuracy: 0.7320

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.7221 - accuracy: 0.7314

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.7209 - accuracy: 0.7309

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.7189 - accuracy: 0.7323

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.7183 - accuracy: 0.7325

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.7167 - accuracy: 0.7332

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.7147 - accuracy: 0.7341

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.7145 - accuracy: 0.7347

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.7114 - accuracy: 0.7356

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.7109 - accuracy: 0.7362

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.7104 - accuracy: 0.7360

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.7121 - accuracy: 0.7358

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.7141 - accuracy: 0.7352

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.7151 - accuracy: 0.7350

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.7158 - accuracy: 0.7342

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.7167 - accuracy: 0.7333

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.7175 - accuracy: 0.7338

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.7160 - accuracy: 0.7350

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.7160 - accuracy: 0.7350 - val_loss: 0.7492 - val_accuracy: 0.7193


.. parsed-literal::

    Epoch 8/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.6519 - accuracy: 0.7188

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.6333 - accuracy: 0.7344

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6408 - accuracy: 0.7500

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6623 - accuracy: 0.7422

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.6180 - accuracy: 0.7563

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.6154 - accuracy: 0.7760

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6369 - accuracy: 0.7679

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6529 - accuracy: 0.7617

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6430 - accuracy: 0.7604

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6368 - accuracy: 0.7625

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6383 - accuracy: 0.7670

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6329 - accuracy: 0.7708

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6388 - accuracy: 0.7692

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6427 - accuracy: 0.7701

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6641 - accuracy: 0.7646

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6638 - accuracy: 0.7637

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6802 - accuracy: 0.7518

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6744 - accuracy: 0.7569

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6854 - accuracy: 0.7533

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6836 - accuracy: 0.7531

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6816 - accuracy: 0.7515

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6804 - accuracy: 0.7528

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.6800 - accuracy: 0.7568

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6701 - accuracy: 0.7617

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6681 - accuracy: 0.7638

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6695 - accuracy: 0.7632

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6720 - accuracy: 0.7616

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6759 - accuracy: 0.7600

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6772 - accuracy: 0.7586

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6797 - accuracy: 0.7530

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6859 - accuracy: 0.7470

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6841 - accuracy: 0.7462

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6779 - accuracy: 0.7491

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6740 - accuracy: 0.7509

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6742 - accuracy: 0.7535

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6724 - accuracy: 0.7543

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6726 - accuracy: 0.7533

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6681 - accuracy: 0.7556

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.6696 - accuracy: 0.7571

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6681 - accuracy: 0.7584

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6655 - accuracy: 0.7575

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6646 - accuracy: 0.7573

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6622 - accuracy: 0.7564

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6635 - accuracy: 0.7556

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6695 - accuracy: 0.7534

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6665 - accuracy: 0.7547

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6658 - accuracy: 0.7565

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6662 - accuracy: 0.7577

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6702 - accuracy: 0.7563

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6770 - accuracy: 0.7549

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6764 - accuracy: 0.7554

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6735 - accuracy: 0.7565

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6733 - accuracy: 0.7558

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6761 - accuracy: 0.7557

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6766 - accuracy: 0.7545

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6779 - accuracy: 0.7539

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6773 - accuracy: 0.7527

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6807 - accuracy: 0.7511

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6798 - accuracy: 0.7510

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6820 - accuracy: 0.7490

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6829 - accuracy: 0.7490

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6798 - accuracy: 0.7505

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6814 - accuracy: 0.7485

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6840 - accuracy: 0.7466

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6827 - accuracy: 0.7471

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6839 - accuracy: 0.7463

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6840 - accuracy: 0.7449

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6840 - accuracy: 0.7450

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6862 - accuracy: 0.7437

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6870 - accuracy: 0.7438

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6878 - accuracy: 0.7435

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6937 - accuracy: 0.7405

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6934 - accuracy: 0.7407

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6937 - accuracy: 0.7408

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6920 - accuracy: 0.7413

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6901 - accuracy: 0.7427

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6893 - accuracy: 0.7436

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6927 - accuracy: 0.7425

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6910 - accuracy: 0.7429

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6922 - accuracy: 0.7423

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6909 - accuracy: 0.7424

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6938 - accuracy: 0.7413

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6921 - accuracy: 0.7422

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6931 - accuracy: 0.7419

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6908 - accuracy: 0.7431

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6901 - accuracy: 0.7417

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6873 - accuracy: 0.7422

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6847 - accuracy: 0.7440

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6847 - accuracy: 0.7437

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6847 - accuracy: 0.7435

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6845 - accuracy: 0.7439

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.6845 - accuracy: 0.7439 - val_loss: 0.7754 - val_accuracy: 0.7125


.. parsed-literal::

    Epoch 9/15


.. parsed-literal::


 1/92 [..............................] - ETA: 6s - loss: 0.4831 - accuracy: 0.8438

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5184 - accuracy: 0.8281

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6743 - accuracy: 0.7604

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6740 - accuracy: 0.7500

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 4s - loss: 0.6225 - accuracy: 0.7625

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.6291 - accuracy: 0.7552

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6330 - accuracy: 0.7545

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6225 - accuracy: 0.7695

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6262 - accuracy: 0.7674

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6388 - accuracy: 0.7625

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6322 - accuracy: 0.7642

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6194 - accuracy: 0.7708

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6162 - accuracy: 0.7692

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6355 - accuracy: 0.7567

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6206 - accuracy: 0.7604

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6189 - accuracy: 0.7598

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6123 - accuracy: 0.7629

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6230 - accuracy: 0.7587

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6323 - accuracy: 0.7549

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6287 - accuracy: 0.7563

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6264 - accuracy: 0.7574

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6262 - accuracy: 0.7571

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.6238 - accuracy: 0.7582

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6178 - accuracy: 0.7617

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6176 - accuracy: 0.7613

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6135 - accuracy: 0.7644

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6194 - accuracy: 0.7604

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6242 - accuracy: 0.7578

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6228 - accuracy: 0.7586

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6273 - accuracy: 0.7552

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6265 - accuracy: 0.7581

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6296 - accuracy: 0.7557

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6304 - accuracy: 0.7556

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6294 - accuracy: 0.7590

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6305 - accuracy: 0.7605

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6265 - accuracy: 0.7619

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6214 - accuracy: 0.7641

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6199 - accuracy: 0.7645

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.6268 - accuracy: 0.7626

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6289 - accuracy: 0.7638

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6317 - accuracy: 0.7620

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6330 - accuracy: 0.7617

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6342 - accuracy: 0.7600

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6288 - accuracy: 0.7612

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6256 - accuracy: 0.7623

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6212 - accuracy: 0.7647

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6192 - accuracy: 0.7651

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6158 - accuracy: 0.7660

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6147 - accuracy: 0.7663

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6160 - accuracy: 0.7654

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6187 - accuracy: 0.7651

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6163 - accuracy: 0.7666

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6152 - accuracy: 0.7669

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6181 - accuracy: 0.7660

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6160 - accuracy: 0.7668

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6148 - accuracy: 0.7671

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6194 - accuracy: 0.7673

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6182 - accuracy: 0.7670

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6233 - accuracy: 0.7641

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6230 - accuracy: 0.7639

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6240 - accuracy: 0.7642

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6243 - accuracy: 0.7639

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6224 - accuracy: 0.7632

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6222 - accuracy: 0.7640

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6213 - accuracy: 0.7633

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6214 - accuracy: 0.7631

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6251 - accuracy: 0.7629

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6331 - accuracy: 0.7586

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6328 - accuracy: 0.7585

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6308 - accuracy: 0.7593

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6331 - accuracy: 0.7583

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6314 - accuracy: 0.7590

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6361 - accuracy: 0.7559

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6366 - accuracy: 0.7554

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6377 - accuracy: 0.7545

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6400 - accuracy: 0.7545

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6419 - accuracy: 0.7540

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6400 - accuracy: 0.7540

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6385 - accuracy: 0.7543

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6388 - accuracy: 0.7543

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6391 - accuracy: 0.7542

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6368 - accuracy: 0.7557

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6362 - accuracy: 0.7563

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6385 - accuracy: 0.7563

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6404 - accuracy: 0.7562

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6419 - accuracy: 0.7547

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6433 - accuracy: 0.7543

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6441 - accuracy: 0.7539

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6461 - accuracy: 0.7538

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6451 - accuracy: 0.7545

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6462 - accuracy: 0.7534

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.6462 - accuracy: 0.7534 - val_loss: 0.7766 - val_accuracy: 0.7071


.. parsed-literal::

    Epoch 10/15


.. parsed-literal::


 1/92 [..............................] - ETA: 6s - loss: 0.3661 - accuracy: 0.9062

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.4720 - accuracy: 0.8281

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.4717 - accuracy: 0.8333

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5884 - accuracy: 0.7812

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 4s - loss: 0.5973 - accuracy: 0.7812

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.5721 - accuracy: 0.7917

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6203 - accuracy: 0.7812

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6320 - accuracy: 0.7695

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6134 - accuracy: 0.7778

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5992 - accuracy: 0.7844

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6097 - accuracy: 0.7812

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6404 - accuracy: 0.7656

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6326 - accuracy: 0.7668

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6360 - accuracy: 0.7634

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6342 - accuracy: 0.7625

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6245 - accuracy: 0.7656

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6220 - accuracy: 0.7629

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6323 - accuracy: 0.7604

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6365 - accuracy: 0.7599

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6344 - accuracy: 0.7656

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6527 - accuracy: 0.7545

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6585 - accuracy: 0.7514

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.6530 - accuracy: 0.7514

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6602 - accuracy: 0.7513

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6636 - accuracy: 0.7475

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6718 - accuracy: 0.7464

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6675 - accuracy: 0.7488

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6750 - accuracy: 0.7467

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6696 - accuracy: 0.7489

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6616 - accuracy: 0.7500

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6584 - accuracy: 0.7520

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6562 - accuracy: 0.7559

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6550 - accuracy: 0.7566

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6556 - accuracy: 0.7555

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6564 - accuracy: 0.7545

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6536 - accuracy: 0.7543

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6533 - accuracy: 0.7542

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6577 - accuracy: 0.7500

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.6550 - accuracy: 0.7508

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6564 - accuracy: 0.7485

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6538 - accuracy: 0.7493

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6466 - accuracy: 0.7522

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6419 - accuracy: 0.7543

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6396 - accuracy: 0.7556

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6381 - accuracy: 0.7561

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6386 - accuracy: 0.7574

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6356 - accuracy: 0.7579

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6387 - accuracy: 0.7564

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6381 - accuracy: 0.7563

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6334 - accuracy: 0.7574

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6343 - accuracy: 0.7572

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6304 - accuracy: 0.7601

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6249 - accuracy: 0.7628

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6286 - accuracy: 0.7626

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6288 - accuracy: 0.7618

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6269 - accuracy: 0.7632

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6295 - accuracy: 0.7614

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6306 - accuracy: 0.7612

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6322 - accuracy: 0.7599

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6297 - accuracy: 0.7618

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6271 - accuracy: 0.7632

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6286 - accuracy: 0.7610

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6279 - accuracy: 0.7613

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6284 - accuracy: 0.7606

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6300 - accuracy: 0.7600

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6307 - accuracy: 0.7594

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6307 - accuracy: 0.7592

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6321 - accuracy: 0.7577

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6326 - accuracy: 0.7572

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6319 - accuracy: 0.7593

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6326 - accuracy: 0.7583

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6325 - accuracy: 0.7573

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6302 - accuracy: 0.7585

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6305 - accuracy: 0.7575

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6318 - accuracy: 0.7570

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6329 - accuracy: 0.7569

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6324 - accuracy: 0.7564

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6320 - accuracy: 0.7563

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6301 - accuracy: 0.7571

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6269 - accuracy: 0.7577

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6254 - accuracy: 0.7580

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6252 - accuracy: 0.7576

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6233 - accuracy: 0.7586

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6229 - accuracy: 0.7581

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6211 - accuracy: 0.7587

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6209 - accuracy: 0.7594

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6181 - accuracy: 0.7607

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6166 - accuracy: 0.7613

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6153 - accuracy: 0.7618

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6136 - accuracy: 0.7631

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6121 - accuracy: 0.7633

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.6121 - accuracy: 0.7633 - val_loss: 0.7282 - val_accuracy: 0.7180


.. parsed-literal::

    Epoch 11/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.6727 - accuracy: 0.7812

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.6855 - accuracy: 0.7344

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6026 - accuracy: 0.7812

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6569 - accuracy: 0.7578

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.6585 - accuracy: 0.7500

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.6519 - accuracy: 0.7500

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6303 - accuracy: 0.7545

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6420 - accuracy: 0.7422

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6264 - accuracy: 0.7361

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6096 - accuracy: 0.7469

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5882 - accuracy: 0.7557

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5965 - accuracy: 0.7526

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5747 - accuracy: 0.7644

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5677 - accuracy: 0.7701

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5667 - accuracy: 0.7688

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5688 - accuracy: 0.7656

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5646 - accuracy: 0.7702

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5651 - accuracy: 0.7708

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5828 - accuracy: 0.7664

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5959 - accuracy: 0.7625

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5927 - accuracy: 0.7619

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5953 - accuracy: 0.7614

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.5906 - accuracy: 0.7649

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5871 - accuracy: 0.7669

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5804 - accuracy: 0.7700

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5891 - accuracy: 0.7692

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5911 - accuracy: 0.7697

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5890 - accuracy: 0.7734

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5820 - accuracy: 0.7780

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5904 - accuracy: 0.7771

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5890 - accuracy: 0.7772

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5860 - accuracy: 0.7793

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5860 - accuracy: 0.7794

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5843 - accuracy: 0.7803

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5811 - accuracy: 0.7821

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5918 - accuracy: 0.7734

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5891 - accuracy: 0.7753

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5890 - accuracy: 0.7755

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5869 - accuracy: 0.7780

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.5852 - accuracy: 0.7789

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5839 - accuracy: 0.7790

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5855 - accuracy: 0.7783

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5864 - accuracy: 0.7776

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5879 - accuracy: 0.7770

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5857 - accuracy: 0.7785

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5892 - accuracy: 0.7765

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5914 - accuracy: 0.7766

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5874 - accuracy: 0.7780

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5851 - accuracy: 0.7787

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5893 - accuracy: 0.7769

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5883 - accuracy: 0.7782

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5888 - accuracy: 0.7776

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5985 - accuracy: 0.7742

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5946 - accuracy: 0.7749

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5922 - accuracy: 0.7756

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5914 - accuracy: 0.7768

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5887 - accuracy: 0.7791

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5913 - accuracy: 0.7780

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5918 - accuracy: 0.7781

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5922 - accuracy: 0.7781

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5891 - accuracy: 0.7797

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5869 - accuracy: 0.7807

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5886 - accuracy: 0.7788

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5870 - accuracy: 0.7793

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5889 - accuracy: 0.7784

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5852 - accuracy: 0.7794

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5849 - accuracy: 0.7794

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5838 - accuracy: 0.7794

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5856 - accuracy: 0.7790

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5864 - accuracy: 0.7781

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5889 - accuracy: 0.7786

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5868 - accuracy: 0.7795

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5864 - accuracy: 0.7800

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5879 - accuracy: 0.7787

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5890 - accuracy: 0.7788

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5883 - accuracy: 0.7788

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5889 - accuracy: 0.7780

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5888 - accuracy: 0.7788

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5888 - accuracy: 0.7797

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5878 - accuracy: 0.7794

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5874 - accuracy: 0.7794

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5878 - accuracy: 0.7787

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5863 - accuracy: 0.7795

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5862 - accuracy: 0.7788

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5836 - accuracy: 0.7802

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5845 - accuracy: 0.7795

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5871 - accuracy: 0.7781

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5863 - accuracy: 0.7782

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5906 - accuracy: 0.7772

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5928 - accuracy: 0.7758

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5915 - accuracy: 0.7755

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.5915 - accuracy: 0.7755 - val_loss: 0.6808 - val_accuracy: 0.7357


.. parsed-literal::

    Epoch 12/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.4877 - accuracy: 0.7500

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.4417 - accuracy: 0.8125

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5186 - accuracy: 0.7917

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6121 - accuracy: 0.7500

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.6493 - accuracy: 0.7437

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.6482 - accuracy: 0.7500

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6296 - accuracy: 0.7545

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6570 - accuracy: 0.7383

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6370 - accuracy: 0.7465

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6218 - accuracy: 0.7500

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6278 - accuracy: 0.7472

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6193 - accuracy: 0.7474

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5997 - accuracy: 0.7596

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6107 - accuracy: 0.7545

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6133 - accuracy: 0.7583

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6027 - accuracy: 0.7637

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6091 - accuracy: 0.7592

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5975 - accuracy: 0.7622

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5948 - accuracy: 0.7632

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5954 - accuracy: 0.7641

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6061 - accuracy: 0.7589

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6076 - accuracy: 0.7585

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.5998 - accuracy: 0.7609

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5995 - accuracy: 0.7630

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5948 - accuracy: 0.7663

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5989 - accuracy: 0.7644

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5921 - accuracy: 0.7674

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5870 - accuracy: 0.7723

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5832 - accuracy: 0.7759

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5844 - accuracy: 0.7740

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5831 - accuracy: 0.7762

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5845 - accuracy: 0.7764

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5825 - accuracy: 0.7784

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5843 - accuracy: 0.7794

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5916 - accuracy: 0.7795

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5877 - accuracy: 0.7804

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5887 - accuracy: 0.7821

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5873 - accuracy: 0.7804

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5897 - accuracy: 0.7788

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.5876 - accuracy: 0.7789

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5878 - accuracy: 0.7752

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5900 - accuracy: 0.7738

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5943 - accuracy: 0.7711

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5938 - accuracy: 0.7720

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5926 - accuracy: 0.7715

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5920 - accuracy: 0.7717

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5908 - accuracy: 0.7733

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5909 - accuracy: 0.7734

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5888 - accuracy: 0.7739

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5891 - accuracy: 0.7746

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5907 - accuracy: 0.7748

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5907 - accuracy: 0.7737

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5895 - accuracy: 0.7750

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5914 - accuracy: 0.7728

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5929 - accuracy: 0.7707

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5920 - accuracy: 0.7720

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5922 - accuracy: 0.7716

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5902 - accuracy: 0.7718

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5897 - accuracy: 0.7709

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5882 - accuracy: 0.7716

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5874 - accuracy: 0.7723

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5867 - accuracy: 0.7729

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5886 - accuracy: 0.7725

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5876 - accuracy: 0.7727

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5886 - accuracy: 0.7714

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5860 - accuracy: 0.7729

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5828 - accuracy: 0.7749

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5801 - accuracy: 0.7759

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5791 - accuracy: 0.7760

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5764 - accuracy: 0.7769

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5766 - accuracy: 0.7770

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5774 - accuracy: 0.7766

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5779 - accuracy: 0.7763

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5801 - accuracy: 0.7747

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5817 - accuracy: 0.7731

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5840 - accuracy: 0.7720

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5849 - accuracy: 0.7701

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5853 - accuracy: 0.7698

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5860 - accuracy: 0.7696

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5883 - accuracy: 0.7693

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5916 - accuracy: 0.7680

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5897 - accuracy: 0.7693

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5926 - accuracy: 0.7679

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5928 - accuracy: 0.7692

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5907 - accuracy: 0.7704

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5890 - accuracy: 0.7705

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5899 - accuracy: 0.7707

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5890 - accuracy: 0.7708

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5891 - accuracy: 0.7712

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5895 - accuracy: 0.7713

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5885 - accuracy: 0.7721

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.5885 - accuracy: 0.7721 - val_loss: 0.7104 - val_accuracy: 0.7384


.. parsed-literal::

    Epoch 13/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.6603 - accuracy: 0.7188

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.6015 - accuracy: 0.7656

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5565 - accuracy: 0.7917

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5470 - accuracy: 0.8047

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5798 - accuracy: 0.7937

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.6187 - accuracy: 0.7812

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.5944 - accuracy: 0.7857

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5649 - accuracy: 0.7857

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5609 - accuracy: 0.7885

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5495 - accuracy: 0.7936

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5513 - accuracy: 0.8005

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5459 - accuracy: 0.8088

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5329 - accuracy: 0.8159

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5391 - accuracy: 0.8072

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5313 - accuracy: 0.8095

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5334 - accuracy: 0.8097

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5361 - accuracy: 0.8081

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5405 - accuracy: 0.8067

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5349 - accuracy: 0.8070

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5406 - accuracy: 0.7997

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 3s - loss: 0.5461 - accuracy: 0.7974

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.5405 - accuracy: 0.8008

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5371 - accuracy: 0.8026

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5321 - accuracy: 0.8043

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5283 - accuracy: 0.8058

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5234 - accuracy: 0.8072

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5281 - accuracy: 0.8063

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5225 - accuracy: 0.8076

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5179 - accuracy: 0.8099

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5234 - accuracy: 0.8089

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5249 - accuracy: 0.8091

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5295 - accuracy: 0.8101

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5285 - accuracy: 0.8120

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5337 - accuracy: 0.8112

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5331 - accuracy: 0.8094

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5339 - accuracy: 0.8078

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5326 - accuracy: 0.8088

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5316 - accuracy: 0.8073

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.5316 - accuracy: 0.8074

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5321 - accuracy: 0.8060

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5283 - accuracy: 0.8069

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5248 - accuracy: 0.8085

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5235 - accuracy: 0.8086

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5314 - accuracy: 0.8052

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5301 - accuracy: 0.8040

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5261 - accuracy: 0.8055

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5262 - accuracy: 0.8037

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5275 - accuracy: 0.8032

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5258 - accuracy: 0.8040

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5369 - accuracy: 0.7999

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5336 - accuracy: 0.8025

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5363 - accuracy: 0.8015

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5338 - accuracy: 0.8023

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5326 - accuracy: 0.8025

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5333 - accuracy: 0.8021

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5352 - accuracy: 0.7996

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5355 - accuracy: 0.7982

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5337 - accuracy: 0.7995

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5323 - accuracy: 0.7997

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5347 - accuracy: 0.7989

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5353 - accuracy: 0.7991

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5324 - accuracy: 0.8003

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5381 - accuracy: 0.7980

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5381 - accuracy: 0.7978

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5391 - accuracy: 0.7961

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5370 - accuracy: 0.7978

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5412 - accuracy: 0.7957

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5406 - accuracy: 0.7950

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5407 - accuracy: 0.7944

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5390 - accuracy: 0.7946

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5424 - accuracy: 0.7927

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5444 - accuracy: 0.7921

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5484 - accuracy: 0.7919

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5519 - accuracy: 0.7910

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5510 - accuracy: 0.7921

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5525 - accuracy: 0.7919

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5511 - accuracy: 0.7922

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5532 - accuracy: 0.7921

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5519 - accuracy: 0.7923

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5507 - accuracy: 0.7937

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5496 - accuracy: 0.7943

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5488 - accuracy: 0.7949

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5496 - accuracy: 0.7940

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5476 - accuracy: 0.7950

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5463 - accuracy: 0.7956

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5469 - accuracy: 0.7947

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5471 - accuracy: 0.7949

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5482 - accuracy: 0.7944

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5480 - accuracy: 0.7942

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5500 - accuracy: 0.7930

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5497 - accuracy: 0.7936

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.5497 - accuracy: 0.7936 - val_loss: 0.7047 - val_accuracy: 0.7357


.. parsed-literal::

    Epoch 14/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.3726 - accuracy: 0.8750

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5405 - accuracy: 0.7969

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.4558 - accuracy: 0.8438

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.4742 - accuracy: 0.8438

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5545 - accuracy: 0.8125

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.5416 - accuracy: 0.8229

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.5139 - accuracy: 0.8259

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.4912 - accuracy: 0.8320

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5076 - accuracy: 0.8333

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.4818 - accuracy: 0.8438

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.4946 - accuracy: 0.8239

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5098 - accuracy: 0.8229

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.4978 - accuracy: 0.8293

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.4989 - accuracy: 0.8259

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5078 - accuracy: 0.8250

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.4975 - accuracy: 0.8320

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5014 - accuracy: 0.8327

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.4964 - accuracy: 0.8281

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5025 - accuracy: 0.8273

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.4927 - accuracy: 0.8297

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.4811 - accuracy: 0.8348

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.4792 - accuracy: 0.8352

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.4689 - accuracy: 0.8397

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.4722 - accuracy: 0.8385

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.4754 - accuracy: 0.8363

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.4740 - accuracy: 0.8353

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.4771 - accuracy: 0.8322

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.4924 - accuracy: 0.8225

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.4928 - accuracy: 0.8211

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.4959 - accuracy: 0.8198

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.4910 - accuracy: 0.8226

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.4955 - accuracy: 0.8193

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5048 - accuracy: 0.8134

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5035 - accuracy: 0.8125

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5011 - accuracy: 0.8143

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5014 - accuracy: 0.8134

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5045 - accuracy: 0.8117

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5056 - accuracy: 0.8092

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5110 - accuracy: 0.8069

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.5072 - accuracy: 0.8078

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5025 - accuracy: 0.8087

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5020 - accuracy: 0.8095

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5049 - accuracy: 0.8110

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5088 - accuracy: 0.8089

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5053 - accuracy: 0.8111

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5006 - accuracy: 0.8132

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5024 - accuracy: 0.8145

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5030 - accuracy: 0.8138

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5021 - accuracy: 0.8131

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5095 - accuracy: 0.8100

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5085 - accuracy: 0.8113

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5095 - accuracy: 0.8119

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5100 - accuracy: 0.8119

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5081 - accuracy: 0.8131

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5072 - accuracy: 0.8139

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5057 - accuracy: 0.8150

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5043 - accuracy: 0.8160

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5034 - accuracy: 0.8165

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5057 - accuracy: 0.8143

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5068 - accuracy: 0.8138

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5066 - accuracy: 0.8143

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5097 - accuracy: 0.8123

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5091 - accuracy: 0.8123

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5128 - accuracy: 0.8108

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5155 - accuracy: 0.8094

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5179 - accuracy: 0.8095

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5152 - accuracy: 0.8104

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5152 - accuracy: 0.8091

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5142 - accuracy: 0.8096

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5103 - accuracy: 0.8114

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5090 - accuracy: 0.8123

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5089 - accuracy: 0.8123

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5088 - accuracy: 0.8119

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5103 - accuracy: 0.8115

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5109 - accuracy: 0.8115

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5101 - accuracy: 0.8119

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5094 - accuracy: 0.8111

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5116 - accuracy: 0.8099

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5132 - accuracy: 0.8088

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5134 - accuracy: 0.8088

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5129 - accuracy: 0.8093

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5145 - accuracy: 0.8078

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5145 - accuracy: 0.8075

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5148 - accuracy: 0.8072

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5156 - accuracy: 0.8065

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5174 - accuracy: 0.8055

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5204 - accuracy: 0.8041

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5185 - accuracy: 0.8046

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5182 - accuracy: 0.8050

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5214 - accuracy: 0.8037

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5209 - accuracy: 0.8035

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.5209 - accuracy: 0.8035 - val_loss: 0.7335 - val_accuracy: 0.7289


.. parsed-literal::

    Epoch 15/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.6594 - accuracy: 0.6562

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.4822 - accuracy: 0.7812

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.4401 - accuracy: 0.8125

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.4288 - accuracy: 0.8359

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.4711 - accuracy: 0.8188

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.4834 - accuracy: 0.8177

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 5s - loss: 0.5001 - accuracy: 0.8080

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5052 - accuracy: 0.8047

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.4861 - accuracy: 0.8160

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.4778 - accuracy: 0.8156

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.4763 - accuracy: 0.8210

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.4871 - accuracy: 0.8177

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.4874 - accuracy: 0.8149

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.4838 - accuracy: 0.8192

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.4898 - accuracy: 0.8104

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.4858 - accuracy: 0.8086

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.4969 - accuracy: 0.8033

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.4890 - accuracy: 0.8090

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.4904 - accuracy: 0.8092

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.4901 - accuracy: 0.8078

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.4958 - accuracy: 0.8080

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5060 - accuracy: 0.8054

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.5097 - accuracy: 0.8057

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5168 - accuracy: 0.8021

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5153 - accuracy: 0.8037

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5116 - accuracy: 0.8077

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5084 - accuracy: 0.8102

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5053 - accuracy: 0.8125

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.4987 - accuracy: 0.8147

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5008 - accuracy: 0.8125

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5025 - accuracy: 0.8105

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5017 - accuracy: 0.8105

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5009 - accuracy: 0.8116

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5038 - accuracy: 0.8107

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5014 - accuracy: 0.8116

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5072 - accuracy: 0.8108

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5048 - accuracy: 0.8133

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5011 - accuracy: 0.8150

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5023 - accuracy: 0.8149

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.5066 - accuracy: 0.8133

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5049 - accuracy: 0.8140

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5006 - accuracy: 0.8155

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5024 - accuracy: 0.8140

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.4984 - accuracy: 0.8161

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.4925 - accuracy: 0.8174

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.4905 - accuracy: 0.8173

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.4909 - accuracy: 0.8165

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.4896 - accuracy: 0.8171

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.4898 - accuracy: 0.8170

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.4902 - accuracy: 0.8163

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.4979 - accuracy: 0.8131

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.4944 - accuracy: 0.8155

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.4957 - accuracy: 0.8149

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.4978 - accuracy: 0.8137

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5011 - accuracy: 0.8114

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5001 - accuracy: 0.8119

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5007 - accuracy: 0.8114

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.4989 - accuracy: 0.8120

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.4995 - accuracy: 0.8114

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.4980 - accuracy: 0.8130

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.4998 - accuracy: 0.8130

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5034 - accuracy: 0.8120

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5046 - accuracy: 0.8115

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5087 - accuracy: 0.8091

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5083 - accuracy: 0.8096

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5101 - accuracy: 0.8081

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5102 - accuracy: 0.8081

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5135 - accuracy: 0.8059

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5148 - accuracy: 0.8047

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5175 - accuracy: 0.8048

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5148 - accuracy: 0.8057

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5148 - accuracy: 0.8054

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5139 - accuracy: 0.8059

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5114 - accuracy: 0.8073

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5127 - accuracy: 0.8073

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5141 - accuracy: 0.8050

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5172 - accuracy: 0.8043

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5145 - accuracy: 0.8056

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5121 - accuracy: 0.8064

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5125 - accuracy: 0.8065

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5121 - accuracy: 0.8062

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5113 - accuracy: 0.8078

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5138 - accuracy: 0.8078

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5138 - accuracy: 0.8075

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5155 - accuracy: 0.8079

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5141 - accuracy: 0.8076

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5125 - accuracy: 0.8084

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5126 - accuracy: 0.8077

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5144 - accuracy: 0.8075

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5161 - accuracy: 0.8072

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5170 - accuracy: 0.8076

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.5170 - accuracy: 0.8076 - val_loss: 0.7975 - val_accuracy: 0.7207


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
   1/1 [==============================] - 0s 78ms/step


.. parsed-literal::

    This image most likely belongs to sunflowers with a 79.79 percent confidence.


Save the TensorFlow Model
-------------------------



.. code:: ipython3

    #save the trained model - a new folder flower will be created
    #and the file "saved_model.pb" is the pre-trained model
    model_dir = "model"
    saved_model_dir = f"{model_dir}/flower/saved_model"
    model.save(saved_model_dir)


.. parsed-literal::

    2024-01-19 00:33:01.483101: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'random_flip_input' with dtype float and shape [?,180,180,3]
    	 [[{{node random_flip_input}}]]
    2024-01-19 00:33:01.569356: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-01-19 00:33:01.579464: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'random_flip_input' with dtype float and shape [?,180,180,3]
    	 [[{{node random_flip_input}}]]
    2024-01-19 00:33:01.590590: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-01-19 00:33:01.597408: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-01-19 00:33:01.604146: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-01-19 00:33:01.614913: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-01-19 00:33:01.654056: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'sequential_1_input' with dtype float and shape [?,180,180,3]
    	 [[{{node sequential_1_input}}]]


.. parsed-literal::

    2024-01-19 00:33:01.720868: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-01-19 00:33:01.741100: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'sequential_1_input' with dtype float and shape [?,180,180,3]
    	 [[{{node sequential_1_input}}]]
    2024-01-19 00:33:01.780037: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,22,22,64]
    	 [[{{node inputs}}]]
    2024-01-19 00:33:01.804995: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-01-19 00:33:01.877797: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]


.. parsed-literal::

    2024-01-19 00:33:02.021425: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-01-19 00:33:02.159309: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,22,22,64]
    	 [[{{node inputs}}]]
    2024-01-19 00:33:02.193674: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-01-19 00:33:02.221410: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]


.. parsed-literal::

    2024-01-19 00:33:02.268517: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
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
    This image most likely belongs to dandelion with a 96.64 percent confidence.



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
