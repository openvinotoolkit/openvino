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

    import platform

    %pip install -q "openvino>=2023.1.0" "pillow"
    if platform.system() != "Windows":
        %pip install -q "matplotlib>=3.4"
    else:
        %pip install -q "matplotlib>=3.4,<3.7"
    %pip install -q "tensorflow-macos>=2.5; sys_platform == 'darwin' and platform_machine == 'arm64' and python_version > '3.8'" # macOS M1 and M2
    %pip install -q "tensorflow-macos>=2.5,<=2.12.0; sys_platform == 'darwin' and platform_machine == 'arm64' and python_version <= '3.8'" # macOS M1 and M2
    %pip install -q "tensorflow>=2.5; sys_platform == 'darwin' and platform_machine != 'arm64' and python_version > '3.8'" # macOS x86
    %pip install -q "tensorflow>=2.5,<=2.12.0; sys_platform == 'darwin' and platform_machine != 'arm64' and python_version <= '3.8'" # macOS x86
    %pip install -q "tensorflow>=2.5; sys_platform != 'darwin' and python_version > '3.8'"
    %pip install -q "tensorflow>=2.5,<=2.12.0; sys_platform != 'darwin' and python_version <= '3.8'"
    %pip install -q tf_keras


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


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

    os.environ["TF_USE_LEGACY_KERAS"] = "1"

    import PIL
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    from PIL import Image
    import openvino as ov

    sys.path.append("../utils")
    from notebook_utils import download_file


.. parsed-literal::

    2024-03-26 00:50:17.842006: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-03-26 00:50:17.877129: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


.. parsed-literal::

    2024-03-26 00:50:18.408118: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


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

    2024-03-26 00:50:21.463087: E tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:266] failed call to cuInit: CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: forward compatibility was attempted on non supported HW
    2024-03-26 00:50:21.463124: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:168] retrieving CUDA diagnostic information for host: iotg-dev-workstation-07
    2024-03-26 00:50:21.463129: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:175] hostname: iotg-dev-workstation-07
    2024-03-26 00:50:21.463280: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:199] libcuda reported version is: 470.223.2
    2024-03-26 00:50:21.463298: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:203] kernel reported version is: 470.182.3
    2024-03-26 00:50:21.463301: E tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:312] kernel version 470.182.3 does not match DSO version 470.223.2 -- cannot find working devices in this configuration


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

    2024-03-26 00:50:21.820556: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-03-26 00:50:21.821002: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]



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

    2024-03-26 00:50:22.652480: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-03-26 00:50:22.652838: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]


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

    normalization_layer = tf.keras.layers.Rescaling(1./255)

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

    2024-03-26 00:50:22.860632: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-03-26 00:50:22.861210: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]


.. parsed-literal::

    0.0 0.8430284


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

    model = tf.keras.Sequential([
      tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
      tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(num_classes)
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

    data_augmentation = tf.keras.Sequential(
      [
        tf.keras.layers.RandomFlip("horizontal",
                          input_shape=(img_height,
                                       img_width,
                                       3)),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
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

    2024-03-26 00:50:23.745614: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]
    2024-03-26 00:50:23.746353: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
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

    model = tf.keras.Sequential([
        data_augmentation,
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, name="outputs")
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

    2024-03-26 00:50:24.898708: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]
    2024-03-26 00:50:24.899131: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]


.. parsed-literal::


 1/92 [..............................] - ETA: 1:28 - loss: 1.6484 - accuracy: 0.1562

.. parsed-literal::

    
 2/92 [..............................] - ETA: 6s - loss: 2.1134 - accuracy: 0.1719

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 2.0551 - accuracy: 0.1979

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 1.9821 - accuracy: 0.1875

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 1.9178 - accuracy: 0.2125

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 1.8789 - accuracy: 0.2083

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 5s - loss: 1.8331 - accuracy: 0.2054

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 5s - loss: 1.8104 - accuracy: 0.2188

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 1.7853 - accuracy: 0.2292

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 1.7660 - accuracy: 0.2281

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 1.7475 - accuracy: 0.2301

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 1.7340 - accuracy: 0.2266

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 1.7185 - accuracy: 0.2308

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 1.7093 - accuracy: 0.2277

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 1.7024 - accuracy: 0.2250

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 1.6931 - accuracy: 0.2285

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 1.6861 - accuracy: 0.2261

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 1.6774 - accuracy: 0.2240

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 1.6686 - accuracy: 0.2319

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 1.6593 - accuracy: 0.2484

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 1.6491 - accuracy: 0.2560

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 1.6411 - accuracy: 0.2614

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 1.6354 - accuracy: 0.2609

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 1.6285 - accuracy: 0.2630

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 1.6182 - accuracy: 0.2675

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 1.6077 - accuracy: 0.2716

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 1.6013 - accuracy: 0.2708

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 1.5909 - accuracy: 0.2734

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 1.5805 - accuracy: 0.2812

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 1.5713 - accuracy: 0.2854

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 1.5618 - accuracy: 0.2903

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 1.5606 - accuracy: 0.2881

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 1.5533 - accuracy: 0.2917

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 1.5410 - accuracy: 0.2978

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 1.5302 - accuracy: 0.3089

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 1.5209 - accuracy: 0.3134

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 1.5141 - accuracy: 0.3184

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 1.5135 - accuracy: 0.3207

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 1.5090 - accuracy: 0.3253

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 1.5047 - accuracy: 0.3281

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 1.4977 - accuracy: 0.3331

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 1.4888 - accuracy: 0.3378

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 1.4828 - accuracy: 0.3416

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 1.4780 - accuracy: 0.3452

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 1.4745 - accuracy: 0.3458

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 1.4741 - accuracy: 0.3465

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 1.4738 - accuracy: 0.3484

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 1.4749 - accuracy: 0.3477

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 1.4720 - accuracy: 0.3489

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 1.4686 - accuracy: 0.3531

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 1.4636 - accuracy: 0.3560

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 1.4573 - accuracy: 0.3570

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 1.4537 - accuracy: 0.3603

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 1.4481 - accuracy: 0.3628

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 1.4451 - accuracy: 0.3653

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 1.4407 - accuracy: 0.3666

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 1.4397 - accuracy: 0.3673

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 1.4377 - accuracy: 0.3707

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 1.4346 - accuracy: 0.3734

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 1.4298 - accuracy: 0.3750

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 1.4228 - accuracy: 0.3770

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 1.4186 - accuracy: 0.3765

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 1.4128 - accuracy: 0.3795

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 1.4101 - accuracy: 0.3794

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 1.4069 - accuracy: 0.3817

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 1.3991 - accuracy: 0.3854

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 1.3931 - accuracy: 0.3885

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 1.3907 - accuracy: 0.3906

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 1.3893 - accuracy: 0.3913

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 1.3867 - accuracy: 0.3920

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 1.3822 - accuracy: 0.3930

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 1.3779 - accuracy: 0.3950

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 1.3760 - accuracy: 0.3947

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 1.3763 - accuracy: 0.3970

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 1.3775 - accuracy: 0.3979

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 1.3753 - accuracy: 0.4009

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 1.3721 - accuracy: 0.4018

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 1.3693 - accuracy: 0.4022

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 1.3644 - accuracy: 0.4051

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 1.3620 - accuracy: 0.4067

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 1.3578 - accuracy: 0.4083

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 1.3536 - accuracy: 0.4113

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 1.3505 - accuracy: 0.4131

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 1.3472 - accuracy: 0.4153

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 1.3443 - accuracy: 0.4167

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 1.3402 - accuracy: 0.4180

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 1.3377 - accuracy: 0.4186

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 1.3342 - accuracy: 0.4206

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 1.3325 - accuracy: 0.4222

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 1.3309 - accuracy: 0.4244

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 1.3314 - accuracy: 0.4229

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 1.3307 - accuracy: 0.4247

.. parsed-literal::

    2024-03-26 00:50:31.159874: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [734]
    	 [[{{node Placeholder/_4}}]]
    2024-03-26 00:50:31.160195: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [734]
    	 [[{{node Placeholder/_0}}]]


.. parsed-literal::

    
92/92 [==============================] - 7s 66ms/step - loss: 1.3307 - accuracy: 0.4247 - val_loss: 1.1092 - val_accuracy: 0.5341


.. parsed-literal::

    Epoch 2/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.9461 - accuracy: 0.6875

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.9892 - accuracy: 0.6562

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 1.0266 - accuracy: 0.6458

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 1.0735 - accuracy: 0.6406

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 1.1036 - accuracy: 0.5938

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 1.0757 - accuracy: 0.5938

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 1.0544 - accuracy: 0.6027

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 1.0392 - accuracy: 0.6172

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 1.0338 - accuracy: 0.6250

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 1.0705 - accuracy: 0.6000

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 1.0623 - accuracy: 0.5994

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 1.0565 - accuracy: 0.5938

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 1.0558 - accuracy: 0.5865

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 1.0509 - accuracy: 0.5938

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 1.0399 - accuracy: 0.6012

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 1.0628 - accuracy: 0.5858

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 1.0614 - accuracy: 0.5880

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 1.0551 - accuracy: 0.5867

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 1.0607 - accuracy: 0.5854

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 1.0575 - accuracy: 0.5843

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 1.0538 - accuracy: 0.5848

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 1.0485 - accuracy: 0.5852

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 1.0541 - accuracy: 0.5776

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 1.0575 - accuracy: 0.5770

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 1.0630 - accuracy: 0.5752

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 1.0703 - accuracy: 0.5724

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 1.0801 - accuracy: 0.5653

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 1.0788 - accuracy: 0.5641

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 1.0706 - accuracy: 0.5714

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 1.0651 - accuracy: 0.5762

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 1.0579 - accuracy: 0.5787

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 1.0561 - accuracy: 0.5802

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 1.0563 - accuracy: 0.5778

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 1.0578 - accuracy: 0.5755

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 1.0565 - accuracy: 0.5795

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 1.0598 - accuracy: 0.5782

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 1.0554 - accuracy: 0.5795

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 1.0584 - accuracy: 0.5758

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 1.0603 - accuracy: 0.5763

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 1.0637 - accuracy: 0.5759

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 1.0621 - accuracy: 0.5778

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 1.0576 - accuracy: 0.5789

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 1.0594 - accuracy: 0.5786

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 1.0588 - accuracy: 0.5782

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 1.0549 - accuracy: 0.5792

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 1.0522 - accuracy: 0.5802

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 1.0454 - accuracy: 0.5857

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 1.0445 - accuracy: 0.5865

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 1.0397 - accuracy: 0.5886

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 1.0467 - accuracy: 0.5856

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 1.0485 - accuracy: 0.5839

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 1.0440 - accuracy: 0.5871

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 1.0450 - accuracy: 0.5890

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 1.0424 - accuracy: 0.5902

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 1.0423 - accuracy: 0.5891

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 1.0444 - accuracy: 0.5898

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 1.0445 - accuracy: 0.5915

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 1.0421 - accuracy: 0.5931

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 1.0480 - accuracy: 0.5915

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 1.0467 - accuracy: 0.5921

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 1.0461 - accuracy: 0.5931

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 1.0467 - accuracy: 0.5941

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 1.0472 - accuracy: 0.5946

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 1.0466 - accuracy: 0.5975

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 1.0462 - accuracy: 0.5965

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 1.0452 - accuracy: 0.5960

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 1.0441 - accuracy: 0.5969

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 1.0439 - accuracy: 0.5973

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 1.0459 - accuracy: 0.5950

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 1.0477 - accuracy: 0.5941

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 1.0442 - accuracy: 0.5954

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 1.0449 - accuracy: 0.5949

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 1.0420 - accuracy: 0.5953

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 1.0398 - accuracy: 0.5966

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 1.0392 - accuracy: 0.5978

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 1.0417 - accuracy: 0.5969

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 1.0423 - accuracy: 0.5969

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 1.0423 - accuracy: 0.5960

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 1.0398 - accuracy: 0.5964

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 1.0421 - accuracy: 0.5964

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 1.0428 - accuracy: 0.5956

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 1.0453 - accuracy: 0.5937

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 1.0459 - accuracy: 0.5925

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 1.0466 - accuracy: 0.5926

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 1.0447 - accuracy: 0.5937

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 1.0449 - accuracy: 0.5933

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 1.0483 - accuracy: 0.5912

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 1.0471 - accuracy: 0.5915

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 1.0473 - accuracy: 0.5909

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 1.0472 - accuracy: 0.5913

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 1.0468 - accuracy: 0.5916

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 1.0468 - accuracy: 0.5916 - val_loss: 0.9766 - val_accuracy: 0.6240


.. parsed-literal::

    Epoch 3/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 1.3307 - accuracy: 0.4688

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 1.2158 - accuracy: 0.5156

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 1.0995 - accuracy: 0.5729

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 1.0678 - accuracy: 0.5781

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 1.0514 - accuracy: 0.5938

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 1.0371 - accuracy: 0.6146

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 1.0182 - accuracy: 0.6116

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.9948 - accuracy: 0.6172

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.9778 - accuracy: 0.6354

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.9767 - accuracy: 0.6250

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.9680 - accuracy: 0.6278

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.9518 - accuracy: 0.6328

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.9512 - accuracy: 0.6346

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.9617 - accuracy: 0.6362

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.9557 - accuracy: 0.6375

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.9556 - accuracy: 0.6426

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.9609 - accuracy: 0.6397

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.9557 - accuracy: 0.6372

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.9592 - accuracy: 0.6398

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.9531 - accuracy: 0.6453

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.9538 - accuracy: 0.6458

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.9444 - accuracy: 0.6506

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.9451 - accuracy: 0.6495

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.9420 - accuracy: 0.6510

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.9450 - accuracy: 0.6488

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.9449 - accuracy: 0.6454

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.9454 - accuracy: 0.6470

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.9408 - accuracy: 0.6518

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.9476 - accuracy: 0.6476

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.9519 - accuracy: 0.6458

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.9495 - accuracy: 0.6472

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.9419 - accuracy: 0.6494

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.9358 - accuracy: 0.6515

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.9322 - accuracy: 0.6526

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.9306 - accuracy: 0.6536

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.9321 - accuracy: 0.6536

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.9356 - accuracy: 0.6503

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.9371 - accuracy: 0.6513

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.9325 - accuracy: 0.6538

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.9372 - accuracy: 0.6523

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.9392 - accuracy: 0.6502

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.9373 - accuracy: 0.6518

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.9385 - accuracy: 0.6504

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.9413 - accuracy: 0.6491

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.9386 - accuracy: 0.6479

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.9402 - accuracy: 0.6474

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.9386 - accuracy: 0.6483

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.9375 - accuracy: 0.6478

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.9353 - accuracy: 0.6501

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.9358 - accuracy: 0.6490

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.9345 - accuracy: 0.6492

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.9370 - accuracy: 0.6469

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.9312 - accuracy: 0.6494

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.9311 - accuracy: 0.6484

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.9288 - accuracy: 0.6497

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.9267 - accuracy: 0.6503

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.9279 - accuracy: 0.6499

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.9368 - accuracy: 0.6463

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.9391 - accuracy: 0.6444

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.9356 - accuracy: 0.6456

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.9380 - accuracy: 0.6442

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.9357 - accuracy: 0.6444

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.9347 - accuracy: 0.6441

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.9329 - accuracy: 0.6448

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.9329 - accuracy: 0.6440

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.9329 - accuracy: 0.6442

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.9350 - accuracy: 0.6439

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.9332 - accuracy: 0.6441

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.9334 - accuracy: 0.6452

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.9367 - accuracy: 0.6449

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.9340 - accuracy: 0.6468

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.9351 - accuracy: 0.6465

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.9339 - accuracy: 0.6470

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.9341 - accuracy: 0.6467

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.9345 - accuracy: 0.6465

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.9336 - accuracy: 0.6458

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.9375 - accuracy: 0.6447

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.9331 - accuracy: 0.6464

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.9338 - accuracy: 0.6458

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.9315 - accuracy: 0.6467

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.9299 - accuracy: 0.6468

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.9286 - accuracy: 0.6477

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.9263 - accuracy: 0.6485

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.9253 - accuracy: 0.6482

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.9259 - accuracy: 0.6476

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.9280 - accuracy: 0.6459

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.9311 - accuracy: 0.6435

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.9309 - accuracy: 0.6433

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.9300 - accuracy: 0.6428

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.9286 - accuracy: 0.6433

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.9279 - accuracy: 0.6420

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.9279 - accuracy: 0.6420 - val_loss: 0.9844 - val_accuracy: 0.6144


.. parsed-literal::

    Epoch 4/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 1.2800 - accuracy: 0.4688

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 1.1636 - accuracy: 0.5312

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 1.0874 - accuracy: 0.5833

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 1.0080 - accuracy: 0.6016

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 1.0324 - accuracy: 0.6000

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.9895 - accuracy: 0.6042

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.9674 - accuracy: 0.6250

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.9439 - accuracy: 0.6289

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.9207 - accuracy: 0.6354

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.9032 - accuracy: 0.6406

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.8865 - accuracy: 0.6534

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.8950 - accuracy: 0.6510

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.9139 - accuracy: 0.6394

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.9159 - accuracy: 0.6362

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.9134 - accuracy: 0.6375

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.9139 - accuracy: 0.6328

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.9169 - accuracy: 0.6305

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.9145 - accuracy: 0.6337

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.9166 - accuracy: 0.6316

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.9160 - accuracy: 0.6344

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.9126 - accuracy: 0.6369

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.9027 - accuracy: 0.6449

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.9024 - accuracy: 0.6440

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.8967 - accuracy: 0.6484

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.8957 - accuracy: 0.6500

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.8896 - accuracy: 0.6514

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.8870 - accuracy: 0.6493

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.8796 - accuracy: 0.6529

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.8833 - accuracy: 0.6498

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.8857 - accuracy: 0.6469

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.8827 - accuracy: 0.6482

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.8716 - accuracy: 0.6555

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.8787 - accuracy: 0.6537

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.8940 - accuracy: 0.6475

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.8958 - accuracy: 0.6469

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.9041 - accuracy: 0.6420

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.9127 - accuracy: 0.6374

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.9099 - accuracy: 0.6379

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.9036 - accuracy: 0.6399

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.9045 - accuracy: 0.6396

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.9022 - accuracy: 0.6392

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.9028 - accuracy: 0.6389

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.9029 - accuracy: 0.6386

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.9030 - accuracy: 0.6397

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.9018 - accuracy: 0.6400

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.9041 - accuracy: 0.6390

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.9118 - accuracy: 0.6374

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.9102 - accuracy: 0.6365

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.9084 - accuracy: 0.6394

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.9048 - accuracy: 0.6429

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.9068 - accuracy: 0.6419

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.9066 - accuracy: 0.6428

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.9033 - accuracy: 0.6465

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.9074 - accuracy: 0.6433

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.9110 - accuracy: 0.6418

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.9087 - accuracy: 0.6443

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.9085 - accuracy: 0.6450

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.9085 - accuracy: 0.6447

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.9066 - accuracy: 0.6470

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.9065 - accuracy: 0.6466

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.9096 - accuracy: 0.6442

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.9101 - accuracy: 0.6454

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.9099 - accuracy: 0.6456

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.9116 - accuracy: 0.6462

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.9110 - accuracy: 0.6459

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.9092 - accuracy: 0.6470

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.9061 - accuracy: 0.6490

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.9035 - accuracy: 0.6495

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.9017 - accuracy: 0.6510

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.9029 - accuracy: 0.6506

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.9019 - accuracy: 0.6511

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.8982 - accuracy: 0.6525

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.8991 - accuracy: 0.6542

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.8972 - accuracy: 0.6555

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.8981 - accuracy: 0.6543

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.8987 - accuracy: 0.6539

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.9011 - accuracy: 0.6535

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.8990 - accuracy: 0.6548

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.8998 - accuracy: 0.6544

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.8960 - accuracy: 0.6571

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.8954 - accuracy: 0.6571

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.8975 - accuracy: 0.6560

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.8963 - accuracy: 0.6563

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.8979 - accuracy: 0.6571

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.8935 - accuracy: 0.6600

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.8936 - accuracy: 0.6599

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.8922 - accuracy: 0.6606

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.8895 - accuracy: 0.6613

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.8899 - accuracy: 0.6612

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.8906 - accuracy: 0.6615

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.8915 - accuracy: 0.6611

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.8915 - accuracy: 0.6611 - val_loss: 0.8685 - val_accuracy: 0.6594


.. parsed-literal::

    Epoch 5/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.6932 - accuracy: 0.6875

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.8816 - accuracy: 0.6875

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.8848 - accuracy: 0.6562

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.8206 - accuracy: 0.6953

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.8488 - accuracy: 0.6812

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.8538 - accuracy: 0.6771

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.8381 - accuracy: 0.6830

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.8435 - accuracy: 0.6914

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.8396 - accuracy: 0.6910

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.8538 - accuracy: 0.6781

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.8590 - accuracy: 0.6790

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.8613 - accuracy: 0.6745

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.8805 - accuracy: 0.6659

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.8919 - accuracy: 0.6540

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.8723 - accuracy: 0.6646

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.8682 - accuracy: 0.6660

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.8521 - accuracy: 0.6710

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.8478 - accuracy: 0.6719

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.8365 - accuracy: 0.6793

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.8293 - accuracy: 0.6844

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.8262 - accuracy: 0.6830

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.8231 - accuracy: 0.6832

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.8289 - accuracy: 0.6793

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.8204 - accuracy: 0.6849

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.8231 - accuracy: 0.6837

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.8263 - accuracy: 0.6839

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.8254 - accuracy: 0.6852

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.8260 - accuracy: 0.6830

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.8211 - accuracy: 0.6843

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.8329 - accuracy: 0.6771

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.8356 - accuracy: 0.6764

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.8360 - accuracy: 0.6777

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.8349 - accuracy: 0.6799

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.8278 - accuracy: 0.6829

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.8324 - accuracy: 0.6821

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.8377 - accuracy: 0.6832

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.8366 - accuracy: 0.6841

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.8373 - accuracy: 0.6850

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.8305 - accuracy: 0.6899

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.8309 - accuracy: 0.6891

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.8264 - accuracy: 0.6905

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.8243 - accuracy: 0.6920

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.8204 - accuracy: 0.6940

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.8206 - accuracy: 0.6953

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.8248 - accuracy: 0.6924

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.8263 - accuracy: 0.6929

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.8295 - accuracy: 0.6902

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.8266 - accuracy: 0.6901

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.8238 - accuracy: 0.6920

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.8219 - accuracy: 0.6925

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.8292 - accuracy: 0.6906

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.8291 - accuracy: 0.6911

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.8271 - accuracy: 0.6934

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.8245 - accuracy: 0.6927

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.8262 - accuracy: 0.6909

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.8217 - accuracy: 0.6920

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.8216 - accuracy: 0.6924

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.8273 - accuracy: 0.6897

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.8275 - accuracy: 0.6886

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.8241 - accuracy: 0.6896

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.8262 - accuracy: 0.6893

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.8250 - accuracy: 0.6897

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.8234 - accuracy: 0.6907

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.8230 - accuracy: 0.6902

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.8221 - accuracy: 0.6892

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.8253 - accuracy: 0.6877

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.8303 - accuracy: 0.6859

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.8292 - accuracy: 0.6868

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.8295 - accuracy: 0.6868

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.8283 - accuracy: 0.6882

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.8282 - accuracy: 0.6877

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.8285 - accuracy: 0.6860

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.8306 - accuracy: 0.6847

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.8327 - accuracy: 0.6856

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.8331 - accuracy: 0.6836

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.8350 - accuracy: 0.6828

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.8311 - accuracy: 0.6833

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.8335 - accuracy: 0.6829

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.8317 - accuracy: 0.6842

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.8310 - accuracy: 0.6838

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.8311 - accuracy: 0.6835

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.8313 - accuracy: 0.6828

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.8320 - accuracy: 0.6821

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.8305 - accuracy: 0.6822

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.8298 - accuracy: 0.6829

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.8286 - accuracy: 0.6834

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.8283 - accuracy: 0.6838

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.8271 - accuracy: 0.6849

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.8295 - accuracy: 0.6849

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.8277 - accuracy: 0.6856

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.8275 - accuracy: 0.6873

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.8275 - accuracy: 0.6873 - val_loss: 0.8407 - val_accuracy: 0.6580


.. parsed-literal::

    Epoch 6/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.7882 - accuracy: 0.7500

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.7719 - accuracy: 0.7344

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.8159 - accuracy: 0.7188

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.7549 - accuracy: 0.7422

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 4s - loss: 0.7784 - accuracy: 0.7250

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.7792 - accuracy: 0.7240

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.8406 - accuracy: 0.6741

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.8102 - accuracy: 0.6914

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.8059 - accuracy: 0.6979

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.8040 - accuracy: 0.7063

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.7988 - accuracy: 0.7017

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.7908 - accuracy: 0.7083

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.7813 - accuracy: 0.7067

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.7916 - accuracy: 0.6987

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.7892 - accuracy: 0.6958

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.7844 - accuracy: 0.7031

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.7688 - accuracy: 0.7096

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.7609 - accuracy: 0.7135

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.7624 - accuracy: 0.7105

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.7626 - accuracy: 0.7094

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.7614 - accuracy: 0.7113

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.7593 - accuracy: 0.7088

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.7717 - accuracy: 0.7052

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.7790 - accuracy: 0.6979

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.7743 - accuracy: 0.6988

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.7675 - accuracy: 0.7007

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.7617 - accuracy: 0.7037

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.7616 - accuracy: 0.7031

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.7532 - accuracy: 0.7058

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.7544 - accuracy: 0.7052

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.7475 - accuracy: 0.7077

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.7606 - accuracy: 0.7051

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.7661 - accuracy: 0.7027

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.7676 - accuracy: 0.7004

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.7767 - accuracy: 0.6964

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.7773 - accuracy: 0.6997

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.7791 - accuracy: 0.6985

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.7765 - accuracy: 0.7015

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.7797 - accuracy: 0.6995

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.7812 - accuracy: 0.6992

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.7829 - accuracy: 0.7005

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.7817 - accuracy: 0.7016

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.7785 - accuracy: 0.7042

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.7829 - accuracy: 0.7031

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.7810 - accuracy: 0.7035

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.7819 - accuracy: 0.7018

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.7783 - accuracy: 0.7028

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.7800 - accuracy: 0.7012

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.7771 - accuracy: 0.7022

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.7720 - accuracy: 0.7050

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.7759 - accuracy: 0.7010

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.7760 - accuracy: 0.7019

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.7731 - accuracy: 0.7046

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.7727 - accuracy: 0.7043

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.7768 - accuracy: 0.7046

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.7768 - accuracy: 0.7043

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.7764 - accuracy: 0.7051

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.7804 - accuracy: 0.7037

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.7771 - accuracy: 0.7050

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.7754 - accuracy: 0.7063

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.7782 - accuracy: 0.7039

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.7781 - accuracy: 0.7052

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.7774 - accuracy: 0.7059

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.7804 - accuracy: 0.7056

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.7825 - accuracy: 0.7044

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.7816 - accuracy: 0.7037

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.7815 - accuracy: 0.7030

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.7810 - accuracy: 0.7018

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.7794 - accuracy: 0.7021

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.7830 - accuracy: 0.7019

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.7829 - accuracy: 0.7017

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.7801 - accuracy: 0.7019

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.7783 - accuracy: 0.7034

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.7828 - accuracy: 0.7003

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.7838 - accuracy: 0.6988

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.7845 - accuracy: 0.6983

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.7841 - accuracy: 0.6982

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.7820 - accuracy: 0.6984

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.7784 - accuracy: 0.6998

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.7769 - accuracy: 0.7012

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.7771 - accuracy: 0.7011

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.7765 - accuracy: 0.7017

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.7748 - accuracy: 0.7034

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.7752 - accuracy: 0.7028

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.7746 - accuracy: 0.7034

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.7754 - accuracy: 0.7035

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.7730 - accuracy: 0.7044

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.7703 - accuracy: 0.7056

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.7684 - accuracy: 0.7061

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.7685 - accuracy: 0.7059

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.7660 - accuracy: 0.7074

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.7660 - accuracy: 0.7074 - val_loss: 0.8043 - val_accuracy: 0.6730


.. parsed-literal::

    Epoch 7/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 1.1985 - accuracy: 0.4688

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.9185 - accuracy: 0.6719

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.8828 - accuracy: 0.6562

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.7929 - accuracy: 0.6875

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 4s - loss: 0.7598 - accuracy: 0.7125

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.7768 - accuracy: 0.7083

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.7301 - accuracy: 0.7321

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.7394 - accuracy: 0.7148

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.7574 - accuracy: 0.7083

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.7430 - accuracy: 0.7188

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.7547 - accuracy: 0.7074

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.7472 - accuracy: 0.7109

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.7435 - accuracy: 0.7115

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.7621 - accuracy: 0.6964

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.7714 - accuracy: 0.6917

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.7825 - accuracy: 0.6855

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.7780 - accuracy: 0.6893

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.7760 - accuracy: 0.6892

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.7763 - accuracy: 0.6891

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.7808 - accuracy: 0.6844

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.7800 - accuracy: 0.6905

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.7932 - accuracy: 0.6847

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.7873 - accuracy: 0.6861

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.7853 - accuracy: 0.6849

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.7825 - accuracy: 0.6862

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.7772 - accuracy: 0.6887

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.7864 - accuracy: 0.6852

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.7842 - accuracy: 0.6853

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.7779 - accuracy: 0.6897

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.7767 - accuracy: 0.6917

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.7769 - accuracy: 0.6915

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.7732 - accuracy: 0.6914

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.7708 - accuracy: 0.6913

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.7760 - accuracy: 0.6921

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.7773 - accuracy: 0.6929

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.7760 - accuracy: 0.6927

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.7742 - accuracy: 0.6934

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.7745 - accuracy: 0.6941

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.7744 - accuracy: 0.6971

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.7693 - accuracy: 0.6977

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.7653 - accuracy: 0.6989

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.7641 - accuracy: 0.7009

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.7710 - accuracy: 0.6977

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.7742 - accuracy: 0.6974

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.7711 - accuracy: 0.6986

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.7680 - accuracy: 0.6997

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.7660 - accuracy: 0.7001

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.7606 - accuracy: 0.7031

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.7565 - accuracy: 0.7047

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.7609 - accuracy: 0.7019

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.7578 - accuracy: 0.7022

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.7585 - accuracy: 0.7031

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.7564 - accuracy: 0.7034

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.7538 - accuracy: 0.7049

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.7550 - accuracy: 0.7045

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.7543 - accuracy: 0.7054

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.7517 - accuracy: 0.7056

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.7523 - accuracy: 0.7047

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.7564 - accuracy: 0.7029

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.7545 - accuracy: 0.7036

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.7532 - accuracy: 0.7029

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.7550 - accuracy: 0.7011

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.7543 - accuracy: 0.7019

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.7515 - accuracy: 0.7041

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.7529 - accuracy: 0.7048

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.7530 - accuracy: 0.7045

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.7498 - accuracy: 0.7057

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.7492 - accuracy: 0.7073

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.7452 - accuracy: 0.7101

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.7459 - accuracy: 0.7089

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.7449 - accuracy: 0.7095

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.7417 - accuracy: 0.7114

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.7411 - accuracy: 0.7110

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.7403 - accuracy: 0.7103

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.7394 - accuracy: 0.7113

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.7396 - accuracy: 0.7109

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.7412 - accuracy: 0.7110

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.7448 - accuracy: 0.7099

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.7438 - accuracy: 0.7104

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.7448 - accuracy: 0.7101

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.7446 - accuracy: 0.7114

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.7434 - accuracy: 0.7122

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.7422 - accuracy: 0.7127

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.7447 - accuracy: 0.7124

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.7437 - accuracy: 0.7128

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.7449 - accuracy: 0.7122

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.7448 - accuracy: 0.7112

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.7439 - accuracy: 0.7120

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.7444 - accuracy: 0.7124

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.7442 - accuracy: 0.7121

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.7428 - accuracy: 0.7125

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.7428 - accuracy: 0.7125 - val_loss: 0.8436 - val_accuracy: 0.6635


.. parsed-literal::

    Epoch 8/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.5652 - accuracy: 0.7500

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5497 - accuracy: 0.7812

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6242 - accuracy: 0.7708

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6359 - accuracy: 0.7734

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.7140 - accuracy: 0.7500

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.7034 - accuracy: 0.7604

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.7016 - accuracy: 0.7589

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.7208 - accuracy: 0.7461

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.7424 - accuracy: 0.7361

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.7369 - accuracy: 0.7406

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.7381 - accuracy: 0.7386

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.7288 - accuracy: 0.7396

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.7209 - accuracy: 0.7428

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.7149 - accuracy: 0.7411

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.7053 - accuracy: 0.7437

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.7179 - accuracy: 0.7402

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.7203 - accuracy: 0.7390

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.7223 - accuracy: 0.7378

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.7183 - accuracy: 0.7352

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.7141 - accuracy: 0.7319

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 3s - loss: 0.7074 - accuracy: 0.7356

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.7059 - accuracy: 0.7349

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.7081 - accuracy: 0.7342

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.7116 - accuracy: 0.7348

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.7086 - accuracy: 0.7379

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.7088 - accuracy: 0.7336

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.7088 - accuracy: 0.7342

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.7144 - accuracy: 0.7293

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.7085 - accuracy: 0.7311

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.7084 - accuracy: 0.7317

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.7051 - accuracy: 0.7343

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.7025 - accuracy: 0.7347

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.7024 - accuracy: 0.7343

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.7022 - accuracy: 0.7329

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.7089 - accuracy: 0.7299

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.7163 - accuracy: 0.7245

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.7113 - accuracy: 0.7260

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.7110 - accuracy: 0.7250

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.7140 - accuracy: 0.7241

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.7143 - accuracy: 0.7232

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.7181 - accuracy: 0.7216

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.7145 - accuracy: 0.7222

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.7127 - accuracy: 0.7214

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.7132 - accuracy: 0.7214

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.7122 - accuracy: 0.7227

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.7083 - accuracy: 0.7246

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.7066 - accuracy: 0.7245

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.7063 - accuracy: 0.7256

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.7046 - accuracy: 0.7255

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.7086 - accuracy: 0.7241

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.7090 - accuracy: 0.7246

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.7090 - accuracy: 0.7245

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.7083 - accuracy: 0.7250

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.7071 - accuracy: 0.7255

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.7077 - accuracy: 0.7253

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.7054 - accuracy: 0.7263

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.7010 - accuracy: 0.7284

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.7041 - accuracy: 0.7282

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.7044 - accuracy: 0.7286

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.7021 - accuracy: 0.7299

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.7026 - accuracy: 0.7303

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.7062 - accuracy: 0.7276

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.7095 - accuracy: 0.7255

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.7084 - accuracy: 0.7264

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.7142 - accuracy: 0.7224

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.7126 - accuracy: 0.7224

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.7128 - accuracy: 0.7228

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.7130 - accuracy: 0.7227

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.7122 - accuracy: 0.7236

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.7127 - accuracy: 0.7239

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.7094 - accuracy: 0.7256

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.7127 - accuracy: 0.7242

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.7127 - accuracy: 0.7246

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.7137 - accuracy: 0.7237

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.7123 - accuracy: 0.7244

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.7103 - accuracy: 0.7252

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.7080 - accuracy: 0.7263

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.7076 - accuracy: 0.7266

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.7069 - accuracy: 0.7277

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.7058 - accuracy: 0.7272

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.7081 - accuracy: 0.7274

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.7047 - accuracy: 0.7289

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.7043 - accuracy: 0.7302

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.7028 - accuracy: 0.7308

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.7036 - accuracy: 0.7307

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.7040 - accuracy: 0.7298

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.7038 - accuracy: 0.7301

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.7032 - accuracy: 0.7303

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6998 - accuracy: 0.7322

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6991 - accuracy: 0.7324

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6999 - accuracy: 0.7323

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.6999 - accuracy: 0.7323 - val_loss: 0.7656 - val_accuracy: 0.6962


.. parsed-literal::

    Epoch 9/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.4719 - accuracy: 0.8125

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.4584 - accuracy: 0.8281

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5582 - accuracy: 0.7917

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5620 - accuracy: 0.7812

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5813 - accuracy: 0.7688

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.6006 - accuracy: 0.7552

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6183 - accuracy: 0.7500

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5989 - accuracy: 0.7578

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5841 - accuracy: 0.7639

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6371 - accuracy: 0.7500

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6255 - accuracy: 0.7585

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6558 - accuracy: 0.7474

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6518 - accuracy: 0.7500

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6666 - accuracy: 0.7478

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6595 - accuracy: 0.7417

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6511 - accuracy: 0.7461

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6490 - accuracy: 0.7445

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6442 - accuracy: 0.7465

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6436 - accuracy: 0.7451

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6419 - accuracy: 0.7469

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6406 - accuracy: 0.7485

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6399 - accuracy: 0.7486

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.6386 - accuracy: 0.7486

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6381 - accuracy: 0.7513

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6418 - accuracy: 0.7462

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6521 - accuracy: 0.7404

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6526 - accuracy: 0.7419

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6533 - accuracy: 0.7433

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6500 - accuracy: 0.7468

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6476 - accuracy: 0.7469

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6458 - accuracy: 0.7480

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6556 - accuracy: 0.7471

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6494 - accuracy: 0.7509

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6543 - accuracy: 0.7500

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6508 - accuracy: 0.7509

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6458 - accuracy: 0.7526

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6468 - accuracy: 0.7517

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6479 - accuracy: 0.7525

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6497 - accuracy: 0.7532

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.6514 - accuracy: 0.7508

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6500 - accuracy: 0.7508

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6477 - accuracy: 0.7522

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6480 - accuracy: 0.7522

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6559 - accuracy: 0.7493

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6623 - accuracy: 0.7444

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6663 - accuracy: 0.7425

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6655 - accuracy: 0.7420

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6673 - accuracy: 0.7415

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6723 - accuracy: 0.7385

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6738 - accuracy: 0.7381

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6704 - accuracy: 0.7396

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6744 - accuracy: 0.7380

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6773 - accuracy: 0.7358

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6746 - accuracy: 0.7373

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6725 - accuracy: 0.7386

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6729 - accuracy: 0.7394

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6718 - accuracy: 0.7396

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6763 - accuracy: 0.7360

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6748 - accuracy: 0.7362

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6744 - accuracy: 0.7359

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6737 - accuracy: 0.7367

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6748 - accuracy: 0.7361

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6744 - accuracy: 0.7368

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6739 - accuracy: 0.7384

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6729 - accuracy: 0.7395

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6711 - accuracy: 0.7406

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6712 - accuracy: 0.7412

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6729 - accuracy: 0.7414

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6704 - accuracy: 0.7424

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6683 - accuracy: 0.7429

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6662 - accuracy: 0.7430

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6638 - accuracy: 0.7444

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6667 - accuracy: 0.7415

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6658 - accuracy: 0.7421

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6652 - accuracy: 0.7426

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6666 - accuracy: 0.7423

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6649 - accuracy: 0.7432

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6666 - accuracy: 0.7421

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6634 - accuracy: 0.7441

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6645 - accuracy: 0.7438

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6656 - accuracy: 0.7427

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6659 - accuracy: 0.7424

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6668 - accuracy: 0.7429

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6673 - accuracy: 0.7423

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6665 - accuracy: 0.7431

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6641 - accuracy: 0.7450

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6648 - accuracy: 0.7454

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6623 - accuracy: 0.7458

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6638 - accuracy: 0.7455

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6673 - accuracy: 0.7438

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6658 - accuracy: 0.7446

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.6658 - accuracy: 0.7446 - val_loss: 0.7885 - val_accuracy: 0.6989


.. parsed-literal::

    Epoch 10/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.7487 - accuracy: 0.8125

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.7439 - accuracy: 0.7812

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.7570 - accuracy: 0.7812

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.7614 - accuracy: 0.7578

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.7051 - accuracy: 0.7625

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.7064 - accuracy: 0.7500

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6823 - accuracy: 0.7589

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6873 - accuracy: 0.7539

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6839 - accuracy: 0.7465

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6830 - accuracy: 0.7531

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6952 - accuracy: 0.7415

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6834 - accuracy: 0.7422

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6874 - accuracy: 0.7452

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6774 - accuracy: 0.7433

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6757 - accuracy: 0.7417

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6730 - accuracy: 0.7402

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6594 - accuracy: 0.7445

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6504 - accuracy: 0.7483

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6422 - accuracy: 0.7516

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6474 - accuracy: 0.7516

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6352 - accuracy: 0.7574

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6269 - accuracy: 0.7599

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.6181 - accuracy: 0.7636

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6195 - accuracy: 0.7617

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6167 - accuracy: 0.7638

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6088 - accuracy: 0.7692

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6065 - accuracy: 0.7708

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6013 - accuracy: 0.7701

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6036 - accuracy: 0.7683

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5992 - accuracy: 0.7708

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6040 - accuracy: 0.7692

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6069 - accuracy: 0.7676

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6121 - accuracy: 0.7652

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6171 - accuracy: 0.7601

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6193 - accuracy: 0.7616

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6251 - accuracy: 0.7587

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6221 - accuracy: 0.7593

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6174 - accuracy: 0.7607

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6255 - accuracy: 0.7564

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.6274 - accuracy: 0.7555

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6293 - accuracy: 0.7553

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6303 - accuracy: 0.7545

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6315 - accuracy: 0.7544

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6355 - accuracy: 0.7514

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6387 - accuracy: 0.7500

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6413 - accuracy: 0.7514

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6405 - accuracy: 0.7533

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6415 - accuracy: 0.7526

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6409 - accuracy: 0.7526

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6397 - accuracy: 0.7531

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6366 - accuracy: 0.7549

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6412 - accuracy: 0.7524

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6465 - accuracy: 0.7512

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6444 - accuracy: 0.7512

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6471 - accuracy: 0.7483

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6488 - accuracy: 0.7472

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6485 - accuracy: 0.7478

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6493 - accuracy: 0.7473

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6458 - accuracy: 0.7489

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6445 - accuracy: 0.7490

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6435 - accuracy: 0.7500

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6474 - accuracy: 0.7485

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6454 - accuracy: 0.7495

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6433 - accuracy: 0.7505

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6459 - accuracy: 0.7486

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6437 - accuracy: 0.7486

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6431 - accuracy: 0.7491

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6443 - accuracy: 0.7491

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6453 - accuracy: 0.7487

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6445 - accuracy: 0.7487

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6446 - accuracy: 0.7496

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6454 - accuracy: 0.7491

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6465 - accuracy: 0.7487

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6439 - accuracy: 0.7496

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6421 - accuracy: 0.7500

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6428 - accuracy: 0.7492

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6445 - accuracy: 0.7488

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6470 - accuracy: 0.7472

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6455 - accuracy: 0.7480

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6473 - accuracy: 0.7481

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6466 - accuracy: 0.7485

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6454 - accuracy: 0.7504

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6420 - accuracy: 0.7522

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6440 - accuracy: 0.7518

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6455 - accuracy: 0.7511

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6450 - accuracy: 0.7514

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6483 - accuracy: 0.7500

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6486 - accuracy: 0.7500

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6486 - accuracy: 0.7493

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6478 - accuracy: 0.7503

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6487 - accuracy: 0.7500

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.6487 - accuracy: 0.7500 - val_loss: 0.8006 - val_accuracy: 0.6894


.. parsed-literal::

    Epoch 11/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.6383 - accuracy: 0.7812

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.6041 - accuracy: 0.7812

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5362 - accuracy: 0.8125

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5368 - accuracy: 0.8125

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5715 - accuracy: 0.7812

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.5573 - accuracy: 0.7969

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6022 - accuracy: 0.7857

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6183 - accuracy: 0.7891

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5983 - accuracy: 0.7882

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5969 - accuracy: 0.7875

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5909 - accuracy: 0.7841

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5897 - accuracy: 0.7865

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6041 - accuracy: 0.7812

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6186 - accuracy: 0.7746

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6045 - accuracy: 0.7771

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6224 - accuracy: 0.7734

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6189 - accuracy: 0.7721

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6136 - accuracy: 0.7726

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6233 - accuracy: 0.7681

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6239 - accuracy: 0.7703

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6204 - accuracy: 0.7738

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6097 - accuracy: 0.7784

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.6070 - accuracy: 0.7799

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6042 - accuracy: 0.7812

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5969 - accuracy: 0.7825

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5911 - accuracy: 0.7849

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5913 - accuracy: 0.7836

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5902 - accuracy: 0.7824

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5867 - accuracy: 0.7834

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5899 - accuracy: 0.7802

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5923 - accuracy: 0.7802

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5858 - accuracy: 0.7822

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5850 - accuracy: 0.7812

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5879 - accuracy: 0.7812

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5896 - accuracy: 0.7795

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5864 - accuracy: 0.7804

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5873 - accuracy: 0.7804

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5873 - accuracy: 0.7812

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5833 - accuracy: 0.7821

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.5859 - accuracy: 0.7805

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5954 - accuracy: 0.7782

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5915 - accuracy: 0.7790

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5910 - accuracy: 0.7783

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6005 - accuracy: 0.7723

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6003 - accuracy: 0.7712

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5992 - accuracy: 0.7707

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5971 - accuracy: 0.7723

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5964 - accuracy: 0.7724

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5955 - accuracy: 0.7726

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5965 - accuracy: 0.7716

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6009 - accuracy: 0.7699

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6030 - accuracy: 0.7695

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6050 - accuracy: 0.7680

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6056 - accuracy: 0.7671

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6042 - accuracy: 0.7668

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6041 - accuracy: 0.7660

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6037 - accuracy: 0.7657

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6079 - accuracy: 0.7617

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6132 - accuracy: 0.7589

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6144 - accuracy: 0.7587

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6140 - accuracy: 0.7581

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6186 - accuracy: 0.7560

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6170 - accuracy: 0.7564

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6149 - accuracy: 0.7572

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6137 - accuracy: 0.7576

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6135 - accuracy: 0.7580

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6127 - accuracy: 0.7597

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6120 - accuracy: 0.7605

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6108 - accuracy: 0.7603

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6110 - accuracy: 0.7588

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6090 - accuracy: 0.7591

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6092 - accuracy: 0.7590

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6130 - accuracy: 0.7585

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6148 - accuracy: 0.7575

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6163 - accuracy: 0.7578

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6135 - accuracy: 0.7581

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6176 - accuracy: 0.7560

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6190 - accuracy: 0.7563

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6183 - accuracy: 0.7567

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6189 - accuracy: 0.7562

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6160 - accuracy: 0.7573

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6161 - accuracy: 0.7572

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6179 - accuracy: 0.7567

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6167 - accuracy: 0.7570

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6163 - accuracy: 0.7577

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6175 - accuracy: 0.7572

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6181 - accuracy: 0.7568

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6179 - accuracy: 0.7574

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6177 - accuracy: 0.7577

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6182 - accuracy: 0.7583

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6183 - accuracy: 0.7585

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.6183 - accuracy: 0.7585 - val_loss: 0.7781 - val_accuracy: 0.6853


.. parsed-literal::

    Epoch 12/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.6038 - accuracy: 0.7188

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.6946 - accuracy: 0.7188

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6952 - accuracy: 0.7396

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6341 - accuracy: 0.7578

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.6013 - accuracy: 0.7812

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.6316 - accuracy: 0.7604

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6176 - accuracy: 0.7679

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6181 - accuracy: 0.7656

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5951 - accuracy: 0.7778

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6056 - accuracy: 0.7750

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5855 - accuracy: 0.7869

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5932 - accuracy: 0.7786

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5820 - accuracy: 0.7812

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5745 - accuracy: 0.7835

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5660 - accuracy: 0.7854

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5672 - accuracy: 0.7891

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5676 - accuracy: 0.7923

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5872 - accuracy: 0.7830

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5823 - accuracy: 0.7862

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5815 - accuracy: 0.7875

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5699 - accuracy: 0.7932

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5707 - accuracy: 0.7912

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.5771 - accuracy: 0.7908

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5718 - accuracy: 0.7904

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5756 - accuracy: 0.7887

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5771 - accuracy: 0.7873

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5839 - accuracy: 0.7824

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5816 - accuracy: 0.7824

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5726 - accuracy: 0.7877

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5759 - accuracy: 0.7875

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5762 - accuracy: 0.7853

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5764 - accuracy: 0.7842

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5787 - accuracy: 0.7841

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5815 - accuracy: 0.7831

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5804 - accuracy: 0.7830

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5762 - accuracy: 0.7847

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5785 - accuracy: 0.7829

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5798 - accuracy: 0.7812

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5847 - accuracy: 0.7764

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.5837 - accuracy: 0.7773

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5903 - accuracy: 0.7759

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5895 - accuracy: 0.7738

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5847 - accuracy: 0.7762

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5816 - accuracy: 0.7791

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5829 - accuracy: 0.7799

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5811 - accuracy: 0.7792

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5765 - accuracy: 0.7806

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5798 - accuracy: 0.7786

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5767 - accuracy: 0.7806

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5764 - accuracy: 0.7812

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5781 - accuracy: 0.7806

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5792 - accuracy: 0.7806

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5826 - accuracy: 0.7789

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5812 - accuracy: 0.7795

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5814 - accuracy: 0.7795

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5805 - accuracy: 0.7801

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5799 - accuracy: 0.7802

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5794 - accuracy: 0.7807

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5763 - accuracy: 0.7823

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5771 - accuracy: 0.7818

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5761 - accuracy: 0.7812

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5794 - accuracy: 0.7797

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5838 - accuracy: 0.7779

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5817 - accuracy: 0.7790

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5830 - accuracy: 0.7785

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5807 - accuracy: 0.7790

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5847 - accuracy: 0.7791

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5862 - accuracy: 0.7777

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5856 - accuracy: 0.7764

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5864 - accuracy: 0.7756

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5920 - accuracy: 0.7735

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5929 - accuracy: 0.7728

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5914 - accuracy: 0.7733

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5946 - accuracy: 0.7730

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5941 - accuracy: 0.7743

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5916 - accuracy: 0.7752

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5929 - accuracy: 0.7753

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5939 - accuracy: 0.7742

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5953 - accuracy: 0.7735

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5938 - accuracy: 0.7732

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5932 - accuracy: 0.7733

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5950 - accuracy: 0.7719

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5956 - accuracy: 0.7713

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5965 - accuracy: 0.7706

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5943 - accuracy: 0.7719

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5935 - accuracy: 0.7723

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5934 - accuracy: 0.7721

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5933 - accuracy: 0.7718

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5957 - accuracy: 0.7716

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5956 - accuracy: 0.7717

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5974 - accuracy: 0.7715

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.5974 - accuracy: 0.7715 - val_loss: 0.7388 - val_accuracy: 0.7207


.. parsed-literal::

    Epoch 13/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.2521 - accuracy: 0.9062

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.4978 - accuracy: 0.8125

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5448 - accuracy: 0.7917

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5292 - accuracy: 0.8047

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5213 - accuracy: 0.8062

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.5083 - accuracy: 0.8125

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.5250 - accuracy: 0.8036

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5078 - accuracy: 0.8086

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5342 - accuracy: 0.8021

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5425 - accuracy: 0.7969

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5871 - accuracy: 0.7841

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5790 - accuracy: 0.7891

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5904 - accuracy: 0.7812

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5850 - accuracy: 0.7835

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5826 - accuracy: 0.7812

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5766 - accuracy: 0.7793

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5750 - accuracy: 0.7776

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5842 - accuracy: 0.7691

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5810 - accuracy: 0.7730

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5892 - accuracy: 0.7688

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5936 - accuracy: 0.7693

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5924 - accuracy: 0.7699

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.5917 - accuracy: 0.7690

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5928 - accuracy: 0.7639

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5913 - accuracy: 0.7646

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5887 - accuracy: 0.7675

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5841 - accuracy: 0.7703

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5846 - accuracy: 0.7696

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5927 - accuracy: 0.7658

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5909 - accuracy: 0.7663

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5896 - accuracy: 0.7667

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5912 - accuracy: 0.7653

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5889 - accuracy: 0.7676

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5905 - accuracy: 0.7680

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5882 - accuracy: 0.7701

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5859 - accuracy: 0.7721

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5854 - accuracy: 0.7715

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5900 - accuracy: 0.7718

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.5961 - accuracy: 0.7697

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6003 - accuracy: 0.7676

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6010 - accuracy: 0.7687

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6092 - accuracy: 0.7646

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6104 - accuracy: 0.7636

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6088 - accuracy: 0.7654

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6066 - accuracy: 0.7664

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6061 - accuracy: 0.7674

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6059 - accuracy: 0.7677

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6067 - accuracy: 0.7667

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6073 - accuracy: 0.7670

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6038 - accuracy: 0.7697

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6005 - accuracy: 0.7705

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5995 - accuracy: 0.7719

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6000 - accuracy: 0.7721

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5973 - accuracy: 0.7734

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6013 - accuracy: 0.7719

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5983 - accuracy: 0.7726

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5964 - accuracy: 0.7733

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5978 - accuracy: 0.7718

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5977 - accuracy: 0.7730

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5951 - accuracy: 0.7731

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5915 - accuracy: 0.7748

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5885 - accuracy: 0.7754

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5901 - accuracy: 0.7750

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5900 - accuracy: 0.7746

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5881 - accuracy: 0.7757

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5828 - accuracy: 0.7781

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5796 - accuracy: 0.7795

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5803 - accuracy: 0.7791

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5769 - accuracy: 0.7809

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5789 - accuracy: 0.7792

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5774 - accuracy: 0.7796

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5764 - accuracy: 0.7805

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5755 - accuracy: 0.7814

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5775 - accuracy: 0.7818

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5780 - accuracy: 0.7818

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5760 - accuracy: 0.7818

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5804 - accuracy: 0.7785

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5793 - accuracy: 0.7794

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5781 - accuracy: 0.7802

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5772 - accuracy: 0.7806

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5764 - accuracy: 0.7817

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5771 - accuracy: 0.7810

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5781 - accuracy: 0.7806

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5765 - accuracy: 0.7813

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5789 - accuracy: 0.7795

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5760 - accuracy: 0.7810

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5763 - accuracy: 0.7806

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5742 - accuracy: 0.7810

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5762 - accuracy: 0.7806

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5780 - accuracy: 0.7796

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5756 - accuracy: 0.7807

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.5756 - accuracy: 0.7807 - val_loss: 0.7732 - val_accuracy: 0.7112


.. parsed-literal::

    Epoch 14/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.4724 - accuracy: 0.7812

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.4677 - accuracy: 0.8438

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.4572 - accuracy: 0.8438

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.4733 - accuracy: 0.8359

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.4459 - accuracy: 0.8438

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.4600 - accuracy: 0.8333

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 5s - loss: 0.4488 - accuracy: 0.8348

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.4775 - accuracy: 0.8203

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.4582 - accuracy: 0.8333

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.4794 - accuracy: 0.8156

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.4702 - accuracy: 0.8182

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.4964 - accuracy: 0.8099

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5159 - accuracy: 0.8077

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5123 - accuracy: 0.8103

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5109 - accuracy: 0.8083

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5095 - accuracy: 0.8105

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5146 - accuracy: 0.8070

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5227 - accuracy: 0.8003

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5209 - accuracy: 0.8010

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5119 - accuracy: 0.8031

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5153 - accuracy: 0.8006

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5334 - accuracy: 0.7912

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.5367 - accuracy: 0.7880

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5335 - accuracy: 0.7891

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5348 - accuracy: 0.7862

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5357 - accuracy: 0.7873

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5435 - accuracy: 0.7870

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5449 - accuracy: 0.7868

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5396 - accuracy: 0.7899

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5440 - accuracy: 0.7896

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5445 - accuracy: 0.7883

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5365 - accuracy: 0.7920

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5339 - accuracy: 0.7936

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5298 - accuracy: 0.7950

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5316 - accuracy: 0.7946

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5394 - accuracy: 0.7908

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5533 - accuracy: 0.7872

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5536 - accuracy: 0.7887

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5529 - accuracy: 0.7885

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.5464 - accuracy: 0.7930

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5496 - accuracy: 0.7934

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5478 - accuracy: 0.7924

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5458 - accuracy: 0.7943

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5462 - accuracy: 0.7947

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5431 - accuracy: 0.7965

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5416 - accuracy: 0.7976

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5400 - accuracy: 0.7979

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5436 - accuracy: 0.7962

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5406 - accuracy: 0.7985

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5384 - accuracy: 0.8006

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5345 - accuracy: 0.8015

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5330 - accuracy: 0.8029

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5363 - accuracy: 0.8013

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5340 - accuracy: 0.8027

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5336 - accuracy: 0.8034

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5298 - accuracy: 0.8047

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5300 - accuracy: 0.8043

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5293 - accuracy: 0.8050

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5309 - accuracy: 0.8040

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5331 - accuracy: 0.8026

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5341 - accuracy: 0.8033

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5359 - accuracy: 0.8029

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5359 - accuracy: 0.8021

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5360 - accuracy: 0.8018

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5354 - accuracy: 0.8019

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5334 - accuracy: 0.8026

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5358 - accuracy: 0.8018

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5344 - accuracy: 0.8023

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5346 - accuracy: 0.8015

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5341 - accuracy: 0.8004

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5348 - accuracy: 0.7997

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5344 - accuracy: 0.7990

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5343 - accuracy: 0.8000

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5338 - accuracy: 0.8002

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5324 - accuracy: 0.8007

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5342 - accuracy: 0.8005

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5330 - accuracy: 0.8002

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5315 - accuracy: 0.8012

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5297 - accuracy: 0.8021

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5321 - accuracy: 0.8007

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5344 - accuracy: 0.7997

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5324 - accuracy: 0.8006

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5323 - accuracy: 0.8011

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5341 - accuracy: 0.8001

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5382 - accuracy: 0.7996

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5364 - accuracy: 0.7997

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5352 - accuracy: 0.7999

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5374 - accuracy: 0.7986

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5366 - accuracy: 0.7987

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5357 - accuracy: 0.7992

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5402 - accuracy: 0.7977

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.5402 - accuracy: 0.7977 - val_loss: 0.7415 - val_accuracy: 0.7262


.. parsed-literal::

    Epoch 15/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.6753 - accuracy: 0.8125

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5525 - accuracy: 0.8594

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.4943 - accuracy: 0.8646

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5050 - accuracy: 0.8359

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5340 - accuracy: 0.8188

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.5539 - accuracy: 0.7917

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.5649 - accuracy: 0.7857

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5729 - accuracy: 0.7812

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5578 - accuracy: 0.7812

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5428 - accuracy: 0.7969

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5413 - accuracy: 0.7955

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5420 - accuracy: 0.7917

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5343 - accuracy: 0.7981

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5238 - accuracy: 0.8013

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5079 - accuracy: 0.8083

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.4981 - accuracy: 0.8164

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5079 - accuracy: 0.8107

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.4957 - accuracy: 0.8160

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.4897 - accuracy: 0.8191

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5031 - accuracy: 0.8156

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 3s - loss: 0.5068 - accuracy: 0.8161

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.5009 - accuracy: 0.8159

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.4993 - accuracy: 0.8158

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.4927 - accuracy: 0.8182

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5030 - accuracy: 0.8143

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5015 - accuracy: 0.8143

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5079 - accuracy: 0.8153

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5098 - accuracy: 0.8130

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5209 - accuracy: 0.8088

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5160 - accuracy: 0.8089

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5169 - accuracy: 0.8100

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5118 - accuracy: 0.8130

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5098 - accuracy: 0.8120

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5224 - accuracy: 0.8094

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5265 - accuracy: 0.8086

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5269 - accuracy: 0.8070

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5314 - accuracy: 0.8046

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5254 - accuracy: 0.8073

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.5210 - accuracy: 0.8097

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5236 - accuracy: 0.8083

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5226 - accuracy: 0.8084

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5256 - accuracy: 0.8056

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5233 - accuracy: 0.8064

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5298 - accuracy: 0.8038

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5333 - accuracy: 0.8040

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5326 - accuracy: 0.8041

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5333 - accuracy: 0.8037

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5372 - accuracy: 0.8019

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5457 - accuracy: 0.7977

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5460 - accuracy: 0.7986

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5445 - accuracy: 0.7995

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5419 - accuracy: 0.8004

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5416 - accuracy: 0.8000

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5430 - accuracy: 0.7991

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5431 - accuracy: 0.7988

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5432 - accuracy: 0.7979

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5451 - accuracy: 0.7976

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5440 - accuracy: 0.7984

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5455 - accuracy: 0.7981

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5462 - accuracy: 0.7973

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5471 - accuracy: 0.7966

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5468 - accuracy: 0.7968

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5442 - accuracy: 0.7975

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5427 - accuracy: 0.7983

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5440 - accuracy: 0.7985

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5431 - accuracy: 0.7992

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5395 - accuracy: 0.8007

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5412 - accuracy: 0.8000

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5390 - accuracy: 0.8020

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5381 - accuracy: 0.8021

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5392 - accuracy: 0.8014

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5366 - accuracy: 0.8028

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5366 - accuracy: 0.8034

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5385 - accuracy: 0.8027

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5374 - accuracy: 0.8032

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5354 - accuracy: 0.8042

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5343 - accuracy: 0.8043

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5340 - accuracy: 0.8044

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5352 - accuracy: 0.8033

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5338 - accuracy: 0.8038

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5352 - accuracy: 0.8031

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5393 - accuracy: 0.8029

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5378 - accuracy: 0.8034

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5369 - accuracy: 0.8035

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5372 - accuracy: 0.8032

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5368 - accuracy: 0.8037

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5371 - accuracy: 0.8038

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5387 - accuracy: 0.8035

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5371 - accuracy: 0.8040

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5400 - accuracy: 0.8023

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5412 - accuracy: 0.8014

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.5412 - accuracy: 0.8014 - val_loss: 0.7033 - val_accuracy: 0.7357


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

    img = tf.keras.preprocessing.image.load_img(
        sunflower_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
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
    1/1 [==============================] - 0s 75ms/step


.. parsed-literal::

    This image most likely belongs to sunflowers with a 99.00 percent confidence.


Save the TensorFlow Model
-------------------------



.. code:: ipython3

    #save the trained model - a new folder flower will be created
    #and the file "saved_model.h5" is the pre-trained model
    model_dir = "model"
    saved_model_path = f"{model_dir}/flower/saved_model"
    model.save(saved_model_path)


.. parsed-literal::

    2024-03-26 00:51:54.338326: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'random_flip_input' with dtype float and shape [?,180,180,3]
    	 [[{{node random_flip_input}}]]
    2024-03-26 00:51:54.424860: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-26 00:51:54.434856: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'random_flip_input' with dtype float and shape [?,180,180,3]
    	 [[{{node random_flip_input}}]]
    2024-03-26 00:51:54.446004: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-26 00:51:54.452857: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-26 00:51:54.459629: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-26 00:51:54.470587: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-26 00:51:54.509602: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'sequential_1_input' with dtype float and shape [?,180,180,3]
    	 [[{{node sequential_1_input}}]]


.. parsed-literal::

    2024-03-26 00:51:54.576782: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-26 00:51:54.597129: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'sequential_1_input' with dtype float and shape [?,180,180,3]
    	 [[{{node sequential_1_input}}]]
    2024-03-26 00:51:54.635836: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,22,22,64]
    	 [[{{node inputs}}]]
    2024-03-26 00:51:54.660639: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-26 00:51:54.733200: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]


.. parsed-literal::

    2024-03-26 00:51:54.876666: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-26 00:51:55.036680: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,22,22,64]
    	 [[{{node inputs}}]]
    2024-03-26 00:51:55.070449: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]


.. parsed-literal::

    2024-03-26 00:51:55.099058: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-26 00:51:55.144920: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
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
    ir_model = ov.convert_model(saved_model_path, input=[1,180,180,3])
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
    This image most likely belongs to dandelion with a 98.99 percent confidence.



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
