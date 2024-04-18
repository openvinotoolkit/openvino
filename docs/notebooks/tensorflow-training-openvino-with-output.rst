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
Classification Model <./tensorflow-training-openvino-nncf.ipynb>`__
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

    %pip install -q "openvino>=2023.1.0" "pillow" "tqdm"
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
    from pathlib import Path

    os.environ["TF_USE_LEGACY_KERAS"] = "1"

    import PIL
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    from PIL import Image
    import openvino as ov

    # Fetch `notebook_utils` module
    import requests

    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )

    open("notebook_utils.py", "w").write(r.text)
    from notebook_utils import download_file


.. parsed-literal::

    2024-04-18 01:14:56.038506: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-04-18 01:14:56.073274: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


.. parsed-literal::

    2024-04-18 01:14:56.589731: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


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
    data_dir = tf.keras.utils.get_file("flower_photos", origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)

After downloading, you should now have a copy of the dataset available.
There are 3,670 total images:

.. code:: ipython3

    image_count = len(list(data_dir.glob("*/*.jpg")))
    print(image_count)


.. parsed-literal::

    3670


Here are some roses:

.. code:: ipython3

    roses = list(data_dir.glob("roses/*"))
    PIL.Image.open(str(roses[0]))




.. image:: tensorflow-training-openvino-with-output_files/tensorflow-training-openvino-with-output_14_0.png



.. code:: ipython3

    PIL.Image.open(str(roses[1]))




.. image:: tensorflow-training-openvino-with-output_files/tensorflow-training-openvino-with-output_15_0.png



And some tulips:

.. code:: ipython3

    tulips = list(data_dir.glob("tulips/*"))
    PIL.Image.open(str(tulips[0]))




.. image:: tensorflow-training-openvino-with-output_files/tensorflow-training-openvino-with-output_17_0.png



.. code:: ipython3

    PIL.Image.open(str(tulips[1]))




.. image:: tensorflow-training-openvino-with-output_files/tensorflow-training-openvino-with-output_18_0.png



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
        batch_size=batch_size,
    )


.. parsed-literal::

    Found 3670 files belonging to 5 classes.


.. parsed-literal::

    Using 2936 files for training.


.. parsed-literal::

    2024-04-18 01:14:59.925007: E tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:266] failed call to cuInit: CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: forward compatibility was attempted on non supported HW
    2024-04-18 01:14:59.925039: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:168] retrieving CUDA diagnostic information for host: iotg-dev-workstation-07
    2024-04-18 01:14:59.925044: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:175] hostname: iotg-dev-workstation-07
    2024-04-18 01:14:59.925173: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:199] libcuda reported version is: 470.223.2
    2024-04-18 01:14:59.925190: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:203] kernel reported version is: 470.182.3
    2024-04-18 01:14:59.925193: E tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:312] kernel version 470.182.3 does not match DSO version 470.223.2 -- cannot find working devices in this configuration


.. code:: ipython3

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )


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

    2024-04-18 01:15:00.258455: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-04-18 01:15:00.259027: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]



.. image:: tensorflow-training-openvino-with-output_files/tensorflow-training-openvino-with-output_29_1.png


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

    2024-04-18 01:15:01.067592: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-04-18 01:15:01.067963: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
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

    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

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

    2024-04-18 01:15:01.259365: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-04-18 01:15:01.259741: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
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

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(1.0 / 255, input_shape=(img_height, img_width, 3)),
            tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(num_classes),
        ]
    )

Compile the Model
-----------------



For this tutorial, choose the ``optimizers.Adam`` optimizer and
``losses.SparseCategoricalCrossentropy`` loss function. To view training
and validation accuracy for each training epoch, pass the ``metrics``
argument.

.. code:: ipython3

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

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
            tf.keras.layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
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

    2024-04-18 01:15:02.167841: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-04-18 01:15:02.168220: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]



.. image:: tensorflow-training-openvino-with-output_files/tensorflow-training-openvino-with-output_57_1.png


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

    model = tf.keras.Sequential(
        [
            data_augmentation,
            tf.keras.layers.Rescaling(1.0 / 255),
            tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(num_classes, name="outputs"),
        ]
    )

Compile and Train the Model
---------------------------



.. code:: ipython3

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

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
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)


.. parsed-literal::

    Epoch 1/15


.. parsed-literal::

    2024-04-18 01:15:03.232166: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]
    2024-04-18 01:15:03.232504: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]


.. parsed-literal::


 1/92 [..............................] - ETA: 1:36 - loss: 1.6139 - accuracy: 0.1562

.. parsed-literal::

    
 2/92 [..............................] - ETA: 6s - loss: 2.6072 - accuracy: 0.2031

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 2.2647 - accuracy: 0.2188

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 2.1299 - accuracy: 0.2266

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 2.0328 - accuracy: 0.2375

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 1.9690 - accuracy: 0.2448

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 5s - loss: 1.9062 - accuracy: 0.2545

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 5s - loss: 1.8712 - accuracy: 0.2461

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 5s - loss: 1.8292 - accuracy: 0.2569

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 1.8010 - accuracy: 0.2594

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 1.7882 - accuracy: 0.2472

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 1.7694 - accuracy: 0.2448

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 1.7518 - accuracy: 0.2500

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 1.7420 - accuracy: 0.2433

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 1.7297 - accuracy: 0.2417

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 1.7182 - accuracy: 0.2578

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 1.7064 - accuracy: 0.2629

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 1.6963 - accuracy: 0.2639

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 1.6857 - accuracy: 0.2664

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 1.6820 - accuracy: 0.2625

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 1.6769 - accuracy: 0.2589

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 1.6687 - accuracy: 0.2670

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 1.6604 - accuracy: 0.2745

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 4s - loss: 1.6586 - accuracy: 0.2734

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 1.6499 - accuracy: 0.2800

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 1.6417 - accuracy: 0.2837

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 1.6396 - accuracy: 0.2812

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 1.6348 - accuracy: 0.2835

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 1.6303 - accuracy: 0.2834

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 1.6242 - accuracy: 0.2854

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 1.6175 - accuracy: 0.2903

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 1.6115 - accuracy: 0.2920

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 1.6049 - accuracy: 0.2926

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 1.6013 - accuracy: 0.2923

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 1.5949 - accuracy: 0.2946

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 1.5943 - accuracy: 0.2925

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 1.5893 - accuracy: 0.2956

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 1.5848 - accuracy: 0.2952

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 1.5784 - accuracy: 0.3005

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 1.5722 - accuracy: 0.3063

.. parsed-literal::

    
41/92 [============>.................] - ETA: 3s - loss: 1.5698 - accuracy: 0.3095

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 1.5657 - accuracy: 0.3110

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 1.5569 - accuracy: 0.3190

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 1.5491 - accuracy: 0.3239

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 1.5441 - accuracy: 0.3264

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 1.5337 - accuracy: 0.3302

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 1.5279 - accuracy: 0.3311

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 1.5244 - accuracy: 0.3327

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 1.5244 - accuracy: 0.3329

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 1.5165 - accuracy: 0.3363

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 1.5052 - accuracy: 0.3412

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 1.5032 - accuracy: 0.3412

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 1.4967 - accuracy: 0.3436

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 1.4917 - accuracy: 0.3453

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 1.4867 - accuracy: 0.3492

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 1.4796 - accuracy: 0.3513

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 1.4692 - accuracy: 0.3588

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 1.4644 - accuracy: 0.3596

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 1.4589 - accuracy: 0.3619

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 1.4557 - accuracy: 0.3652

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 1.4508 - accuracy: 0.3679

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 1.4475 - accuracy: 0.3700

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 1.4455 - accuracy: 0.3721

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 1.4425 - accuracy: 0.3726

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 1.4393 - accuracy: 0.3740

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 1.4328 - accuracy: 0.3773

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 1.4288 - accuracy: 0.3796

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 1.4242 - accuracy: 0.3814

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 1.4179 - accuracy: 0.3849

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 1.4150 - accuracy: 0.3883

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 1.4107 - accuracy: 0.3911

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 1.4078 - accuracy: 0.3935

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 1.4038 - accuracy: 0.3966

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 1.3994 - accuracy: 0.3984

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 1.3970 - accuracy: 0.3985

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 1.3944 - accuracy: 0.3994

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 1.3883 - accuracy: 0.4031

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 1.3831 - accuracy: 0.4056

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 1.3799 - accuracy: 0.4060

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 1.3766 - accuracy: 0.4079

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 1.3725 - accuracy: 0.4109

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 1.3693 - accuracy: 0.4131

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 1.3667 - accuracy: 0.4146

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 1.3630 - accuracy: 0.4170

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 1.3620 - accuracy: 0.4173

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 1.3574 - accuracy: 0.4204

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 1.3518 - accuracy: 0.4234

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 1.3495 - accuracy: 0.4246

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 1.3457 - accuracy: 0.4262

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 1.3471 - accuracy: 0.4256

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 1.3422 - accuracy: 0.4278

.. parsed-literal::

    2024-04-18 01:15:09.636279: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [734]
    	 [[{{node Placeholder/_0}}]]
    2024-04-18 01:15:09.636558: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [734]
    	 [[{{node Placeholder/_4}}]]


.. parsed-literal::

    
92/92 [==============================] - 7s 66ms/step - loss: 1.3422 - accuracy: 0.4278 - val_loss: 1.1363 - val_accuracy: 0.5354


.. parsed-literal::

    Epoch 2/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 1.1986 - accuracy: 0.4375

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 1.2820 - accuracy: 0.4062

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 1.2114 - accuracy: 0.4479

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 1.2021 - accuracy: 0.4844

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 1.1758 - accuracy: 0.5000

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 1.1485 - accuracy: 0.5278

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 1.1516 - accuracy: 0.5323

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 1.1049 - accuracy: 0.5571

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 1.1170 - accuracy: 0.5481

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 1.1096 - accuracy: 0.5581

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 1.0917 - accuracy: 0.5638

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 1.0837 - accuracy: 0.5686

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 1.0854 - accuracy: 0.5659

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 1.0719 - accuracy: 0.5720

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 1.0719 - accuracy: 0.5694

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 1.0607 - accuracy: 0.5802

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 1.0545 - accuracy: 0.5845

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 1.0433 - accuracy: 0.5900

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 1.0469 - accuracy: 0.5870

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 1.0472 - accuracy: 0.5858

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 1.0548 - accuracy: 0.5805

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 1.0484 - accuracy: 0.5852

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 1.0542 - accuracy: 0.5855

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 1.0491 - accuracy: 0.5871

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 1.0484 - accuracy: 0.5886

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 1.0557 - accuracy: 0.5841

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 1.0421 - accuracy: 0.5912

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 1.0484 - accuracy: 0.5859

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 1.0456 - accuracy: 0.5893

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 1.0468 - accuracy: 0.5894

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 1.0443 - accuracy: 0.5945

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 1.0452 - accuracy: 0.5935

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 1.0388 - accuracy: 0.5954

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 1.0333 - accuracy: 0.5971

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 1.0318 - accuracy: 0.5962

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 1.0289 - accuracy: 0.5961

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 1.0255 - accuracy: 0.5977

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 1.0291 - accuracy: 0.5984

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 1.0355 - accuracy: 0.5967

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 1.0413 - accuracy: 0.5928

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 1.0354 - accuracy: 0.5958

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 1.0371 - accuracy: 0.5965

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 1.0384 - accuracy: 0.5950

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 1.0413 - accuracy: 0.5971

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 1.0459 - accuracy: 0.5949

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 1.0455 - accuracy: 0.5949

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 1.0414 - accuracy: 0.5962

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 1.0395 - accuracy: 0.5968

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 1.0412 - accuracy: 0.5948

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 1.0356 - accuracy: 0.5967

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 1.0350 - accuracy: 0.5966

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 1.0338 - accuracy: 0.5977

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 1.0313 - accuracy: 0.5983

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 1.0285 - accuracy: 0.5999

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 1.0268 - accuracy: 0.6003

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 1.0263 - accuracy: 0.6019

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 1.0250 - accuracy: 0.6012

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 1.0237 - accuracy: 0.6027

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 1.0248 - accuracy: 0.6020

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 1.0204 - accuracy: 0.6044

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 1.0172 - accuracy: 0.6073

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 1.0174 - accuracy: 0.6071

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 1.0192 - accuracy: 0.6059

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 1.0171 - accuracy: 0.6067

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 1.0145 - accuracy: 0.6065

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 1.0136 - accuracy: 0.6053

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 1.0121 - accuracy: 0.6070

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 1.0103 - accuracy: 0.6073

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 1.0133 - accuracy: 0.6057

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 1.0128 - accuracy: 0.6047

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 1.0119 - accuracy: 0.6028

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 1.0166 - accuracy: 0.5997

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 1.0152 - accuracy: 0.6013

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 1.0118 - accuracy: 0.6033

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 1.0104 - accuracy: 0.6052

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 1.0095 - accuracy: 0.6046

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 1.0075 - accuracy: 0.6045

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 1.0106 - accuracy: 0.6036

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 1.0089 - accuracy: 0.6034

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 1.0060 - accuracy: 0.6041

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 1.0082 - accuracy: 0.6036

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 1.0097 - accuracy: 0.6031

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 1.0104 - accuracy: 0.6026

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 1.0136 - accuracy: 0.6021

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 1.0142 - accuracy: 0.6006

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 1.0127 - accuracy: 0.6016

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 1.0143 - accuracy: 0.6011

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 1.0151 - accuracy: 0.6000

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 1.0147 - accuracy: 0.6006

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 1.0150 - accuracy: 0.6009

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 1.0155 - accuracy: 0.6005

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 1.0155 - accuracy: 0.6005 - val_loss: 1.0104 - val_accuracy: 0.5926


.. parsed-literal::

    Epoch 3/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.9533 - accuracy: 0.7500

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 1.1267 - accuracy: 0.5938

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 1.0877 - accuracy: 0.6042

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 1.0102 - accuracy: 0.6250

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.9940 - accuracy: 0.6375

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.9650 - accuracy: 0.6510

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.9989 - accuracy: 0.6295

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.9844 - accuracy: 0.6367

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.9845 - accuracy: 0.6250

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.9637 - accuracy: 0.6344

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.9750 - accuracy: 0.6250

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.9728 - accuracy: 0.6302

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.9751 - accuracy: 0.6298

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.9820 - accuracy: 0.6295

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.9981 - accuracy: 0.6229

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.9898 - accuracy: 0.6250

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.9699 - accuracy: 0.6324

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.9650 - accuracy: 0.6372

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.9762 - accuracy: 0.6365

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.9739 - accuracy: 0.6359

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.9694 - accuracy: 0.6354

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.9641 - accuracy: 0.6349

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.9599 - accuracy: 0.6399

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.9665 - accuracy: 0.6367

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.9625 - accuracy: 0.6375

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.9541 - accuracy: 0.6382

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.9530 - accuracy: 0.6377

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.9513 - accuracy: 0.6373

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.9535 - accuracy: 0.6358

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.9461 - accuracy: 0.6375

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.9436 - accuracy: 0.6351

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.9455 - accuracy: 0.6357

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.9432 - accuracy: 0.6345

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.9491 - accuracy: 0.6296

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.9464 - accuracy: 0.6304

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.9462 - accuracy: 0.6293

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.9573 - accuracy: 0.6250

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.9568 - accuracy: 0.6266

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.9532 - accuracy: 0.6242

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.9522 - accuracy: 0.6258

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.9505 - accuracy: 0.6273

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.9511 - accuracy: 0.6257

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.9539 - accuracy: 0.6235

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.9542 - accuracy: 0.6236

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.9552 - accuracy: 0.6236

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.9541 - accuracy: 0.6250

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.9549 - accuracy: 0.6250

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.9550 - accuracy: 0.6257

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.9580 - accuracy: 0.6237

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.9554 - accuracy: 0.6256

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.9494 - accuracy: 0.6292

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.9497 - accuracy: 0.6297

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.9519 - accuracy: 0.6291

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.9542 - accuracy: 0.6273

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.9547 - accuracy: 0.6272

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.9536 - accuracy: 0.6294

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.9581 - accuracy: 0.6272

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.9562 - accuracy: 0.6287

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.9607 - accuracy: 0.6292

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.9581 - accuracy: 0.6291

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.9577 - accuracy: 0.6285

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.9559 - accuracy: 0.6290

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.9565 - accuracy: 0.6284

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.9564 - accuracy: 0.6274

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.9526 - accuracy: 0.6288

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.9530 - accuracy: 0.6287

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.9509 - accuracy: 0.6296

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.9494 - accuracy: 0.6305

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.9521 - accuracy: 0.6281

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.9550 - accuracy: 0.6277

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.9552 - accuracy: 0.6276

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.9556 - accuracy: 0.6263

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.9570 - accuracy: 0.6258

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.9544 - accuracy: 0.6263

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.9524 - accuracy: 0.6271

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.9531 - accuracy: 0.6254

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.9519 - accuracy: 0.6270

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.9517 - accuracy: 0.6274

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.9510 - accuracy: 0.6270

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.9511 - accuracy: 0.6281

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.9523 - accuracy: 0.6281

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.9527 - accuracy: 0.6269

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.9539 - accuracy: 0.6265

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.9536 - accuracy: 0.6265

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.9525 - accuracy: 0.6276

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.9540 - accuracy: 0.6275

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.9547 - accuracy: 0.6275

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.9533 - accuracy: 0.6285

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.9513 - accuracy: 0.6285

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.9500 - accuracy: 0.6284

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.9482 - accuracy: 0.6294

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.9482 - accuracy: 0.6294 - val_loss: 0.9087 - val_accuracy: 0.6376


.. parsed-literal::

    Epoch 4/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.7776 - accuracy: 0.7188

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.8659 - accuracy: 0.6562

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.8840 - accuracy: 0.6667

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.8530 - accuracy: 0.6875

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.8666 - accuracy: 0.6875

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.8646 - accuracy: 0.6771

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 5s - loss: 0.8364 - accuracy: 0.6875

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.8359 - accuracy: 0.6875

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.8548 - accuracy: 0.6806

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.8281 - accuracy: 0.6906

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.8342 - accuracy: 0.6875

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.8389 - accuracy: 0.6875

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.8304 - accuracy: 0.6923

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.8267 - accuracy: 0.6920

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.8444 - accuracy: 0.6750

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.8419 - accuracy: 0.6797

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.8528 - accuracy: 0.6691

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.8604 - accuracy: 0.6632

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.8589 - accuracy: 0.6661

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.8529 - accuracy: 0.6703

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.8637 - accuracy: 0.6741

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.8614 - accuracy: 0.6761

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.8570 - accuracy: 0.6766

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.8603 - accuracy: 0.6719

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.8648 - accuracy: 0.6737

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.8708 - accuracy: 0.6719

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.8721 - accuracy: 0.6667

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.8751 - accuracy: 0.6652

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.8792 - accuracy: 0.6649

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.8823 - accuracy: 0.6625

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.8807 - accuracy: 0.6653

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.8812 - accuracy: 0.6631

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.8865 - accuracy: 0.6610

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.8887 - accuracy: 0.6590

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.8874 - accuracy: 0.6598

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.8898 - accuracy: 0.6597

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.8911 - accuracy: 0.6556

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.8876 - accuracy: 0.6573

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.8879 - accuracy: 0.6580

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.8844 - accuracy: 0.6595

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.8857 - accuracy: 0.6579

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.8992 - accuracy: 0.6506

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.8927 - accuracy: 0.6543

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.8905 - accuracy: 0.6550

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.8858 - accuracy: 0.6578

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.8863 - accuracy: 0.6584

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.8846 - accuracy: 0.6590

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.8819 - accuracy: 0.6609

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.8793 - accuracy: 0.6614

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.8858 - accuracy: 0.6576

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.8833 - accuracy: 0.6600

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.8849 - accuracy: 0.6600

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.8867 - accuracy: 0.6576

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.8851 - accuracy: 0.6587

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.8836 - accuracy: 0.6592

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.8818 - accuracy: 0.6597

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.8795 - accuracy: 0.6596

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.8778 - accuracy: 0.6612

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.8743 - accuracy: 0.6621

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.8741 - accuracy: 0.6626

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.8774 - accuracy: 0.6614

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.8841 - accuracy: 0.6599

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.8877 - accuracy: 0.6583

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.8860 - accuracy: 0.6593

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.8885 - accuracy: 0.6602

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.8856 - accuracy: 0.6615

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.8875 - accuracy: 0.6605

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.8891 - accuracy: 0.6591

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.8886 - accuracy: 0.6595

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.8884 - accuracy: 0.6586

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.8844 - accuracy: 0.6607

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.8866 - accuracy: 0.6598

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.8872 - accuracy: 0.6597

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.8851 - accuracy: 0.6601

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.8898 - accuracy: 0.6584

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.8898 - accuracy: 0.6584

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.8907 - accuracy: 0.6580

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.8876 - accuracy: 0.6587

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.8879 - accuracy: 0.6587

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.8860 - accuracy: 0.6591

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.8842 - accuracy: 0.6598

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.8850 - accuracy: 0.6590

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.8838 - accuracy: 0.6582

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.8817 - accuracy: 0.6597

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.8794 - accuracy: 0.6607

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.8775 - accuracy: 0.6607

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.8772 - accuracy: 0.6599

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.8758 - accuracy: 0.6606

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.8751 - accuracy: 0.6605

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.8748 - accuracy: 0.6605

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.8733 - accuracy: 0.6611

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.8733 - accuracy: 0.6611 - val_loss: 0.8510 - val_accuracy: 0.6567


.. parsed-literal::

    Epoch 5/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.8657 - accuracy: 0.7188

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.7892 - accuracy: 0.7188

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.7836 - accuracy: 0.6979

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.7753 - accuracy: 0.6797

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.7662 - accuracy: 0.6938

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.7787 - accuracy: 0.7083

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.7777 - accuracy: 0.7098

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.7901 - accuracy: 0.7109

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.7951 - accuracy: 0.7014

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.7763 - accuracy: 0.7094

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.8326 - accuracy: 0.6875

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.8178 - accuracy: 0.6901

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.8167 - accuracy: 0.6851

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.8195 - accuracy: 0.6808

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.8092 - accuracy: 0.6854

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.7981 - accuracy: 0.6914

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.7976 - accuracy: 0.6937

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.7904 - accuracy: 0.6983

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.7961 - accuracy: 0.6978

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.7896 - accuracy: 0.7003

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 3s - loss: 0.7887 - accuracy: 0.6968

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.7973 - accuracy: 0.6909

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.7968 - accuracy: 0.6895

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.7883 - accuracy: 0.6932

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.7887 - accuracy: 0.6942

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.7798 - accuracy: 0.6986

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.7829 - accuracy: 0.6959

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.7854 - accuracy: 0.6935

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.7930 - accuracy: 0.6891

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.8000 - accuracy: 0.6860

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.8004 - accuracy: 0.6860

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.8010 - accuracy: 0.6832

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.7980 - accuracy: 0.6843

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.7957 - accuracy: 0.6862

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.7993 - accuracy: 0.6836

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.8022 - accuracy: 0.6845

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.8000 - accuracy: 0.6863

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.8015 - accuracy: 0.6863

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.8010 - accuracy: 0.6871

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.7999 - accuracy: 0.6879

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.7988 - accuracy: 0.6886

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.7963 - accuracy: 0.6908

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.7955 - accuracy: 0.6914

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.7980 - accuracy: 0.6927

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.7975 - accuracy: 0.6913

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.7976 - accuracy: 0.6905

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.7976 - accuracy: 0.6904

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.7978 - accuracy: 0.6923

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.7987 - accuracy: 0.6941

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.7994 - accuracy: 0.6933

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.7948 - accuracy: 0.6944

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.7983 - accuracy: 0.6931

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.7968 - accuracy: 0.6942

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.7969 - accuracy: 0.6941

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.7984 - accuracy: 0.6934

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.7978 - accuracy: 0.6938

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.7939 - accuracy: 0.6943

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.7931 - accuracy: 0.6947

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.7949 - accuracy: 0.6925

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.7953 - accuracy: 0.6929

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.7937 - accuracy: 0.6928

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.7945 - accuracy: 0.6912

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.7966 - accuracy: 0.6907

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.7917 - accuracy: 0.6931

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.7929 - accuracy: 0.6930

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.7913 - accuracy: 0.6943

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.7908 - accuracy: 0.6946

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.7927 - accuracy: 0.6945

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.7939 - accuracy: 0.6927

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.7905 - accuracy: 0.6943

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.7902 - accuracy: 0.6925

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.7924 - accuracy: 0.6916

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.7905 - accuracy: 0.6928

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.7955 - accuracy: 0.6911

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.7980 - accuracy: 0.6898

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.7978 - accuracy: 0.6889

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.7967 - accuracy: 0.6893

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.7995 - accuracy: 0.6885

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.7982 - accuracy: 0.6893

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.7980 - accuracy: 0.6900

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.7979 - accuracy: 0.6911

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.7979 - accuracy: 0.6918

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.7971 - accuracy: 0.6925

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.7951 - accuracy: 0.6940

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.7924 - accuracy: 0.6961

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.7896 - accuracy: 0.6974

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.7934 - accuracy: 0.6952

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.7955 - accuracy: 0.6947

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.7927 - accuracy: 0.6960

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.7930 - accuracy: 0.6959

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.7920 - accuracy: 0.6962

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.7920 - accuracy: 0.6962 - val_loss: 0.8125 - val_accuracy: 0.6744


.. parsed-literal::

    Epoch 6/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.6349 - accuracy: 0.7812

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.8563 - accuracy: 0.6562

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.8303 - accuracy: 0.6771

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.7896 - accuracy: 0.7109

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.7733 - accuracy: 0.7188

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.7531 - accuracy: 0.7344

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 5s - loss: 0.7843 - accuracy: 0.7232

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.7724 - accuracy: 0.7227

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.7519 - accuracy: 0.7292

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.7475 - accuracy: 0.7344

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.7607 - accuracy: 0.7301

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.7461 - accuracy: 0.7370

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.7412 - accuracy: 0.7380

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.7566 - accuracy: 0.7299

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.7400 - accuracy: 0.7354

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.7398 - accuracy: 0.7383

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.7428 - accuracy: 0.7243

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.7577 - accuracy: 0.7170

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.7533 - accuracy: 0.7204

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.7505 - accuracy: 0.7266

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.7473 - accuracy: 0.7247

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.7537 - accuracy: 0.7188

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.7508 - accuracy: 0.7228

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.7521 - accuracy: 0.7227

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.7527 - accuracy: 0.7200

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.7483 - accuracy: 0.7188

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.7496 - accuracy: 0.7176

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.7463 - accuracy: 0.7188

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.7406 - accuracy: 0.7209

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.7375 - accuracy: 0.7240

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.7442 - accuracy: 0.7208

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.7498 - accuracy: 0.7197

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.7430 - accuracy: 0.7235

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.7478 - accuracy: 0.7215

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.7474 - accuracy: 0.7188

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.7460 - accuracy: 0.7179

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.7470 - accuracy: 0.7162

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.7418 - accuracy: 0.7179

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.7378 - accuracy: 0.7196

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.7373 - accuracy: 0.7188

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.7509 - accuracy: 0.7134

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.7482 - accuracy: 0.7165

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.7513 - accuracy: 0.7129

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.7526 - accuracy: 0.7131

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.7525 - accuracy: 0.7125

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.7510 - accuracy: 0.7133

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.7513 - accuracy: 0.7141

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.7500 - accuracy: 0.7135

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.7500 - accuracy: 0.7156

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.7489 - accuracy: 0.7169

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.7478 - accuracy: 0.7181

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.7469 - accuracy: 0.7194

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.7450 - accuracy: 0.7199

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.7469 - accuracy: 0.7176

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.7422 - accuracy: 0.7193

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.7411 - accuracy: 0.7199

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.7420 - accuracy: 0.7193

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.7413 - accuracy: 0.7193

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.7409 - accuracy: 0.7193

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.7432 - accuracy: 0.7182

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.7460 - accuracy: 0.7172

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.7444 - accuracy: 0.7177

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.7509 - accuracy: 0.7148

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.7523 - accuracy: 0.7148

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.7551 - accuracy: 0.7130

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.7576 - accuracy: 0.7125

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.7557 - accuracy: 0.7136

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.7528 - accuracy: 0.7155

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.7536 - accuracy: 0.7142

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.7528 - accuracy: 0.7142

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.7539 - accuracy: 0.7130

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.7555 - accuracy: 0.7126

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.7561 - accuracy: 0.7127

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.7532 - accuracy: 0.7136

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.7523 - accuracy: 0.7153

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.7524 - accuracy: 0.7146

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.7541 - accuracy: 0.7138

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.7554 - accuracy: 0.7143

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.7563 - accuracy: 0.7143

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.7551 - accuracy: 0.7144

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.7536 - accuracy: 0.7152

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.7528 - accuracy: 0.7153

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.7517 - accuracy: 0.7146

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.7486 - accuracy: 0.7150

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.7476 - accuracy: 0.7161

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.7457 - accuracy: 0.7169

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.7444 - accuracy: 0.7169

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.7431 - accuracy: 0.7176

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.7464 - accuracy: 0.7166

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.7473 - accuracy: 0.7149

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.7498 - accuracy: 0.7142

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.7498 - accuracy: 0.7142 - val_loss: 0.7877 - val_accuracy: 0.6785


.. parsed-literal::

    Epoch 7/15


.. parsed-literal::


 1/92 [..............................] - ETA: 8s - loss: 0.8708 - accuracy: 0.6562

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.6871 - accuracy: 0.7188

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6740 - accuracy: 0.7396

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6400 - accuracy: 0.7500

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.7039 - accuracy: 0.7563

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.7394 - accuracy: 0.7344

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.7259 - accuracy: 0.7321

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.7397 - accuracy: 0.7227

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.7078 - accuracy: 0.7396

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6926 - accuracy: 0.7563

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6873 - accuracy: 0.7585

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6861 - accuracy: 0.7604

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6925 - accuracy: 0.7548

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.7057 - accuracy: 0.7455

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.7119 - accuracy: 0.7417

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6981 - accuracy: 0.7461

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.7039 - accuracy: 0.7500

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6966 - accuracy: 0.7517

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.7118 - accuracy: 0.7451

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.7030 - accuracy: 0.7484

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6918 - accuracy: 0.7515

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6896 - accuracy: 0.7514

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.6861 - accuracy: 0.7541

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6834 - accuracy: 0.7539

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6910 - accuracy: 0.7538

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6926 - accuracy: 0.7536

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6992 - accuracy: 0.7512

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.7012 - accuracy: 0.7500

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.7092 - accuracy: 0.7457

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.7087 - accuracy: 0.7469

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.7053 - accuracy: 0.7480

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.7085 - accuracy: 0.7471

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.7160 - accuracy: 0.7434

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.7174 - accuracy: 0.7436

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.7268 - accuracy: 0.7386

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.7247 - accuracy: 0.7389

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.7220 - accuracy: 0.7376

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.7229 - accuracy: 0.7363

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.7237 - accuracy: 0.7358

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.7245 - accuracy: 0.7339

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.7202 - accuracy: 0.7373

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.7169 - accuracy: 0.7390

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.7177 - accuracy: 0.7379

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.7152 - accuracy: 0.7381

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.7230 - accuracy: 0.7350

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.7197 - accuracy: 0.7366

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.7241 - accuracy: 0.7330

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.7235 - accuracy: 0.7340

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.7290 - accuracy: 0.7324

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.7345 - accuracy: 0.7309

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.7325 - accuracy: 0.7325

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.7313 - accuracy: 0.7328

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.7289 - accuracy: 0.7337

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.7273 - accuracy: 0.7352

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.7247 - accuracy: 0.7349

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.7269 - accuracy: 0.7346

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.7265 - accuracy: 0.7354

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.7290 - accuracy: 0.7335

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.7302 - accuracy: 0.7338

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.7289 - accuracy: 0.7346

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.7281 - accuracy: 0.7353

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.7261 - accuracy: 0.7356

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.7252 - accuracy: 0.7353

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.7226 - accuracy: 0.7365

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.7235 - accuracy: 0.7367

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.7224 - accuracy: 0.7392

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.7236 - accuracy: 0.7385

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.7230 - accuracy: 0.7382

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.7252 - accuracy: 0.7375

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.7261 - accuracy: 0.7359

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.7252 - accuracy: 0.7356

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.7230 - accuracy: 0.7363

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.7216 - accuracy: 0.7373

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.7255 - accuracy: 0.7354

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.7237 - accuracy: 0.7360

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.7233 - accuracy: 0.7353

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.7231 - accuracy: 0.7359

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.7238 - accuracy: 0.7353

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.7226 - accuracy: 0.7359

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.7209 - accuracy: 0.7361

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.7205 - accuracy: 0.7370

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.7185 - accuracy: 0.7379

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.7190 - accuracy: 0.7377

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.7177 - accuracy: 0.7389

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.7169 - accuracy: 0.7398

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.7168 - accuracy: 0.7406

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.7168 - accuracy: 0.7407

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.7175 - accuracy: 0.7405

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.7152 - accuracy: 0.7406

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.7162 - accuracy: 0.7404

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.7143 - accuracy: 0.7418

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.7143 - accuracy: 0.7418 - val_loss: 0.7818 - val_accuracy: 0.6894


.. parsed-literal::

    Epoch 8/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.8805 - accuracy: 0.5625

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.8586 - accuracy: 0.6094

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.9007 - accuracy: 0.6562

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.8638 - accuracy: 0.6562

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.8294 - accuracy: 0.6625

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.7996 - accuracy: 0.6771

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.7719 - accuracy: 0.6875

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.7409 - accuracy: 0.7031

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.7085 - accuracy: 0.7083

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6926 - accuracy: 0.7125

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6836 - accuracy: 0.7188

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6705 - accuracy: 0.7344

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6796 - accuracy: 0.7284

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6709 - accuracy: 0.7344

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6820 - accuracy: 0.7354

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6948 - accuracy: 0.7324

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6901 - accuracy: 0.7353

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6826 - accuracy: 0.7344

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6819 - accuracy: 0.7336

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6760 - accuracy: 0.7359

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6767 - accuracy: 0.7366

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6782 - accuracy: 0.7358

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.6750 - accuracy: 0.7378

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6782 - accuracy: 0.7409

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6894 - accuracy: 0.7362

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6848 - accuracy: 0.7380

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6820 - accuracy: 0.7407

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6777 - accuracy: 0.7433

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6821 - accuracy: 0.7414

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6745 - accuracy: 0.7437

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6698 - accuracy: 0.7470

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6688 - accuracy: 0.7461

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6782 - accuracy: 0.7405

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6791 - accuracy: 0.7399

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6923 - accuracy: 0.7366

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6905 - accuracy: 0.7370

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6867 - accuracy: 0.7382

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6859 - accuracy: 0.7393

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6879 - accuracy: 0.7388

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.6823 - accuracy: 0.7406

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6857 - accuracy: 0.7393

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6796 - accuracy: 0.7418

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6786 - accuracy: 0.7420

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6763 - accuracy: 0.7429

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6736 - accuracy: 0.7437

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6795 - accuracy: 0.7418

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6743 - accuracy: 0.7440

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6786 - accuracy: 0.7409

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6918 - accuracy: 0.7360

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6911 - accuracy: 0.7362

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6902 - accuracy: 0.7359

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6892 - accuracy: 0.7356

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6913 - accuracy: 0.7347

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6888 - accuracy: 0.7355

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6879 - accuracy: 0.7352

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6851 - accuracy: 0.7377

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6850 - accuracy: 0.7379

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6881 - accuracy: 0.7381

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6891 - accuracy: 0.7394

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6895 - accuracy: 0.7391

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6929 - accuracy: 0.7387

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6899 - accuracy: 0.7399

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6872 - accuracy: 0.7416

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6868 - accuracy: 0.7422

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6820 - accuracy: 0.7437

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6810 - accuracy: 0.7443

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6796 - accuracy: 0.7444

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6815 - accuracy: 0.7431

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6809 - accuracy: 0.7432

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6813 - accuracy: 0.7433

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6818 - accuracy: 0.7430

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6788 - accuracy: 0.7444

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6774 - accuracy: 0.7440

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6790 - accuracy: 0.7428

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6773 - accuracy: 0.7433

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6752 - accuracy: 0.7443

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6748 - accuracy: 0.7444

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6745 - accuracy: 0.7448

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6752 - accuracy: 0.7441

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6772 - accuracy: 0.7434

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6768 - accuracy: 0.7443

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6776 - accuracy: 0.7443

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6778 - accuracy: 0.7451

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6802 - accuracy: 0.7434

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6784 - accuracy: 0.7438

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6789 - accuracy: 0.7428

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6801 - accuracy: 0.7422

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6778 - accuracy: 0.7437

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6770 - accuracy: 0.7437

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6784 - accuracy: 0.7435

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6814 - accuracy: 0.7415

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.6814 - accuracy: 0.7415 - val_loss: 0.7626 - val_accuracy: 0.7112


.. parsed-literal::

    Epoch 9/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.7865 - accuracy: 0.6875

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.7304 - accuracy: 0.7344

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.7195 - accuracy: 0.7188

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.7386 - accuracy: 0.7344

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.7251 - accuracy: 0.7437

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.7120 - accuracy: 0.7448

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6797 - accuracy: 0.7545

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.7327 - accuracy: 0.7344

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.7338 - accuracy: 0.7396

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.7174 - accuracy: 0.7406

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.7141 - accuracy: 0.7415

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6980 - accuracy: 0.7474

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6912 - accuracy: 0.7524

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6817 - accuracy: 0.7612

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6815 - accuracy: 0.7542

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6619 - accuracy: 0.7617

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6497 - accuracy: 0.7684

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6517 - accuracy: 0.7691

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6459 - accuracy: 0.7714

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6527 - accuracy: 0.7656

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6489 - accuracy: 0.7679

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6447 - accuracy: 0.7727

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.6371 - accuracy: 0.7785

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6404 - accuracy: 0.7760

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6372 - accuracy: 0.7775

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6386 - accuracy: 0.7776

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6373 - accuracy: 0.7766

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6324 - accuracy: 0.7768

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6440 - accuracy: 0.7705

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6414 - accuracy: 0.7729

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6421 - accuracy: 0.7712

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6414 - accuracy: 0.7715

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6405 - accuracy: 0.7727

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6365 - accuracy: 0.7730

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6365 - accuracy: 0.7705

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6303 - accuracy: 0.7726

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6258 - accuracy: 0.7736

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6213 - accuracy: 0.7755

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6228 - accuracy: 0.7748

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.6269 - accuracy: 0.7727

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6282 - accuracy: 0.7713

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6279 - accuracy: 0.7708

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6269 - accuracy: 0.7718

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6277 - accuracy: 0.7720

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6305 - accuracy: 0.7701

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6285 - accuracy: 0.7711

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6261 - accuracy: 0.7726

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6309 - accuracy: 0.7689

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6327 - accuracy: 0.7666

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6296 - accuracy: 0.7675

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6305 - accuracy: 0.7665

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6283 - accuracy: 0.7674

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6304 - accuracy: 0.7653

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6307 - accuracy: 0.7650

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6360 - accuracy: 0.7619

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6391 - accuracy: 0.7606

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6408 - accuracy: 0.7588

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6416 - accuracy: 0.7575

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6362 - accuracy: 0.7606

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6368 - accuracy: 0.7609

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6366 - accuracy: 0.7602

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6377 - accuracy: 0.7591

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6355 - accuracy: 0.7599

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6378 - accuracy: 0.7603

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6359 - accuracy: 0.7601

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6364 - accuracy: 0.7590

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6392 - accuracy: 0.7579

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6401 - accuracy: 0.7578

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6426 - accuracy: 0.7586

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6444 - accuracy: 0.7589

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6480 - accuracy: 0.7575

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6483 - accuracy: 0.7578

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6492 - accuracy: 0.7576

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6487 - accuracy: 0.7584

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6501 - accuracy: 0.7583

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6513 - accuracy: 0.7581

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6507 - accuracy: 0.7588

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6499 - accuracy: 0.7603

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6489 - accuracy: 0.7610

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6523 - accuracy: 0.7597

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6528 - accuracy: 0.7584

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6514 - accuracy: 0.7591

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6515 - accuracy: 0.7593

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6506 - accuracy: 0.7600

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6507 - accuracy: 0.7606

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6514 - accuracy: 0.7601

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6505 - accuracy: 0.7600

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6501 - accuracy: 0.7595

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6492 - accuracy: 0.7587

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6476 - accuracy: 0.7603

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6473 - accuracy: 0.7599

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.6473 - accuracy: 0.7599 - val_loss: 0.7236 - val_accuracy: 0.7207


.. parsed-literal::

    Epoch 10/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.5740 - accuracy: 0.7500

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5935 - accuracy: 0.7656

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5920 - accuracy: 0.7500

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6635 - accuracy: 0.7109

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.6747 - accuracy: 0.7125

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.6427 - accuracy: 0.7448

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6283 - accuracy: 0.7455

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6546 - accuracy: 0.7422

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6286 - accuracy: 0.7569

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6376 - accuracy: 0.7563

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6405 - accuracy: 0.7585

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6312 - accuracy: 0.7630

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6237 - accuracy: 0.7620

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6406 - accuracy: 0.7500

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6407 - accuracy: 0.7500

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6304 - accuracy: 0.7520

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6200 - accuracy: 0.7555

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6103 - accuracy: 0.7587

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6163 - accuracy: 0.7582

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6189 - accuracy: 0.7578

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6263 - accuracy: 0.7515

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6176 - accuracy: 0.7557

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.6378 - accuracy: 0.7459

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6463 - accuracy: 0.7435

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6394 - accuracy: 0.7462

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6360 - accuracy: 0.7512

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6481 - accuracy: 0.7477

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6490 - accuracy: 0.7478

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6455 - accuracy: 0.7500

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6457 - accuracy: 0.7479

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6470 - accuracy: 0.7450

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6387 - accuracy: 0.7490

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6376 - accuracy: 0.7481

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6360 - accuracy: 0.7491

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6348 - accuracy: 0.7500

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6478 - accuracy: 0.7448

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6487 - accuracy: 0.7466

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6503 - accuracy: 0.7459

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6464 - accuracy: 0.7476

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.6422 - accuracy: 0.7500

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6358 - accuracy: 0.7523

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6368 - accuracy: 0.7522

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6329 - accuracy: 0.7536

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6278 - accuracy: 0.7557

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6250 - accuracy: 0.7576

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6247 - accuracy: 0.7588

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6275 - accuracy: 0.7586

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6272 - accuracy: 0.7585

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6232 - accuracy: 0.7602

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6188 - accuracy: 0.7631

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6164 - accuracy: 0.7647

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6179 - accuracy: 0.7650

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6147 - accuracy: 0.7665

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6173 - accuracy: 0.7650

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6130 - accuracy: 0.7665

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6126 - accuracy: 0.7662

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6152 - accuracy: 0.7659

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6182 - accuracy: 0.7629

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6191 - accuracy: 0.7622

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6211 - accuracy: 0.7620

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6216 - accuracy: 0.7618

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6219 - accuracy: 0.7606

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6186 - accuracy: 0.7614

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6187 - accuracy: 0.7607

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6179 - accuracy: 0.7611

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6168 - accuracy: 0.7614

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6160 - accuracy: 0.7607

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6189 - accuracy: 0.7592

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6191 - accuracy: 0.7599

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6182 - accuracy: 0.7602

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6174 - accuracy: 0.7600

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6162 - accuracy: 0.7603

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6152 - accuracy: 0.7614

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6149 - accuracy: 0.7617

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6151 - accuracy: 0.7603

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6145 - accuracy: 0.7610

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6163 - accuracy: 0.7592

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6142 - accuracy: 0.7607

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6116 - accuracy: 0.7625

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6098 - accuracy: 0.7643

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6092 - accuracy: 0.7645

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6145 - accuracy: 0.7647

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6170 - accuracy: 0.7638

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6193 - accuracy: 0.7629

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6220 - accuracy: 0.7609

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6231 - accuracy: 0.7608

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6242 - accuracy: 0.7603

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6252 - accuracy: 0.7599

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6245 - accuracy: 0.7597

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6248 - accuracy: 0.7596

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6268 - accuracy: 0.7592

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.6268 - accuracy: 0.7592 - val_loss: 0.7243 - val_accuracy: 0.7207


.. parsed-literal::

    Epoch 11/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.5004 - accuracy: 0.8438

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5188 - accuracy: 0.8438

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5596 - accuracy: 0.8333

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6025 - accuracy: 0.7812

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.6079 - accuracy: 0.7812

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.5940 - accuracy: 0.7969

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6423 - accuracy: 0.7679

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6230 - accuracy: 0.7773

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6169 - accuracy: 0.7821

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6303 - accuracy: 0.7674

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6434 - accuracy: 0.7580

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6346 - accuracy: 0.7598

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6356 - accuracy: 0.7568

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6246 - accuracy: 0.7606

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6124 - accuracy: 0.7659

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6050 - accuracy: 0.7705

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6044 - accuracy: 0.7729

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5975 - accuracy: 0.7783

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5979 - accuracy: 0.7785

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5977 - accuracy: 0.7771

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5974 - accuracy: 0.7773

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.5837 - accuracy: 0.7843

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5871 - accuracy: 0.7829

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5854 - accuracy: 0.7816

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5881 - accuracy: 0.7755

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5781 - accuracy: 0.7780

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5824 - accuracy: 0.7782

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5826 - accuracy: 0.7783

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5814 - accuracy: 0.7805

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5852 - accuracy: 0.7805

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5898 - accuracy: 0.7746

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5892 - accuracy: 0.7739

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6016 - accuracy: 0.7685

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6007 - accuracy: 0.7689

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5975 - accuracy: 0.7701

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5969 - accuracy: 0.7704

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6047 - accuracy: 0.7666

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6092 - accuracy: 0.7637

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.6073 - accuracy: 0.7626

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6064 - accuracy: 0.7623

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6104 - accuracy: 0.7597

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6046 - accuracy: 0.7639

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6086 - accuracy: 0.7650

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6073 - accuracy: 0.7647

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6072 - accuracy: 0.7650

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6057 - accuracy: 0.7660

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6086 - accuracy: 0.7637

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6050 - accuracy: 0.7667

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6028 - accuracy: 0.7670

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6040 - accuracy: 0.7666

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6074 - accuracy: 0.7645

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6079 - accuracy: 0.7648

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6114 - accuracy: 0.7645

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6097 - accuracy: 0.7671

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6124 - accuracy: 0.7657

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6151 - accuracy: 0.7638

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6176 - accuracy: 0.7624

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6145 - accuracy: 0.7638

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6129 - accuracy: 0.7652

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6118 - accuracy: 0.7654

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6139 - accuracy: 0.7652

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6108 - accuracy: 0.7659

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6083 - accuracy: 0.7667

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6127 - accuracy: 0.7645

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6179 - accuracy: 0.7624

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6177 - accuracy: 0.7626

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6206 - accuracy: 0.7615

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6248 - accuracy: 0.7609

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6231 - accuracy: 0.7625

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6259 - accuracy: 0.7633

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6235 - accuracy: 0.7652

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6247 - accuracy: 0.7650

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6252 - accuracy: 0.7644

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6255 - accuracy: 0.7642

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6241 - accuracy: 0.7644

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6231 - accuracy: 0.7638

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6207 - accuracy: 0.7657

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6195 - accuracy: 0.7671

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6174 - accuracy: 0.7676

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6151 - accuracy: 0.7690

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6135 - accuracy: 0.7699

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6159 - accuracy: 0.7681

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6165 - accuracy: 0.7679

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6142 - accuracy: 0.7692

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6129 - accuracy: 0.7708

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6112 - accuracy: 0.7705

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6089 - accuracy: 0.7714

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6111 - accuracy: 0.7711

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6111 - accuracy: 0.7712

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6089 - accuracy: 0.7724

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6087 - accuracy: 0.7728

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.6087 - accuracy: 0.7728 - val_loss: 0.7648 - val_accuracy: 0.7112


.. parsed-literal::

    Epoch 12/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 1.0327 - accuracy: 0.5625

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.8357 - accuracy: 0.6406

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.7161 - accuracy: 0.7188

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6609 - accuracy: 0.7422

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.6659 - accuracy: 0.7437

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.6649 - accuracy: 0.7396

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6817 - accuracy: 0.7411

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6448 - accuracy: 0.7461

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6305 - accuracy: 0.7569

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6144 - accuracy: 0.7594

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5968 - accuracy: 0.7670

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6061 - accuracy: 0.7604

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6096 - accuracy: 0.7596

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6252 - accuracy: 0.7545

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6285 - accuracy: 0.7604

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6208 - accuracy: 0.7656

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6177 - accuracy: 0.7721

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6112 - accuracy: 0.7778

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6023 - accuracy: 0.7829

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6077 - accuracy: 0.7766

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6062 - accuracy: 0.7730

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.6150 - accuracy: 0.7706

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6158 - accuracy: 0.7711

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6215 - accuracy: 0.7689

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6208 - accuracy: 0.7670

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6196 - accuracy: 0.7640

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6157 - accuracy: 0.7635

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6170 - accuracy: 0.7641

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6153 - accuracy: 0.7668

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6128 - accuracy: 0.7683

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6120 - accuracy: 0.7667

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6120 - accuracy: 0.7653

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6129 - accuracy: 0.7639

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6148 - accuracy: 0.7635

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6105 - accuracy: 0.7640

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6107 - accuracy: 0.7645

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6126 - accuracy: 0.7632

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6090 - accuracy: 0.7669

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.6056 - accuracy: 0.7673

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6074 - accuracy: 0.7676

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6109 - accuracy: 0.7657

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6134 - accuracy: 0.7668

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6080 - accuracy: 0.7686

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6061 - accuracy: 0.7689

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6038 - accuracy: 0.7698

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5996 - accuracy: 0.7714

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6002 - accuracy: 0.7723

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6031 - accuracy: 0.7705

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6054 - accuracy: 0.7695

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6094 - accuracy: 0.7679

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6037 - accuracy: 0.7699

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6016 - accuracy: 0.7701

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6032 - accuracy: 0.7686

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6005 - accuracy: 0.7700

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6032 - accuracy: 0.7679

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6006 - accuracy: 0.7687

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6028 - accuracy: 0.7695

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5982 - accuracy: 0.7723

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5961 - accuracy: 0.7741

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5957 - accuracy: 0.7752

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6028 - accuracy: 0.7713

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6007 - accuracy: 0.7714

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6004 - accuracy: 0.7716

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6022 - accuracy: 0.7717

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5994 - accuracy: 0.7723

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5994 - accuracy: 0.7725

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5960 - accuracy: 0.7731

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5939 - accuracy: 0.7741

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5967 - accuracy: 0.7728

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5957 - accuracy: 0.7721

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5950 - accuracy: 0.7722

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5953 - accuracy: 0.7723

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5941 - accuracy: 0.7733

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5940 - accuracy: 0.7742

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5924 - accuracy: 0.7743

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5925 - accuracy: 0.7744

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5912 - accuracy: 0.7749

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5919 - accuracy: 0.7746

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5908 - accuracy: 0.7743

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5877 - accuracy: 0.7759

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5862 - accuracy: 0.7764

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5887 - accuracy: 0.7764

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5872 - accuracy: 0.7772

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5902 - accuracy: 0.7754

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5876 - accuracy: 0.7766

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5871 - accuracy: 0.7777

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5874 - accuracy: 0.7785

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5870 - accuracy: 0.7789

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5838 - accuracy: 0.7806

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5832 - accuracy: 0.7813

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5850 - accuracy: 0.7800

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.5850 - accuracy: 0.7800 - val_loss: 0.7342 - val_accuracy: 0.7343


.. parsed-literal::

    Epoch 13/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.4550 - accuracy: 0.7812

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5313 - accuracy: 0.7812

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5090 - accuracy: 0.8021

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5141 - accuracy: 0.7812

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.4895 - accuracy: 0.7875

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.5004 - accuracy: 0.7812

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.5125 - accuracy: 0.7902

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5179 - accuracy: 0.7852

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5274 - accuracy: 0.7882

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5448 - accuracy: 0.7844

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5538 - accuracy: 0.7812

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5582 - accuracy: 0.7786

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5492 - accuracy: 0.7812

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5755 - accuracy: 0.7746

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5715 - accuracy: 0.7792

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5620 - accuracy: 0.7871

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5613 - accuracy: 0.7886

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5706 - accuracy: 0.7847

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5721 - accuracy: 0.7862

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5707 - accuracy: 0.7844

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5784 - accuracy: 0.7827

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5686 - accuracy: 0.7869

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.5670 - accuracy: 0.7853

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5748 - accuracy: 0.7812

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5766 - accuracy: 0.7775

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5703 - accuracy: 0.7812

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5644 - accuracy: 0.7824

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5593 - accuracy: 0.7868

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5536 - accuracy: 0.7888

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5467 - accuracy: 0.7917

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5404 - accuracy: 0.7933

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5435 - accuracy: 0.7910

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5474 - accuracy: 0.7888

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5518 - accuracy: 0.7868

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5481 - accuracy: 0.7884

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5445 - accuracy: 0.7908

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5481 - accuracy: 0.7889

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5563 - accuracy: 0.7845

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5566 - accuracy: 0.7869

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.5576 - accuracy: 0.7867

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5621 - accuracy: 0.7835

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5599 - accuracy: 0.7842

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5591 - accuracy: 0.7834

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5536 - accuracy: 0.7862

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5502 - accuracy: 0.7889

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5504 - accuracy: 0.7880

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5547 - accuracy: 0.7852

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5600 - accuracy: 0.7826

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5573 - accuracy: 0.7832

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5540 - accuracy: 0.7837

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5500 - accuracy: 0.7862

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5476 - accuracy: 0.7867

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5456 - accuracy: 0.7877

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5471 - accuracy: 0.7870

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5454 - accuracy: 0.7881

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5413 - accuracy: 0.7891

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5381 - accuracy: 0.7917

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5395 - accuracy: 0.7915

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5427 - accuracy: 0.7903

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5418 - accuracy: 0.7911

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5434 - accuracy: 0.7910

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5438 - accuracy: 0.7903

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5446 - accuracy: 0.7897

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5444 - accuracy: 0.7905

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5426 - accuracy: 0.7913

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5415 - accuracy: 0.7912

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5454 - accuracy: 0.7896

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5460 - accuracy: 0.7900

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5460 - accuracy: 0.7894

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5460 - accuracy: 0.7893

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5484 - accuracy: 0.7879

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5515 - accuracy: 0.7869

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5512 - accuracy: 0.7872

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5525 - accuracy: 0.7860

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5531 - accuracy: 0.7863

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5537 - accuracy: 0.7854

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5524 - accuracy: 0.7858

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5567 - accuracy: 0.7837

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5574 - accuracy: 0.7829

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5557 - accuracy: 0.7829

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5553 - accuracy: 0.7833

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5557 - accuracy: 0.7817

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5565 - accuracy: 0.7810

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5543 - accuracy: 0.7824

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5543 - accuracy: 0.7828

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5536 - accuracy: 0.7821

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5523 - accuracy: 0.7831

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5519 - accuracy: 0.7831

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5509 - accuracy: 0.7838

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5502 - accuracy: 0.7841

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5513 - accuracy: 0.7837

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.5513 - accuracy: 0.7837 - val_loss: 0.6990 - val_accuracy: 0.7480


.. parsed-literal::

    Epoch 14/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.6649 - accuracy: 0.7188

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5909 - accuracy: 0.7969

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5531 - accuracy: 0.8125

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5504 - accuracy: 0.8047

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5108 - accuracy: 0.8125

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.5102 - accuracy: 0.8021

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 5s - loss: 0.5049 - accuracy: 0.8036

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5017 - accuracy: 0.8086

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5248 - accuracy: 0.7986

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5043 - accuracy: 0.8094

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5471 - accuracy: 0.7955

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5545 - accuracy: 0.7865

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5579 - accuracy: 0.7788

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5583 - accuracy: 0.7790

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5389 - accuracy: 0.7854

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5264 - accuracy: 0.7930

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5270 - accuracy: 0.7960

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5333 - accuracy: 0.7951

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5256 - accuracy: 0.7961

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5310 - accuracy: 0.7937

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5334 - accuracy: 0.7917

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5285 - accuracy: 0.7940

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.5268 - accuracy: 0.7948

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5174 - accuracy: 0.7982

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5106 - accuracy: 0.8037

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5186 - accuracy: 0.8005

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5255 - accuracy: 0.7975

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5254 - accuracy: 0.7980

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5252 - accuracy: 0.7974

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5198 - accuracy: 0.7979

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5220 - accuracy: 0.7974

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5228 - accuracy: 0.7979

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5241 - accuracy: 0.7973

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5291 - accuracy: 0.7978

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5341 - accuracy: 0.7955

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5377 - accuracy: 0.7934

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5363 - accuracy: 0.7931

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5349 - accuracy: 0.7936

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5333 - accuracy: 0.7957

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.5298 - accuracy: 0.7977

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5392 - accuracy: 0.7950

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5368 - accuracy: 0.7939

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5331 - accuracy: 0.7958

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5363 - accuracy: 0.7955

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5342 - accuracy: 0.7937

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5367 - accuracy: 0.7928

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5397 - accuracy: 0.7912

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5435 - accuracy: 0.7884

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5432 - accuracy: 0.7889

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5404 - accuracy: 0.7900

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5429 - accuracy: 0.7892

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5479 - accuracy: 0.7873

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5478 - accuracy: 0.7871

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5484 - accuracy: 0.7876

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5471 - accuracy: 0.7881

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5453 - accuracy: 0.7891

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5419 - accuracy: 0.7900

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5436 - accuracy: 0.7883

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5506 - accuracy: 0.7855

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5525 - accuracy: 0.7859

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5489 - accuracy: 0.7879

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5478 - accuracy: 0.7888

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5464 - accuracy: 0.7897

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5490 - accuracy: 0.7881

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5478 - accuracy: 0.7880

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5468 - accuracy: 0.7884

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5454 - accuracy: 0.7882

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5442 - accuracy: 0.7886

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5493 - accuracy: 0.7862

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5502 - accuracy: 0.7862

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5501 - accuracy: 0.7865

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5506 - accuracy: 0.7869

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5485 - accuracy: 0.7877

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5482 - accuracy: 0.7880

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5511 - accuracy: 0.7854

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5502 - accuracy: 0.7858

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5512 - accuracy: 0.7853

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5493 - accuracy: 0.7857

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5512 - accuracy: 0.7845

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5499 - accuracy: 0.7848

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5493 - accuracy: 0.7859

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5494 - accuracy: 0.7866

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5511 - accuracy: 0.7869

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5510 - accuracy: 0.7865

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5523 - accuracy: 0.7864

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5506 - accuracy: 0.7871

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5502 - accuracy: 0.7870

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5542 - accuracy: 0.7849

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5533 - accuracy: 0.7852

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5532 - accuracy: 0.7855

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5526 - accuracy: 0.7861

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.5526 - accuracy: 0.7861 - val_loss: 0.7637 - val_accuracy: 0.7139


.. parsed-literal::

    Epoch 15/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.5513 - accuracy: 0.7812

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5177 - accuracy: 0.8125

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5096 - accuracy: 0.8125

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 4s - loss: 0.4752 - accuracy: 0.8355

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.4979 - accuracy: 0.8261

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.4825 - accuracy: 0.8194

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.4852 - accuracy: 0.8145

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.4888 - accuracy: 0.8036

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.4903 - accuracy: 0.8045

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.4805 - accuracy: 0.8081

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.4807 - accuracy: 0.8112

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.4831 - accuracy: 0.8113

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.4690 - accuracy: 0.8159

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.4656 - accuracy: 0.8199

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.4811 - accuracy: 0.8155

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.4729 - accuracy: 0.8209

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.4906 - accuracy: 0.8169

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.4903 - accuracy: 0.8183

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.4945 - accuracy: 0.8165

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.4933 - accuracy: 0.8178

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.4921 - accuracy: 0.8175

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.4917 - accuracy: 0.8173

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.4913 - accuracy: 0.8197

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.4971 - accuracy: 0.8182

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.4916 - accuracy: 0.8216

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.4904 - accuracy: 0.8236

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.4831 - accuracy: 0.8266

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.4818 - accuracy: 0.8272

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.4766 - accuracy: 0.8288

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.4793 - accuracy: 0.8262

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.4847 - accuracy: 0.8287

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.4831 - accuracy: 0.8282

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.4851 - accuracy: 0.8269

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.4880 - accuracy: 0.8255

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.4946 - accuracy: 0.8217

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.4900 - accuracy: 0.8231

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.4949 - accuracy: 0.8187

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.4969 - accuracy: 0.8169

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.4973 - accuracy: 0.8160

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.4963 - accuracy: 0.8152

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5040 - accuracy: 0.8136

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5071 - accuracy: 0.8129

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5116 - accuracy: 0.8114

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5125 - accuracy: 0.8115

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5142 - accuracy: 0.8108

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5187 - accuracy: 0.8082

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5264 - accuracy: 0.8043

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5268 - accuracy: 0.8045

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5309 - accuracy: 0.8040

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5279 - accuracy: 0.8048

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5281 - accuracy: 0.8050

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5301 - accuracy: 0.8045

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5325 - accuracy: 0.8052

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5327 - accuracy: 0.8048

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5275 - accuracy: 0.8072

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5242 - accuracy: 0.8084

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5237 - accuracy: 0.8074

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5269 - accuracy: 0.8064

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5248 - accuracy: 0.8075

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5285 - accuracy: 0.8050

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5285 - accuracy: 0.8041

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5277 - accuracy: 0.8038

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5287 - accuracy: 0.8034

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5295 - accuracy: 0.8036

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5283 - accuracy: 0.8047

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5282 - accuracy: 0.8052

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5254 - accuracy: 0.8063

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5269 - accuracy: 0.8055

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5242 - accuracy: 0.8065

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5244 - accuracy: 0.8052

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5234 - accuracy: 0.8053

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5232 - accuracy: 0.8050

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5213 - accuracy: 0.8064

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5197 - accuracy: 0.8077

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5190 - accuracy: 0.8078

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5211 - accuracy: 0.8066

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5191 - accuracy: 0.8079

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5176 - accuracy: 0.8079

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5250 - accuracy: 0.8056

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5235 - accuracy: 0.8065

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5239 - accuracy: 0.8054

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5253 - accuracy: 0.8040

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5238 - accuracy: 0.8045

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5205 - accuracy: 0.8057

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5228 - accuracy: 0.8043

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5234 - accuracy: 0.8037

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5244 - accuracy: 0.8031

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5243 - accuracy: 0.8032

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5238 - accuracy: 0.8029

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5240 - accuracy: 0.8027

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5243 - accuracy: 0.8028

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.5243 - accuracy: 0.8028 - val_loss: 0.6708 - val_accuracy: 0.7520


Visualize Training Results
--------------------------



After applying data augmentation and Dropout, there is less overfitting
than before, and training and validation accuracy are closer aligned.

.. code:: ipython3

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.show()



.. image:: tensorflow-training-openvino-with-output_files/tensorflow-training-openvino-with-output_66_0.png


Predict on New Data
-------------------



Finally, let us use the model to classify an image that was not included
in the training or validation sets.

   **Note**: Data augmentation and Dropout layers are inactive at
   inference time.

.. code:: ipython3

    sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
    sunflower_path = tf.keras.utils.get_file("Red_sunflower", origin=sunflower_url)

    img = tf.keras.preprocessing.image.load_img(sunflower_path, target_size=(img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))


.. parsed-literal::


    1/1 [==============================] - ETA: 0s

.. parsed-literal::

    
1/1 [==============================] - 0s 76ms/step


.. parsed-literal::

    This image most likely belongs to sunflowers with a 95.17 percent confidence.


Save the TensorFlow Model
-------------------------



.. code:: ipython3

    # save the trained model - a new folder flower will be created
    # and the file "saved_model.h5" is the pre-trained model
    model_dir = "model"
    saved_model_path = f"{model_dir}/flower/saved_model"
    model.save(saved_model_path)


.. parsed-literal::

    2024-04-18 01:16:33.043830: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'random_flip_input' with dtype float and shape [?,180,180,3]
    	 [[{{node random_flip_input}}]]
    2024-04-18 01:16:33.140167: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-04-18 01:16:33.150389: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'random_flip_input' with dtype float and shape [?,180,180,3]
    	 [[{{node random_flip_input}}]]
    2024-04-18 01:16:33.162042: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-04-18 01:16:33.169331: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-04-18 01:16:33.176583: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-04-18 01:16:33.188085: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-04-18 01:16:33.229110: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'sequential_1_input' with dtype float and shape [?,180,180,3]
    	 [[{{node sequential_1_input}}]]


.. parsed-literal::

    2024-04-18 01:16:33.301829: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-04-18 01:16:33.323839: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'sequential_1_input' with dtype float and shape [?,180,180,3]
    	 [[{{node sequential_1_input}}]]
    2024-04-18 01:16:33.365804: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,22,22,64]
    	 [[{{node inputs}}]]
    2024-04-18 01:16:33.390823: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-04-18 01:16:33.480089: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]


.. parsed-literal::

    2024-04-18 01:16:33.638699: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-04-18 01:16:33.790660: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,22,22,64]
    	 [[{{node inputs}}]]
    2024-04-18 01:16:33.827134: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]


.. parsed-literal::

    2024-04-18 01:16:33.858808: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-04-18 01:16:33.909722: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
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
    ir_model = ov.convert_model(saved_model_path, input=[1, 180, 180, 3])
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
        value="AUTO",
        description="Device:",
        disabled=False,
    )

    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



.. code:: ipython3

    class_names = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

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
    file_path = Path(OUTPUT_DIR) / Path(inp_file_name)

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
    print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))


.. parsed-literal::

    'output/A_Close_Up_Photo_of_a_Dandelion.jpg' already exists.
    (1, 180, 180, 3)
    [1,180,180,3]
    This image most likely belongs to dandelion with a 99.97 percent confidence.



.. image:: tensorflow-training-openvino-with-output_files/tensorflow-training-openvino-with-output_79_1.png


The Next Steps
--------------



This tutorial showed how to train a TensorFlow model, how to convert
that model to OpenVINO’s IR format, and how to do inference on the
converted model. For faster inference speed, you can quantize the IR
model. To see how to quantize this model with OpenVINO’s `Post-training
Quantization with NNCF
Tool <https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.html>`__,
check out the `Post-Training Quantization with TensorFlow Classification
Model <./tensorflow-training-openvino-nncf.ipynb>`__ notebook.
