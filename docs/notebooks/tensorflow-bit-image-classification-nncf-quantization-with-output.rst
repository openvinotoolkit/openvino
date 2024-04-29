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

    import platform
    
    %pip install -q "tensorflow-macos>=2.5; sys_platform == 'darwin' and platform_machine == 'arm64' and python_version > '3.8'" # macOS M1 and M2
    %pip install -q "tensorflow-macos>=2.5,<=2.12.0; sys_platform == 'darwin' and platform_machine == 'arm64' and python_version <= '3.8'" # macOS M1 and M2
    %pip install -q "tensorflow>=2.5; sys_platform == 'darwin' and platform_machine != 'arm64' and python_version > '3.8'" # macOS x86
    %pip install -q "tensorflow>=2.5,<=2.12.0; sys_platform == 'darwin' and platform_machine != 'arm64' and python_version <= '3.8'" # macOS x86
    %pip install -q "tensorflow>=2.5; sys_platform != 'darwin' and python_version > '3.8'"
    %pip install -q "tensorflow>=2.5,<=2.12.0; sys_platform != 'darwin' and python_version <= '3.8'"
    
    %pip install -q "openvino>=2024.0.0" "nncf>=2.7.0" "tensorflow-hub>=0.15.0" "tensorflow_datasets" tf_keras
    %pip install -q "scikit-learn>=1.3.2"
    
    if platform.system() != "Windows":
        %pip install -q "matplotlib>=3.4"
    else:
        %pip install -q "matplotlib>=3.4,<3.7"


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


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. code:: ipython3

    import os
    import numpy as np
    from pathlib import Path
    
    from openvino.runtime import Core
    import openvino as ov
    import nncf
    import logging
    
    from nncf.common.logging.logger import set_log_level
    
    set_log_level(logging.ERROR)
    
    from sklearn.metrics import accuracy_score
    
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TFHUB_CACHE_DIR"] = str(Path("./tfhub_modules").resolve())
    
    import tensorflow as tf
    import tensorflow_datasets as tfds
    import tensorflow_hub as hub
    
    tfds.core.utils.gcs_utils._is_gcs_disabled = True
    os.environ["NO_GCE_CHECK"] = "true"


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

    datasets, datasets_info = tfds.load(
        "imagenette/160px",
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
        read_config=tfds.ReadConfig(shuffle_seed=0),
    )
    train_ds, validation_ds = datasets["train"], datasets["validation"]


.. parsed-literal::

    2024-04-17 23:07:11.869026: E tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:266] failed call to cuInit: CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: forward compatibility was attempted on non supported HW
    2024-04-17 23:07:11.869251: E tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:312] kernel version 470.182.3 does not match DSO version 470.223.2 -- cannot find working devices in this configuration


.. code:: ipython3

    def preprocessing(image, label):
        image = tf.image.resize(image, IMG_SIZE)
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.one_hot(label, NUM_CLASSES)
        return image, label
    
    
    train_dataset = train_ds.map(preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(TRAINING_BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    validation_dataset = (
        validation_ds.map(preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(TRAINING_BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    )

.. code:: ipython3

    # Class labels dictionary with imagenette sample names and classes
    lbl_dict = dict(
        n01440764="tench",
        n02102040="English springer",
        n02979186="cassette player",
        n03000684="chain saw",
        n03028079="church",
        n03394916="French horn",
        n03417042="garbage truck",
        n03425413="gas pump",
        n03445777="golf ball",
        n03888257="parachute",
    )
    
    # Imagenette samples name index
    class_idx_dict = [
        "n01440764",
        "n02102040",
        "n02979186",
        "n03000684",
        "n03028079",
        "n03394916",
        "n03417042",
        "n03425413",
        "n03445777",
        "n03888257",
    ]
    
    
    def label_func(key):
        return lbl_dict[key]

Plotting data samples
~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    import matplotlib.pyplot as plt
    
    # Get the class labels from the dataset info
    class_labels = datasets_info.features["label"].names
    
    # Display labels along with the examples
    num_examples_to_display = 4
    fig, axes = plt.subplots(nrows=1, ncols=num_examples_to_display, figsize=(10, 5))
    
    for i, (image, label_index) in enumerate(train_ds.take(num_examples_to_display)):
        label_name = class_labels[label_index.numpy()]
    
        axes[i].imshow(image.numpy())
        axes[i].set_title(f"{label_func(label_name)}")
        axes[i].axis("off")
        plt.tight_layout()
    plt.show()



.. image:: tensorflow-bit-image-classification-nncf-quantization-with-output_files/tensorflow-bit-image-classification-nncf-quantization-with-output_9_0.png


.. code:: ipython3

    # Get the class labels from the dataset info
    class_labels = datasets_info.features["label"].names
    
    # Display labels along with the examples
    num_examples_to_display = 4
    fig, axes = plt.subplots(nrows=1, ncols=num_examples_to_display, figsize=(10, 5))
    
    for i, (image, label_index) in enumerate(validation_ds.take(num_examples_to_display)):
        label_name = class_labels[label_index.numpy()]
    
        axes[i].imshow(image.numpy())
        axes[i].set_title(f"{label_func(label_name)}")
        axes[i].axis("off")
        plt.tight_layout()
    plt.show()



.. image:: tensorflow-bit-image-classification-nncf-quantization-with-output_files/tensorflow-bit-image-classification-nncf-quantization-with-output_10_0.png


Model Fine-tuning
~~~~~~~~~~~~~~~~~



.. code:: ipython3

    # Load the Big Transfer model
    bit_model_url = "https://www.kaggle.com/models/google/bit/frameworks/TensorFlow2/variations/m-r50x1/versions/1"
    bit_m = hub.KerasLayer(bit_model_url, trainable=True)
    
    # Customize the model for the new task
    model = tf.keras.Sequential([bit_m, tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")])
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    # Fine-tune the model
    model.fit(
        train_dataset.take(3000),
        epochs=FINE_TUNING_STEPS,
        validation_data=validation_dataset.take(1000),
    )
    model.save("./bit_tf_model/", save_format="tf")


.. parsed-literal::

    
  1/101 [..............................] - ETA: 45:24 - loss: 6.0479 - accuracy: 0.0469

.. parsed-literal::

    
  2/101 [..............................] - ETA: 15:13 - loss: 5.6065 - accuracy: 0.0820

.. parsed-literal::

    
  3/101 [..............................] - ETA: 15:03 - loss: 5.2138 - accuracy: 0.0964

.. parsed-literal::

    
  4/101 [>.............................] - ETA: 14:54 - loss: 4.8722 - accuracy: 0.1016

.. parsed-literal::

    
  5/101 [>.............................] - ETA: 14:44 - loss: 4.4679 - accuracy: 0.1219

.. parsed-literal::

    
  6/101 [>.............................] - ETA: 14:35 - loss: 4.1461 - accuracy: 0.1510

.. parsed-literal::

    
  7/101 [=>............................] - ETA: 14:26 - loss: 3.8548 - accuracy: 0.1953

.. parsed-literal::

    
  8/101 [=>............................] - ETA: 14:16 - loss: 3.5996 - accuracy: 0.2373

.. parsed-literal::

    
  9/101 [=>............................] - ETA: 14:07 - loss: 3.4021 - accuracy: 0.2700

.. parsed-literal::

    
 10/101 [=>............................] - ETA: 13:58 - loss: 3.1909 - accuracy: 0.3063

.. parsed-literal::

    
 11/101 [==>...........................] - ETA: 13:48 - loss: 2.9949 - accuracy: 0.3381

.. parsed-literal::

    
 12/101 [==>...........................] - ETA: 13:39 - loss: 2.8219 - accuracy: 0.3698

.. parsed-literal::

    
 13/101 [==>...........................] - ETA: 13:30 - loss: 2.6598 - accuracy: 0.4020

.. parsed-literal::

    
 14/101 [===>..........................] - ETA: 13:20 - loss: 2.5312 - accuracy: 0.4269

.. parsed-literal::

    
 15/101 [===>..........................] - ETA: 13:11 - loss: 2.4153 - accuracy: 0.4510

.. parsed-literal::

    
 16/101 [===>..........................] - ETA: 13:02 - loss: 2.3011 - accuracy: 0.4761

.. parsed-literal::

    
 17/101 [====>.........................] - ETA: 12:53 - loss: 2.1989 - accuracy: 0.4959

.. parsed-literal::

    
 18/101 [====>.........................] - ETA: 12:44 - loss: 2.1012 - accuracy: 0.5161

.. parsed-literal::

    
 19/101 [====>.........................] - ETA: 12:35 - loss: 2.0128 - accuracy: 0.5354

.. parsed-literal::

    
 20/101 [====>.........................] - ETA: 12:26 - loss: 1.9297 - accuracy: 0.5543

.. parsed-literal::

    
 21/101 [=====>........................] - ETA: 12:16 - loss: 1.8573 - accuracy: 0.5707

.. parsed-literal::

    
 22/101 [=====>........................] - ETA: 12:07 - loss: 1.7837 - accuracy: 0.5870

.. parsed-literal::

    
 23/101 [=====>........................] - ETA: 11:58 - loss: 1.7245 - accuracy: 0.5999

.. parsed-literal::

    
 24/101 [======>.......................] - ETA: 11:49 - loss: 1.6608 - accuracy: 0.6146

.. parsed-literal::

    
 25/101 [======>.......................] - ETA: 11:39 - loss: 1.6048 - accuracy: 0.6263

.. parsed-literal::

    
 26/101 [======>.......................] - ETA: 11:30 - loss: 1.5509 - accuracy: 0.6385

.. parsed-literal::

    
 27/101 [=======>......................] - ETA: 11:21 - loss: 1.5013 - accuracy: 0.6493

.. parsed-literal::

    
 28/101 [=======>......................] - ETA: 11:12 - loss: 1.4595 - accuracy: 0.6585

.. parsed-literal::

    
 29/101 [=======>......................] - ETA: 11:03 - loss: 1.4204 - accuracy: 0.6676

.. parsed-literal::

    
 30/101 [=======>......................] - ETA: 10:54 - loss: 1.3813 - accuracy: 0.6766

.. parsed-literal::

    
 31/101 [========>.....................] - ETA: 10:44 - loss: 1.3424 - accuracy: 0.6852

.. parsed-literal::

    
 32/101 [========>.....................] - ETA: 10:35 - loss: 1.3107 - accuracy: 0.6926

.. parsed-literal::

    
 33/101 [========>.....................] - ETA: 10:26 - loss: 1.2751 - accuracy: 0.7005

.. parsed-literal::

    
 34/101 [=========>....................] - ETA: 10:17 - loss: 1.2508 - accuracy: 0.7063

.. parsed-literal::

    
 35/101 [=========>....................] - ETA: 10:08 - loss: 1.2218 - accuracy: 0.7123

.. parsed-literal::

    
 36/101 [=========>....................] - ETA: 9:58 - loss: 1.1916 - accuracy: 0.7192 

.. parsed-literal::

    
 37/101 [=========>....................] - ETA: 9:49 - loss: 1.1619 - accuracy: 0.7259

.. parsed-literal::

    
 38/101 [==========>...................] - ETA: 9:40 - loss: 1.1351 - accuracy: 0.7317

.. parsed-literal::

    
 39/101 [==========>...................] - ETA: 9:31 - loss: 1.1082 - accuracy: 0.7378

.. parsed-literal::

    
 40/101 [==========>...................] - ETA: 9:22 - loss: 1.0851 - accuracy: 0.7428

.. parsed-literal::

    
 41/101 [===========>..................] - ETA: 9:12 - loss: 1.0637 - accuracy: 0.7471

.. parsed-literal::

    
 42/101 [===========>..................] - ETA: 9:03 - loss: 1.0405 - accuracy: 0.7524

.. parsed-literal::

    
 43/101 [===========>..................] - ETA: 8:54 - loss: 1.0197 - accuracy: 0.7571

.. parsed-literal::

    
 44/101 [============>.................] - ETA: 8:45 - loss: 0.9996 - accuracy: 0.7617

.. parsed-literal::

    
 45/101 [============>.................] - ETA: 8:35 - loss: 0.9804 - accuracy: 0.7665

.. parsed-literal::

    
 46/101 [============>.................] - ETA: 8:26 - loss: 0.9625 - accuracy: 0.7706

.. parsed-literal::

    
 47/101 [============>.................] - ETA: 8:17 - loss: 0.9440 - accuracy: 0.7748

.. parsed-literal::

    
 48/101 [=============>................] - ETA: 8:08 - loss: 0.9260 - accuracy: 0.7790

.. parsed-literal::

    
 49/101 [=============>................] - ETA: 7:59 - loss: 0.9093 - accuracy: 0.7828

.. parsed-literal::

    
 50/101 [=============>................] - ETA: 7:49 - loss: 0.8950 - accuracy: 0.7859

.. parsed-literal::

    
 51/101 [==============>...............] - ETA: 7:40 - loss: 0.8806 - accuracy: 0.7891

.. parsed-literal::

    
 52/101 [==============>...............] - ETA: 7:31 - loss: 0.8659 - accuracy: 0.7924

.. parsed-literal::

    
 53/101 [==============>...............] - ETA: 7:22 - loss: 0.8514 - accuracy: 0.7953

.. parsed-literal::

    
 54/101 [===============>..............] - ETA: 7:13 - loss: 0.8371 - accuracy: 0.7986

.. parsed-literal::

    
 55/101 [===============>..............] - ETA: 7:03 - loss: 0.8234 - accuracy: 0.8018

.. parsed-literal::

    
 56/101 [===============>..............] - ETA: 6:54 - loss: 0.8093 - accuracy: 0.8054

.. parsed-literal::

    
 57/101 [===============>..............] - ETA: 6:45 - loss: 0.7989 - accuracy: 0.8070

.. parsed-literal::

    
 58/101 [================>.............] - ETA: 6:36 - loss: 0.7859 - accuracy: 0.8098

.. parsed-literal::

    
 59/101 [================>.............] - ETA: 6:27 - loss: 0.7743 - accuracy: 0.8124

.. parsed-literal::

    
 60/101 [================>.............] - ETA: 6:17 - loss: 0.7642 - accuracy: 0.8148

.. parsed-literal::

    
 61/101 [=================>............] - ETA: 6:08 - loss: 0.7527 - accuracy: 0.8175

.. parsed-literal::

    
 62/101 [=================>............] - ETA: 5:59 - loss: 0.7433 - accuracy: 0.8194

.. parsed-literal::

    
 63/101 [=================>............] - ETA: 5:50 - loss: 0.7338 - accuracy: 0.8214

.. parsed-literal::

    
 64/101 [==================>...........] - ETA: 5:40 - loss: 0.7265 - accuracy: 0.8232

.. parsed-literal::

    
 65/101 [==================>...........] - ETA: 5:31 - loss: 0.7195 - accuracy: 0.8251

.. parsed-literal::

    
 66/101 [==================>...........] - ETA: 5:22 - loss: 0.7111 - accuracy: 0.8269

.. parsed-literal::

    
 67/101 [==================>...........] - ETA: 5:13 - loss: 0.7020 - accuracy: 0.8289

.. parsed-literal::

    
 68/101 [===================>..........] - ETA: 5:04 - loss: 0.6924 - accuracy: 0.8313

.. parsed-literal::

    
 69/101 [===================>..........] - ETA: 4:54 - loss: 0.6833 - accuracy: 0.8334

.. parsed-literal::

    
 70/101 [===================>..........] - ETA: 4:45 - loss: 0.6755 - accuracy: 0.8352

.. parsed-literal::

    
 71/101 [====================>.........] - ETA: 4:36 - loss: 0.6679 - accuracy: 0.8369

.. parsed-literal::

    
 72/101 [====================>.........] - ETA: 4:27 - loss: 0.6598 - accuracy: 0.8388

.. parsed-literal::

    
 73/101 [====================>.........] - ETA: 4:18 - loss: 0.6516 - accuracy: 0.8408

.. parsed-literal::

    
 74/101 [====================>.........] - ETA: 4:08 - loss: 0.6436 - accuracy: 0.8426

.. parsed-literal::

    
 75/101 [=====================>........] - ETA: 3:59 - loss: 0.6378 - accuracy: 0.8441

.. parsed-literal::

    
 76/101 [=====================>........] - ETA: 3:50 - loss: 0.6309 - accuracy: 0.8457

.. parsed-literal::

    
 77/101 [=====================>........] - ETA: 3:41 - loss: 0.6255 - accuracy: 0.8469

.. parsed-literal::

    
 78/101 [======================>.......] - ETA: 3:31 - loss: 0.6188 - accuracy: 0.8484

.. parsed-literal::

    
 79/101 [======================>.......] - ETA: 3:22 - loss: 0.6117 - accuracy: 0.8500

.. parsed-literal::

    
 80/101 [======================>.......] - ETA: 3:13 - loss: 0.6043 - accuracy: 0.8518

.. parsed-literal::

    
 81/101 [=======================>......] - ETA: 3:04 - loss: 0.5972 - accuracy: 0.8534

.. parsed-literal::

    
 82/101 [=======================>......] - ETA: 2:55 - loss: 0.5905 - accuracy: 0.8549

.. parsed-literal::

    
 83/101 [=======================>......] - ETA: 2:45 - loss: 0.5847 - accuracy: 0.8562

.. parsed-literal::

    
 84/101 [=======================>......] - ETA: 2:36 - loss: 0.5788 - accuracy: 0.8575

.. parsed-literal::

    
 85/101 [========================>.....] - ETA: 2:27 - loss: 0.5726 - accuracy: 0.8590

.. parsed-literal::

    
 86/101 [========================>.....] - ETA: 2:18 - loss: 0.5673 - accuracy: 0.8606

.. parsed-literal::

    
 87/101 [========================>.....] - ETA: 2:09 - loss: 0.5613 - accuracy: 0.8619

.. parsed-literal::

    
 88/101 [=========================>....] - ETA: 1:59 - loss: 0.5554 - accuracy: 0.8634

.. parsed-literal::

    
 89/101 [=========================>....] - ETA: 1:50 - loss: 0.5494 - accuracy: 0.8649

.. parsed-literal::

    
 90/101 [=========================>....] - ETA: 1:41 - loss: 0.5439 - accuracy: 0.8661

.. parsed-literal::

    
 91/101 [==========================>...] - ETA: 1:32 - loss: 0.5382 - accuracy: 0.8674

.. parsed-literal::

    
 92/101 [==========================>...] - ETA: 1:22 - loss: 0.5326 - accuracy: 0.8688

.. parsed-literal::

    
 93/101 [==========================>...] - ETA: 1:13 - loss: 0.5273 - accuracy: 0.8700

.. parsed-literal::

    
 94/101 [==========================>...] - ETA: 1:04 - loss: 0.5231 - accuracy: 0.8711

.. parsed-literal::

    
 95/101 [===========================>..] - ETA: 55s - loss: 0.5177 - accuracy: 0.8725 

.. parsed-literal::

    
 96/101 [===========================>..] - ETA: 46s - loss: 0.5131 - accuracy: 0.8736

.. parsed-literal::

    
 97/101 [===========================>..] - ETA: 36s - loss: 0.5086 - accuracy: 0.8748

.. parsed-literal::

    
 98/101 [============================>.] - ETA: 27s - loss: 0.5044 - accuracy: 0.8758

.. parsed-literal::

    
 99/101 [============================>.] - ETA: 18s - loss: 0.5013 - accuracy: 0.8767

.. parsed-literal::

    
100/101 [============================>.] - ETA: 9s - loss: 0.4974 - accuracy: 0.8775 

.. parsed-literal::

    
101/101 [==============================] - ETA: 0s - loss: 0.4945 - accuracy: 0.8782

.. parsed-literal::

    
101/101 [==============================] - 962s 9s/step - loss: 0.4945 - accuracy: 0.8782 - val_loss: 0.0819 - val_accuracy: 0.9800


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
        value="AUTO",
        description="Device:",
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
    
    
    val_ds = validation_ds.map(nncf_preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(1).prefetch(tf.data.experimental.AUTOTUNE)
    
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

    Accuracy of the tensorflow model (fp32):  98.00%
    Accuracy of the OpenVINO optimized model (fp32):  98.20%
    Accuracy of the OpenVINO quantized model (int8):  97.00%
    Accuracy drop between OV FP32 and INT8 model: 1.2% 


Compare inference results on one picture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    # Accessing validation sample
    sample_idx = 50
    vds = datasets["validation"]
    
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
    plt.axis("off")
    plt.show()


.. parsed-literal::

    Predicted label for the sample picture by float (fp32) model: gas pump
    


.. parsed-literal::

    Predicted label for the sample picture by qunatized (int8) model: gas pump
    



.. image:: tensorflow-bit-image-classification-nncf-quantization-with-output_files/tensorflow-bit-image-classification-nncf-quantization-with-output_27_2.png

