Post-Training Quantization with TensorFlow Classification Model
===============================================================

This example demonstrates how to quantize the OpenVINO model that was
created in `tensorflow-training-openvino
notebook <tensorflow-training-openvino.ipynb>`__, to improve inference
speed. Quantization is performed with `Post-training Quantization with
NNCF <https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.html>`__.
A custom dataloader and metric will be defined, and accuracy and
performance will be computed for the original IR model and the quantized
model.

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Preparation <#preparation>`__

   -  `Imports <#imports>`__

-  `Post-training Quantization with
   NNCF <#post-training-quantization-with-nncf>`__

   -  `Select inference device <#select-inference-device>`__

-  `Compare Metrics <#compare-metrics>`__
-  `Run Inference on Quantized
   Model <#run-inference-on-quantized-model>`__
-  `Compare Inference Speed <#compare-inference-speed>`__

Preparation
-----------



The notebook requires that the training notebook has been run and that
the Intermediate Representation (IR) models are created. If the IR
models do not exist, running the next cell will run the training
notebook. This will take a while.

.. code:: ipython3

    import platform

    %pip install -q Pillow numpy tqdm nncf "openvino>=2023.1"

    if platform.system() != "Windows":
        %pip install -q "matplotlib>=3.4"
    else:
        %pip install -q "matplotlib>=3.4,<3.7"

    %pip install -q "tensorflow-macos>=2.5; sys_platform == 'darwin' and platform_machine == 'arm64' and python_version > '3.8'" # macOS M1 and M2
    %pip install -q "tensorflow-macos>=2.5,<=2.12.0; sys_platform == 'darwin' and platform_machine == 'arm64' and python_version <= '3.8'" # macOS M1 and M2
    %pip install -q "tensorflow>=2.5; sys_platform == 'darwin' and platform_machine != 'arm64' and python_version > '3.8'" # macOS x86
    %pip install -q "tensorflow>=2.5,<=2.12.0; sys_platform == 'darwin' and platform_machine != 'arm64' and python_version <= '3.8'" # macOS x86
    %pip install -q "tensorflow>=2.5; sys_platform != 'darwin' and python_version > '3.8'"
    %pip install -q "tensorflow>=2.5; sys_platform != 'darwin' and python_version <= '3.8'"
    %pip install -q tf_keras tqdm


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


.. code:: ipython3

    from pathlib import Path
    import os

    os.environ["TF_USE_LEGACY_KERAS"] = "1"


    import tensorflow as tf

    model_xml = Path("model/flower/flower_ir.xml")
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = Path(tf.keras.utils.get_file("flower_photos", origin=dataset_url, untar=True))

    if not model_xml.exists():
        print("Executing training notebook. This will take a while...")
        %run tensorflow-training-openvino.ipynb


.. parsed-literal::

    2024-04-18 01:11:39.893197: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-04-18 01:11:39.928932: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


.. parsed-literal::

    2024-04-18 01:11:40.523686: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


.. parsed-literal::

    Executing training notebook. This will take a while...


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


.. parsed-literal::

    3670


.. parsed-literal::

    Found 3670 files belonging to 5 classes.


.. parsed-literal::

    Using 2936 files for training.


.. parsed-literal::

    2024-04-18 01:12:09.395264: E tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:266] failed call to cuInit: CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: forward compatibility was attempted on non supported HW
    2024-04-18 01:12:09.395300: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:168] retrieving CUDA diagnostic information for host: iotg-dev-workstation-07
    2024-04-18 01:12:09.395304: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:175] hostname: iotg-dev-workstation-07
    2024-04-18 01:12:09.395435: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:199] libcuda reported version is: 470.223.2
    2024-04-18 01:12:09.395449: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:203] kernel reported version is: 470.182.3
    2024-04-18 01:12:09.395453: E tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:312] kernel version 470.182.3 does not match DSO version 470.223.2 -- cannot find working devices in this configuration


.. parsed-literal::

    Found 3670 files belonging to 5 classes.


.. parsed-literal::

    Using 734 files for validation.
    ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']


.. parsed-literal::

    2024-04-18 01:12:09.711710: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]
    2024-04-18 01:12:09.711989: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]



.. image:: tensorflow-training-openvino-nncf-with-output_files/tensorflow-training-openvino-nncf-with-output_3_28.png


.. parsed-literal::

    2024-04-18 01:12:10.676936: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-04-18 01:12:10.677180: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-04-18 01:12:10.818263: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]
    2024-04-18 01:12:10.818550: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]


.. parsed-literal::

    (32, 180, 180, 3)
    (32,)


.. parsed-literal::

    0.0 1.0


.. parsed-literal::

    2024-04-18 01:12:11.693375: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]
    2024-04-18 01:12:11.693703: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]



.. image:: tensorflow-training-openvino-nncf-with-output_files/tensorflow-training-openvino-nncf-with-output_3_33.png


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


.. parsed-literal::

    Epoch 1/15


.. parsed-literal::

    2024-04-18 01:12:12.768860: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-04-18 01:12:12.769608: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]


.. parsed-literal::


 1/92 [..............................] - ETA: 1:32 - loss: 1.6453 - accuracy: 0.1250

.. parsed-literal::

    
 2/92 [..............................] - ETA: 6s - loss: 3.1595 - accuracy: 0.1875

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 3.0478 - accuracy: 0.1771

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 3.0508 - accuracy: 0.1875

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 2.8188 - accuracy: 0.2000

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 2.6227 - accuracy: 0.2135

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 5s - loss: 2.4793 - accuracy: 0.2277

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 5s - loss: 2.3750 - accuracy: 0.2070

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 2.2907 - accuracy: 0.2014

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 2.2244 - accuracy: 0.1906

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 2.1686 - accuracy: 0.1903

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 2.1208 - accuracy: 0.1953

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 2.0802 - accuracy: 0.2019

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 2.0465 - accuracy: 0.2031

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 2.0149 - accuracy: 0.2042

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 1.9883 - accuracy: 0.2090

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 1.9660 - accuracy: 0.2040

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 1.9437 - accuracy: 0.2031

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 1.9263 - accuracy: 0.2039

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 1.9101 - accuracy: 0.2078

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 1.8940 - accuracy: 0.2098

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 1.8793 - accuracy: 0.2116

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 1.8656 - accuracy: 0.2106

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 4s - loss: 1.8528 - accuracy: 0.2109

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 1.8416 - accuracy: 0.2125

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 1.8309 - accuracy: 0.2103

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 1.8192 - accuracy: 0.2164

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 1.8101 - accuracy: 0.2188

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 1.7988 - accuracy: 0.2220

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 1.7844 - accuracy: 0.2240

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 1.7738 - accuracy: 0.2248

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 1.7688 - accuracy: 0.2275

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 1.7630 - accuracy: 0.2254

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 1.7550 - accuracy: 0.2270

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 1.7470 - accuracy: 0.2286

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 1.7391 - accuracy: 0.2370

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 1.7348 - accuracy: 0.2424

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 1.7286 - accuracy: 0.2459

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 1.7229 - accuracy: 0.2532

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 1.7152 - accuracy: 0.2562

.. parsed-literal::

    
41/92 [============>.................] - ETA: 3s - loss: 1.7083 - accuracy: 0.2599

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 1.6998 - accuracy: 0.2612

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 1.6929 - accuracy: 0.2653

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 1.6868 - accuracy: 0.2670

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 1.6784 - accuracy: 0.2701

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 1.6715 - accuracy: 0.2758

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 1.6668 - accuracy: 0.2773

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 1.6591 - accuracy: 0.2799

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 1.6517 - accuracy: 0.2851

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 1.6514 - accuracy: 0.2850

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 1.6496 - accuracy: 0.2886

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 1.6477 - accuracy: 0.2897

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 1.6395 - accuracy: 0.2936

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 1.6327 - accuracy: 0.2957

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 1.6294 - accuracy: 0.2989

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 1.6205 - accuracy: 0.3007

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 1.6147 - accuracy: 0.3025

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 1.6101 - accuracy: 0.3021

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 1.6038 - accuracy: 0.3028

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 1.5967 - accuracy: 0.3076

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 1.5905 - accuracy: 0.3097

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 1.5886 - accuracy: 0.3113

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 1.5849 - accuracy: 0.3113

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 1.5793 - accuracy: 0.3127

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 1.5717 - accuracy: 0.3165

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 1.5682 - accuracy: 0.3169

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 1.5641 - accuracy: 0.3220

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 1.5572 - accuracy: 0.3250

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 1.5522 - accuracy: 0.3280

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 1.5501 - accuracy: 0.3269

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 1.5454 - accuracy: 0.3306

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 1.5428 - accuracy: 0.3325

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 1.5386 - accuracy: 0.3326

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 1.5324 - accuracy: 0.3353

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 1.5266 - accuracy: 0.3383

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 1.5249 - accuracy: 0.3400

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 1.5187 - accuracy: 0.3428

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 1.5143 - accuracy: 0.3448

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 1.5138 - accuracy: 0.3444

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 1.5123 - accuracy: 0.3444

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 1.5104 - accuracy: 0.3448

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 1.5065 - accuracy: 0.3471

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 1.5016 - accuracy: 0.3504

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 1.4957 - accuracy: 0.3544

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 1.4916 - accuracy: 0.3560

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 1.4868 - accuracy: 0.3570

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 1.4843 - accuracy: 0.3586

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 1.4796 - accuracy: 0.3613

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 1.4778 - accuracy: 0.3632

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 1.4775 - accuracy: 0.3633

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 1.4749 - accuracy: 0.3655

.. parsed-literal::

    2024-04-18 01:12:19.128053: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [734]
    	 [[{{node Placeholder/_0}}]]
    2024-04-18 01:12:19.128326: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [734]
    	 [[{{node Placeholder/_0}}]]


.. parsed-literal::

    
92/92 [==============================] - 7s 66ms/step - loss: 1.4749 - accuracy: 0.3655 - val_loss: 1.1456 - val_accuracy: 0.5341


.. parsed-literal::

    Epoch 2/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 1.4784 - accuracy: 0.3125

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 1.3069 - accuracy: 0.4062

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 1.2792 - accuracy: 0.4792

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 1.2735 - accuracy: 0.4375

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 1.2419 - accuracy: 0.4625

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 1.2061 - accuracy: 0.4896

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 1.1802 - accuracy: 0.5000

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 1.2022 - accuracy: 0.4883

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 1.2132 - accuracy: 0.4965

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 1.1943 - accuracy: 0.5156

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 1.2038 - accuracy: 0.5085

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 1.2072 - accuracy: 0.5104

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 1.2088 - accuracy: 0.5120

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 1.1953 - accuracy: 0.5223

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 1.1934 - accuracy: 0.5250

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 1.1870 - accuracy: 0.5312

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 1.1830 - accuracy: 0.5331

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 1.1699 - accuracy: 0.5399

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 1.1824 - accuracy: 0.5312

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 1.1714 - accuracy: 0.5328

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 1.1673 - accuracy: 0.5372

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 1.1658 - accuracy: 0.5369

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 1.1660 - accuracy: 0.5408

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 1.1668 - accuracy: 0.5378

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 1.1689 - accuracy: 0.5362

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 1.1677 - accuracy: 0.5350

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 1.1673 - accuracy: 0.5327

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 1.1636 - accuracy: 0.5348

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 1.1597 - accuracy: 0.5315

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 1.1546 - accuracy: 0.5325

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 1.1615 - accuracy: 0.5266

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 1.1623 - accuracy: 0.5219

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 1.1588 - accuracy: 0.5231

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 1.1526 - accuracy: 0.5279

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 1.1511 - accuracy: 0.5262

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 1.1447 - accuracy: 0.5272

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 1.1423 - accuracy: 0.5265

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 1.1494 - accuracy: 0.5266

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 1.1540 - accuracy: 0.5220

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 1.1530 - accuracy: 0.5215

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 1.1578 - accuracy: 0.5172

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 1.1597 - accuracy: 0.5183

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 1.1599 - accuracy: 0.5171

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 1.1571 - accuracy: 0.5189

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 1.1546 - accuracy: 0.5212

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 1.1557 - accuracy: 0.5214

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 1.1557 - accuracy: 0.5229

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 1.1541 - accuracy: 0.5250

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 1.1559 - accuracy: 0.5214

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 1.1577 - accuracy: 0.5209

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 1.1585 - accuracy: 0.5205

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 1.1588 - accuracy: 0.5225

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 1.1564 - accuracy: 0.5238

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 1.1559 - accuracy: 0.5228

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 1.1582 - accuracy: 0.5219

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 1.1575 - accuracy: 0.5209

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 1.1617 - accuracy: 0.5179

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 1.1581 - accuracy: 0.5197

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 1.1561 - accuracy: 0.5204

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 1.1529 - accuracy: 0.5216

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 1.1494 - accuracy: 0.5233

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 1.1464 - accuracy: 0.5239

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 1.1477 - accuracy: 0.5230

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 1.1483 - accuracy: 0.5236

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 1.1470 - accuracy: 0.5238

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 1.1475 - accuracy: 0.5229

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 1.1462 - accuracy: 0.5235

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 1.1447 - accuracy: 0.5250

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 1.1460 - accuracy: 0.5255

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 1.1434 - accuracy: 0.5265

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 1.1393 - accuracy: 0.5283

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 1.1366 - accuracy: 0.5301

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 1.1359 - accuracy: 0.5305

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 1.1332 - accuracy: 0.5326

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 1.1300 - accuracy: 0.5342

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 1.1273 - accuracy: 0.5338

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 1.1315 - accuracy: 0.5334

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 1.1328 - accuracy: 0.5325

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 1.1328 - accuracy: 0.5333

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 1.1373 - accuracy: 0.5329

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 1.1383 - accuracy: 0.5317

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 1.1364 - accuracy: 0.5321

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 1.1378 - accuracy: 0.5313

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 1.1357 - accuracy: 0.5324

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 1.1355 - accuracy: 0.5317

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 1.1341 - accuracy: 0.5328

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 1.1338 - accuracy: 0.5324

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 1.1349 - accuracy: 0.5324

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 1.1361 - accuracy: 0.5324

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 1.1342 - accuracy: 0.5327

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 1.1322 - accuracy: 0.5341

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 1.1322 - accuracy: 0.5341 - val_loss: 1.0712 - val_accuracy: 0.5708


.. parsed-literal::

    Epoch 3/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 1.0558 - accuracy: 0.6250

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.9953 - accuracy: 0.6875

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.9741 - accuracy: 0.6667

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.9762 - accuracy: 0.6406

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 1.0409 - accuracy: 0.6000

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 1.0258 - accuracy: 0.6146

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 1.0239 - accuracy: 0.6116

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 1.0139 - accuracy: 0.6133

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.9907 - accuracy: 0.6319

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.9815 - accuracy: 0.6344

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.9741 - accuracy: 0.6364

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.9891 - accuracy: 0.6276

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.9823 - accuracy: 0.6250

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.9847 - accuracy: 0.6183

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.9844 - accuracy: 0.6187

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.9729 - accuracy: 0.6211

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.9792 - accuracy: 0.6250

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.9805 - accuracy: 0.6215

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.9759 - accuracy: 0.6250

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 1.0010 - accuracy: 0.6156

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.9915 - accuracy: 0.6161

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.9885 - accuracy: 0.6179

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.9939 - accuracy: 0.6155

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.9814 - accuracy: 0.6172

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.9855 - accuracy: 0.6150

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.9876 - accuracy: 0.6154

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.9934 - accuracy: 0.6123

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.9933 - accuracy: 0.6105

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.9953 - accuracy: 0.6099

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 1.0016 - accuracy: 0.6042

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 1.0062 - accuracy: 0.6008

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 1.0193 - accuracy: 0.5928

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 1.0180 - accuracy: 0.5919

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 1.0210 - accuracy: 0.5928

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 1.0225 - accuracy: 0.5964

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 1.0209 - accuracy: 0.5998

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 1.0220 - accuracy: 0.6014

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 1.0265 - accuracy: 0.6020

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 1.0314 - accuracy: 0.5978

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 1.0301 - accuracy: 0.5992

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 1.0326 - accuracy: 0.5983

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 1.0321 - accuracy: 0.5975

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 1.0316 - accuracy: 0.5959

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 1.0263 - accuracy: 0.5994

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 1.0292 - accuracy: 0.6000

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 1.0306 - accuracy: 0.6012

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 1.0407 - accuracy: 0.5964

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 1.0408 - accuracy: 0.5957

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 1.0388 - accuracy: 0.5957

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 1.0398 - accuracy: 0.5944

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 1.0420 - accuracy: 0.5925

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 1.0420 - accuracy: 0.5931

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 1.0412 - accuracy: 0.5926

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 1.0406 - accuracy: 0.5932

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 1.0416 - accuracy: 0.5932

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 1.0418 - accuracy: 0.5938

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 1.0389 - accuracy: 0.5943

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 1.0415 - accuracy: 0.5938

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 1.0417 - accuracy: 0.5938

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 1.0401 - accuracy: 0.5938

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 1.0405 - accuracy: 0.5953

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 1.0400 - accuracy: 0.5958

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 1.0420 - accuracy: 0.5938

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 1.0483 - accuracy: 0.5913

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 1.0446 - accuracy: 0.5923

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 1.0439 - accuracy: 0.5923

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 1.0423 - accuracy: 0.5928

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 1.0439 - accuracy: 0.5928

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 1.0424 - accuracy: 0.5933

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 1.0417 - accuracy: 0.5951

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 1.0414 - accuracy: 0.5954

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 1.0385 - accuracy: 0.5962

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 1.0407 - accuracy: 0.5970

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 1.0381 - accuracy: 0.5970

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 1.0351 - accuracy: 0.5982

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 1.0349 - accuracy: 0.5989

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 1.0354 - accuracy: 0.5993

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 1.0332 - accuracy: 0.5996

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 1.0320 - accuracy: 0.6003

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 1.0328 - accuracy: 0.6002

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 1.0326 - accuracy: 0.5998

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 1.0308 - accuracy: 0.6012

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 1.0285 - accuracy: 0.6022

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 1.0258 - accuracy: 0.6025

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 1.0265 - accuracy: 0.6013

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 1.0275 - accuracy: 0.6001

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 1.0258 - accuracy: 0.6004

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 1.0281 - accuracy: 0.6004

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 1.0270 - accuracy: 0.6006

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 1.0262 - accuracy: 0.6006

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 1.0246 - accuracy: 0.6008

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 1.0246 - accuracy: 0.6008 - val_loss: 0.9583 - val_accuracy: 0.6349


.. parsed-literal::

    Epoch 4/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 1.2949 - accuracy: 0.4375

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 1.1851 - accuracy: 0.5312

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 1.0969 - accuracy: 0.5521

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 1.0954 - accuracy: 0.5625

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 1.0742 - accuracy: 0.5625

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 1.0892 - accuracy: 0.5573

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 5s - loss: 1.0770 - accuracy: 0.5580

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 1.0740 - accuracy: 0.5586

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 1.0555 - accuracy: 0.5660

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 1.0522 - accuracy: 0.5688

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 1.0447 - accuracy: 0.5795

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 1.0230 - accuracy: 0.5859

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 1.0052 - accuracy: 0.5962

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 1.0053 - accuracy: 0.6004

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.9968 - accuracy: 0.6021

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.9991 - accuracy: 0.5977

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.9962 - accuracy: 0.5956

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.9968 - accuracy: 0.5972

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.9931 - accuracy: 0.5954

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.9957 - accuracy: 0.6016

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.9932 - accuracy: 0.6012

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.9864 - accuracy: 0.6051

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.9823 - accuracy: 0.6087

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.9748 - accuracy: 0.6120

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.9821 - accuracy: 0.6125

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.9895 - accuracy: 0.6070

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.9889 - accuracy: 0.6134

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.9822 - accuracy: 0.6172

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.9763 - accuracy: 0.6207

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.9738 - accuracy: 0.6208

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.9703 - accuracy: 0.6240

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.9664 - accuracy: 0.6270

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.9625 - accuracy: 0.6288

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.9663 - accuracy: 0.6296

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.9708 - accuracy: 0.6277

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.9707 - accuracy: 0.6276

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.9713 - accuracy: 0.6284

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.9679 - accuracy: 0.6308

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.9612 - accuracy: 0.6338

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.9624 - accuracy: 0.6313

.. parsed-literal::

    
41/92 [============>.................] - ETA: 3s - loss: 0.9625 - accuracy: 0.6311

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.9611 - accuracy: 0.6302

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.9582 - accuracy: 0.6330

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.9686 - accuracy: 0.6293

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.9665 - accuracy: 0.6285

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.9652 - accuracy: 0.6284

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.9583 - accuracy: 0.6323

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.9600 - accuracy: 0.6315

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.9625 - accuracy: 0.6314

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.9624 - accuracy: 0.6319

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.9600 - accuracy: 0.6330

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.9607 - accuracy: 0.6322

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.9606 - accuracy: 0.6315

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.9571 - accuracy: 0.6331

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.9575 - accuracy: 0.6347

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.9535 - accuracy: 0.6367

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.9556 - accuracy: 0.6354

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.9570 - accuracy: 0.6342

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.9584 - accuracy: 0.6361

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.9560 - accuracy: 0.6359

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.9540 - accuracy: 0.6368

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.9507 - accuracy: 0.6386

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.9495 - accuracy: 0.6394

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.9518 - accuracy: 0.6382

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.9512 - accuracy: 0.6394

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.9484 - accuracy: 0.6411

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.9505 - accuracy: 0.6399

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.9517 - accuracy: 0.6392

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.9484 - accuracy: 0.6404

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.9501 - accuracy: 0.6402

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.9461 - accuracy: 0.6417

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.9494 - accuracy: 0.6398

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.9549 - accuracy: 0.6370

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.9564 - accuracy: 0.6356

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.9540 - accuracy: 0.6367

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.9537 - accuracy: 0.6365

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.9539 - accuracy: 0.6372

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.9534 - accuracy: 0.6365

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.9550 - accuracy: 0.6364

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.9559 - accuracy: 0.6347

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.9589 - accuracy: 0.6346

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.9601 - accuracy: 0.6337

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.9580 - accuracy: 0.6340

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.9577 - accuracy: 0.6331

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.9584 - accuracy: 0.6319

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.9568 - accuracy: 0.6329

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.9564 - accuracy: 0.6321

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.9570 - accuracy: 0.6324

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.9557 - accuracy: 0.6334

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.9555 - accuracy: 0.6340

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.9604 - accuracy: 0.6318

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.9604 - accuracy: 0.6318 - val_loss: 0.9705 - val_accuracy: 0.6308


.. parsed-literal::

    Epoch 5/15


.. parsed-literal::


 1/92 [..............................] - ETA: 6s - loss: 0.9688 - accuracy: 0.7500

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.8982 - accuracy: 0.7500

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.8791 - accuracy: 0.7188

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.8947 - accuracy: 0.7031

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.9190 - accuracy: 0.6687

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.8789 - accuracy: 0.6771

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 5s - loss: 0.8793 - accuracy: 0.6830

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 5s - loss: 0.8757 - accuracy: 0.6797

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 5s - loss: 0.8925 - accuracy: 0.6701

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 5s - loss: 0.9010 - accuracy: 0.6594

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.9055 - accuracy: 0.6562

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.9024 - accuracy: 0.6536

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.8993 - accuracy: 0.6611

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.8962 - accuracy: 0.6585

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.8879 - accuracy: 0.6583

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.9131 - accuracy: 0.6426

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.9255 - accuracy: 0.6379

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.9209 - accuracy: 0.6406

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.9128 - accuracy: 0.6414

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.9060 - accuracy: 0.6453

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.9091 - accuracy: 0.6488

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.9110 - accuracy: 0.6491

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.9085 - accuracy: 0.6508

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 4s - loss: 0.9065 - accuracy: 0.6523

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.8992 - accuracy: 0.6525

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.9081 - accuracy: 0.6466

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.9077 - accuracy: 0.6481

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.9098 - accuracy: 0.6496

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.9049 - accuracy: 0.6530

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.9001 - accuracy: 0.6552

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.8972 - accuracy: 0.6573

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.8884 - accuracy: 0.6611

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.8936 - accuracy: 0.6610

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.8929 - accuracy: 0.6608

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.8989 - accuracy: 0.6571

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.9004 - accuracy: 0.6562

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.8991 - accuracy: 0.6571

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.8939 - accuracy: 0.6587

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.9007 - accuracy: 0.6554

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.8991 - accuracy: 0.6555

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.8958 - accuracy: 0.6555

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.8924 - accuracy: 0.6577

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.8966 - accuracy: 0.6541

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.8950 - accuracy: 0.6555

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.8990 - accuracy: 0.6549

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.8978 - accuracy: 0.6562

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.8962 - accuracy: 0.6549

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.8979 - accuracy: 0.6536

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.9008 - accuracy: 0.6537

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.9017 - accuracy: 0.6531

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.8987 - accuracy: 0.6556

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.8980 - accuracy: 0.6562

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.8973 - accuracy: 0.6557

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.8989 - accuracy: 0.6545

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.9012 - accuracy: 0.6523

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.9044 - accuracy: 0.6512

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.9028 - accuracy: 0.6530

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.9081 - accuracy: 0.6514

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.9063 - accuracy: 0.6515

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.9033 - accuracy: 0.6536

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.9079 - accuracy: 0.6506

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.9081 - accuracy: 0.6507

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.9059 - accuracy: 0.6518

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.9039 - accuracy: 0.6509

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.9062 - accuracy: 0.6495

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.9062 - accuracy: 0.6501

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.9043 - accuracy: 0.6488

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.9044 - accuracy: 0.6494

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.9044 - accuracy: 0.6495

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.9053 - accuracy: 0.6496

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.9019 - accuracy: 0.6514

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.9011 - accuracy: 0.6528

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.9018 - accuracy: 0.6528

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.9024 - accuracy: 0.6533

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.9009 - accuracy: 0.6546

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.9038 - accuracy: 0.6517

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.9029 - accuracy: 0.6526

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.9031 - accuracy: 0.6532

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.9024 - accuracy: 0.6532

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.9035 - accuracy: 0.6521

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.9061 - accuracy: 0.6506

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.9040 - accuracy: 0.6503

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.9017 - accuracy: 0.6504

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.9030 - accuracy: 0.6497

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.9042 - accuracy: 0.6487

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.9049 - accuracy: 0.6491

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.9035 - accuracy: 0.6499

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.9008 - accuracy: 0.6514

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.9014 - accuracy: 0.6518

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.9004 - accuracy: 0.6515

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.9002 - accuracy: 0.6516

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.9002 - accuracy: 0.6516 - val_loss: 0.8861 - val_accuracy: 0.6512


.. parsed-literal::

    Epoch 6/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.7376 - accuracy: 0.7812

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.8019 - accuracy: 0.6562

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.7662 - accuracy: 0.6875

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.8261 - accuracy: 0.6797

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.8438 - accuracy: 0.6438

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.8973 - accuracy: 0.6198

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 5s - loss: 0.8784 - accuracy: 0.6339

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.8579 - accuracy: 0.6445

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.8367 - accuracy: 0.6562

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.8337 - accuracy: 0.6625

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.8314 - accuracy: 0.6676

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.8254 - accuracy: 0.6745

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.8425 - accuracy: 0.6635

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.8482 - accuracy: 0.6607

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.8387 - accuracy: 0.6667

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.8313 - accuracy: 0.6660

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.8200 - accuracy: 0.6710

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.8306 - accuracy: 0.6649

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.8374 - accuracy: 0.6612

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.8466 - accuracy: 0.6594

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.8448 - accuracy: 0.6577

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.8360 - accuracy: 0.6619

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.8271 - accuracy: 0.6671

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.8202 - accuracy: 0.6732

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.8271 - accuracy: 0.6762

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.8231 - accuracy: 0.6755

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.8248 - accuracy: 0.6736

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.8303 - accuracy: 0.6741

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.8456 - accuracy: 0.6659

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.8445 - accuracy: 0.6677

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.8365 - accuracy: 0.6734

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.8398 - accuracy: 0.6758

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.8439 - accuracy: 0.6752

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.8499 - accuracy: 0.6737

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.8491 - accuracy: 0.6750

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.8461 - accuracy: 0.6771

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.8472 - accuracy: 0.6748

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.8525 - accuracy: 0.6719

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.8485 - accuracy: 0.6723

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.8532 - accuracy: 0.6703

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.8519 - accuracy: 0.6707

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.8528 - accuracy: 0.6704

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.8500 - accuracy: 0.6708

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.8484 - accuracy: 0.6726

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.8451 - accuracy: 0.6757

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.8471 - accuracy: 0.6732

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.8485 - accuracy: 0.6729

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.8485 - accuracy: 0.6719

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.8500 - accuracy: 0.6728

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.8477 - accuracy: 0.6744

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.8469 - accuracy: 0.6740

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.8459 - accuracy: 0.6755

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.8441 - accuracy: 0.6763

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.8458 - accuracy: 0.6748

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.8429 - accuracy: 0.6750

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.8460 - accuracy: 0.6724

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.8432 - accuracy: 0.6732

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.8427 - accuracy: 0.6730

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.8456 - accuracy: 0.6706

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.8450 - accuracy: 0.6698

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.8427 - accuracy: 0.6711

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.8430 - accuracy: 0.6719

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.8443 - accuracy: 0.6701

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.8467 - accuracy: 0.6699

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.8466 - accuracy: 0.6692

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.8451 - accuracy: 0.6695

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.8464 - accuracy: 0.6679

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.8453 - accuracy: 0.6682

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.8421 - accuracy: 0.6694

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.8406 - accuracy: 0.6696

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.8443 - accuracy: 0.6673

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.8451 - accuracy: 0.6658

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.8436 - accuracy: 0.6648

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.8403 - accuracy: 0.6660

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.8419 - accuracy: 0.6662

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.8418 - accuracy: 0.6665

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.8434 - accuracy: 0.6668

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.8415 - accuracy: 0.6679

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.8398 - accuracy: 0.6689

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.8380 - accuracy: 0.6707

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.8374 - accuracy: 0.6709

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.8344 - accuracy: 0.6711

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.8336 - accuracy: 0.6720

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.8384 - accuracy: 0.6700

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.8382 - accuracy: 0.6709

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.8397 - accuracy: 0.6711

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.8417 - accuracy: 0.6717

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.8452 - accuracy: 0.6697

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.8454 - accuracy: 0.6703

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.8470 - accuracy: 0.6708

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.8468 - accuracy: 0.6703

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.8468 - accuracy: 0.6703 - val_loss: 0.8756 - val_accuracy: 0.6553


.. parsed-literal::

    Epoch 7/15


.. parsed-literal::


 1/92 [..............................] - ETA: 6s - loss: 0.6264 - accuracy: 0.7812

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.6836 - accuracy: 0.7500

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6947 - accuracy: 0.7396

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.7650 - accuracy: 0.6875

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.7629 - accuracy: 0.6687

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.7479 - accuracy: 0.6719

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.7367 - accuracy: 0.6830

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.7399 - accuracy: 0.6875

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.7255 - accuracy: 0.6944

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.7219 - accuracy: 0.7000

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.7231 - accuracy: 0.7017

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.7262 - accuracy: 0.6979

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.7422 - accuracy: 0.6995

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.7450 - accuracy: 0.6987

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.7604 - accuracy: 0.6917

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.7630 - accuracy: 0.6914

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.7670 - accuracy: 0.6857

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.7710 - accuracy: 0.6875

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.7752 - accuracy: 0.6842

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.7717 - accuracy: 0.6906

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.7747 - accuracy: 0.6935

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.7742 - accuracy: 0.6946

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.7775 - accuracy: 0.6957

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.7746 - accuracy: 0.6966

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.7811 - accuracy: 0.6963

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.7955 - accuracy: 0.6911

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.7988 - accuracy: 0.6921

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.7998 - accuracy: 0.6897

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.7936 - accuracy: 0.6929

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.7915 - accuracy: 0.6958

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.7968 - accuracy: 0.6946

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.8040 - accuracy: 0.6924

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.8010 - accuracy: 0.6932

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.7984 - accuracy: 0.6958

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.7943 - accuracy: 0.6982

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.7935 - accuracy: 0.6997

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.7985 - accuracy: 0.6993

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.8004 - accuracy: 0.6982

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.7993 - accuracy: 0.6987

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.8016 - accuracy: 0.6969

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.7994 - accuracy: 0.6974

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.7999 - accuracy: 0.6964

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.8057 - accuracy: 0.6962

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.8052 - accuracy: 0.6967

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.8028 - accuracy: 0.6972

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.8059 - accuracy: 0.6957

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.8029 - accuracy: 0.6961

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.8031 - accuracy: 0.6960

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.7996 - accuracy: 0.6971

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.7977 - accuracy: 0.6981

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.8009 - accuracy: 0.6961

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.8005 - accuracy: 0.6953

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.8015 - accuracy: 0.6963

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.8041 - accuracy: 0.6956

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.8068 - accuracy: 0.6938

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.8128 - accuracy: 0.6914

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.8190 - accuracy: 0.6870

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.8181 - accuracy: 0.6870

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.8184 - accuracy: 0.6867

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.8173 - accuracy: 0.6872

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.8147 - accuracy: 0.6888

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.8145 - accuracy: 0.6877

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.8142 - accuracy: 0.6873

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.8139 - accuracy: 0.6863

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.8131 - accuracy: 0.6863

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.8115 - accuracy: 0.6877

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.8116 - accuracy: 0.6868

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.8075 - accuracy: 0.6886

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.8098 - accuracy: 0.6877

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.8080 - accuracy: 0.6886

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.8115 - accuracy: 0.6860

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.8074 - accuracy: 0.6864

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.8086 - accuracy: 0.6873

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.8111 - accuracy: 0.6848

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.8099 - accuracy: 0.6852

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.8094 - accuracy: 0.6853

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.8082 - accuracy: 0.6849

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.8071 - accuracy: 0.6853

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.8077 - accuracy: 0.6857

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.8084 - accuracy: 0.6850

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.8135 - accuracy: 0.6827

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.8137 - accuracy: 0.6824

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.8151 - accuracy: 0.6813

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.8135 - accuracy: 0.6825

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.8145 - accuracy: 0.6826

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.8161 - accuracy: 0.6812

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.8168 - accuracy: 0.6813

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.8165 - accuracy: 0.6810

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.8168 - accuracy: 0.6818

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.8162 - accuracy: 0.6825

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.8148 - accuracy: 0.6832

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.8148 - accuracy: 0.6832 - val_loss: 0.7976 - val_accuracy: 0.6689


.. parsed-literal::

    Epoch 8/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.7238 - accuracy: 0.7188

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.6960 - accuracy: 0.7500

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.7456 - accuracy: 0.7500

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 4s - loss: 0.8287 - accuracy: 0.7105

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.7804 - accuracy: 0.7228

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.7723 - accuracy: 0.7176

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.7302 - accuracy: 0.7379

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.7499 - accuracy: 0.7286

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.7572 - accuracy: 0.7212

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.7766 - accuracy: 0.7122

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.7523 - accuracy: 0.7234

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.7439 - accuracy: 0.7304

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.7451 - accuracy: 0.7250

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.7425 - accuracy: 0.7267

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.7410 - accuracy: 0.7262

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.7452 - accuracy: 0.7220

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.7387 - accuracy: 0.7254

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.7468 - accuracy: 0.7133

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.7516 - accuracy: 0.7136

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.7829 - accuracy: 0.6973

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.7891 - accuracy: 0.6940

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.7870 - accuracy: 0.6964

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.7964 - accuracy: 0.6947

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.7866 - accuracy: 0.6970

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.7880 - accuracy: 0.6978

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.7870 - accuracy: 0.6998

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.7793 - accuracy: 0.7038

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.7790 - accuracy: 0.7065

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.7753 - accuracy: 0.7069

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.7794 - accuracy: 0.7043

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.7778 - accuracy: 0.7067

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.7690 - accuracy: 0.7090

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.7702 - accuracy: 0.7083

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.7670 - accuracy: 0.7077

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.7700 - accuracy: 0.7098

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.7703 - accuracy: 0.7109

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.7710 - accuracy: 0.7103

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.7732 - accuracy: 0.7081

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.7725 - accuracy: 0.7099

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.7691 - accuracy: 0.7109

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.7689 - accuracy: 0.7096

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.7684 - accuracy: 0.7105

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.7658 - accuracy: 0.7129

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.7679 - accuracy: 0.7116

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.7704 - accuracy: 0.7077

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.7701 - accuracy: 0.7086

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.7674 - accuracy: 0.7088

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.7658 - accuracy: 0.7090

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.7639 - accuracy: 0.7117

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.7641 - accuracy: 0.7112

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.7636 - accuracy: 0.7107

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.7595 - accuracy: 0.7127

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.7654 - accuracy: 0.7093

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.7638 - accuracy: 0.7100

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.7652 - accuracy: 0.7108

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.7661 - accuracy: 0.7098

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.7630 - accuracy: 0.7105

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.7630 - accuracy: 0.7096

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.7610 - accuracy: 0.7092

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.7629 - accuracy: 0.7083

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.7626 - accuracy: 0.7075

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.7682 - accuracy: 0.7062

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.7657 - accuracy: 0.7059

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.7649 - accuracy: 0.7056

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.7652 - accuracy: 0.7058

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.7662 - accuracy: 0.7055

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.7665 - accuracy: 0.7053

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.7670 - accuracy: 0.7041

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.7708 - accuracy: 0.7043

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.7739 - accuracy: 0.7041

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.7738 - accuracy: 0.7038

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.7722 - accuracy: 0.7040

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.7715 - accuracy: 0.7038

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.7732 - accuracy: 0.7036

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.7711 - accuracy: 0.7046

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.7713 - accuracy: 0.7048

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.7710 - accuracy: 0.7054

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.7702 - accuracy: 0.7056

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.7708 - accuracy: 0.7061

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.7702 - accuracy: 0.7067

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.7697 - accuracy: 0.7060

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.7708 - accuracy: 0.7054

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.7705 - accuracy: 0.7056

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.7713 - accuracy: 0.7043

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.7728 - accuracy: 0.7023

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.7723 - accuracy: 0.7032

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.7720 - accuracy: 0.7041

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.7728 - accuracy: 0.7039

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.7748 - accuracy: 0.7030

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.7780 - accuracy: 0.7018

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.7789 - accuracy: 0.7023

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.7789 - accuracy: 0.7023 - val_loss: 0.8257 - val_accuracy: 0.6826


.. parsed-literal::

    Epoch 9/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.7737 - accuracy: 0.6875

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.6998 - accuracy: 0.6875

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.7219 - accuracy: 0.6875

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.7040 - accuracy: 0.6953

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 4s - loss: 0.6988 - accuracy: 0.7000

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.7033 - accuracy: 0.7031

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.7014 - accuracy: 0.7054

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6923 - accuracy: 0.6992

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.7400 - accuracy: 0.6840

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.7474 - accuracy: 0.6875

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.7315 - accuracy: 0.6960

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.7483 - accuracy: 0.6953

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.7540 - accuracy: 0.6947

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.7483 - accuracy: 0.6964

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.7436 - accuracy: 0.7021

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.7494 - accuracy: 0.7012

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.7471 - accuracy: 0.7022

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.7628 - accuracy: 0.6927

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.7589 - accuracy: 0.6957

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.7572 - accuracy: 0.6953

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.7487 - accuracy: 0.7009

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.7454 - accuracy: 0.7045

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.7385 - accuracy: 0.7120

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.7333 - accuracy: 0.7174

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.7299 - accuracy: 0.7188

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.7285 - accuracy: 0.7224

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.7258 - accuracy: 0.7199

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.7238 - accuracy: 0.7254

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.7281 - accuracy: 0.7263

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.7260 - accuracy: 0.7271

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.7353 - accuracy: 0.7258

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.7290 - accuracy: 0.7266

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.7266 - accuracy: 0.7273

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.7233 - accuracy: 0.7289

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.7190 - accuracy: 0.7304

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.7154 - accuracy: 0.7300

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.7150 - accuracy: 0.7331

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.7147 - accuracy: 0.7344

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.7127 - accuracy: 0.7372

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.7139 - accuracy: 0.7367

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.7199 - accuracy: 0.7340

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.7243 - accuracy: 0.7339

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.7293 - accuracy: 0.7321

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.7308 - accuracy: 0.7304

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.7367 - accuracy: 0.7275

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.7397 - accuracy: 0.7253

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.7377 - accuracy: 0.7277

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.7324 - accuracy: 0.7288

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.7353 - accuracy: 0.7274

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.7405 - accuracy: 0.7254

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.7486 - accuracy: 0.7234

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.7479 - accuracy: 0.7227

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.7472 - accuracy: 0.7233

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.7504 - accuracy: 0.7226

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.7469 - accuracy: 0.7231

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.7515 - accuracy: 0.7197

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.7508 - accuracy: 0.7197

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.7488 - accuracy: 0.7202

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.7478 - accuracy: 0.7207

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.7478 - accuracy: 0.7207

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.7452 - accuracy: 0.7212

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.7421 - accuracy: 0.7226

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.7453 - accuracy: 0.7225

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.7479 - accuracy: 0.7225

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.7489 - accuracy: 0.7210

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.7490 - accuracy: 0.7214

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.7494 - accuracy: 0.7219

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.7588 - accuracy: 0.7173

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.7586 - accuracy: 0.7155

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.7602 - accuracy: 0.7142

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.7576 - accuracy: 0.7160

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.7577 - accuracy: 0.7152

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.7604 - accuracy: 0.7127

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.7575 - accuracy: 0.7136

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.7558 - accuracy: 0.7141

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.7562 - accuracy: 0.7142

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.7547 - accuracy: 0.7142

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.7529 - accuracy: 0.7151

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.7530 - accuracy: 0.7155

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.7548 - accuracy: 0.7140

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.7544 - accuracy: 0.7152

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.7539 - accuracy: 0.7153

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.7547 - accuracy: 0.7146

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.7555 - accuracy: 0.7139

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.7560 - accuracy: 0.7132

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.7601 - accuracy: 0.7115

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.7603 - accuracy: 0.7108

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.7611 - accuracy: 0.7109

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.7607 - accuracy: 0.7117

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.7599 - accuracy: 0.7121

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.7587 - accuracy: 0.7122

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.7587 - accuracy: 0.7122 - val_loss: 0.7773 - val_accuracy: 0.6894


.. parsed-literal::

    Epoch 10/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.8874 - accuracy: 0.6562

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.7006 - accuracy: 0.7656

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6894 - accuracy: 0.7396

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6788 - accuracy: 0.7578

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.7055 - accuracy: 0.7625

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.7129 - accuracy: 0.7500

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 5s - loss: 0.7416 - accuracy: 0.7232

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 5s - loss: 0.7553 - accuracy: 0.7188

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.7519 - accuracy: 0.7153

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.7384 - accuracy: 0.7156

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.7344 - accuracy: 0.7159

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.7319 - accuracy: 0.7214

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.7205 - accuracy: 0.7236

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.7497 - accuracy: 0.7098

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.7527 - accuracy: 0.7167

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.7523 - accuracy: 0.7168

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.7440 - accuracy: 0.7206

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.7410 - accuracy: 0.7257

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.7339 - accuracy: 0.7286

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.7494 - accuracy: 0.7188

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.7449 - accuracy: 0.7232

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.7454 - accuracy: 0.7202

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.7433 - accuracy: 0.7242

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 4s - loss: 0.7340 - accuracy: 0.7292

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.7369 - accuracy: 0.7300

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.7384 - accuracy: 0.7278

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.7361 - accuracy: 0.7264

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.7327 - accuracy: 0.7261

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.7345 - accuracy: 0.7248

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.7249 - accuracy: 0.7266

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.7248 - accuracy: 0.7283

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.7269 - accuracy: 0.7281

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.7327 - accuracy: 0.7287

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.7334 - accuracy: 0.7266

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.7328 - accuracy: 0.7264

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.7316 - accuracy: 0.7279

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.7343 - accuracy: 0.7260

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.7310 - accuracy: 0.7290

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.7339 - accuracy: 0.7256

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.7305 - accuracy: 0.7270

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.7295 - accuracy: 0.7268

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.7279 - accuracy: 0.7273

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.7285 - accuracy: 0.7264

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.7294 - accuracy: 0.7270

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.7298 - accuracy: 0.7268

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.7260 - accuracy: 0.7279

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.7220 - accuracy: 0.7284

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.7237 - accuracy: 0.7288

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.7265 - accuracy: 0.7280

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.7256 - accuracy: 0.7284

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.7280 - accuracy: 0.7264

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.7303 - accuracy: 0.7245

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.7296 - accuracy: 0.7256

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.7284 - accuracy: 0.7255

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.7272 - accuracy: 0.7259

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.7230 - accuracy: 0.7274

.. parsed-literal::

    
58/92 [=================>............] - ETA: 2s - loss: 0.7235 - accuracy: 0.7262

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.7210 - accuracy: 0.7277

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.7240 - accuracy: 0.7265

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.7238 - accuracy: 0.7263

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.7262 - accuracy: 0.7247

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.7257 - accuracy: 0.7251

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.7239 - accuracy: 0.7255

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.7250 - accuracy: 0.7249

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.7224 - accuracy: 0.7267

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.7277 - accuracy: 0.7257

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.7265 - accuracy: 0.7256

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.7299 - accuracy: 0.7250

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.7311 - accuracy: 0.7254

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.7294 - accuracy: 0.7266

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.7259 - accuracy: 0.7274

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.7251 - accuracy: 0.7277

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.7268 - accuracy: 0.7263

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.7253 - accuracy: 0.7262

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.7283 - accuracy: 0.7261

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.7267 - accuracy: 0.7264

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.7264 - accuracy: 0.7259

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.7262 - accuracy: 0.7262

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.7274 - accuracy: 0.7241

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.7283 - accuracy: 0.7233

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.7279 - accuracy: 0.7232

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.7274 - accuracy: 0.7236

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.7249 - accuracy: 0.7246

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.7240 - accuracy: 0.7253

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.7254 - accuracy: 0.7245

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.7259 - accuracy: 0.7248

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.7289 - accuracy: 0.7247

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.7260 - accuracy: 0.7254

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.7266 - accuracy: 0.7246

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.7275 - accuracy: 0.7242

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.7271 - accuracy: 0.7234

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.7271 - accuracy: 0.7234 - val_loss: 0.7713 - val_accuracy: 0.7016


.. parsed-literal::

    Epoch 11/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.7591 - accuracy: 0.7188

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.6976 - accuracy: 0.7500

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6632 - accuracy: 0.7292

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6253 - accuracy: 0.7500

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.6267 - accuracy: 0.7500

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.6051 - accuracy: 0.7656

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6296 - accuracy: 0.7500

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6541 - accuracy: 0.7422

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6716 - accuracy: 0.7292

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6663 - accuracy: 0.7344

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6532 - accuracy: 0.7358

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6776 - accuracy: 0.7240

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6726 - accuracy: 0.7236

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6662 - accuracy: 0.7299

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6528 - accuracy: 0.7375

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6668 - accuracy: 0.7344

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6780 - accuracy: 0.7279

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6714 - accuracy: 0.7292

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6778 - accuracy: 0.7253

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6799 - accuracy: 0.7234

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6754 - accuracy: 0.7262

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6751 - accuracy: 0.7287

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.6790 - accuracy: 0.7283

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6767 - accuracy: 0.7318

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6739 - accuracy: 0.7312

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6909 - accuracy: 0.7260

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6983 - accuracy: 0.7222

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6931 - accuracy: 0.7266

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6875 - accuracy: 0.7295

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6899 - accuracy: 0.7302

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6882 - accuracy: 0.7339

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6861 - accuracy: 0.7324

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6830 - accuracy: 0.7330

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6861 - accuracy: 0.7307

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6815 - accuracy: 0.7330

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6788 - accuracy: 0.7352

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6785 - accuracy: 0.7373

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6758 - accuracy: 0.7410

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6824 - accuracy: 0.7412

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.6827 - accuracy: 0.7422

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6811 - accuracy: 0.7439

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6788 - accuracy: 0.7448

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6804 - accuracy: 0.7456

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6769 - accuracy: 0.7464

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6806 - accuracy: 0.7465

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6837 - accuracy: 0.7446

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6849 - accuracy: 0.7447

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6853 - accuracy: 0.7448

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6829 - accuracy: 0.7449

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6938 - accuracy: 0.7406

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6922 - accuracy: 0.7414

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6915 - accuracy: 0.7410

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6885 - accuracy: 0.7423

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6872 - accuracy: 0.7413

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6881 - accuracy: 0.7415

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6871 - accuracy: 0.7422

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6871 - accuracy: 0.7412

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6875 - accuracy: 0.7408

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6912 - accuracy: 0.7389

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6900 - accuracy: 0.7396

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6859 - accuracy: 0.7413

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6863 - accuracy: 0.7399

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6835 - accuracy: 0.7411

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6851 - accuracy: 0.7407

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6833 - accuracy: 0.7409

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6826 - accuracy: 0.7415

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6816 - accuracy: 0.7407

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6814 - accuracy: 0.7413

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6807 - accuracy: 0.7423

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6807 - accuracy: 0.7429

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6836 - accuracy: 0.7412

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6837 - accuracy: 0.7418

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6834 - accuracy: 0.7423

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6851 - accuracy: 0.7424

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6832 - accuracy: 0.7434

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6834 - accuracy: 0.7427

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6821 - accuracy: 0.7436

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6825 - accuracy: 0.7437

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6828 - accuracy: 0.7437

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6808 - accuracy: 0.7446

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6818 - accuracy: 0.7431

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6806 - accuracy: 0.7436

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6778 - accuracy: 0.7444

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6763 - accuracy: 0.7456

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6765 - accuracy: 0.7445

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6780 - accuracy: 0.7439

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6778 - accuracy: 0.7436

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6771 - accuracy: 0.7444

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6771 - accuracy: 0.7444

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6797 - accuracy: 0.7435

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6786 - accuracy: 0.7435

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.6786 - accuracy: 0.7435 - val_loss: 0.7715 - val_accuracy: 0.6798


.. parsed-literal::

    Epoch 12/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.5204 - accuracy: 0.8125

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5876 - accuracy: 0.7812

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5540 - accuracy: 0.7917

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5630 - accuracy: 0.7734

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.6064 - accuracy: 0.7500

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.6603 - accuracy: 0.7135

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6659 - accuracy: 0.7098

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6532 - accuracy: 0.7109

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6953 - accuracy: 0.6910

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6888 - accuracy: 0.6969

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6750 - accuracy: 0.7074

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6779 - accuracy: 0.7109

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6687 - accuracy: 0.7188

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6736 - accuracy: 0.7210

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6661 - accuracy: 0.7250

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6907 - accuracy: 0.7129

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6828 - accuracy: 0.7169

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6764 - accuracy: 0.7240

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6708 - accuracy: 0.7286

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6711 - accuracy: 0.7234

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6707 - accuracy: 0.7247

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6612 - accuracy: 0.7287

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.6664 - accuracy: 0.7269

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6589 - accuracy: 0.7318

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6555 - accuracy: 0.7362

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6615 - accuracy: 0.7368

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6585 - accuracy: 0.7419

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6605 - accuracy: 0.7411

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6640 - accuracy: 0.7403

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6651 - accuracy: 0.7385

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6663 - accuracy: 0.7379

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6644 - accuracy: 0.7393

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6585 - accuracy: 0.7424

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6641 - accuracy: 0.7399

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6600 - accuracy: 0.7411

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6574 - accuracy: 0.7413

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6605 - accuracy: 0.7416

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6604 - accuracy: 0.7410

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6616 - accuracy: 0.7404

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.6610 - accuracy: 0.7406

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6643 - accuracy: 0.7416

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6661 - accuracy: 0.7396

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6706 - accuracy: 0.7369

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6685 - accuracy: 0.7379

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6685 - accuracy: 0.7389

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6644 - accuracy: 0.7398

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6602 - accuracy: 0.7420

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6569 - accuracy: 0.7422

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6579 - accuracy: 0.7430

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6618 - accuracy: 0.7412

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6665 - accuracy: 0.7396

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6646 - accuracy: 0.7404

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6622 - accuracy: 0.7423

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6613 - accuracy: 0.7442

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6608 - accuracy: 0.7443

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6585 - accuracy: 0.7455

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6573 - accuracy: 0.7467

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6552 - accuracy: 0.7489

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6562 - accuracy: 0.7484

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6584 - accuracy: 0.7474

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6567 - accuracy: 0.7480

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6583 - accuracy: 0.7470

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6578 - accuracy: 0.7470

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6588 - accuracy: 0.7461

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6590 - accuracy: 0.7462

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6583 - accuracy: 0.7457

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6572 - accuracy: 0.7458

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6573 - accuracy: 0.7440

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6538 - accuracy: 0.7455

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6536 - accuracy: 0.7464

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6533 - accuracy: 0.7474

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6555 - accuracy: 0.7470

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6560 - accuracy: 0.7462

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6576 - accuracy: 0.7450

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6595 - accuracy: 0.7434

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6634 - accuracy: 0.7423

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6615 - accuracy: 0.7432

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6644 - accuracy: 0.7429

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6643 - accuracy: 0.7422

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6657 - accuracy: 0.7415

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6679 - accuracy: 0.7404

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6686 - accuracy: 0.7394

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6643 - accuracy: 0.7410

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6666 - accuracy: 0.7389

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6648 - accuracy: 0.7398

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6629 - accuracy: 0.7406

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6634 - accuracy: 0.7411

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6615 - accuracy: 0.7419

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6613 - accuracy: 0.7416

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6610 - accuracy: 0.7414

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6607 - accuracy: 0.7408

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.6607 - accuracy: 0.7408 - val_loss: 0.8130 - val_accuracy: 0.6880


.. parsed-literal::

    Epoch 13/15


.. parsed-literal::


 1/92 [..............................] - ETA: 6s - loss: 0.7036 - accuracy: 0.7500

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5775 - accuracy: 0.7969

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5625 - accuracy: 0.8021

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5585 - accuracy: 0.7812

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5784 - accuracy: 0.7688

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.6100 - accuracy: 0.7604

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6105 - accuracy: 0.7634

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6258 - accuracy: 0.7617

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6514 - accuracy: 0.7535

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6498 - accuracy: 0.7563

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6625 - accuracy: 0.7472

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6400 - accuracy: 0.7552

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6212 - accuracy: 0.7644

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6370 - accuracy: 0.7634

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6449 - accuracy: 0.7563

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6506 - accuracy: 0.7520

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6472 - accuracy: 0.7537

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6407 - accuracy: 0.7569

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6400 - accuracy: 0.7582

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6378 - accuracy: 0.7578

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6425 - accuracy: 0.7560

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6402 - accuracy: 0.7571

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.6362 - accuracy: 0.7582

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6312 - accuracy: 0.7617

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6326 - accuracy: 0.7600

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6350 - accuracy: 0.7584

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6295 - accuracy: 0.7604

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6258 - accuracy: 0.7600

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6203 - accuracy: 0.7608

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6228 - accuracy: 0.7604

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6210 - accuracy: 0.7631

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6188 - accuracy: 0.7637

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6210 - accuracy: 0.7633

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6231 - accuracy: 0.7610

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6273 - accuracy: 0.7580

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6277 - accuracy: 0.7587

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6235 - accuracy: 0.7610

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6186 - accuracy: 0.7615

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6256 - accuracy: 0.7588

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.6243 - accuracy: 0.7609

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6233 - accuracy: 0.7599

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6234 - accuracy: 0.7597

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6257 - accuracy: 0.7587

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6267 - accuracy: 0.7578

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6262 - accuracy: 0.7576

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6265 - accuracy: 0.7568

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6259 - accuracy: 0.7573

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6294 - accuracy: 0.7565

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6340 - accuracy: 0.7550

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6325 - accuracy: 0.7568

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6328 - accuracy: 0.7566

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6327 - accuracy: 0.7565

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6322 - accuracy: 0.7570

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6305 - accuracy: 0.7591

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6305 - accuracy: 0.7584

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6342 - accuracy: 0.7572

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6363 - accuracy: 0.7565

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6343 - accuracy: 0.7569

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6325 - accuracy: 0.7578

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6320 - accuracy: 0.7577

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6308 - accuracy: 0.7586

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6304 - accuracy: 0.7590

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6276 - accuracy: 0.7608

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6340 - accuracy: 0.7582

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6340 - accuracy: 0.7590

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6347 - accuracy: 0.7580

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6398 - accuracy: 0.7560

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6405 - accuracy: 0.7541

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6445 - accuracy: 0.7518

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6482 - accuracy: 0.7513

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6499 - accuracy: 0.7500

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6494 - accuracy: 0.7509

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6503 - accuracy: 0.7513

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6487 - accuracy: 0.7525

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6495 - accuracy: 0.7529

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6519 - accuracy: 0.7529

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6511 - accuracy: 0.7536

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6495 - accuracy: 0.7552

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6484 - accuracy: 0.7551

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6463 - accuracy: 0.7562

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6455 - accuracy: 0.7569

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6443 - accuracy: 0.7579

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6464 - accuracy: 0.7556

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6444 - accuracy: 0.7566

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6430 - accuracy: 0.7573

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6428 - accuracy: 0.7568

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6419 - accuracy: 0.7564

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6398 - accuracy: 0.7567

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6389 - accuracy: 0.7570

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6400 - accuracy: 0.7562

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6398 - accuracy: 0.7558

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.6398 - accuracy: 0.7558 - val_loss: 0.7777 - val_accuracy: 0.6948


.. parsed-literal::

    Epoch 14/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.4261 - accuracy: 0.8750

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5132 - accuracy: 0.8438

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5510 - accuracy: 0.7708

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5898 - accuracy: 0.7422

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5904 - accuracy: 0.7500

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.5960 - accuracy: 0.7552

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.5943 - accuracy: 0.7500

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5583 - accuracy: 0.7695

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5586 - accuracy: 0.7674

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5563 - accuracy: 0.7781

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5574 - accuracy: 0.7756

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5506 - accuracy: 0.7786

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5592 - accuracy: 0.7788

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5578 - accuracy: 0.7746

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5770 - accuracy: 0.7667

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5725 - accuracy: 0.7695

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5689 - accuracy: 0.7739

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5653 - accuracy: 0.7778

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5729 - accuracy: 0.7730

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5793 - accuracy: 0.7719

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5779 - accuracy: 0.7738

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5852 - accuracy: 0.7685

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.5810 - accuracy: 0.7704

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5964 - accuracy: 0.7604

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6061 - accuracy: 0.7600

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6080 - accuracy: 0.7572

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6155 - accuracy: 0.7569

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6166 - accuracy: 0.7567

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6179 - accuracy: 0.7565

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6113 - accuracy: 0.7583

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6072 - accuracy: 0.7621

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6094 - accuracy: 0.7627

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6160 - accuracy: 0.7630

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6133 - accuracy: 0.7653

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6064 - accuracy: 0.7692

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6016 - accuracy: 0.7713

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5976 - accuracy: 0.7732

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5938 - accuracy: 0.7758

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.5896 - accuracy: 0.7775

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5981 - accuracy: 0.7745

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5944 - accuracy: 0.7769

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5974 - accuracy: 0.7749

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5971 - accuracy: 0.7743

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5972 - accuracy: 0.7751

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5985 - accuracy: 0.7746

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6010 - accuracy: 0.7721

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6030 - accuracy: 0.7709

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6040 - accuracy: 0.7686

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6054 - accuracy: 0.7701

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6080 - accuracy: 0.7685

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6174 - accuracy: 0.7645

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6172 - accuracy: 0.7660

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6193 - accuracy: 0.7651

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6172 - accuracy: 0.7666

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6150 - accuracy: 0.7674

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6115 - accuracy: 0.7687

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6124 - accuracy: 0.7689

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6117 - accuracy: 0.7697

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6089 - accuracy: 0.7714

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6065 - accuracy: 0.7721

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6043 - accuracy: 0.7733

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6059 - accuracy: 0.7734

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6050 - accuracy: 0.7735

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6022 - accuracy: 0.7751

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6021 - accuracy: 0.7747

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6032 - accuracy: 0.7748

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6043 - accuracy: 0.7749

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6087 - accuracy: 0.7736

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6058 - accuracy: 0.7746

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6083 - accuracy: 0.7730

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6081 - accuracy: 0.7722

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6062 - accuracy: 0.7728

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6061 - accuracy: 0.7716

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6055 - accuracy: 0.7722

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6044 - accuracy: 0.7723

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6056 - accuracy: 0.7720

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6074 - accuracy: 0.7725

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6077 - accuracy: 0.7722

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6080 - accuracy: 0.7727

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6068 - accuracy: 0.7732

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6091 - accuracy: 0.7733

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6109 - accuracy: 0.7738

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6099 - accuracy: 0.7739

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6088 - accuracy: 0.7751

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6102 - accuracy: 0.7733

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6095 - accuracy: 0.7734

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6101 - accuracy: 0.7735

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6092 - accuracy: 0.7736

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6088 - accuracy: 0.7740

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6091 - accuracy: 0.7741

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6112 - accuracy: 0.7728

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.6112 - accuracy: 0.7728 - val_loss: 0.7446 - val_accuracy: 0.7180


.. parsed-literal::

    Epoch 15/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.7172 - accuracy: 0.7500

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.6898 - accuracy: 0.7188

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6331 - accuracy: 0.7396

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6351 - accuracy: 0.7344

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.6131 - accuracy: 0.7500

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.6061 - accuracy: 0.7656

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6028 - accuracy: 0.7723

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5966 - accuracy: 0.7773

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5947 - accuracy: 0.7639

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6258 - accuracy: 0.7563

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6195 - accuracy: 0.7557

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6263 - accuracy: 0.7500

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6249 - accuracy: 0.7524

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6133 - accuracy: 0.7567

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5968 - accuracy: 0.7646

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5936 - accuracy: 0.7676

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5926 - accuracy: 0.7665

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5875 - accuracy: 0.7691

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5894 - accuracy: 0.7697

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5951 - accuracy: 0.7703

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5969 - accuracy: 0.7708

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5999 - accuracy: 0.7699

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.5981 - accuracy: 0.7717

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5957 - accuracy: 0.7721

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5867 - accuracy: 0.7750

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5860 - accuracy: 0.7752

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5902 - accuracy: 0.7720

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5927 - accuracy: 0.7679

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5968 - accuracy: 0.7672

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5930 - accuracy: 0.7677

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5927 - accuracy: 0.7692

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5898 - accuracy: 0.7695

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5950 - accuracy: 0.7661

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5884 - accuracy: 0.7693

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5948 - accuracy: 0.7670

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5938 - accuracy: 0.7682

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5942 - accuracy: 0.7686

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5992 - accuracy: 0.7656

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5962 - accuracy: 0.7676

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.5981 - accuracy: 0.7672

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5995 - accuracy: 0.7675

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5977 - accuracy: 0.7693

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5971 - accuracy: 0.7711

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5994 - accuracy: 0.7692

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5988 - accuracy: 0.7694

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5967 - accuracy: 0.7711

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5953 - accuracy: 0.7726

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5901 - accuracy: 0.7756

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5903 - accuracy: 0.7758

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5900 - accuracy: 0.7759

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5938 - accuracy: 0.7748

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5955 - accuracy: 0.7749

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5915 - accuracy: 0.7767

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5903 - accuracy: 0.7768

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5884 - accuracy: 0.7775

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5933 - accuracy: 0.7753

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5904 - accuracy: 0.7765

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5885 - accuracy: 0.7771

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5849 - accuracy: 0.7788

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5820 - accuracy: 0.7798

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5856 - accuracy: 0.7768

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5818 - accuracy: 0.7784

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5835 - accuracy: 0.7779

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5837 - accuracy: 0.7785

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5860 - accuracy: 0.7776

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5855 - accuracy: 0.7776

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5883 - accuracy: 0.7772

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5883 - accuracy: 0.7768

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5879 - accuracy: 0.7764

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5860 - accuracy: 0.7765

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5857 - accuracy: 0.7770

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5873 - accuracy: 0.7766

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5881 - accuracy: 0.7767

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5885 - accuracy: 0.7759

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5900 - accuracy: 0.7748

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5888 - accuracy: 0.7757

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5946 - accuracy: 0.7729

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5968 - accuracy: 0.7722

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5973 - accuracy: 0.7723

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5966 - accuracy: 0.7721

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5950 - accuracy: 0.7722

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5967 - accuracy: 0.7715

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6001 - accuracy: 0.7694

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5977 - accuracy: 0.7710

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5969 - accuracy: 0.7715

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5970 - accuracy: 0.7716

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5964 - accuracy: 0.7717

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5970 - accuracy: 0.7711

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5947 - accuracy: 0.7723

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5965 - accuracy: 0.7713

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5970 - accuracy: 0.7718

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.5970 - accuracy: 0.7718 - val_loss: 0.7530 - val_accuracy: 0.7112



.. image:: tensorflow-training-openvino-nncf-with-output_files/tensorflow-training-openvino-nncf-with-output_3_1467.png


.. parsed-literal::


    1/1 [==============================] - ETA: 0s

.. parsed-literal::

    
1/1 [==============================] - 0s 75ms/step


.. parsed-literal::

    This image most likely belongs to sunflowers with a 86.22 percent confidence.


.. parsed-literal::

    2024-04-18 01:13:42.664989: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'random_flip_input' with dtype float and shape [?,180,180,3]
    	 [[{{node random_flip_input}}]]
    2024-04-18 01:13:42.761394: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-04-18 01:13:42.772169: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'random_flip_input' with dtype float and shape [?,180,180,3]
    	 [[{{node random_flip_input}}]]
    2024-04-18 01:13:42.784031: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-04-18 01:13:42.791410: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-04-18 01:13:42.798592: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-04-18 01:13:42.810122: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-04-18 01:13:42.851227: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'sequential_1_input' with dtype float and shape [?,180,180,3]
    	 [[{{node sequential_1_input}}]]


.. parsed-literal::

    2024-04-18 01:13:42.923722: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-04-18 01:13:42.945679: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'sequential_1_input' with dtype float and shape [?,180,180,3]
    	 [[{{node sequential_1_input}}]]
    2024-04-18 01:13:42.987699: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,22,22,64]
    	 [[{{node inputs}}]]
    2024-04-18 01:13:43.012752: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-04-18 01:13:43.090809: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]


.. parsed-literal::

    2024-04-18 01:13:43.248083: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]


.. parsed-literal::

    2024-04-18 01:13:43.562029: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,22,22,64]
    	 [[{{node inputs}}]]
    2024-04-18 01:13:43.598798: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-04-18 01:13:43.628830: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-04-18 01:13:43.679003: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    WARNING:absl:Found untraced functions such as _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op, _update_step_xla while saving (showing 4 of 4). These functions will not be directly callable after loading.


.. parsed-literal::

    INFO:tensorflow:Assets written to: model/flower/saved_model/assets


.. parsed-literal::

    INFO:tensorflow:Assets written to: model/flower/saved_model/assets



.. parsed-literal::

    output/A_Close_Up_Photo_of_a_Dandelion.jpg:   0%|          | 0.00/21.7k [00:00<?, ?B/s]


.. parsed-literal::

    (1, 180, 180, 3)
    [1,180,180,3]
    This image most likely belongs to dandelion with a 97.36 percent confidence.



.. image:: tensorflow-training-openvino-nncf-with-output_files/tensorflow-training-openvino-nncf-with-output_3_1479.png


Imports
~~~~~~~



The Post Training Quantization API is implemented in the ``nncf``
library.

.. code:: ipython3

    import matplotlib.pyplot as plt
    import numpy as np
    import nncf
    from openvino.runtime import Core
    from openvino.runtime import serialize
    from PIL import Image
    from sklearn.metrics import accuracy_score

    # Fetch `notebook_utils` module
    import requests

    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )

    open("notebook_utils.py", "w").write(r.text)
    from notebook_utils import download_file


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


Post-training Quantization with NNCF
------------------------------------



`NNCF <https://github.com/openvinotoolkit/nncf>`__ provides a suite of
advanced algorithms for Neural Networks inference optimization in
OpenVINO with minimal accuracy drop.

Create a quantized model from the pre-trained FP32 model and the
calibration dataset. The optimization process contains the following
steps:

1. Create a Dataset for quantization.
2. Run nncf.quantize for getting an optimized model.

The validation dataset already defined in the training notebook.

.. code:: ipython3

    img_height = 180
    img_width = 180
    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=1,
    )

    for a, b in val_dataset:
        print(type(a), type(b))
        break


.. parsed-literal::

    Found 3670 files belonging to 5 classes.


.. parsed-literal::

    Using 734 files for validation.
    <class 'tensorflow.python.framework.ops.EagerTensor'> <class 'tensorflow.python.framework.ops.EagerTensor'>


.. parsed-literal::

    2024-04-18 01:13:46.645327: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [734]
    	 [[{{node Placeholder/_4}}]]
    2024-04-18 01:13:46.645607: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [734]
    	 [[{{node Placeholder/_0}}]]


The validation dataset can be reused in quantization process. But it
returns a tuple (images, labels), whereas calibration_dataset should
only return images. The transformation function helps to transform a
user validation dataset to the calibration dataset.

.. code:: ipython3

    def transform_fn(data_item):
        """
        The transformation function transforms a data item into model input data.
        This function should be passed when the data item cannot be used as model's input.
        """
        images, _ = data_item
        return images.numpy()


    calibration_dataset = nncf.Dataset(val_dataset, transform_fn)

Download Intermediate Representation (IR) model.

.. code:: ipython3

    core = Core()
    ir_model = core.read_model(model_xml)

Use `Basic Quantization
Flow <https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.html>`__.
To use the most advanced quantization flow that allows to apply 8-bit
quantization to the model with accuracy control see `Quantizing with
accuracy
control <https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/quantizing-models-post-training/quantizing-with-accuracy-control.html>`__.

.. code:: ipython3

    quantized_model = nncf.quantize(ir_model, calibration_dataset, subset_size=1000)



.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Exception in thread Thread-88:
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Traceback (most recent call last):
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File "/usr/lib/python3.8/threading.py", line 932, in _bootstrap_inner
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    self.run()
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/live.py", line 32, in run
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    self.live.refresh()
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/live.py", line 223, in refresh
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    self._live_render.set_renderable(self.renderable)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/live.py", line 203, in renderable
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    renderable = self.get_renderable()
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/live.py", line 98, in get_renderable
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    self._get_renderable()
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1537, in get_renderable
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    renderable = Group(*self.get_renderables())
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1542, in get_renderables
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    table = self.make_tasks_table(self.tasks)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1566, in make_tasks_table
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    table.add_row(
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1571, in &lt;genexpr&gt;
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    else column(task)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 528, in __call__
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    renderable = self.render(task)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/nncf/common/logging/track_progress.py", line 58, in render
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    text = super().render(task)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 787, in render
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    task_time = task.time_remaining
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1039, in time_remaining
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    estimate = ceil(remaining / speed)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/tensorflow/python/util/traceback_utils.py", line 153, in error_handler
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    raise e.with_traceback(filtered_tb) from None
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/tensorflow/python/ops/math_ops.py", line 1569, in _truediv_python3
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    raise TypeError(f"`x` and `y` must have the same dtype, "
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">TypeError: `x` and `y` must have the same dtype, got tf.int64 != tf.float32.
    </pre>




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



Save quantized model to benchmark.

.. code:: ipython3

    compressed_model_dir = Path("model/optimized")
    compressed_model_dir.mkdir(parents=True, exist_ok=True)
    compressed_model_xml = compressed_model_dir / "flower_ir.xml"
    serialize(quantized_model, str(compressed_model_xml))

Select inference device
~~~~~~~~~~~~~~~~~~~~~~~



select device from dropdown list for running inference using OpenVINO

.. code:: ipython3

    import ipywidgets as widgets

    device = widgets.Dropdown(
        options=(core.available_devices + ["AUTO"] if not "GPU" in core.available_devices else ["AUTO", "MULTY:CPU,GPU"]),
        value="AUTO",
        description="Device:",
        disabled=False,
    )

    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



Compare Metrics
---------------



Define a metric to determine the performance of the model.

For this demo we define validate function to compute accuracy metrics.

.. code:: ipython3

    def validate(model, validation_loader):
        """
        Evaluate model and compute accuracy metrics.

        :param model: Model to validate
        :param validation_loader: Validation dataset
        :returns: Accuracy scores
        """
        predictions = []
        references = []

        output = model.outputs[0]

        for images, target in validation_loader:
            pred = model(images.numpy())[output]

            predictions.append(np.argmax(pred, axis=1))
            references.append(target)

        predictions = np.concatenate(predictions, axis=0)
        references = np.concatenate(references, axis=0)

        scores = accuracy_score(references, predictions)

        return scores

Calculate accuracy for the original model and the quantized model.

.. code:: ipython3

    original_compiled_model = core.compile_model(model=ir_model, device_name=device.value)
    quantized_compiled_model = core.compile_model(model=quantized_model, device_name=device.value)

    original_accuracy = validate(original_compiled_model, val_dataset)
    quantized_accuracy = validate(quantized_compiled_model, val_dataset)

    print(f"Accuracy of the original model: {original_accuracy:.3f}")
    print(f"Accuracy of the quantized model: {quantized_accuracy:.3f}")


.. parsed-literal::

    Accuracy of the original model: 0.711
    Accuracy of the quantized model: 0.707


Compare file size of the models.

.. code:: ipython3

    original_model_size = model_xml.with_suffix(".bin").stat().st_size / 1024
    quantized_model_size = compressed_model_xml.with_suffix(".bin").stat().st_size / 1024

    print(f"Original model size: {original_model_size:.2f} KB")
    print(f"Quantized model size: {quantized_model_size:.2f} KB")


.. parsed-literal::

    Original model size: 7791.65 KB
    Quantized model size: 3897.08 KB


So, we can see that the original and quantized models have similar
accuracy with a much smaller size of the quantized model.

Run Inference on Quantized Model
--------------------------------



Copy the preprocess function from the training notebook and run
inference on the quantized model with Inference Engine. See the
`OpenVINO API tutorial <openvino-api-with-output.html>`__ for more
information about running inference with Inference Engine Python API.

.. code:: ipython3

    def pre_process_image(imagePath, img_height=180):
        # Model input format
        n, c, h, w = [1, 3, img_height, img_height]
        image = Image.open(imagePath)
        image = image.resize((h, w), resample=Image.BILINEAR)

        # Convert to array and change data layout from HWC to CHW
        image = np.array(image)

        input_image = image.reshape((n, h, w, c))

        return input_image

.. code:: ipython3

    # Get the names of the input and output layer
    input_layer = quantized_compiled_model.input(0)
    output_layer = quantized_compiled_model.output(0)

    # Get the class names: a list of directory names in alphabetical order
    class_names = sorted([item.name for item in Path(data_dir).iterdir() if item.is_dir()])

    # Run inference on an input image...
    inp_img_url = "https://upload.wikimedia.org/wikipedia/commons/4/48/A_Close_Up_Photo_of_a_Dandelion.jpg"
    directory = "output"
    inp_file_name = "A_Close_Up_Photo_of_a_Dandelion.jpg"
    file_path = Path(directory) / Path(inp_file_name)
    # Download the image if it does not exist yet
    if not Path(inp_file_name).exists():
        download_file(inp_img_url, inp_file_name, directory=directory)

    # Pre-process the image and get it ready for inference.
    input_image = pre_process_image(imagePath=file_path)
    print(f"input image shape: {input_image.shape}")
    print(f"input layer shape: {input_layer.shape}")

    res = quantized_compiled_model([input_image])[output_layer]

    score = tf.nn.softmax(res[0])

    # Show the results
    image = Image.open(file_path)
    plt.imshow(image)
    print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))


.. parsed-literal::

    'output/A_Close_Up_Photo_of_a_Dandelion.jpg' already exists.
    input image shape: (1, 180, 180, 3)
    input layer shape: [1,180,180,3]
    This image most likely belongs to dandelion with a 97.54 percent confidence.



.. image:: tensorflow-training-openvino-nncf-with-output_files/tensorflow-training-openvino-nncf-with-output_27_1.png


Compare Inference Speed
-----------------------



Measure inference speed with the `OpenVINO Benchmark
App <https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html>`__.

Benchmark App is a command line tool that measures raw inference
performance for a specified OpenVINO IR model. Run
``benchmark_app --help`` to see a list of available parameters. By
default, Benchmark App tests the performance of the model specified with
the ``-m`` parameter with asynchronous inference on CPU, for one minute.
Use the ``-d`` parameter to test performance on a different device, for
example an Intel integrated Graphics (iGPU), and ``-t`` to set the
number of seconds to run inference. See the
`documentation <https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html>`__
for more information.

This tutorial uses a wrapper function from `Notebook
Utils <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/utils/notebook_utils.ipynb>`__.
It prints the ``benchmark_app`` command with the chosen parameters.

In the next cells, inference speed will be measured for the original and
quantized model on CPU. If an iGPU is available, inference speed will be
measured for CPU+GPU as well. The number of seconds is set to 15.

   **NOTE**: For the most accurate performance estimation, it is
   recommended to run ``benchmark_app`` in a terminal/command prompt
   after closing other applications.

.. code:: ipython3

    # print the available devices on this system
    print("Device information:")

    for ov_device in core.available_devices:
        print(f'{ov_device} - {core.get_property(ov_device, "FULL_DEVICE_NAME")}')


.. parsed-literal::

    Device information:
    CPU - Intel(R) Core(TM) i9-10920X CPU @ 3.50GHz


.. code:: ipython3

    # Original model benchmarking
    ! benchmark_app -m $model_xml -d $device.value -t 15 -api async


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2024.0.0-14509-34caeefd078-releases/2024/0
    [ INFO ]
    [ INFO ] Device info:
    [ INFO ] AUTO


.. parsed-literal::

    [ INFO ] Build ................................. 2024.0.0-14509-34caeefd078-releases/2024/0
    [ INFO ]
    [ INFO ]
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(AUTO) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 4.36 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     sequential_1_input (node: sequential_1_input) : f32 / [...] / [1,180,180,3]
    [ INFO ] Model outputs:
    [ INFO ]     outputs (node: sequential_2/outputs/BiasAdd) : f32 / [...] / [1,5]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     sequential_1_input (node: sequential_1_input) : u8 / [N,H,W,C] / [1,180,180,3]
    [ INFO ] Model outputs:
    [ INFO ]     outputs (node: sequential_2/outputs/BiasAdd) : f32 / [...] / [1,5]
    [Step 7/11] Loading the model to the device


.. parsed-literal::

    [ INFO ] Compile model took 119.88 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: TensorFlow_Frontend_IR
    [ INFO ]   EXECUTION_DEVICES: ['CPU']
    [ INFO ]   PERFORMANCE_HINT: PerformanceMode.THROUGHPUT
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 12
    [ INFO ]   MULTI_DEVICE_PRIORITIES: CPU
    [ INFO ]   CPU:
    [ INFO ]     AFFINITY: Affinity.CORE
    [ INFO ]     CPU_DENORMALS_OPTIMIZATION: False
    [ INFO ]     CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE: 1.0
    [ INFO ]     DYNAMIC_QUANTIZATION_GROUP_SIZE: 0
    [ INFO ]     ENABLE_CPU_PINNING: True
    [ INFO ]     ENABLE_HYPER_THREADING: True
    [ INFO ]     EXECUTION_DEVICES: ['CPU']
    [ INFO ]     EXECUTION_MODE_HINT: ExecutionMode.PERFORMANCE
    [ INFO ]     INFERENCE_NUM_THREADS: 24
    [ INFO ]     INFERENCE_PRECISION_HINT: <Type: 'float32'>
    [ INFO ]     KV_CACHE_PRECISION: <Type: 'float16'>
    [ INFO ]     LOG_LEVEL: Level.NO
    [ INFO ]     NETWORK_NAME: TensorFlow_Frontend_IR
    [ INFO ]     NUM_STREAMS: 12
    [ INFO ]     OPTIMAL_NUMBER_OF_INFER_REQUESTS: 12
    [ INFO ]     PERFORMANCE_HINT: THROUGHPUT
    [ INFO ]     PERFORMANCE_HINT_NUM_REQUESTS: 0
    [ INFO ]     PERF_COUNT: NO
    [ INFO ]     SCHEDULING_CORE_TYPE: SchedulingCoreType.ANY_CORE
    [ INFO ]   MODEL_PRIORITY: Priority.MEDIUM
    [ INFO ]   LOADED_FROM_CACHE: False
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'sequential_1_input'!. This input will be filled with random values!
    [ INFO ] Fill input 'sequential_1_input' with random values
    [Step 10/11] Measuring performance (Start inference asynchronously, 12 inference requests, limits: 15000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 4.01 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            55860 iterations
    [ INFO ] Duration:         15002.77 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        3.03 ms
    [ INFO ]    Average:       3.04 ms
    [ INFO ]    Min:           1.44 ms
    [ INFO ]    Max:           13.17 ms
    [ INFO ] Throughput:   3723.31 FPS


.. code:: ipython3

    # Quantized model benchmarking
    ! benchmark_app -m $compressed_model_xml -d $device.value -t 15 -api async


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2024.0.0-14509-34caeefd078-releases/2024/0
    [ INFO ]
    [ INFO ] Device info:
    [ INFO ] AUTO
    [ INFO ] Build ................................. 2024.0.0-14509-34caeefd078-releases/2024/0
    [ INFO ]
    [ INFO ]
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(AUTO) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 4.80 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     sequential_1_input (node: sequential_1_input) : f32 / [...] / [1,180,180,3]
    [ INFO ] Model outputs:
    [ INFO ]     outputs (node: sequential_2/outputs/BiasAdd) : f32 / [...] / [1,5]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     sequential_1_input (node: sequential_1_input) : u8 / [N,H,W,C] / [1,180,180,3]
    [ INFO ] Model outputs:
    [ INFO ]     outputs (node: sequential_2/outputs/BiasAdd) : f32 / [...] / [1,5]
    [Step 7/11] Loading the model to the device


.. parsed-literal::

    [ INFO ] Compile model took 105.05 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: TensorFlow_Frontend_IR
    [ INFO ]   EXECUTION_DEVICES: ['CPU']
    [ INFO ]   PERFORMANCE_HINT: PerformanceMode.THROUGHPUT
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 12
    [ INFO ]   MULTI_DEVICE_PRIORITIES: CPU
    [ INFO ]   CPU:
    [ INFO ]     AFFINITY: Affinity.CORE
    [ INFO ]     CPU_DENORMALS_OPTIMIZATION: False
    [ INFO ]     CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE: 1.0
    [ INFO ]     DYNAMIC_QUANTIZATION_GROUP_SIZE: 0
    [ INFO ]     ENABLE_CPU_PINNING: True
    [ INFO ]     ENABLE_HYPER_THREADING: True
    [ INFO ]     EXECUTION_DEVICES: ['CPU']
    [ INFO ]     EXECUTION_MODE_HINT: ExecutionMode.PERFORMANCE
    [ INFO ]     INFERENCE_NUM_THREADS: 24
    [ INFO ]     INFERENCE_PRECISION_HINT: <Type: 'float32'>
    [ INFO ]     KV_CACHE_PRECISION: <Type: 'float16'>
    [ INFO ]     LOG_LEVEL: Level.NO
    [ INFO ]     NETWORK_NAME: TensorFlow_Frontend_IR
    [ INFO ]     NUM_STREAMS: 12
    [ INFO ]     OPTIMAL_NUMBER_OF_INFER_REQUESTS: 12
    [ INFO ]     PERFORMANCE_HINT: THROUGHPUT
    [ INFO ]     PERFORMANCE_HINT_NUM_REQUESTS: 0
    [ INFO ]     PERF_COUNT: NO
    [ INFO ]     SCHEDULING_CORE_TYPE: SchedulingCoreType.ANY_CORE
    [ INFO ]   MODEL_PRIORITY: Priority.MEDIUM
    [ INFO ]   LOADED_FROM_CACHE: False
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'sequential_1_input'!. This input will be filled with random values!
    [ INFO ] Fill input 'sequential_1_input' with random values
    [Step 10/11] Measuring performance (Start inference asynchronously, 12 inference requests, limits: 15000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).


.. parsed-literal::

    [ INFO ] First inference took 2.15 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            178632 iterations
    [ INFO ] Duration:         15001.70 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        0.94 ms
    [ INFO ]    Average:       0.97 ms
    [ INFO ]    Min:           0.58 ms
    [ INFO ]    Max:           7.07 ms
    [ INFO ] Throughput:   11907.45 FPS

