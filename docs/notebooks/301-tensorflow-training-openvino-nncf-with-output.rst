Post-Training Quantization with TensorFlow Classification Model
===============================================================

This example demonstrates how to quantize the OpenVINO model that was
created in `301-tensorflow-training-openvino
notebook <301-tensorflow-training-openvino-with-output.html>`__, to improve
inference speed. Quantization is performed with `Post-training
Quantization with
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

    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    magika 0.5.1 requires numpy<2.0,>=1.24; python_version >= "3.8" and python_version < "3.9", but you have numpy 1.23.5 which is incompatible.
    mobileclip 0.1.0 requires torch==1.13.1, but you have torch 2.2.1+cpu which is incompatible.
    mobileclip 0.1.0 requires torchvision==0.14.1, but you have torchvision 0.17.1+cpu which is incompatible.
    paddleclas 2.5.2 requires gast==0.3.3, but you have gast 0.4.0 which is incompatible.
    paddleclas 2.5.2 requires opencv-python==4.6.0.66, but you have opencv-python 4.9.0.80 which is incompatible.
    ppgan 2.1.0 requires imageio==2.9.0, but you have imageio 2.34.0 which is incompatible.
    ppgan 2.1.0 requires librosa==0.8.1, but you have librosa 0.9.2 which is incompatible.
    ppgan 2.1.0 requires opencv-python<=4.6.0.66, but you have opencv-python 4.9.0.80 which is incompatible.
    pyannote-audio 2.0.1 requires torchaudio<1.0,>=0.10, but you have torchaudio 2.2.1+cpu which is incompatible.
    pytorch-lightning 1.6.5 requires protobuf<=3.20.1, but you have protobuf 4.25.3 which is incompatible.
    tensorflow-metadata 1.14.0 requires protobuf<4.21,>=3.20.3, but you have protobuf 4.25.3 which is incompatible.
    tf2onnx 1.16.1 requires protobuf~=3.20, but you have protobuf 4.25.3 which is incompatible.


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
    dataset_url = (
        "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    )
    data_dir = Path(tf.keras.utils.get_file("flower_photos", origin=dataset_url, untar=True))

    if not model_xml.exists():
        print("Executing training notebook. This will take a while...")
        %run 301-tensorflow-training-openvino.ipynb


.. parsed-literal::

    2024-03-26 00:47:04.242300: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-03-26 00:47:04.277485: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


.. parsed-literal::

    2024-03-26 00:47:04.868172: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


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

    2024-03-26 00:47:32.693589: E tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:266] failed call to cuInit: CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: forward compatibility was attempted on non supported HW
    2024-03-26 00:47:32.693624: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:168] retrieving CUDA diagnostic information for host: iotg-dev-workstation-07
    2024-03-26 00:47:32.693628: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:175] hostname: iotg-dev-workstation-07
    2024-03-26 00:47:32.693762: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:199] libcuda reported version is: 470.223.2
    2024-03-26 00:47:32.693778: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:203] kernel reported version is: 470.182.3
    2024-03-26 00:47:32.693782: E tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:312] kernel version 470.182.3 does not match DSO version 470.223.2 -- cannot find working devices in this configuration


.. parsed-literal::

    Found 3670 files belonging to 5 classes.


.. parsed-literal::

    Using 734 files for validation.
    ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']


.. parsed-literal::

    2024-03-26 00:47:32.991785: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-03-26 00:47:32.992049: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]



.. image:: 301-tensorflow-training-openvino-nncf-with-output_files/301-tensorflow-training-openvino-nncf-with-output_3_28.png


.. parsed-literal::

    2024-03-26 00:47:33.939058: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]
    2024-03-26 00:47:33.939298: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-03-26 00:47:34.102086: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-03-26 00:47:34.102377: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]


.. parsed-literal::

    (32, 180, 180, 3)
    (32,)


.. parsed-literal::

    0.0 1.0


.. parsed-literal::

    2024-03-26 00:47:34.773051: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]
    2024-03-26 00:47:34.773358: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]



.. image:: 301-tensorflow-training-openvino-nncf-with-output_files/301-tensorflow-training-openvino-nncf-with-output_3_33.png


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

    2024-03-26 00:47:35.848318: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]
    2024-03-26 00:47:35.848816: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]


.. parsed-literal::


 1/92 [..............................] - ETA: 1:24 - loss: 1.5954 - accuracy: 0.2812

.. parsed-literal::

    
 2/92 [..............................] - ETA: 6s - loss: 2.7563 - accuracy: 0.2188

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 2.3804 - accuracy: 0.2396

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 2.2234 - accuracy: 0.2344

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 2.1300 - accuracy: 0.2438

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 2.0498 - accuracy: 0.2448

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 5s - loss: 1.9889 - accuracy: 0.2545

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 1.9468 - accuracy: 0.2782

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 1.9051 - accuracy: 0.2821

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 1.8738 - accuracy: 0.2756

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 1.8497 - accuracy: 0.2645

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 1.8266 - accuracy: 0.2553

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 1.8030 - accuracy: 0.2598

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 1.7786 - accuracy: 0.2727

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 1.7551 - accuracy: 0.2903

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 1.7377 - accuracy: 0.2996

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 1.7297 - accuracy: 0.3041

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 1.7181 - accuracy: 0.3046

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 1.7063 - accuracy: 0.3133

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 1.6953 - accuracy: 0.3149

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 1.6822 - accuracy: 0.3178

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 1.6752 - accuracy: 0.3218

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 1.6677 - accuracy: 0.3228

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 1.6641 - accuracy: 0.3184

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 1.6566 - accuracy: 0.3182

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 1.6512 - accuracy: 0.3216

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 1.6478 - accuracy: 0.3224

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 1.6397 - accuracy: 0.3277

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 1.6358 - accuracy: 0.3283

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 1.6298 - accuracy: 0.3256

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 1.6242 - accuracy: 0.3262

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 1.6188 - accuracy: 0.3248

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 1.6147 - accuracy: 0.3273

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 1.6106 - accuracy: 0.3287

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 1.6061 - accuracy: 0.3300

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 1.6015 - accuracy: 0.3304

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 1.5991 - accuracy: 0.3282

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 1.5903 - accuracy: 0.3286

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 1.5831 - accuracy: 0.3306

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 1.5777 - accuracy: 0.3349

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 1.5714 - accuracy: 0.3374

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 1.5611 - accuracy: 0.3428

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 1.5562 - accuracy: 0.3428

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 1.5524 - accuracy: 0.3464

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 1.5513 - accuracy: 0.3464

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 1.5450 - accuracy: 0.3490

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 1.5395 - accuracy: 0.3509

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 1.5321 - accuracy: 0.3521

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 1.5269 - accuracy: 0.3545

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 1.5250 - accuracy: 0.3543

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 1.5215 - accuracy: 0.3565

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 1.5183 - accuracy: 0.3563

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 1.5152 - accuracy: 0.3578

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 1.5081 - accuracy: 0.3622

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 1.5040 - accuracy: 0.3653

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 1.4995 - accuracy: 0.3672

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 1.4963 - accuracy: 0.3667

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 1.4907 - accuracy: 0.3690

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 1.4856 - accuracy: 0.3723

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 1.4852 - accuracy: 0.3734

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 1.4820 - accuracy: 0.3735

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 1.4812 - accuracy: 0.3755

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 1.4774 - accuracy: 0.3775

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 1.4755 - accuracy: 0.3765

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 1.4712 - accuracy: 0.3784

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 1.4687 - accuracy: 0.3793

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 1.4630 - accuracy: 0.3811

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 1.4585 - accuracy: 0.3824

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 1.4552 - accuracy: 0.3827

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 1.4520 - accuracy: 0.3840

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 1.4478 - accuracy: 0.3869

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 1.4474 - accuracy: 0.3855

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 1.4438 - accuracy: 0.3857

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 1.4389 - accuracy: 0.3890

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 1.4347 - accuracy: 0.3905

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 1.4319 - accuracy: 0.3911

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 1.4287 - accuracy: 0.3921

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 1.4215 - accuracy: 0.3967

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 1.4191 - accuracy: 0.3976

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 1.4145 - accuracy: 0.3993

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 1.4107 - accuracy: 0.4017

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 1.4085 - accuracy: 0.4021

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 1.4084 - accuracy: 0.4026

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 1.4045 - accuracy: 0.4049

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 1.4013 - accuracy: 0.4063

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 1.3977 - accuracy: 0.4085

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 1.3934 - accuracy: 0.4107

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 1.3891 - accuracy: 0.4131

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 1.3853 - accuracy: 0.4155

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 1.3837 - accuracy: 0.4168

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 1.3819 - accuracy: 0.4177

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 1.3749 - accuracy: 0.4230

.. parsed-literal::

    2024-03-26 00:47:42.053665: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [734]
    	 [[{{node Placeholder/_0}}]]
    2024-03-26 00:47:42.053930: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [734]
    	 [[{{node Placeholder/_4}}]]


.. parsed-literal::

    
92/92 [==============================] - 7s 66ms/step - loss: 1.3749 - accuracy: 0.4230 - val_loss: 1.0496 - val_accuracy: 0.5858


.. parsed-literal::

    Epoch 2/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 1.2158 - accuracy: 0.5000

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 1.1093 - accuracy: 0.5469

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 1.1151 - accuracy: 0.5208

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 1.1434 - accuracy: 0.4844

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 1.1226 - accuracy: 0.5125

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 1.0869 - accuracy: 0.5312

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 1.0718 - accuracy: 0.5446

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 1.0686 - accuracy: 0.5508

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 1.0964 - accuracy: 0.5625

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 1.0734 - accuracy: 0.5813

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 1.0613 - accuracy: 0.5881

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 1.0456 - accuracy: 0.5885

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 1.0370 - accuracy: 0.5865

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 1.0504 - accuracy: 0.5871

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 1.0584 - accuracy: 0.5813

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 1.0699 - accuracy: 0.5801

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 1.0713 - accuracy: 0.5809

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 1.0746 - accuracy: 0.5747

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 1.0811 - accuracy: 0.5740

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 1.0937 - accuracy: 0.5625

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 1.0941 - accuracy: 0.5595

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 1.0937 - accuracy: 0.5582

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 1.0983 - accuracy: 0.5543

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 1.0947 - accuracy: 0.5547

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 1.0953 - accuracy: 0.5512

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 1.1013 - accuracy: 0.5577

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 1.0940 - accuracy: 0.5613

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 1.0965 - accuracy: 0.5558

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 1.0971 - accuracy: 0.5560

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 1.1022 - accuracy: 0.5542

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 1.1058 - accuracy: 0.5544

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 1.1012 - accuracy: 0.5586

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 1.1004 - accuracy: 0.5597

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 1.1063 - accuracy: 0.5542

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 1.1028 - accuracy: 0.5536

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 1.1016 - accuracy: 0.5556

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 1.0999 - accuracy: 0.5566

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 1.0936 - accuracy: 0.5617

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 1.0913 - accuracy: 0.5617

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 1.0896 - accuracy: 0.5641

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 1.0853 - accuracy: 0.5655

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 1.0936 - accuracy: 0.5618

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 1.0962 - accuracy: 0.5640

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 1.0918 - accuracy: 0.5668

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 1.0889 - accuracy: 0.5701

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 1.0910 - accuracy: 0.5693

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 1.0931 - accuracy: 0.5685

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 1.0922 - accuracy: 0.5684

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 1.0875 - accuracy: 0.5695

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 1.0930 - accuracy: 0.5644

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 1.0956 - accuracy: 0.5631

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 1.0944 - accuracy: 0.5649

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 1.0917 - accuracy: 0.5654

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 1.0860 - accuracy: 0.5660

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 1.0878 - accuracy: 0.5648

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 1.0887 - accuracy: 0.5631

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 1.0883 - accuracy: 0.5614

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 1.0834 - accuracy: 0.5620

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 1.0846 - accuracy: 0.5614

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 1.0833 - accuracy: 0.5635

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 1.0831 - accuracy: 0.5645

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 1.0812 - accuracy: 0.5670

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 1.0808 - accuracy: 0.5660

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 1.0805 - accuracy: 0.5664

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 1.0777 - accuracy: 0.5663

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 1.0800 - accuracy: 0.5658

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 1.0765 - accuracy: 0.5681

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 1.0779 - accuracy: 0.5676

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 1.0745 - accuracy: 0.5697

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 1.0739 - accuracy: 0.5688

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 1.0725 - accuracy: 0.5687

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 1.0761 - accuracy: 0.5668

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 1.0739 - accuracy: 0.5668

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 1.0733 - accuracy: 0.5671

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 1.0714 - accuracy: 0.5679

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 1.0691 - accuracy: 0.5687

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 1.0705 - accuracy: 0.5690

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 1.0693 - accuracy: 0.5693

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 1.0649 - accuracy: 0.5704

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 1.0645 - accuracy: 0.5711

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 1.0668 - accuracy: 0.5706

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 1.0644 - accuracy: 0.5720

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 1.0626 - accuracy: 0.5727

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 1.0635 - accuracy: 0.5718

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 1.0618 - accuracy: 0.5721

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 1.0602 - accuracy: 0.5723

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 1.0580 - accuracy: 0.5740

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 1.0588 - accuracy: 0.5732

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 1.0594 - accuracy: 0.5727

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 1.0572 - accuracy: 0.5729

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 1.0568 - accuracy: 0.5729

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 1.0568 - accuracy: 0.5729 - val_loss: 0.9544 - val_accuracy: 0.6172


.. parsed-literal::

    Epoch 3/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.8932 - accuracy: 0.6250

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 1.0382 - accuracy: 0.5781

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.9613 - accuracy: 0.6354

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.9476 - accuracy: 0.6250

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.9552 - accuracy: 0.6250

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.9476 - accuracy: 0.6354

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.9461 - accuracy: 0.6295

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.9649 - accuracy: 0.6250

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.9386 - accuracy: 0.6389

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.9255 - accuracy: 0.6469

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.9200 - accuracy: 0.6477

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.9522 - accuracy: 0.6380

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.9553 - accuracy: 0.6322

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.9529 - accuracy: 0.6272

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.9535 - accuracy: 0.6292

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.9572 - accuracy: 0.6270

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.9458 - accuracy: 0.6305

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.9372 - accuracy: 0.6372

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.9347 - accuracy: 0.6414

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.9224 - accuracy: 0.6484

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.9234 - accuracy: 0.6473

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.9182 - accuracy: 0.6463

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.9069 - accuracy: 0.6495

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.9007 - accuracy: 0.6510

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.9019 - accuracy: 0.6513

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.9011 - accuracy: 0.6514

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.9008 - accuracy: 0.6516

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.9049 - accuracy: 0.6507

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.9021 - accuracy: 0.6487

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.9038 - accuracy: 0.6479

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.9021 - accuracy: 0.6472

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.8938 - accuracy: 0.6484

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.8967 - accuracy: 0.6506

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.8945 - accuracy: 0.6526

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.8912 - accuracy: 0.6518

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.8826 - accuracy: 0.6545

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.8882 - accuracy: 0.6529

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.8977 - accuracy: 0.6488

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.8997 - accuracy: 0.6466

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.8986 - accuracy: 0.6461

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.8930 - accuracy: 0.6494

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.8932 - accuracy: 0.6481

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.8902 - accuracy: 0.6490

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.8953 - accuracy: 0.6484

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.8926 - accuracy: 0.6537

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.8919 - accuracy: 0.6537

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.8932 - accuracy: 0.6525

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.8973 - accuracy: 0.6538

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.8989 - accuracy: 0.6539

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.8996 - accuracy: 0.6533

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.9021 - accuracy: 0.6522

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.9101 - accuracy: 0.6475

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.9099 - accuracy: 0.6459

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.9096 - accuracy: 0.6455

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.9117 - accuracy: 0.6457

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.9149 - accuracy: 0.6432

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.9152 - accuracy: 0.6423

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.9146 - accuracy: 0.6420

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.9184 - accuracy: 0.6391

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.9198 - accuracy: 0.6389

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.9175 - accuracy: 0.6392

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.9189 - accuracy: 0.6379

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.9163 - accuracy: 0.6397

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.9138 - accuracy: 0.6404

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.9160 - accuracy: 0.6393

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.9171 - accuracy: 0.6390

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.9208 - accuracy: 0.6379

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.9226 - accuracy: 0.6373

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.9210 - accuracy: 0.6384

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.9202 - accuracy: 0.6391

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.9238 - accuracy: 0.6385

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.9283 - accuracy: 0.6366

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.9320 - accuracy: 0.6364

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.9310 - accuracy: 0.6380

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.9334 - accuracy: 0.6370

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.9343 - accuracy: 0.6384

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.9338 - accuracy: 0.6383

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.9356 - accuracy: 0.6377

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.9378 - accuracy: 0.6356

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.9351 - accuracy: 0.6362

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.9342 - accuracy: 0.6365

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.9348 - accuracy: 0.6367

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.9351 - accuracy: 0.6358

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.9359 - accuracy: 0.6361

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.9368 - accuracy: 0.6356

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.9389 - accuracy: 0.6344

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.9385 - accuracy: 0.6350

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.9391 - accuracy: 0.6349

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.9405 - accuracy: 0.6337

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.9415 - accuracy: 0.6343

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.9398 - accuracy: 0.6356

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.9398 - accuracy: 0.6356 - val_loss: 0.8804 - val_accuracy: 0.6689


.. parsed-literal::

    Epoch 4/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.8340 - accuracy: 0.6562

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.9211 - accuracy: 0.6562

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.9262 - accuracy: 0.6458

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.9397 - accuracy: 0.6484

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.9500 - accuracy: 0.6313

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.9335 - accuracy: 0.6354

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.9298 - accuracy: 0.6339

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.9676 - accuracy: 0.6143

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.9409 - accuracy: 0.6250

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.9048 - accuracy: 0.6453

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.8883 - accuracy: 0.6569

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.8859 - accuracy: 0.6618

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.9022 - accuracy: 0.6500

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.8911 - accuracy: 0.6547

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.8799 - accuracy: 0.6587

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.8776 - accuracy: 0.6623

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.8785 - accuracy: 0.6637

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.8803 - accuracy: 0.6633

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.8831 - accuracy: 0.6677

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.8808 - accuracy: 0.6657

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 3s - loss: 0.8681 - accuracy: 0.6724

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.8723 - accuracy: 0.6676

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.8618 - accuracy: 0.6724

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.8522 - accuracy: 0.6780

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.8574 - accuracy: 0.6748

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.8587 - accuracy: 0.6729

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.8643 - accuracy: 0.6678

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.8745 - accuracy: 0.6620

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.8839 - accuracy: 0.6618

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.8883 - accuracy: 0.6616

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.8845 - accuracy: 0.6654

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.8835 - accuracy: 0.6641

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.8779 - accuracy: 0.6667

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.8777 - accuracy: 0.6637

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.8757 - accuracy: 0.6652

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.8775 - accuracy: 0.6641

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.8796 - accuracy: 0.6639

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.8783 - accuracy: 0.6653

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.8746 - accuracy: 0.6667

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.8762 - accuracy: 0.6656

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.8707 - accuracy: 0.6669

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.8661 - accuracy: 0.6674

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.8651 - accuracy: 0.6671

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.8664 - accuracy: 0.6662

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.8672 - accuracy: 0.6660

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.8657 - accuracy: 0.6651

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.8687 - accuracy: 0.6656

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.8674 - accuracy: 0.6654

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.8713 - accuracy: 0.6646

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.8718 - accuracy: 0.6638

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.8716 - accuracy: 0.6630

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.8738 - accuracy: 0.6600

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.8750 - accuracy: 0.6599

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.8747 - accuracy: 0.6598

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.8710 - accuracy: 0.6603

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.8711 - accuracy: 0.6597

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.8750 - accuracy: 0.6585

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.8806 - accuracy: 0.6559

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.8796 - accuracy: 0.6569

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.8815 - accuracy: 0.6559

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.8842 - accuracy: 0.6538

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.8830 - accuracy: 0.6559

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.8841 - accuracy: 0.6564

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.8822 - accuracy: 0.6593

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.8795 - accuracy: 0.6602

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.8811 - accuracy: 0.6601

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.8813 - accuracy: 0.6601

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.8800 - accuracy: 0.6586

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.8808 - accuracy: 0.6586

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.8787 - accuracy: 0.6595

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.8784 - accuracy: 0.6590

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.8777 - accuracy: 0.6589

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.8766 - accuracy: 0.6589

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.8747 - accuracy: 0.6605

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.8743 - accuracy: 0.6617

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.8723 - accuracy: 0.6629

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.8737 - accuracy: 0.6632

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.8727 - accuracy: 0.6631

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.8740 - accuracy: 0.6638

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.8769 - accuracy: 0.6637

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.8777 - accuracy: 0.6632

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.8792 - accuracy: 0.6628

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.8829 - accuracy: 0.6608

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.8809 - accuracy: 0.6611

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.8816 - accuracy: 0.6607

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.8833 - accuracy: 0.6621

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.8801 - accuracy: 0.6635

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.8791 - accuracy: 0.6641

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.8781 - accuracy: 0.6640

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.8780 - accuracy: 0.6649

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.8791 - accuracy: 0.6638

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.8791 - accuracy: 0.6638 - val_loss: 0.9852 - val_accuracy: 0.6035


.. parsed-literal::

    Epoch 5/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.5466 - accuracy: 0.8125

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.7630 - accuracy: 0.7656

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6892 - accuracy: 0.8125

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.7431 - accuracy: 0.7734

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 4s - loss: 0.7409 - accuracy: 0.7625

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.7505 - accuracy: 0.7604

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.7826 - accuracy: 0.7277

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.7637 - accuracy: 0.7344

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.7487 - accuracy: 0.7361

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.7520 - accuracy: 0.7188

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.7805 - accuracy: 0.7188

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.7978 - accuracy: 0.7109

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.8237 - accuracy: 0.6971

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.8482 - accuracy: 0.6853

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.8479 - accuracy: 0.6854

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.8502 - accuracy: 0.6855

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.8559 - accuracy: 0.6893

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.8547 - accuracy: 0.6823

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.8485 - accuracy: 0.6851

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.8495 - accuracy: 0.6837

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.8508 - accuracy: 0.6810

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.8414 - accuracy: 0.6813

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.8424 - accuracy: 0.6842

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.8322 - accuracy: 0.6869

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.8395 - accuracy: 0.6808

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.8347 - accuracy: 0.6834

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.8311 - accuracy: 0.6847

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.8262 - accuracy: 0.6870

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.8346 - accuracy: 0.6817

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.8378 - accuracy: 0.6799

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.8385 - accuracy: 0.6801

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.8405 - accuracy: 0.6746

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.8407 - accuracy: 0.6769

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.8352 - accuracy: 0.6799

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.8353 - accuracy: 0.6801

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.8346 - accuracy: 0.6803

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.8316 - accuracy: 0.6805

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.8365 - accuracy: 0.6766

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.8380 - accuracy: 0.6769

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.8331 - accuracy: 0.6794

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.8278 - accuracy: 0.6804

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.8263 - accuracy: 0.6791

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.8269 - accuracy: 0.6793

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.8218 - accuracy: 0.6809

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.8185 - accuracy: 0.6824

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.8156 - accuracy: 0.6825

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.8168 - accuracy: 0.6819

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.8177 - accuracy: 0.6827

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.8191 - accuracy: 0.6796

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.8165 - accuracy: 0.6810

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.8192 - accuracy: 0.6806

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.8170 - accuracy: 0.6825

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.8212 - accuracy: 0.6797

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.8222 - accuracy: 0.6798

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.8181 - accuracy: 0.6822

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.8208 - accuracy: 0.6817

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.8212 - accuracy: 0.6824

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.8207 - accuracy: 0.6835

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.8239 - accuracy: 0.6820

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.8243 - accuracy: 0.6826

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.8232 - accuracy: 0.6842

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.8212 - accuracy: 0.6853

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.8211 - accuracy: 0.6848

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.8224 - accuracy: 0.6834

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.8197 - accuracy: 0.6844

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.8179 - accuracy: 0.6854

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.8179 - accuracy: 0.6854

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.8183 - accuracy: 0.6841

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.8188 - accuracy: 0.6841

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.8175 - accuracy: 0.6842

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.8156 - accuracy: 0.6851

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.8143 - accuracy: 0.6860

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.8115 - accuracy: 0.6864

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.8145 - accuracy: 0.6852

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.8133 - accuracy: 0.6856

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.8129 - accuracy: 0.6865

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.8126 - accuracy: 0.6865

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.8117 - accuracy: 0.6869

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.8105 - accuracy: 0.6873

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.8124 - accuracy: 0.6877

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.8122 - accuracy: 0.6877

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.8093 - accuracy: 0.6892

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.8108 - accuracy: 0.6892

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.8113 - accuracy: 0.6892

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.8111 - accuracy: 0.6891

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.8087 - accuracy: 0.6895

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.8074 - accuracy: 0.6905

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.8073 - accuracy: 0.6898

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.8070 - accuracy: 0.6905

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.8080 - accuracy: 0.6904

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.8054 - accuracy: 0.6918

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.8054 - accuracy: 0.6918 - val_loss: 0.8109 - val_accuracy: 0.7016


.. parsed-literal::

    Epoch 6/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.6900 - accuracy: 0.7500

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.7109 - accuracy: 0.7188

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6937 - accuracy: 0.7292

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6976 - accuracy: 0.7344

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.6722 - accuracy: 0.7500

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.6701 - accuracy: 0.7448

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6857 - accuracy: 0.7321

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.7234 - accuracy: 0.7070

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.7514 - accuracy: 0.6840

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.7883 - accuracy: 0.6656

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.7983 - accuracy: 0.6648

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.8040 - accuracy: 0.6719

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.7887 - accuracy: 0.6803

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.7917 - accuracy: 0.6786

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.7840 - accuracy: 0.6833

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.7688 - accuracy: 0.6914

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.7827 - accuracy: 0.6893

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.7923 - accuracy: 0.6858

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.7798 - accuracy: 0.6875

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.7905 - accuracy: 0.6828

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.7957 - accuracy: 0.6815

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.7912 - accuracy: 0.6832

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.7831 - accuracy: 0.6889

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.7847 - accuracy: 0.6875

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.7832 - accuracy: 0.6888

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.7906 - accuracy: 0.6827

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.7990 - accuracy: 0.6829

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.7916 - accuracy: 0.6864

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.7887 - accuracy: 0.6897

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.7866 - accuracy: 0.6885

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.7820 - accuracy: 0.6925

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.7746 - accuracy: 0.6953

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.7654 - accuracy: 0.7017

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.7649 - accuracy: 0.7022

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.7593 - accuracy: 0.7045

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.7520 - accuracy: 0.7083

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.7508 - accuracy: 0.7111

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.7546 - accuracy: 0.7105

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.7508 - accuracy: 0.7115

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.7509 - accuracy: 0.7109

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.7466 - accuracy: 0.7111

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.7520 - accuracy: 0.7068

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.7536 - accuracy: 0.7064

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.7545 - accuracy: 0.7074

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.7528 - accuracy: 0.7076

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.7521 - accuracy: 0.7086

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.7509 - accuracy: 0.7074

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.7539 - accuracy: 0.7077

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.7503 - accuracy: 0.7105

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.7517 - accuracy: 0.7106

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.7527 - accuracy: 0.7102

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.7554 - accuracy: 0.7091

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.7572 - accuracy: 0.7081

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.7577 - accuracy: 0.7078

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.7573 - accuracy: 0.7068

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.7535 - accuracy: 0.7087

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.7591 - accuracy: 0.7089

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.7573 - accuracy: 0.7112

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.7594 - accuracy: 0.7108

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.7622 - accuracy: 0.7094

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.7594 - accuracy: 0.7100

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.7587 - accuracy: 0.7102

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.7590 - accuracy: 0.7098

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.7585 - accuracy: 0.7100

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.7583 - accuracy: 0.7096

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.7595 - accuracy: 0.7088

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.7574 - accuracy: 0.7090

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.7585 - accuracy: 0.7091

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.7600 - accuracy: 0.7083

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.7602 - accuracy: 0.7089

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.7620 - accuracy: 0.7073

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.7609 - accuracy: 0.7092

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.7617 - accuracy: 0.7093

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.7619 - accuracy: 0.7086

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.7623 - accuracy: 0.7104

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.7644 - accuracy: 0.7093

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.7646 - accuracy: 0.7082

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.7632 - accuracy: 0.7095

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.7665 - accuracy: 0.7073

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.7662 - accuracy: 0.7070

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.7684 - accuracy: 0.7060

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.7678 - accuracy: 0.7062

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.7678 - accuracy: 0.7059

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.7693 - accuracy: 0.7043

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.7681 - accuracy: 0.7048

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.7661 - accuracy: 0.7064

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.7687 - accuracy: 0.7044

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.7682 - accuracy: 0.7049

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.7681 - accuracy: 0.7044

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.7684 - accuracy: 0.7042

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.7662 - accuracy: 0.7050

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.7662 - accuracy: 0.7050 - val_loss: 0.7632 - val_accuracy: 0.7071


.. parsed-literal::

    Epoch 7/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.6328 - accuracy: 0.7812

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.7247 - accuracy: 0.7188

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.7307 - accuracy: 0.7188

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.7388 - accuracy: 0.7188

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.7150 - accuracy: 0.7312

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.6998 - accuracy: 0.7188

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6832 - accuracy: 0.7321

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6904 - accuracy: 0.7344

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6921 - accuracy: 0.7292

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6749 - accuracy: 0.7375

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6683 - accuracy: 0.7415

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6621 - accuracy: 0.7448

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6653 - accuracy: 0.7380

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6871 - accuracy: 0.7254

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.7092 - accuracy: 0.7167

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6993 - accuracy: 0.7168

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6995 - accuracy: 0.7224

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6889 - accuracy: 0.7257

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6934 - accuracy: 0.7168

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6905 - accuracy: 0.7199

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 3s - loss: 0.6892 - accuracy: 0.7213

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.6816 - accuracy: 0.7253

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6918 - accuracy: 0.7197

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6971 - accuracy: 0.7159

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6960 - accuracy: 0.7148

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6963 - accuracy: 0.7161

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.7110 - accuracy: 0.7117

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.7138 - accuracy: 0.7076

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.7236 - accuracy: 0.7048

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.7243 - accuracy: 0.7073

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.7250 - accuracy: 0.7077

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.7260 - accuracy: 0.7061

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.7321 - accuracy: 0.7056

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.7325 - accuracy: 0.7077

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.7330 - accuracy: 0.7063

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.7354 - accuracy: 0.7083

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.7360 - accuracy: 0.7086

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.7343 - accuracy: 0.7097

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.7366 - accuracy: 0.7083

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.7455 - accuracy: 0.7078

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.7437 - accuracy: 0.7103

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.7437 - accuracy: 0.7113

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.7547 - accuracy: 0.7057

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.7554 - accuracy: 0.7060

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.7525 - accuracy: 0.7097

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.7520 - accuracy: 0.7086

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.7560 - accuracy: 0.7075

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.7571 - accuracy: 0.7064

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.7570 - accuracy: 0.7060

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.7561 - accuracy: 0.7081

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.7582 - accuracy: 0.7077

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.7554 - accuracy: 0.7085

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.7585 - accuracy: 0.7076

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.7583 - accuracy: 0.7089

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.7572 - accuracy: 0.7102

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.7550 - accuracy: 0.7109

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.7561 - accuracy: 0.7116

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.7545 - accuracy: 0.7128

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.7538 - accuracy: 0.7129

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.7544 - accuracy: 0.7130

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.7574 - accuracy: 0.7115

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.7535 - accuracy: 0.7136

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.7508 - accuracy: 0.7147

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.7520 - accuracy: 0.7138

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.7524 - accuracy: 0.7139

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.7563 - accuracy: 0.7116

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.7566 - accuracy: 0.7108

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.7546 - accuracy: 0.7118

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.7511 - accuracy: 0.7124

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.7523 - accuracy: 0.7107

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.7522 - accuracy: 0.7104

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.7540 - accuracy: 0.7109

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.7531 - accuracy: 0.7114

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.7506 - accuracy: 0.7124

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.7493 - accuracy: 0.7116

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.7497 - accuracy: 0.7121

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.7466 - accuracy: 0.7134

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.7463 - accuracy: 0.7143

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.7482 - accuracy: 0.7136

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.7471 - accuracy: 0.7125

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.7440 - accuracy: 0.7129

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.7411 - accuracy: 0.7141

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.7460 - accuracy: 0.7127

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.7469 - accuracy: 0.7131

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.7460 - accuracy: 0.7136

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.7449 - accuracy: 0.7140

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.7437 - accuracy: 0.7140

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.7413 - accuracy: 0.7144

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.7412 - accuracy: 0.7148

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.7407 - accuracy: 0.7149

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.7415 - accuracy: 0.7146

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.7415 - accuracy: 0.7146 - val_loss: 0.8283 - val_accuracy: 0.6553


.. parsed-literal::

    Epoch 8/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.6520 - accuracy: 0.7812

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.6357 - accuracy: 0.7656

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6933 - accuracy: 0.7292

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6895 - accuracy: 0.7344

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.7363 - accuracy: 0.7000

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.7471 - accuracy: 0.6771

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.7416 - accuracy: 0.6830

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.7830 - accuracy: 0.6680

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.7593 - accuracy: 0.6771

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.7410 - accuracy: 0.6875

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.7380 - accuracy: 0.6875

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.7562 - accuracy: 0.6771

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.7746 - accuracy: 0.6779

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.7669 - accuracy: 0.6786

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.7647 - accuracy: 0.6854

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.7576 - accuracy: 0.6934

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.7656 - accuracy: 0.6857

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.7675 - accuracy: 0.6858

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.7482 - accuracy: 0.6990

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.7391 - accuracy: 0.7047

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.7346 - accuracy: 0.7068

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.7338 - accuracy: 0.7116

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.7403 - accuracy: 0.7065

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.7402 - accuracy: 0.7083

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.7446 - accuracy: 0.7025

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.7456 - accuracy: 0.7067

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.7452 - accuracy: 0.7083

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.7543 - accuracy: 0.7042

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.7510 - accuracy: 0.7058

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.7464 - accuracy: 0.7042

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.7414 - accuracy: 0.7046

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.7346 - accuracy: 0.7090

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.7336 - accuracy: 0.7093

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.7332 - accuracy: 0.7077

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.7324 - accuracy: 0.7071

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.7332 - accuracy: 0.7057

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.7360 - accuracy: 0.7027

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.7402 - accuracy: 0.7023

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.7358 - accuracy: 0.7027

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.7288 - accuracy: 0.7078

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.7285 - accuracy: 0.7073

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.7255 - accuracy: 0.7098

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.7254 - accuracy: 0.7100

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.7273 - accuracy: 0.7102

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.7240 - accuracy: 0.7117

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.7266 - accuracy: 0.7099

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.7296 - accuracy: 0.7081

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.7252 - accuracy: 0.7096

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.7257 - accuracy: 0.7073

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.7229 - accuracy: 0.7106

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.7219 - accuracy: 0.7114

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.7225 - accuracy: 0.7127

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.7227 - accuracy: 0.7128

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.7278 - accuracy: 0.7123

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.7240 - accuracy: 0.7130

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.7242 - accuracy: 0.7137

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.7279 - accuracy: 0.7132

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.7304 - accuracy: 0.7122

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.7270 - accuracy: 0.7139

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.7238 - accuracy: 0.7160

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.7252 - accuracy: 0.7166

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.7276 - accuracy: 0.7156

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.7311 - accuracy: 0.7152

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.7321 - accuracy: 0.7138

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.7306 - accuracy: 0.7139

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.7297 - accuracy: 0.7135

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.7269 - accuracy: 0.7140

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.7230 - accuracy: 0.7159

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.7215 - accuracy: 0.7164

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.7235 - accuracy: 0.7169

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.7221 - accuracy: 0.7173

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.7204 - accuracy: 0.7195

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.7213 - accuracy: 0.7195

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.7211 - accuracy: 0.7195

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.7206 - accuracy: 0.7195

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.7190 - accuracy: 0.7203

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.7173 - accuracy: 0.7215

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.7180 - accuracy: 0.7214

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.7162 - accuracy: 0.7222

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.7137 - accuracy: 0.7229

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.7143 - accuracy: 0.7232

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.7134 - accuracy: 0.7243

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.7146 - accuracy: 0.7235

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.7173 - accuracy: 0.7220

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.7144 - accuracy: 0.7230

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.7133 - accuracy: 0.7237

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.7105 - accuracy: 0.7247

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.7123 - accuracy: 0.7243

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.7108 - accuracy: 0.7253

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.7113 - accuracy: 0.7238

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.7112 - accuracy: 0.7245

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.7112 - accuracy: 0.7245 - val_loss: 0.7494 - val_accuracy: 0.7057


.. parsed-literal::

    Epoch 9/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.7932 - accuracy: 0.6562

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5997 - accuracy: 0.7656

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6476 - accuracy: 0.7292

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6560 - accuracy: 0.7188

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 4s - loss: 0.6975 - accuracy: 0.7000

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.7102 - accuracy: 0.6979

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.7020 - accuracy: 0.7098

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6966 - accuracy: 0.7070

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6830 - accuracy: 0.7118

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6645 - accuracy: 0.7250

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6627 - accuracy: 0.7273

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6652 - accuracy: 0.7318

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6745 - accuracy: 0.7356

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6869 - accuracy: 0.7232

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6750 - accuracy: 0.7292

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6725 - accuracy: 0.7246

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6770 - accuracy: 0.7206

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6754 - accuracy: 0.7205

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6757 - accuracy: 0.7237

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6635 - accuracy: 0.7344

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6588 - accuracy: 0.7381

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6645 - accuracy: 0.7344

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.6590 - accuracy: 0.7364

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 4s - loss: 0.6579 - accuracy: 0.7357

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6714 - accuracy: 0.7300

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6688 - accuracy: 0.7296

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6885 - accuracy: 0.7222

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.7007 - accuracy: 0.7243

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6971 - accuracy: 0.7274

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6927 - accuracy: 0.7302

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6957 - accuracy: 0.7288

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6906 - accuracy: 0.7314

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6846 - accuracy: 0.7358

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6820 - accuracy: 0.7329

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6775 - accuracy: 0.7360

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6820 - accuracy: 0.7355

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6822 - accuracy: 0.7359

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6774 - accuracy: 0.7371

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.6830 - accuracy: 0.7358

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6795 - accuracy: 0.7370

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6812 - accuracy: 0.7373

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6812 - accuracy: 0.7383

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6825 - accuracy: 0.7400

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6822 - accuracy: 0.7402

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6825 - accuracy: 0.7411

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6848 - accuracy: 0.7393

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6832 - accuracy: 0.7402

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6843 - accuracy: 0.7391

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6817 - accuracy: 0.7393

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6851 - accuracy: 0.7389

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6847 - accuracy: 0.7391

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6831 - accuracy: 0.7405

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6789 - accuracy: 0.7424

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6819 - accuracy: 0.7414

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6825 - accuracy: 0.7410

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6830 - accuracy: 0.7406

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6816 - accuracy: 0.7413

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6862 - accuracy: 0.7399

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6845 - accuracy: 0.7406

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6854 - accuracy: 0.7397

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6865 - accuracy: 0.7384

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6857 - accuracy: 0.7380

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6818 - accuracy: 0.7397

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6813 - accuracy: 0.7408

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6880 - accuracy: 0.7391

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6902 - accuracy: 0.7388

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6906 - accuracy: 0.7389

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6936 - accuracy: 0.7382

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6952 - accuracy: 0.7384

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6953 - accuracy: 0.7390

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6965 - accuracy: 0.7391

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6969 - accuracy: 0.7384

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6978 - accuracy: 0.7386

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6967 - accuracy: 0.7391

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6970 - accuracy: 0.7389

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6964 - accuracy: 0.7378

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6972 - accuracy: 0.7371

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6957 - accuracy: 0.7385

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6959 - accuracy: 0.7379

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6972 - accuracy: 0.7361

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6956 - accuracy: 0.7362

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6984 - accuracy: 0.7345

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6963 - accuracy: 0.7351

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6983 - accuracy: 0.7349

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6967 - accuracy: 0.7362

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6961 - accuracy: 0.7363

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6971 - accuracy: 0.7347

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6957 - accuracy: 0.7349

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6962 - accuracy: 0.7340

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6941 - accuracy: 0.7348

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6934 - accuracy: 0.7354

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.6934 - accuracy: 0.7354 - val_loss: 0.7232 - val_accuracy: 0.7180


.. parsed-literal::

    Epoch 10/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.7562 - accuracy: 0.7188

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.6408 - accuracy: 0.7812

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6744 - accuracy: 0.7812

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6956 - accuracy: 0.7734

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 4s - loss: 0.6923 - accuracy: 0.7688

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.6556 - accuracy: 0.7656

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6520 - accuracy: 0.7589

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6403 - accuracy: 0.7617

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6329 - accuracy: 0.7604

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6245 - accuracy: 0.7625

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6135 - accuracy: 0.7699

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6075 - accuracy: 0.7760

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5945 - accuracy: 0.7812

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5930 - accuracy: 0.7835

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6062 - accuracy: 0.7750

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6069 - accuracy: 0.7773

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6067 - accuracy: 0.7757

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6012 - accuracy: 0.7778

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5956 - accuracy: 0.7763

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5915 - accuracy: 0.7781

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5990 - accuracy: 0.7768

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6097 - accuracy: 0.7784

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.6074 - accuracy: 0.7785

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6132 - accuracy: 0.7773

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6095 - accuracy: 0.7800

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6018 - accuracy: 0.7837

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6183 - accuracy: 0.7778

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6185 - accuracy: 0.7779

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6152 - accuracy: 0.7780

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6166 - accuracy: 0.7771

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6182 - accuracy: 0.7772

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6195 - accuracy: 0.7744

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6270 - accuracy: 0.7689

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6307 - accuracy: 0.7665

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6311 - accuracy: 0.7661

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6345 - accuracy: 0.7648

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6323 - accuracy: 0.7660

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6425 - accuracy: 0.7632

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6429 - accuracy: 0.7620

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.6383 - accuracy: 0.7641

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6387 - accuracy: 0.7645

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6417 - accuracy: 0.7641

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6451 - accuracy: 0.7616

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6465 - accuracy: 0.7598

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6492 - accuracy: 0.7596

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6470 - accuracy: 0.7607

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6493 - accuracy: 0.7592

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6533 - accuracy: 0.7583

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6546 - accuracy: 0.7575

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6551 - accuracy: 0.7580

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6535 - accuracy: 0.7579

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6527 - accuracy: 0.7583

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6549 - accuracy: 0.7576

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6552 - accuracy: 0.7580

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6529 - accuracy: 0.7590

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6555 - accuracy: 0.7577

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6533 - accuracy: 0.7592

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6535 - accuracy: 0.7596

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6570 - accuracy: 0.7578

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6528 - accuracy: 0.7593

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6540 - accuracy: 0.7581

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6516 - accuracy: 0.7590

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6522 - accuracy: 0.7598

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6531 - accuracy: 0.7597

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6570 - accuracy: 0.7576

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6557 - accuracy: 0.7570

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6563 - accuracy: 0.7560

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6551 - accuracy: 0.7559

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6580 - accuracy: 0.7549

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6593 - accuracy: 0.7540

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6627 - accuracy: 0.7526

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6625 - accuracy: 0.7530

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6618 - accuracy: 0.7530

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6606 - accuracy: 0.7533

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6633 - accuracy: 0.7529

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6613 - accuracy: 0.7524

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6582 - accuracy: 0.7532

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6555 - accuracy: 0.7544

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6550 - accuracy: 0.7539

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6543 - accuracy: 0.7543

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6547 - accuracy: 0.7550

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6563 - accuracy: 0.7542

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6565 - accuracy: 0.7537

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6559 - accuracy: 0.7537

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6530 - accuracy: 0.7551

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6518 - accuracy: 0.7554

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6531 - accuracy: 0.7553

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6534 - accuracy: 0.7549

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6563 - accuracy: 0.7535

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6563 - accuracy: 0.7534

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6571 - accuracy: 0.7527

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.6571 - accuracy: 0.7527 - val_loss: 0.6993 - val_accuracy: 0.7057


.. parsed-literal::

    Epoch 11/15


.. parsed-literal::


 1/92 [..............................] - ETA: 6s - loss: 0.6069 - accuracy: 0.7500

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5452 - accuracy: 0.8125

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.4984 - accuracy: 0.8438

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.4999 - accuracy: 0.8281

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5367 - accuracy: 0.8188

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.5187 - accuracy: 0.8177

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.5356 - accuracy: 0.8080

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5444 - accuracy: 0.8086

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5540 - accuracy: 0.8056

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5527 - accuracy: 0.8000

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5680 - accuracy: 0.7955

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5843 - accuracy: 0.7865

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5843 - accuracy: 0.7837

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5896 - accuracy: 0.7790

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6007 - accuracy: 0.7771

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5975 - accuracy: 0.7793

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5952 - accuracy: 0.7831

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5944 - accuracy: 0.7795

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5909 - accuracy: 0.7796

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5920 - accuracy: 0.7734

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5971 - accuracy: 0.7693

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6000 - accuracy: 0.7713

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.5885 - accuracy: 0.7745

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5929 - accuracy: 0.7734

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5968 - accuracy: 0.7738

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5975 - accuracy: 0.7728

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5973 - accuracy: 0.7720

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5997 - accuracy: 0.7734

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5915 - accuracy: 0.7780

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5929 - accuracy: 0.7750

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5929 - accuracy: 0.7742

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5914 - accuracy: 0.7734

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5941 - accuracy: 0.7718

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5879 - accuracy: 0.7748

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5890 - accuracy: 0.7741

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5854 - accuracy: 0.7760

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5858 - accuracy: 0.7762

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5909 - accuracy: 0.7738

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5917 - accuracy: 0.7724

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.5950 - accuracy: 0.7703

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5940 - accuracy: 0.7706

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6041 - accuracy: 0.7671

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6109 - accuracy: 0.7667

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6098 - accuracy: 0.7678

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6116 - accuracy: 0.7667

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6164 - accuracy: 0.7663

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6115 - accuracy: 0.7686

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6110 - accuracy: 0.7702

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6072 - accuracy: 0.7723

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6052 - accuracy: 0.7750

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6090 - accuracy: 0.7721

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6116 - accuracy: 0.7704

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6112 - accuracy: 0.7718

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6103 - accuracy: 0.7726

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6090 - accuracy: 0.7722

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6135 - accuracy: 0.7690

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6145 - accuracy: 0.7681

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6142 - accuracy: 0.7667

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6183 - accuracy: 0.7632

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6179 - accuracy: 0.7625

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6158 - accuracy: 0.7633

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6158 - accuracy: 0.7636

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6153 - accuracy: 0.7639

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6165 - accuracy: 0.7637

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6152 - accuracy: 0.7635

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6203 - accuracy: 0.7604

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6192 - accuracy: 0.7612

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6158 - accuracy: 0.7629

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6172 - accuracy: 0.7622

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6170 - accuracy: 0.7629

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6177 - accuracy: 0.7619

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6159 - accuracy: 0.7626

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6175 - accuracy: 0.7616

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6219 - accuracy: 0.7610

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6235 - accuracy: 0.7617

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6244 - accuracy: 0.7623

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6239 - accuracy: 0.7622

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6271 - accuracy: 0.7608

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6253 - accuracy: 0.7615

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6237 - accuracy: 0.7620

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6300 - accuracy: 0.7611

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6304 - accuracy: 0.7613

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6311 - accuracy: 0.7619

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6318 - accuracy: 0.7614

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6323 - accuracy: 0.7628

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6375 - accuracy: 0.7608

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6386 - accuracy: 0.7607

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6366 - accuracy: 0.7613

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6362 - accuracy: 0.7611

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6383 - accuracy: 0.7607

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6369 - accuracy: 0.7612

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.6369 - accuracy: 0.7612 - val_loss: 0.7075 - val_accuracy: 0.7330


.. parsed-literal::

    Epoch 12/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.4500 - accuracy: 0.8125

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5225 - accuracy: 0.7969

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5612 - accuracy: 0.7708

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5623 - accuracy: 0.7578

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5879 - accuracy: 0.7500

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.6115 - accuracy: 0.7240

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.5935 - accuracy: 0.7455

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6019 - accuracy: 0.7461

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5915 - accuracy: 0.7535

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5786 - accuracy: 0.7625

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5800 - accuracy: 0.7614

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5674 - accuracy: 0.7682

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5575 - accuracy: 0.7740

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5640 - accuracy: 0.7768

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5483 - accuracy: 0.7854

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5637 - accuracy: 0.7793

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5567 - accuracy: 0.7849

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5617 - accuracy: 0.7830

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5567 - accuracy: 0.7862

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5607 - accuracy: 0.7828

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5654 - accuracy: 0.7798

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5709 - accuracy: 0.7798

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.5717 - accuracy: 0.7785

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5671 - accuracy: 0.7812

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5721 - accuracy: 0.7775

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5865 - accuracy: 0.7728

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5852 - accuracy: 0.7743

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5858 - accuracy: 0.7723

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5935 - accuracy: 0.7705

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5968 - accuracy: 0.7693

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5985 - accuracy: 0.7687

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6084 - accuracy: 0.7662

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6138 - accuracy: 0.7657

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6163 - accuracy: 0.7635

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6112 - accuracy: 0.7657

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6113 - accuracy: 0.7679

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6139 - accuracy: 0.7690

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6181 - accuracy: 0.7685

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.6173 - accuracy: 0.7689

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6202 - accuracy: 0.7669

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6262 - accuracy: 0.7657

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6253 - accuracy: 0.7654

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6219 - accuracy: 0.7657

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6242 - accuracy: 0.7647

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6208 - accuracy: 0.7657

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6192 - accuracy: 0.7654

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6229 - accuracy: 0.7644

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6225 - accuracy: 0.7654

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6199 - accuracy: 0.7657

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6173 - accuracy: 0.7672

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6151 - accuracy: 0.7675

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6180 - accuracy: 0.7648

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6156 - accuracy: 0.7651

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6121 - accuracy: 0.7666

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6136 - accuracy: 0.7651

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6120 - accuracy: 0.7665

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6111 - accuracy: 0.7668

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6113 - accuracy: 0.7644

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6098 - accuracy: 0.7641

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6097 - accuracy: 0.7639

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6110 - accuracy: 0.7632

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6096 - accuracy: 0.7644

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6070 - accuracy: 0.7657

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6122 - accuracy: 0.7640

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6125 - accuracy: 0.7638

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6122 - accuracy: 0.7640

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6086 - accuracy: 0.7657

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6067 - accuracy: 0.7664

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6076 - accuracy: 0.7666

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6112 - accuracy: 0.7646

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6084 - accuracy: 0.7648

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6083 - accuracy: 0.7650

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6055 - accuracy: 0.7661

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6020 - accuracy: 0.7671

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6048 - accuracy: 0.7661

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6051 - accuracy: 0.7659

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6077 - accuracy: 0.7645

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6132 - accuracy: 0.7623

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6122 - accuracy: 0.7625

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6145 - accuracy: 0.7612

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6131 - accuracy: 0.7619

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6116 - accuracy: 0.7617

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6121 - accuracy: 0.7608

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6159 - accuracy: 0.7596

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6137 - accuracy: 0.7602

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6162 - accuracy: 0.7594

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6169 - accuracy: 0.7593

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6170 - accuracy: 0.7588

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6162 - accuracy: 0.7594

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6194 - accuracy: 0.7590

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6199 - accuracy: 0.7585

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.6199 - accuracy: 0.7585 - val_loss: 0.6783 - val_accuracy: 0.7371


.. parsed-literal::

    Epoch 13/15


.. parsed-literal::


 1/92 [..............................] - ETA: 6s - loss: 0.4247 - accuracy: 0.8438

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5067 - accuracy: 0.8125

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5223 - accuracy: 0.8438

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5302 - accuracy: 0.8203

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5584 - accuracy: 0.8000

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.5410 - accuracy: 0.8073

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.5143 - accuracy: 0.8170

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5017 - accuracy: 0.8242

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.4913 - accuracy: 0.8299

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.4800 - accuracy: 0.8313

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5101 - accuracy: 0.8182

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5003 - accuracy: 0.8203

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5007 - accuracy: 0.8173

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.4951 - accuracy: 0.8170

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.4949 - accuracy: 0.8188

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.4955 - accuracy: 0.8184

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.4891 - accuracy: 0.8162

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.4886 - accuracy: 0.8142

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5130 - accuracy: 0.8026

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5140 - accuracy: 0.8000

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5142 - accuracy: 0.7976

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.5191 - accuracy: 0.7940

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5168 - accuracy: 0.7934

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5140 - accuracy: 0.7942

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5200 - accuracy: 0.7900

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5167 - accuracy: 0.7921

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5171 - accuracy: 0.7917

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5214 - accuracy: 0.7913

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5272 - accuracy: 0.7899

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5230 - accuracy: 0.7927

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5300 - accuracy: 0.7923

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5234 - accuracy: 0.7948

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5278 - accuracy: 0.7954

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5300 - accuracy: 0.7941

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5338 - accuracy: 0.7911

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5366 - accuracy: 0.7900

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5330 - accuracy: 0.7914

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5358 - accuracy: 0.7879

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.5427 - accuracy: 0.7877

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5526 - accuracy: 0.7868

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5505 - accuracy: 0.7874

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5502 - accuracy: 0.7873

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5470 - accuracy: 0.7900

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5507 - accuracy: 0.7877

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5515 - accuracy: 0.7883

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5516 - accuracy: 0.7881

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5527 - accuracy: 0.7886

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5528 - accuracy: 0.7878

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5537 - accuracy: 0.7877

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5574 - accuracy: 0.7863

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5581 - accuracy: 0.7844

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5544 - accuracy: 0.7861

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5531 - accuracy: 0.7872

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5585 - accuracy: 0.7842

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5577 - accuracy: 0.7842

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5553 - accuracy: 0.7852

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5561 - accuracy: 0.7841

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5532 - accuracy: 0.7851

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5544 - accuracy: 0.7850

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5563 - accuracy: 0.7845

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5567 - accuracy: 0.7849

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5594 - accuracy: 0.7844

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5615 - accuracy: 0.7843

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5602 - accuracy: 0.7857

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5610 - accuracy: 0.7847

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5599 - accuracy: 0.7856

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5606 - accuracy: 0.7860

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5628 - accuracy: 0.7850

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5611 - accuracy: 0.7854

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5623 - accuracy: 0.7849

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5618 - accuracy: 0.7853

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5589 - accuracy: 0.7869

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5613 - accuracy: 0.7847

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5635 - accuracy: 0.7834

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5646 - accuracy: 0.7826

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5672 - accuracy: 0.7822

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5681 - accuracy: 0.7818

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5687 - accuracy: 0.7821

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5646 - accuracy: 0.7845

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5644 - accuracy: 0.7848

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5635 - accuracy: 0.7859

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5641 - accuracy: 0.7866

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5678 - accuracy: 0.7843

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5668 - accuracy: 0.7847

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5726 - accuracy: 0.7828

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5716 - accuracy: 0.7835

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5735 - accuracy: 0.7835

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5745 - accuracy: 0.7835

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5756 - accuracy: 0.7827

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5761 - accuracy: 0.7817

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5735 - accuracy: 0.7827

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.5735 - accuracy: 0.7827 - val_loss: 0.6753 - val_accuracy: 0.7330


.. parsed-literal::

    Epoch 14/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.4192 - accuracy: 0.8750

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.4099 - accuracy: 0.8750

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.4549 - accuracy: 0.8542

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.4530 - accuracy: 0.8281

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 4s - loss: 0.4456 - accuracy: 0.8250

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.4710 - accuracy: 0.8125

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.4541 - accuracy: 0.8170

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.4605 - accuracy: 0.8203

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.4598 - accuracy: 0.8160

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.4631 - accuracy: 0.8188

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.4695 - accuracy: 0.8210

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.4936 - accuracy: 0.8151

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5028 - accuracy: 0.8149

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.4845 - accuracy: 0.8214

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.4870 - accuracy: 0.8208

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.4892 - accuracy: 0.8184

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5016 - accuracy: 0.8107

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5174 - accuracy: 0.8073

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5177 - accuracy: 0.8059

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5200 - accuracy: 0.8078

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5129 - accuracy: 0.8125

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5092 - accuracy: 0.8111

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.5073 - accuracy: 0.8125

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5048 - accuracy: 0.8125

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5064 - accuracy: 0.8100

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5021 - accuracy: 0.8113

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5028 - accuracy: 0.8102

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5087 - accuracy: 0.8092

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5151 - accuracy: 0.8060

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5193 - accuracy: 0.8031

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5207 - accuracy: 0.8024

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5195 - accuracy: 0.8018

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5174 - accuracy: 0.8011

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5230 - accuracy: 0.8006

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5219 - accuracy: 0.8000

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5220 - accuracy: 0.8012

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5271 - accuracy: 0.7990

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5268 - accuracy: 0.7977

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5340 - accuracy: 0.7941

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.5349 - accuracy: 0.7930

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5361 - accuracy: 0.7927

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5399 - accuracy: 0.7909

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5357 - accuracy: 0.7936

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5393 - accuracy: 0.7933

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5406 - accuracy: 0.7937

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5409 - accuracy: 0.7928

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5368 - accuracy: 0.7952

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5322 - accuracy: 0.7975

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5313 - accuracy: 0.7985

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5299 - accuracy: 0.7994

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5313 - accuracy: 0.7978

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5363 - accuracy: 0.7969

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5358 - accuracy: 0.7966

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5366 - accuracy: 0.7951

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5380 - accuracy: 0.7949

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5360 - accuracy: 0.7946

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5373 - accuracy: 0.7950

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5361 - accuracy: 0.7947

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5322 - accuracy: 0.7966

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5340 - accuracy: 0.7948

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5322 - accuracy: 0.7956

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5336 - accuracy: 0.7944

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5315 - accuracy: 0.7951

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5324 - accuracy: 0.7935

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5319 - accuracy: 0.7942

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5321 - accuracy: 0.7945

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5301 - accuracy: 0.7948

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5282 - accuracy: 0.7964

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5280 - accuracy: 0.7966

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5278 - accuracy: 0.7960

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5277 - accuracy: 0.7967

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5283 - accuracy: 0.7956

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5298 - accuracy: 0.7949

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5295 - accuracy: 0.7952

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5305 - accuracy: 0.7950

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5325 - accuracy: 0.7948

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5331 - accuracy: 0.7946

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5371 - accuracy: 0.7929

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5378 - accuracy: 0.7923

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5400 - accuracy: 0.7914

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5432 - accuracy: 0.7897

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5451 - accuracy: 0.7893

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5438 - accuracy: 0.7899

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5431 - accuracy: 0.7898

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5430 - accuracy: 0.7901

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5415 - accuracy: 0.7911

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5441 - accuracy: 0.7895

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5451 - accuracy: 0.7894

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5499 - accuracy: 0.7872

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5515 - accuracy: 0.7865

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5513 - accuracy: 0.7864

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.5513 - accuracy: 0.7864 - val_loss: 0.6839 - val_accuracy: 0.7234


.. parsed-literal::

    Epoch 15/15


.. parsed-literal::


 1/92 [..............................] - ETA: 8s - loss: 0.5391 - accuracy: 0.7812

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5023 - accuracy: 0.8438

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.4915 - accuracy: 0.8542

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.4998 - accuracy: 0.8438

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5063 - accuracy: 0.8375

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.4881 - accuracy: 0.8490

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.4849 - accuracy: 0.8438

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5109 - accuracy: 0.8359

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5077 - accuracy: 0.8403

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.4938 - accuracy: 0.8406

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5151 - accuracy: 0.8267

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5234 - accuracy: 0.8255

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5285 - accuracy: 0.8173

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5221 - accuracy: 0.8192

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5158 - accuracy: 0.8188

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5290 - accuracy: 0.8047

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5373 - accuracy: 0.8033

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5453 - accuracy: 0.8003

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5433 - accuracy: 0.7961

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5366 - accuracy: 0.7953

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5397 - accuracy: 0.7932

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5389 - accuracy: 0.7926

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.5322 - accuracy: 0.7935

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5405 - accuracy: 0.7852

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5459 - accuracy: 0.7812

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5415 - accuracy: 0.7837

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5426 - accuracy: 0.7836

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5382 - accuracy: 0.7857

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5341 - accuracy: 0.7888

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5394 - accuracy: 0.7854

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5412 - accuracy: 0.7843

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5460 - accuracy: 0.7812

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5516 - accuracy: 0.7794

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5546 - accuracy: 0.7776

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5576 - accuracy: 0.7768

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5531 - accuracy: 0.7795

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5484 - accuracy: 0.7812

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5456 - accuracy: 0.7854

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5418 - accuracy: 0.7877

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.5389 - accuracy: 0.7883

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5333 - accuracy: 0.7896

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5354 - accuracy: 0.7909

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5381 - accuracy: 0.7907

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5346 - accuracy: 0.7926

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5357 - accuracy: 0.7924

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5316 - accuracy: 0.7935

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5303 - accuracy: 0.7932

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5313 - accuracy: 0.7943

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5311 - accuracy: 0.7940

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5341 - accuracy: 0.7925

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5357 - accuracy: 0.7923

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5349 - accuracy: 0.7921

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5330 - accuracy: 0.7942

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5301 - accuracy: 0.7951

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5302 - accuracy: 0.7943

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5279 - accuracy: 0.7946

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5301 - accuracy: 0.7928

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5267 - accuracy: 0.7942

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5252 - accuracy: 0.7956

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5303 - accuracy: 0.7927

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5301 - accuracy: 0.7930

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5283 - accuracy: 0.7939

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5347 - accuracy: 0.7941

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5352 - accuracy: 0.7939

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5356 - accuracy: 0.7933

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5423 - accuracy: 0.7917

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5417 - accuracy: 0.7915

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5405 - accuracy: 0.7918

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5390 - accuracy: 0.7930

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5358 - accuracy: 0.7942

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5349 - accuracy: 0.7940

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5345 - accuracy: 0.7947

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5341 - accuracy: 0.7953

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5307 - accuracy: 0.7968

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5280 - accuracy: 0.7983

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5283 - accuracy: 0.7989

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5288 - accuracy: 0.7994

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5304 - accuracy: 0.7988

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5299 - accuracy: 0.7990

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5299 - accuracy: 0.7991

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5311 - accuracy: 0.7993

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5313 - accuracy: 0.8002

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5309 - accuracy: 0.8007

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5313 - accuracy: 0.8013

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5342 - accuracy: 0.8007

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5353 - accuracy: 0.7997

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5329 - accuracy: 0.8009

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5358 - accuracy: 0.7993

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5360 - accuracy: 0.7998

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5349 - accuracy: 0.8003

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5370 - accuracy: 0.8001

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.5370 - accuracy: 0.8001 - val_loss: 0.7213 - val_accuracy: 0.7180



.. image:: 301-tensorflow-training-openvino-nncf-with-output_files/301-tensorflow-training-openvino-nncf-with-output_3_1468.png


.. parsed-literal::


    1/1 [==============================] - ETA: 0s

.. parsed-literal::

    
1/1 [==============================] - 0s 83ms/step


.. parsed-literal::

    This image most likely belongs to sunflowers with a 97.26 percent confidence.


.. parsed-literal::

    2024-03-26 00:49:04.906709: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'random_flip_input' with dtype float and shape [?,180,180,3]
    	 [[{{node random_flip_input}}]]
    2024-03-26 00:49:04.992095: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-26 00:49:05.002116: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'random_flip_input' with dtype float and shape [?,180,180,3]
    	 [[{{node random_flip_input}}]]
    2024-03-26 00:49:05.012936: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-26 00:49:05.020868: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-26 00:49:05.027814: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-26 00:49:05.038593: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-26 00:49:05.077750: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'sequential_1_input' with dtype float and shape [?,180,180,3]
    	 [[{{node sequential_1_input}}]]


.. parsed-literal::

    2024-03-26 00:49:05.144626: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-26 00:49:05.164951: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'sequential_1_input' with dtype float and shape [?,180,180,3]
    	 [[{{node sequential_1_input}}]]
    2024-03-26 00:49:05.203424: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,22,22,64]
    	 [[{{node inputs}}]]


.. parsed-literal::

    2024-03-26 00:49:05.389708: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-26 00:49:05.490765: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]


.. parsed-literal::

    2024-03-26 00:49:05.632044: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-26 00:49:05.769720: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,22,22,64]
    	 [[{{node inputs}}]]
    2024-03-26 00:49:05.803555: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-26 00:49:05.831194: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]


.. parsed-literal::

    2024-03-26 00:49:05.878489: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
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
    This image most likely belongs to dandelion with a 98.83 percent confidence.



.. image:: 301-tensorflow-training-openvino-nncf-with-output_files/301-tensorflow-training-openvino-nncf-with-output_3_1481.png


Imports
~~~~~~~



The Post Training Quantization API is implemented in the ``nncf``
library.

.. code:: ipython3

    import sys

    import matplotlib.pyplot as plt
    import numpy as np
    import nncf
    from openvino.runtime import Core
    from openvino.runtime import serialize
    from PIL import Image
    from sklearn.metrics import accuracy_score

    sys.path.append("../utils")
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
      batch_size=1
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

    2024-03-26 00:49:08.750442: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [734]
    	 [[{{node Placeholder/_4}}]]
    2024-03-26 00:49:08.750752: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [734]
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

    quantized_model = nncf.quantize(
        ir_model,
        calibration_dataset,
        subset_size=1000
    )



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
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-642/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/live.py", line 32, in run
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    self.live.refresh()
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-642/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/live.py", line 223, in refresh
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    self._live_render.set_renderable(self.renderable)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-642/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/live.py", line 203, in renderable
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    renderable = self.get_renderable()
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-642/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/live.py", line 98, in get_renderable
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    self._get_renderable()
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-642/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1537, in get_renderable
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    renderable = Group(*self.get_renderables())
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-642/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1542, in get_renderables
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    table = self.make_tasks_table(self.tasks)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-642/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1566, in make_tasks_table
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    table.add_row(
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-642/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1571, in &lt;genexpr&gt;
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    else column(task)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-642/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 528, in __call__
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    renderable = self.render(task)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-642/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/nncf/common/logging/track_progress.py", line 58, in render
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    text = super().render(task)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-642/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 787, in render
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    task_time = task.time_remaining
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-642/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1039, in time_remaining
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    estimate = ceil(remaining / speed)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-642/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/tensorflow/python/util/traceback_utils.py", line 153, in error_handler
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    raise e.with_traceback(filtered_tb) from None
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-642/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
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
        options=core.available_devices + ["AUTO"] if not "GPU" in core.available_devices else ["AUTO", "MULTY:CPU,GPU"],
        value='AUTO',
        description='Device:',
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

    Accuracy of the original model: 0.718
    Accuracy of the quantized model: 0.726


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
`OpenVINO API tutorial <002-openvino-api-with-output.html>`__
for more information about running inference with Inference Engine
Python API.

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
    inp_img_url = (
        "https://upload.wikimedia.org/wikipedia/commons/4/48/A_Close_Up_Photo_of_a_Dandelion.jpg"
    )
    directory = "output"
    inp_file_name = "A_Close_Up_Photo_of_a_Dandelion.jpg"
    file_path = Path(directory)/Path(inp_file_name)
    # Download the image if it does not exist yet
    if not Path(inp_file_name).exists():
        download_file(inp_img_url, inp_file_name, directory=directory)

    # Pre-process the image and get it ready for inference.
    input_image = pre_process_image(imagePath=file_path)
    print(f'input image shape: {input_image.shape}')
    print(f'input layer shape: {input_layer.shape}')

    res = quantized_compiled_model([input_image])[output_layer]

    score = tf.nn.softmax(res[0])

    # Show the results
    image = Image.open(file_path)
    plt.imshow(image)
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence.".format(
            class_names[np.argmax(score)], 100 * np.max(score)
        )
    )


.. parsed-literal::

    'output/A_Close_Up_Photo_of_a_Dandelion.jpg' already exists.
    input image shape: (1, 180, 180, 3)
    input layer shape: [1,180,180,3]
    This image most likely belongs to dandelion with a 98.81 percent confidence.



.. image:: 301-tensorflow-training-openvino-nncf-with-output_files/301-tensorflow-training-openvino-nncf-with-output_27_1.png


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
Utils <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/utils/notebook_utils.ipynb>`__.
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
    [ INFO ] Build ................................. 2024.0.0-14509-34caeefd078-releases/2024/0
    [ INFO ]
    [ INFO ]
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(AUTO) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 4.24 ms
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

    [ INFO ] Compile model took 122.88 ms
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
    [ INFO ] First inference took 4.14 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            55908 iterations
    [ INFO ] Duration:         15005.05 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        3.03 ms
    [ INFO ]    Average:       3.03 ms
    [ INFO ]    Min:           2.00 ms
    [ INFO ]    Max:           12.20 ms
    [ INFO ] Throughput:   3725.95 FPS


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
    [ INFO ] Read model took 4.68 ms
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

    [ INFO ] Compile model took 120.45 ms
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
    [ INFO ] First inference took 2.08 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            178896 iterations
    [ INFO ] Duration:         15001.54 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        0.94 ms
    [ INFO ]    Average:       0.97 ms
    [ INFO ]    Min:           0.60 ms
    [ INFO ]    Max:           6.64 ms
    [ INFO ] Throughput:   11925.17 FPS

