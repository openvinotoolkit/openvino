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

    %pip install -q tensorflow Pillow numpy tqdm nncf

    if platform.system() != "Windows":
        %pip install -q "matplotlib>=3.4"
    else:
        %pip install -q "matplotlib>=3.4,<3.7"


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

    2024-03-13 00:59:54.212886: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-03-13 00:59:54.247629: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


.. parsed-literal::

    2024-03-13 00:59:54.839388: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


.. parsed-literal::

    Executing training notebook. This will take a while...


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

    2024-03-13 01:00:00.957194: E tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:266] failed call to cuInit: CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: forward compatibility was attempted on non supported HW
    2024-03-13 01:00:00.957232: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:168] retrieving CUDA diagnostic information for host: iotg-dev-workstation-07
    2024-03-13 01:00:00.957237: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:175] hostname: iotg-dev-workstation-07
    2024-03-13 01:00:00.957362: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:199] libcuda reported version is: 470.223.2
    2024-03-13 01:00:00.957378: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:203] kernel reported version is: 470.182.3
    2024-03-13 01:00:00.957382: E tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:312] kernel version 470.182.3 does not match DSO version 470.223.2 -- cannot find working devices in this configuration


.. parsed-literal::

    Found 3670 files belonging to 5 classes.


.. parsed-literal::

    Using 734 files for validation.
    ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']


.. parsed-literal::

    2024-03-13 01:00:01.273972: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-03-13 01:00:01.274232: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]



.. image:: 301-tensorflow-training-openvino-nncf-with-output_files/301-tensorflow-training-openvino-nncf-with-output_3_12.png


.. parsed-literal::

    2024-03-13 01:00:02.318594: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-03-13 01:00:02.319077: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-03-13 01:00:02.512508: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-03-13 01:00:02.512891: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]


.. parsed-literal::

    (32, 180, 180, 3)
    (32,)


.. parsed-literal::

    0.008872573 0.7322078


.. parsed-literal::

    2024-03-13 01:00:03.177759: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]
    2024-03-13 01:00:03.178061: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]



.. image:: 301-tensorflow-training-openvino-nncf-with-output_files/301-tensorflow-training-openvino-nncf-with-output_3_17.png


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

    2024-03-13 01:00:04.215942: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-03-13 01:00:04.216563: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]


.. parsed-literal::


 1/92 [..............................] - ETA: 1:27 - loss: 1.6034 - accuracy: 0.2812

.. parsed-literal::

    
 2/92 [..............................] - ETA: 6s - loss: 1.8268 - accuracy: 0.2812

.. parsed-literal::

    
 3/92 [..............................] - ETA: 6s - loss: 1.9325 - accuracy: 0.2500

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 1.9389 - accuracy: 0.2422

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 1.8737 - accuracy: 0.2375

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 1.8344 - accuracy: 0.2188

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 5s - loss: 1.7918 - accuracy: 0.2321

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 5s - loss: 1.7671 - accuracy: 0.2383

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 5s - loss: 1.7399 - accuracy: 0.2569

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 1.7224 - accuracy: 0.2562

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 1.7058 - accuracy: 0.2699

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 1.6920 - accuracy: 0.2786

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 1.6738 - accuracy: 0.2933

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 1.6602 - accuracy: 0.2946

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 1.6393 - accuracy: 0.3000

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 1.6261 - accuracy: 0.3008

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 1.6120 - accuracy: 0.3107

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 1.6005 - accuracy: 0.3108

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 1.5823 - accuracy: 0.3174

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 1.5752 - accuracy: 0.3172

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 1.5543 - accuracy: 0.3289

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 1.5440 - accuracy: 0.3338

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 1.5303 - accuracy: 0.3407

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 4s - loss: 1.5142 - accuracy: 0.3500

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 1.4981 - accuracy: 0.3523

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 1.4926 - accuracy: 0.3580

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 1.4827 - accuracy: 0.3586

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 1.4825 - accuracy: 0.3570

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 1.4812 - accuracy: 0.3576

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 1.4728 - accuracy: 0.3561

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 1.4755 - accuracy: 0.3587

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 1.4716 - accuracy: 0.3543

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 1.4667 - accuracy: 0.3569

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 1.4663 - accuracy: 0.3602

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 1.4650 - accuracy: 0.3606

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 1.4604 - accuracy: 0.3601

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 1.4550 - accuracy: 0.3614

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 1.4499 - accuracy: 0.3667

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 1.4472 - accuracy: 0.3685

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 1.4436 - accuracy: 0.3711

.. parsed-literal::

    
41/92 [============>.................] - ETA: 3s - loss: 1.4408 - accuracy: 0.3735

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 1.4339 - accuracy: 0.3757

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 1.4296 - accuracy: 0.3772

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 1.4223 - accuracy: 0.3821

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 1.4189 - accuracy: 0.3855

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 1.4150 - accuracy: 0.3887

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 1.4034 - accuracy: 0.3944

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 1.4024 - accuracy: 0.3946

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 1.4005 - accuracy: 0.3949

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 1.4008 - accuracy: 0.3957

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 1.3970 - accuracy: 0.3959

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 1.3879 - accuracy: 0.4016

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 1.3819 - accuracy: 0.4052

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 1.3761 - accuracy: 0.4081

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 1.3762 - accuracy: 0.4081

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 1.3707 - accuracy: 0.4126

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 1.3653 - accuracy: 0.4163

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 1.3623 - accuracy: 0.4194

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 1.3580 - accuracy: 0.4229

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 1.3568 - accuracy: 0.4231

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 1.3522 - accuracy: 0.4254

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 1.3477 - accuracy: 0.4281

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 1.3447 - accuracy: 0.4298

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 1.3389 - accuracy: 0.4343

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 1.3347 - accuracy: 0.4363

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 1.3310 - accuracy: 0.4377

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 1.3324 - accuracy: 0.4368

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 1.3264 - accuracy: 0.4410

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 1.3251 - accuracy: 0.4409

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 1.3276 - accuracy: 0.4404

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 1.3246 - accuracy: 0.4417

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 1.3187 - accuracy: 0.4425

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 1.3160 - accuracy: 0.4437

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 1.3133 - accuracy: 0.4453

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 1.3088 - accuracy: 0.4473

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 1.3031 - accuracy: 0.4501

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 1.3020 - accuracy: 0.4491

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 1.2994 - accuracy: 0.4494

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 1.2970 - accuracy: 0.4500

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 1.2982 - accuracy: 0.4510

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 1.2962 - accuracy: 0.4512

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 1.2922 - accuracy: 0.4537

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 1.2894 - accuracy: 0.4543

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 1.2885 - accuracy: 0.4549

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 1.2850 - accuracy: 0.4558

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 1.2820 - accuracy: 0.4563

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 1.2782 - accuracy: 0.4593

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 1.2757 - accuracy: 0.4605

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 1.2736 - accuracy: 0.4613

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 1.2737 - accuracy: 0.4614

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 1.2724 - accuracy: 0.4621

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 1.2689 - accuracy: 0.4642

.. parsed-literal::

    2024-03-13 01:00:10.495383: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [734]
    	 [[{{node Placeholder/_0}}]]
    2024-03-13 01:00:10.495657: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [734]
    	 [[{{node Placeholder/_4}}]]


.. parsed-literal::

    
92/92 [==============================] - 7s 66ms/step - loss: 1.2689 - accuracy: 0.4642 - val_loss: 0.9877 - val_accuracy: 0.5954


.. parsed-literal::

    Epoch 2/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.9592 - accuracy: 0.5312

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.9494 - accuracy: 0.5312

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.9880 - accuracy: 0.5521

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 1.0088 - accuracy: 0.5391

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 1.0084 - accuracy: 0.5500

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.9770 - accuracy: 0.5833

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 5s - loss: 1.0209 - accuracy: 0.5670

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 1.0146 - accuracy: 0.5781

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.9849 - accuracy: 0.5903

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.9816 - accuracy: 0.6031

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.9651 - accuracy: 0.6080

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.9724 - accuracy: 0.5990

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.9674 - accuracy: 0.6034

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.9582 - accuracy: 0.6027

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.9618 - accuracy: 0.5979

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.9806 - accuracy: 0.5898

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.9973 - accuracy: 0.5882

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.9917 - accuracy: 0.5903

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.9847 - accuracy: 0.5938

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.9966 - accuracy: 0.5938

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.9942 - accuracy: 0.5908

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 1.0058 - accuracy: 0.5923

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 1.0056 - accuracy: 0.5938

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.9982 - accuracy: 0.5990

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.9984 - accuracy: 0.6037

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 1.0006 - accuracy: 0.6022

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.9982 - accuracy: 0.6030

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 1.0047 - accuracy: 0.5982

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 1.0096 - accuracy: 0.5938

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 1.0082 - accuracy: 0.5927

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 1.0093 - accuracy: 0.5938

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 1.0069 - accuracy: 0.5957

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 1.0024 - accuracy: 0.5994

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 1.0045 - accuracy: 0.5983

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 1.0002 - accuracy: 0.6000

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 1.0049 - accuracy: 0.5964

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 1.0047 - accuracy: 0.5997

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 1.0040 - accuracy: 0.5995

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 1.0111 - accuracy: 0.5962

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 1.0087 - accuracy: 0.5992

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 1.0157 - accuracy: 0.5960

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 1.0148 - accuracy: 0.5945

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 1.0224 - accuracy: 0.5908

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 1.0306 - accuracy: 0.5852

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 1.0325 - accuracy: 0.5896

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 1.0310 - accuracy: 0.5938

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 1.0277 - accuracy: 0.5957

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 1.0227 - accuracy: 0.5983

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 1.0222 - accuracy: 0.6001

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 1.0194 - accuracy: 0.6000

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 1.0189 - accuracy: 0.5993

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 1.0204 - accuracy: 0.5986

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 1.0189 - accuracy: 0.5985

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 1.0253 - accuracy: 0.5961

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 1.0253 - accuracy: 0.5972

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 1.0253 - accuracy: 0.5982

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 1.0231 - accuracy: 0.5998

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 1.0205 - accuracy: 0.6008

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 1.0206 - accuracy: 0.6022

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 1.0246 - accuracy: 0.6005

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 1.0248 - accuracy: 0.5999

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 1.0223 - accuracy: 0.6018

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 1.0188 - accuracy: 0.6047

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 1.0164 - accuracy: 0.6060

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 1.0137 - accuracy: 0.6062

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 1.0164 - accuracy: 0.6037

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 1.0162 - accuracy: 0.6054

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 1.0145 - accuracy: 0.6066

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 1.0131 - accuracy: 0.6073

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 1.0129 - accuracy: 0.6076

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 1.0136 - accuracy: 0.6080

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 1.0118 - accuracy: 0.6087

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 1.0102 - accuracy: 0.6085

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 1.0104 - accuracy: 0.6079

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 1.0093 - accuracy: 0.6077

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 1.0111 - accuracy: 0.6067

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 1.0106 - accuracy: 0.6069

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 1.0101 - accuracy: 0.6083

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 1.0098 - accuracy: 0.6089

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 1.0103 - accuracy: 0.6080

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 1.0114 - accuracy: 0.6074

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 1.0102 - accuracy: 0.6080

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 1.0084 - accuracy: 0.6090

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 1.0105 - accuracy: 0.6077

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 1.0073 - accuracy: 0.6086

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 1.0080 - accuracy: 0.6077

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 1.0059 - accuracy: 0.6097

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 1.0064 - accuracy: 0.6088

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 1.0058 - accuracy: 0.6090

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 1.0035 - accuracy: 0.6102

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 1.0031 - accuracy: 0.6107

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 1.0031 - accuracy: 0.6107 - val_loss: 0.9459 - val_accuracy: 0.6362


.. parsed-literal::

    Epoch 3/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.9821 - accuracy: 0.6250

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.9723 - accuracy: 0.6562

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.8494 - accuracy: 0.7083

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.8594 - accuracy: 0.7109

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 4s - loss: 0.8248 - accuracy: 0.7188

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.7995 - accuracy: 0.7240

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.7832 - accuracy: 0.7277

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.7810 - accuracy: 0.7266

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.7997 - accuracy: 0.7188

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.8188 - accuracy: 0.7125

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.8115 - accuracy: 0.7159

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.8429 - accuracy: 0.7057

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.8557 - accuracy: 0.6947

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.8773 - accuracy: 0.6920

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.8872 - accuracy: 0.6896

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.8785 - accuracy: 0.6895

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.8851 - accuracy: 0.6857

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.8821 - accuracy: 0.6892

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.8844 - accuracy: 0.6842

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.8843 - accuracy: 0.6828

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.8778 - accuracy: 0.6845

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.8890 - accuracy: 0.6804

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.8845 - accuracy: 0.6834

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.8785 - accuracy: 0.6862

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.8871 - accuracy: 0.6808

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.8890 - accuracy: 0.6764

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.8872 - accuracy: 0.6757

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.8884 - accuracy: 0.6750

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.8841 - accuracy: 0.6765

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.8817 - accuracy: 0.6758

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.8775 - accuracy: 0.6742

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.8771 - accuracy: 0.6737

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.8715 - accuracy: 0.6750

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.8819 - accuracy: 0.6718

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.9015 - accuracy: 0.6670

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.9032 - accuracy: 0.6641

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.9129 - accuracy: 0.6581

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.9161 - accuracy: 0.6548

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.9155 - accuracy: 0.6557

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.9134 - accuracy: 0.6572

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.9135 - accuracy: 0.6564

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.9153 - accuracy: 0.6557

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.9133 - accuracy: 0.6564

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.9115 - accuracy: 0.6557

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.9087 - accuracy: 0.6578

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.9028 - accuracy: 0.6604

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.9003 - accuracy: 0.6616

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.8983 - accuracy: 0.6622

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.8977 - accuracy: 0.6602

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.8962 - accuracy: 0.6613

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.8955 - accuracy: 0.6612

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.8943 - accuracy: 0.6605

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.8947 - accuracy: 0.6593

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.8937 - accuracy: 0.6610

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.8920 - accuracy: 0.6620

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.8889 - accuracy: 0.6630

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.8896 - accuracy: 0.6613

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.8939 - accuracy: 0.6612

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.8903 - accuracy: 0.6627

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.8897 - accuracy: 0.6620

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.8934 - accuracy: 0.6599

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.8916 - accuracy: 0.6609

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.8933 - accuracy: 0.6598

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.8913 - accuracy: 0.6612

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.8902 - accuracy: 0.6621

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.8875 - accuracy: 0.6620

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.8855 - accuracy: 0.6624

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.8892 - accuracy: 0.6605

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.8907 - accuracy: 0.6591

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.8893 - accuracy: 0.6608

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.8888 - accuracy: 0.6616

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.8861 - accuracy: 0.6619

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.8853 - accuracy: 0.6610

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.8887 - accuracy: 0.6610

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.8943 - accuracy: 0.6584

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.8933 - accuracy: 0.6588

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.8936 - accuracy: 0.6592

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.8903 - accuracy: 0.6603

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.8899 - accuracy: 0.6587

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.8911 - accuracy: 0.6587

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.8899 - accuracy: 0.6590

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.8901 - accuracy: 0.6597

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.8904 - accuracy: 0.6586

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.8881 - accuracy: 0.6593

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.8884 - accuracy: 0.6589

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.8889 - accuracy: 0.6585

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.8901 - accuracy: 0.6578

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.8902 - accuracy: 0.6567

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.8907 - accuracy: 0.6563

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.8901 - accuracy: 0.6567

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.8904 - accuracy: 0.6567

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.8904 - accuracy: 0.6567 - val_loss: 0.8648 - val_accuracy: 0.6444


.. parsed-literal::

    Epoch 4/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.9896 - accuracy: 0.5938

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.9483 - accuracy: 0.6250

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.8749 - accuracy: 0.6667

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.8280 - accuracy: 0.6953

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.8490 - accuracy: 0.6875

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.9025 - accuracy: 0.6719

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.8749 - accuracy: 0.6741

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.8837 - accuracy: 0.6719

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.8878 - accuracy: 0.6701

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.8915 - accuracy: 0.6687

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.8794 - accuracy: 0.6705

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.8726 - accuracy: 0.6641

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.8571 - accuracy: 0.6659

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.8522 - accuracy: 0.6696

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.8456 - accuracy: 0.6687

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.8426 - accuracy: 0.6699

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.8576 - accuracy: 0.6562

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.8540 - accuracy: 0.6580

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.8630 - accuracy: 0.6595

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.8560 - accuracy: 0.6609

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.8498 - accuracy: 0.6622

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.8452 - accuracy: 0.6648

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.8392 - accuracy: 0.6671

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.8392 - accuracy: 0.6667

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.8538 - accuracy: 0.6625

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.8538 - accuracy: 0.6600

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.8436 - accuracy: 0.6655

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.8437 - accuracy: 0.6674

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.8434 - accuracy: 0.6670

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.8447 - accuracy: 0.6697

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.8511 - accuracy: 0.6693

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.8546 - accuracy: 0.6689

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.8529 - accuracy: 0.6694

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.8472 - accuracy: 0.6727

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.8461 - accuracy: 0.6740

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.8527 - accuracy: 0.6726

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.8548 - accuracy: 0.6714

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.8513 - accuracy: 0.6742

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.8520 - accuracy: 0.6737

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.8454 - accuracy: 0.6756

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.8508 - accuracy: 0.6737

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.8491 - accuracy: 0.6740

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.8516 - accuracy: 0.6736

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.8572 - accuracy: 0.6711

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.8577 - accuracy: 0.6708

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.8573 - accuracy: 0.6718

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.8560 - accuracy: 0.6734

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.8557 - accuracy: 0.6744

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.8599 - accuracy: 0.6734

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.8599 - accuracy: 0.6730

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.8576 - accuracy: 0.6733

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.8552 - accuracy: 0.6736

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.8510 - accuracy: 0.6738

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.8471 - accuracy: 0.6752

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.8438 - accuracy: 0.6771

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.8413 - accuracy: 0.6779

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.8420 - accuracy: 0.6780

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.8412 - accuracy: 0.6787

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.8412 - accuracy: 0.6799

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.8434 - accuracy: 0.6800

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.8447 - accuracy: 0.6797

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.8454 - accuracy: 0.6783

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.8441 - accuracy: 0.6799

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.8414 - accuracy: 0.6810

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.8409 - accuracy: 0.6792

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.8392 - accuracy: 0.6793

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.8382 - accuracy: 0.6799

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.8391 - accuracy: 0.6786

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.8388 - accuracy: 0.6788

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.8393 - accuracy: 0.6776

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.8384 - accuracy: 0.6781

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.8382 - accuracy: 0.6796

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.8368 - accuracy: 0.6797

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.8399 - accuracy: 0.6789

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.8411 - accuracy: 0.6782

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.8425 - accuracy: 0.6771

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.8402 - accuracy: 0.6785

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.8427 - accuracy: 0.6786

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.8431 - accuracy: 0.6787

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.8441 - accuracy: 0.6776

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.8435 - accuracy: 0.6778

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.8424 - accuracy: 0.6775

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.8411 - accuracy: 0.6784

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.8421 - accuracy: 0.6781

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.8446 - accuracy: 0.6778

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.8441 - accuracy: 0.6790

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.8434 - accuracy: 0.6795

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.8418 - accuracy: 0.6796

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.8403 - accuracy: 0.6804

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.8420 - accuracy: 0.6787

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.8425 - accuracy: 0.6788

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.8425 - accuracy: 0.6788 - val_loss: 0.7927 - val_accuracy: 0.6948


.. parsed-literal::

    Epoch 5/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.6404 - accuracy: 0.7812

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.6825 - accuracy: 0.7500

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6366 - accuracy: 0.7708

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.7350 - accuracy: 0.7344

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.7050 - accuracy: 0.7500

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.7362 - accuracy: 0.7448

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.7464 - accuracy: 0.7455

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.7201 - accuracy: 0.7617

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.7403 - accuracy: 0.7535

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.7423 - accuracy: 0.7531

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.7348 - accuracy: 0.7614

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.7754 - accuracy: 0.7344

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.7602 - accuracy: 0.7404

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.7555 - accuracy: 0.7411

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.7509 - accuracy: 0.7417

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.7544 - accuracy: 0.7363

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.7430 - accuracy: 0.7390

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.7442 - accuracy: 0.7378

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.7523 - accuracy: 0.7303

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.7570 - accuracy: 0.7266

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.7640 - accuracy: 0.7232

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.7590 - accuracy: 0.7216

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.7601 - accuracy: 0.7224

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.7649 - accuracy: 0.7172

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.7715 - accuracy: 0.7184

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.7758 - accuracy: 0.7196

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.7727 - accuracy: 0.7185

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.7745 - accuracy: 0.7152

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.7760 - accuracy: 0.7143

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.7723 - accuracy: 0.7154

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.7750 - accuracy: 0.7146

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.7828 - accuracy: 0.7109

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.7795 - accuracy: 0.7111

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.7856 - accuracy: 0.7086

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.7864 - accuracy: 0.7063

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.7845 - accuracy: 0.7058

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.7867 - accuracy: 0.7036

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.7945 - accuracy: 0.6984

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.7895 - accuracy: 0.7028

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.7913 - accuracy: 0.7009

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.7863 - accuracy: 0.7043

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.7840 - accuracy: 0.7032

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.7820 - accuracy: 0.7057

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.7812 - accuracy: 0.7053

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.7850 - accuracy: 0.7042

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.7802 - accuracy: 0.7052

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.7799 - accuracy: 0.7068

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.7815 - accuracy: 0.7071

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.7813 - accuracy: 0.7073

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.7782 - accuracy: 0.7100

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.7786 - accuracy: 0.7077

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.7769 - accuracy: 0.7097

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.7835 - accuracy: 0.7070

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.7795 - accuracy: 0.7072

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.7789 - accuracy: 0.7080

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.7770 - accuracy: 0.7081

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.7789 - accuracy: 0.7056

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.7809 - accuracy: 0.7059

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.7800 - accuracy: 0.7071

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.7785 - accuracy: 0.7088

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.7782 - accuracy: 0.7085

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.7784 - accuracy: 0.7082

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.7779 - accuracy: 0.7083

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.7796 - accuracy: 0.7090

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.7810 - accuracy: 0.7067

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.7796 - accuracy: 0.7069

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.7797 - accuracy: 0.7085

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.7767 - accuracy: 0.7100

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.7788 - accuracy: 0.7101

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.7799 - accuracy: 0.7107

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.7791 - accuracy: 0.7112

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.7787 - accuracy: 0.7118

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.7769 - accuracy: 0.7123

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.7788 - accuracy: 0.7115

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.7791 - accuracy: 0.7116

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.7799 - accuracy: 0.7109

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.7783 - accuracy: 0.7118

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.7791 - accuracy: 0.7115

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.7828 - accuracy: 0.7096

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.7833 - accuracy: 0.7094

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.7866 - accuracy: 0.7068

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.7864 - accuracy: 0.7069

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.7894 - accuracy: 0.7052

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.7870 - accuracy: 0.7061

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.7859 - accuracy: 0.7070

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.7847 - accuracy: 0.7075

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.7860 - accuracy: 0.7062

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.7849 - accuracy: 0.7067

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.7823 - accuracy: 0.7072

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.7810 - accuracy: 0.7073

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.7821 - accuracy: 0.7074

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.7821 - accuracy: 0.7074 - val_loss: 0.7956 - val_accuracy: 0.6907


.. parsed-literal::

    Epoch 6/15


.. parsed-literal::


 1/92 [..............................] - ETA: 6s - loss: 0.5259 - accuracy: 0.8125

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.6810 - accuracy: 0.7344

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6532 - accuracy: 0.7708

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6841 - accuracy: 0.7578

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.7076 - accuracy: 0.7437

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.7284 - accuracy: 0.7292

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.7835 - accuracy: 0.6964

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.7509 - accuracy: 0.7109

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.7550 - accuracy: 0.7188

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.7711 - accuracy: 0.7063

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.7442 - accuracy: 0.7216

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.7518 - accuracy: 0.7292

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.7561 - accuracy: 0.7236

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.7499 - accuracy: 0.7246

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.7461 - accuracy: 0.7282

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.7449 - accuracy: 0.7276

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.7416 - accuracy: 0.7236

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.7333 - accuracy: 0.7283

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.7369 - accuracy: 0.7263

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.7392 - accuracy: 0.7229

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.7365 - accuracy: 0.7227

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.7411 - accuracy: 0.7225

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.7425 - accuracy: 0.7211

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.7413 - accuracy: 0.7222

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.7361 - accuracy: 0.7233

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.7333 - accuracy: 0.7231

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.7304 - accuracy: 0.7230

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.7243 - accuracy: 0.7250

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.7290 - accuracy: 0.7237

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.7319 - accuracy: 0.7236

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.7240 - accuracy: 0.7274

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.7212 - accuracy: 0.7290

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.7181 - accuracy: 0.7287

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.7166 - accuracy: 0.7302

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.7104 - accuracy: 0.7343

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.7107 - accuracy: 0.7355

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.7034 - accuracy: 0.7392

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.7021 - accuracy: 0.7395

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.7043 - accuracy: 0.7382

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.7039 - accuracy: 0.7362

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.7067 - accuracy: 0.7373

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.7059 - accuracy: 0.7376

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.7084 - accuracy: 0.7371

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.7206 - accuracy: 0.7311

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.7171 - accuracy: 0.7322

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.7173 - accuracy: 0.7320

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.7160 - accuracy: 0.7330

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.7152 - accuracy: 0.7333

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.7160 - accuracy: 0.7337

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.7235 - accuracy: 0.7303

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.7200 - accuracy: 0.7331

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.7276 - accuracy: 0.7316

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.7316 - accuracy: 0.7297

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.7338 - accuracy: 0.7277

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.7332 - accuracy: 0.7270

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.7309 - accuracy: 0.7285

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.7302 - accuracy: 0.7284

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.7292 - accuracy: 0.7303

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.7316 - accuracy: 0.7296

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.7343 - accuracy: 0.7269

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.7400 - accuracy: 0.7247

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.7407 - accuracy: 0.7236

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.7397 - accuracy: 0.7230

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.7405 - accuracy: 0.7220

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.7435 - accuracy: 0.7196

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.7468 - accuracy: 0.7186

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.7461 - accuracy: 0.7182

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.7459 - accuracy: 0.7182

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.7456 - accuracy: 0.7177

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.7468 - accuracy: 0.7164

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.7508 - accuracy: 0.7147

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.7517 - accuracy: 0.7143

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.7538 - accuracy: 0.7131

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.7568 - accuracy: 0.7111

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.7555 - accuracy: 0.7120

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.7541 - accuracy: 0.7134

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.7545 - accuracy: 0.7134

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.7556 - accuracy: 0.7127

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.7560 - accuracy: 0.7132

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.7551 - accuracy: 0.7148

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.7565 - accuracy: 0.7129

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.7572 - accuracy: 0.7130

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.7562 - accuracy: 0.7131

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.7581 - accuracy: 0.7120

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.7602 - accuracy: 0.7114

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.7630 - accuracy: 0.7097

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.7602 - accuracy: 0.7112

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.7615 - accuracy: 0.7113

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.7615 - accuracy: 0.7114

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.7614 - accuracy: 0.7125

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.7639 - accuracy: 0.7112

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.7639 - accuracy: 0.7112 - val_loss: 0.7952 - val_accuracy: 0.6744


.. parsed-literal::

    Epoch 7/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.9636 - accuracy: 0.6250

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.9083 - accuracy: 0.6406

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.7801 - accuracy: 0.6979

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.7920 - accuracy: 0.6875

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.7895 - accuracy: 0.6875

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.7849 - accuracy: 0.6927

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.8100 - accuracy: 0.6964

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.7832 - accuracy: 0.7109

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.7856 - accuracy: 0.7118

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.7685 - accuracy: 0.7188

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.7446 - accuracy: 0.7301

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.7175 - accuracy: 0.7422

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.7149 - accuracy: 0.7452

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.7050 - accuracy: 0.7478

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.7073 - accuracy: 0.7479

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.7272 - accuracy: 0.7363

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.7160 - accuracy: 0.7371

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.7210 - accuracy: 0.7309

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.7214 - accuracy: 0.7352

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.7259 - accuracy: 0.7312

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.7191 - accuracy: 0.7321

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.7161 - accuracy: 0.7301

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.7124 - accuracy: 0.7310

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.7126 - accuracy: 0.7279

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.7065 - accuracy: 0.7325

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.7084 - accuracy: 0.7308

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.7033 - accuracy: 0.7315

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.7063 - accuracy: 0.7277

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.7023 - accuracy: 0.7274

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6984 - accuracy: 0.7292

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6987 - accuracy: 0.7278

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.7093 - accuracy: 0.7227

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.7059 - accuracy: 0.7254

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6989 - accuracy: 0.7298

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6994 - accuracy: 0.7312

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.7021 - accuracy: 0.7309

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.7004 - accuracy: 0.7314

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.7004 - accuracy: 0.7311

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.7057 - accuracy: 0.7276

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.7024 - accuracy: 0.7297

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.7071 - accuracy: 0.7271

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.7082 - accuracy: 0.7269

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.7072 - accuracy: 0.7282

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.7054 - accuracy: 0.7287

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.7071 - accuracy: 0.7299

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.7043 - accuracy: 0.7317

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.7034 - accuracy: 0.7320

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6984 - accuracy: 0.7331

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.7049 - accuracy: 0.7309

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.7024 - accuracy: 0.7319

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.7011 - accuracy: 0.7341

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.7025 - accuracy: 0.7350

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.7032 - accuracy: 0.7358

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.7029 - accuracy: 0.7367

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.7054 - accuracy: 0.7341

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.7026 - accuracy: 0.7349

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.7042 - accuracy: 0.7341

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.7071 - accuracy: 0.7311

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.7123 - accuracy: 0.7288

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.7103 - accuracy: 0.7292

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.7111 - accuracy: 0.7300

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.7075 - accuracy: 0.7314

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.7080 - accuracy: 0.7302

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.7108 - accuracy: 0.7285

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.7126 - accuracy: 0.7269

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.7159 - accuracy: 0.7273

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.7148 - accuracy: 0.7267

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.7123 - accuracy: 0.7275

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.7158 - accuracy: 0.7255

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.7161 - accuracy: 0.7254

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.7149 - accuracy: 0.7252

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.7120 - accuracy: 0.7259

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.7105 - accuracy: 0.7267

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.7126 - accuracy: 0.7258

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.7162 - accuracy: 0.7236

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.7144 - accuracy: 0.7239

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.7134 - accuracy: 0.7251

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.7143 - accuracy: 0.7250

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.7147 - accuracy: 0.7249

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.7174 - accuracy: 0.7248

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.7177 - accuracy: 0.7244

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.7171 - accuracy: 0.7247

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.7166 - accuracy: 0.7243

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.7137 - accuracy: 0.7257

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.7127 - accuracy: 0.7252

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.7127 - accuracy: 0.7251

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.7157 - accuracy: 0.7247

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.7150 - accuracy: 0.7257

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.7155 - accuracy: 0.7246

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.7126 - accuracy: 0.7252

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.7125 - accuracy: 0.7251

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.7125 - accuracy: 0.7251 - val_loss: 0.7162 - val_accuracy: 0.7248


.. parsed-literal::

    Epoch 8/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.5581 - accuracy: 0.8438

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.6446 - accuracy: 0.7656

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.7033 - accuracy: 0.7396

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.7630 - accuracy: 0.7109

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.7291 - accuracy: 0.7250

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.7279 - accuracy: 0.7240

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.7209 - accuracy: 0.7232

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6963 - accuracy: 0.7266

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6854 - accuracy: 0.7396

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6670 - accuracy: 0.7500

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6627 - accuracy: 0.7528

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6783 - accuracy: 0.7474

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6823 - accuracy: 0.7404

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6710 - accuracy: 0.7388

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6574 - accuracy: 0.7437

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6521 - accuracy: 0.7441

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6599 - accuracy: 0.7445

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6677 - accuracy: 0.7396

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6584 - accuracy: 0.7434

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6603 - accuracy: 0.7422

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6654 - accuracy: 0.7396

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6753 - accuracy: 0.7344

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.6743 - accuracy: 0.7364

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6789 - accuracy: 0.7331

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6771 - accuracy: 0.7350

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6825 - accuracy: 0.7308

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6790 - accuracy: 0.7350

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6770 - accuracy: 0.7355

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6779 - accuracy: 0.7317

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6757 - accuracy: 0.7354

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6785 - accuracy: 0.7339

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6766 - accuracy: 0.7344

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6734 - accuracy: 0.7367

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6699 - accuracy: 0.7399

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6635 - accuracy: 0.7429

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6601 - accuracy: 0.7439

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6605 - accuracy: 0.7432

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6638 - accuracy: 0.7410

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6585 - accuracy: 0.7444

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.6588 - accuracy: 0.7437

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6618 - accuracy: 0.7447

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6625 - accuracy: 0.7440

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6595 - accuracy: 0.7456

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6654 - accuracy: 0.7429

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6737 - accuracy: 0.7375

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6708 - accuracy: 0.7378

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6708 - accuracy: 0.7387

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6719 - accuracy: 0.7396

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6694 - accuracy: 0.7417

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6721 - accuracy: 0.7377

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6690 - accuracy: 0.7391

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6640 - accuracy: 0.7417

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6672 - accuracy: 0.7401

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6716 - accuracy: 0.7386

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6668 - accuracy: 0.7416

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6656 - accuracy: 0.7417

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6651 - accuracy: 0.7419

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6625 - accuracy: 0.7436

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6657 - accuracy: 0.7432

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6652 - accuracy: 0.7428

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6649 - accuracy: 0.7424

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6657 - accuracy: 0.7405

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6674 - accuracy: 0.7397

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6649 - accuracy: 0.7413

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6624 - accuracy: 0.7433

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6634 - accuracy: 0.7439

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6614 - accuracy: 0.7454

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6606 - accuracy: 0.7468

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6631 - accuracy: 0.7451

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6641 - accuracy: 0.7443

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6667 - accuracy: 0.7439

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6679 - accuracy: 0.7436

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6648 - accuracy: 0.7441

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6643 - accuracy: 0.7433

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6630 - accuracy: 0.7434

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6650 - accuracy: 0.7423

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6686 - accuracy: 0.7412

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6679 - accuracy: 0.7417

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6681 - accuracy: 0.7426

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6656 - accuracy: 0.7430

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6639 - accuracy: 0.7435

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6623 - accuracy: 0.7447

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6632 - accuracy: 0.7451

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6643 - accuracy: 0.7448

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6628 - accuracy: 0.7449

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6647 - accuracy: 0.7446

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6657 - accuracy: 0.7439

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6687 - accuracy: 0.7430

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6678 - accuracy: 0.7444

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6687 - accuracy: 0.7435

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6669 - accuracy: 0.7442

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.6669 - accuracy: 0.7442 - val_loss: 0.7692 - val_accuracy: 0.6771


.. parsed-literal::

    Epoch 9/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.4666 - accuracy: 0.8750

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5074 - accuracy: 0.8125

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6037 - accuracy: 0.7292

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6106 - accuracy: 0.7344

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.6262 - accuracy: 0.7250

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.6256 - accuracy: 0.7448

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6184 - accuracy: 0.7500

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6006 - accuracy: 0.7500

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5930 - accuracy: 0.7500

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5971 - accuracy: 0.7469

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6612 - accuracy: 0.7188

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6702 - accuracy: 0.7109

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6750 - accuracy: 0.7115

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6804 - accuracy: 0.7121

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6854 - accuracy: 0.7208

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6925 - accuracy: 0.7168

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6867 - accuracy: 0.7224

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6865 - accuracy: 0.7257

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6822 - accuracy: 0.7286

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6723 - accuracy: 0.7297

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6809 - accuracy: 0.7292

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6765 - accuracy: 0.7287

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.6683 - accuracy: 0.7310

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6606 - accuracy: 0.7344

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6633 - accuracy: 0.7325

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6632 - accuracy: 0.7332

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6571 - accuracy: 0.7361

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6496 - accuracy: 0.7402

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6536 - accuracy: 0.7395

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6551 - accuracy: 0.7409

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6531 - accuracy: 0.7411

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6463 - accuracy: 0.7433

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6405 - accuracy: 0.7444

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6514 - accuracy: 0.7401

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6462 - accuracy: 0.7430

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6519 - accuracy: 0.7389

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6589 - accuracy: 0.7359

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6558 - accuracy: 0.7379

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.6556 - accuracy: 0.7382

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6555 - accuracy: 0.7377

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6549 - accuracy: 0.7403

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6603 - accuracy: 0.7368

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6559 - accuracy: 0.7393

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6576 - accuracy: 0.7374

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6604 - accuracy: 0.7343

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6591 - accuracy: 0.7340

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6579 - accuracy: 0.7356

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6563 - accuracy: 0.7372

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6519 - accuracy: 0.7393

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6505 - accuracy: 0.7395

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6492 - accuracy: 0.7409

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6489 - accuracy: 0.7423

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6449 - accuracy: 0.7453

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6403 - accuracy: 0.7477

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6427 - accuracy: 0.7472

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6399 - accuracy: 0.7483

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6413 - accuracy: 0.7478

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6419 - accuracy: 0.7479

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6413 - accuracy: 0.7495

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6396 - accuracy: 0.7510

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6437 - accuracy: 0.7500

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6450 - accuracy: 0.7485

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6502 - accuracy: 0.7461

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6505 - accuracy: 0.7466

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6498 - accuracy: 0.7471

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6504 - accuracy: 0.7467

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6483 - accuracy: 0.7482

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6474 - accuracy: 0.7482

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6461 - accuracy: 0.7496

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6461 - accuracy: 0.7504

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6436 - accuracy: 0.7513

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6450 - accuracy: 0.7500

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6449 - accuracy: 0.7504

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6429 - accuracy: 0.7513

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6423 - accuracy: 0.7517

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6398 - accuracy: 0.7537

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6399 - accuracy: 0.7528

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6387 - accuracy: 0.7532

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6405 - accuracy: 0.7512

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6428 - accuracy: 0.7508

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6419 - accuracy: 0.7511

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6398 - accuracy: 0.7515

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6395 - accuracy: 0.7519

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6384 - accuracy: 0.7522

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6401 - accuracy: 0.7515

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6416 - accuracy: 0.7511

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6405 - accuracy: 0.7518

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6423 - accuracy: 0.7511

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6423 - accuracy: 0.7514

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6411 - accuracy: 0.7528

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6401 - accuracy: 0.7531

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.6401 - accuracy: 0.7531 - val_loss: 0.7722 - val_accuracy: 0.6880


.. parsed-literal::

    Epoch 10/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.5518 - accuracy: 0.8125

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.6186 - accuracy: 0.7812

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6593 - accuracy: 0.7708

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6392 - accuracy: 0.7812

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.6786 - accuracy: 0.7500

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.6991 - accuracy: 0.7448

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6717 - accuracy: 0.7500

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6444 - accuracy: 0.7617

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6269 - accuracy: 0.7674

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6254 - accuracy: 0.7750

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6196 - accuracy: 0.7727

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6111 - accuracy: 0.7734

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6013 - accuracy: 0.7788

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6024 - accuracy: 0.7790

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6018 - accuracy: 0.7738

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5923 - accuracy: 0.7668

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5973 - accuracy: 0.7641

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5998 - accuracy: 0.7600

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6019 - accuracy: 0.7579

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6015 - accuracy: 0.7575

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5965 - accuracy: 0.7615

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.6093 - accuracy: 0.7596

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6157 - accuracy: 0.7566

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6124 - accuracy: 0.7588

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6075 - accuracy: 0.7609

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6021 - accuracy: 0.7629

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5999 - accuracy: 0.7669

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5975 - accuracy: 0.7674

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6022 - accuracy: 0.7637

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5988 - accuracy: 0.7663

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6036 - accuracy: 0.7628

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6038 - accuracy: 0.7615

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6136 - accuracy: 0.7583

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6135 - accuracy: 0.7581

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6176 - accuracy: 0.7570

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6131 - accuracy: 0.7585

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6211 - accuracy: 0.7550

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6234 - accuracy: 0.7556

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.6247 - accuracy: 0.7563

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6280 - accuracy: 0.7554

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6276 - accuracy: 0.7552

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6268 - accuracy: 0.7566

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6207 - accuracy: 0.7600

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6198 - accuracy: 0.7605

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6225 - accuracy: 0.7602

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6208 - accuracy: 0.7614

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6251 - accuracy: 0.7592

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6247 - accuracy: 0.7583

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6270 - accuracy: 0.7588

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6252 - accuracy: 0.7605

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6217 - accuracy: 0.7615

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6216 - accuracy: 0.7624

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6231 - accuracy: 0.7605

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6212 - accuracy: 0.7608

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6195 - accuracy: 0.7629

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6158 - accuracy: 0.7660

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6193 - accuracy: 0.7646

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6177 - accuracy: 0.7638

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6144 - accuracy: 0.7652

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6103 - accuracy: 0.7665

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6104 - accuracy: 0.7657

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6073 - accuracy: 0.7669

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6080 - accuracy: 0.7662

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6158 - accuracy: 0.7635

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6139 - accuracy: 0.7643

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6130 - accuracy: 0.7645

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6111 - accuracy: 0.7648

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6086 - accuracy: 0.7659

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6088 - accuracy: 0.7661

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6090 - accuracy: 0.7663

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6109 - accuracy: 0.7657

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6122 - accuracy: 0.7663

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6128 - accuracy: 0.7665

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6121 - accuracy: 0.7676

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6122 - accuracy: 0.7682

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6122 - accuracy: 0.7671

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6123 - accuracy: 0.7665

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6144 - accuracy: 0.7655

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6142 - accuracy: 0.7653

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6144 - accuracy: 0.7647

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6145 - accuracy: 0.7638

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6156 - accuracy: 0.7636

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6160 - accuracy: 0.7634

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6180 - accuracy: 0.7633

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6182 - accuracy: 0.7631

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6199 - accuracy: 0.7622

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6175 - accuracy: 0.7632

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6170 - accuracy: 0.7634

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6203 - accuracy: 0.7611

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6202 - accuracy: 0.7614

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6217 - accuracy: 0.7606

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.6217 - accuracy: 0.7606 - val_loss: 0.7700 - val_accuracy: 0.7071


.. parsed-literal::

    Epoch 11/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.4981 - accuracy: 0.8125

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5819 - accuracy: 0.7656

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5281 - accuracy: 0.8125

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5673 - accuracy: 0.7812

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.6359 - accuracy: 0.7375

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.5973 - accuracy: 0.7656

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.5857 - accuracy: 0.7723

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5830 - accuracy: 0.7734

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5898 - accuracy: 0.7708

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5826 - accuracy: 0.7750

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5843 - accuracy: 0.7784

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5888 - accuracy: 0.7760

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5808 - accuracy: 0.7812

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5672 - accuracy: 0.7879

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5874 - accuracy: 0.7833

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5875 - accuracy: 0.7832

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5837 - accuracy: 0.7831

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5737 - accuracy: 0.7865

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5835 - accuracy: 0.7812

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5782 - accuracy: 0.7828

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5747 - accuracy: 0.7857

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5807 - accuracy: 0.7812

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.5858 - accuracy: 0.7799

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5895 - accuracy: 0.7773

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5919 - accuracy: 0.7750

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5958 - accuracy: 0.7728

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5871 - accuracy: 0.7778

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5821 - accuracy: 0.7801

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5746 - accuracy: 0.7823

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5767 - accuracy: 0.7802

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5771 - accuracy: 0.7792

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5796 - accuracy: 0.7773

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5823 - accuracy: 0.7746

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5822 - accuracy: 0.7748

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5802 - accuracy: 0.7759

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5800 - accuracy: 0.7769

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5799 - accuracy: 0.7779

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5802 - accuracy: 0.7771

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5765 - accuracy: 0.7788

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.5741 - accuracy: 0.7805

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5748 - accuracy: 0.7812

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5753 - accuracy: 0.7805

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5734 - accuracy: 0.7820

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5732 - accuracy: 0.7834

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5702 - accuracy: 0.7833

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5729 - accuracy: 0.7833

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5711 - accuracy: 0.7839

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5693 - accuracy: 0.7852

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5701 - accuracy: 0.7844

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5679 - accuracy: 0.7850

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5703 - accuracy: 0.7843

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5724 - accuracy: 0.7849

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5751 - accuracy: 0.7830

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5767 - accuracy: 0.7818

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5760 - accuracy: 0.7830

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5725 - accuracy: 0.7840

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5712 - accuracy: 0.7851

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5723 - accuracy: 0.7845

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5718 - accuracy: 0.7839

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5740 - accuracy: 0.7823

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5786 - accuracy: 0.7809

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5783 - accuracy: 0.7804

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5769 - accuracy: 0.7804

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5777 - accuracy: 0.7799

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5778 - accuracy: 0.7804

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5823 - accuracy: 0.7786

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5845 - accuracy: 0.7777

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5846 - accuracy: 0.7768

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5870 - accuracy: 0.7764

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5877 - accuracy: 0.7761

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5879 - accuracy: 0.7766

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5852 - accuracy: 0.7766

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5867 - accuracy: 0.7758

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5868 - accuracy: 0.7755

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5859 - accuracy: 0.7748

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5844 - accuracy: 0.7748

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5848 - accuracy: 0.7745

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5866 - accuracy: 0.7726

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5852 - accuracy: 0.7727

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5915 - accuracy: 0.7697

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5917 - accuracy: 0.7703

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5907 - accuracy: 0.7708

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5925 - accuracy: 0.7705

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5931 - accuracy: 0.7710

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5939 - accuracy: 0.7704

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5973 - accuracy: 0.7687

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5954 - accuracy: 0.7707

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5953 - accuracy: 0.7704

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5971 - accuracy: 0.7692

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5996 - accuracy: 0.7679

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5981 - accuracy: 0.7691

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.5981 - accuracy: 0.7691 - val_loss: 0.7115 - val_accuracy: 0.7221


.. parsed-literal::

    Epoch 12/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.7337 - accuracy: 0.7812

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5403 - accuracy: 0.8594

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.4955 - accuracy: 0.8542

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5461 - accuracy: 0.8281

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5412 - accuracy: 0.8250

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.5209 - accuracy: 0.8333

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 5s - loss: 0.5470 - accuracy: 0.8170

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5296 - accuracy: 0.8281

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5669 - accuracy: 0.8125

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5538 - accuracy: 0.8156

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5553 - accuracy: 0.8097

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5551 - accuracy: 0.8151

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5451 - accuracy: 0.8173

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5719 - accuracy: 0.7991

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5774 - accuracy: 0.8000

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5697 - accuracy: 0.8008

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5602 - accuracy: 0.8070

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5665 - accuracy: 0.8056

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5520 - accuracy: 0.8125

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5532 - accuracy: 0.8156

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5492 - accuracy: 0.8140

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5483 - accuracy: 0.8097

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.5440 - accuracy: 0.8084

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5410 - accuracy: 0.8112

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5350 - accuracy: 0.8112

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5286 - accuracy: 0.8137

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5272 - accuracy: 0.8137

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5255 - accuracy: 0.8092

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5178 - accuracy: 0.8114

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5120 - accuracy: 0.8146

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5063 - accuracy: 0.8155

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5162 - accuracy: 0.8135

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5211 - accuracy: 0.8116

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5212 - accuracy: 0.8107

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5202 - accuracy: 0.8098

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5185 - accuracy: 0.8099

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5184 - accuracy: 0.8083

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5137 - accuracy: 0.8109

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5161 - accuracy: 0.8101

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.5181 - accuracy: 0.8086

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5171 - accuracy: 0.8087

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5211 - accuracy: 0.8080

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5192 - accuracy: 0.8074

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5152 - accuracy: 0.8089

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5146 - accuracy: 0.8090

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5139 - accuracy: 0.8084

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5159 - accuracy: 0.8072

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5143 - accuracy: 0.8079

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5168 - accuracy: 0.8055

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5178 - accuracy: 0.8062

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5166 - accuracy: 0.8070

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5184 - accuracy: 0.8071

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5226 - accuracy: 0.8054

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5204 - accuracy: 0.8079

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5187 - accuracy: 0.8085

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5183 - accuracy: 0.8097

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5201 - accuracy: 0.8092

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5167 - accuracy: 0.8109

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5180 - accuracy: 0.8104

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5233 - accuracy: 0.8078

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5218 - accuracy: 0.8084

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5251 - accuracy: 0.8070

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5261 - accuracy: 0.8051

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5259 - accuracy: 0.8042

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5262 - accuracy: 0.8038

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5260 - accuracy: 0.8034

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5297 - accuracy: 0.8007

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5284 - accuracy: 0.8023

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5309 - accuracy: 0.8029

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5323 - accuracy: 0.8021

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5335 - accuracy: 0.8018

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5331 - accuracy: 0.8011

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5360 - accuracy: 0.8000

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5361 - accuracy: 0.8002

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5369 - accuracy: 0.7995

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5351 - accuracy: 0.8009

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5370 - accuracy: 0.8010

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5377 - accuracy: 0.8004

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5374 - accuracy: 0.8002

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5360 - accuracy: 0.7999

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5367 - accuracy: 0.7993

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5375 - accuracy: 0.7991

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5368 - accuracy: 0.7989

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5372 - accuracy: 0.7979

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5372 - accuracy: 0.7977

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5374 - accuracy: 0.7976

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5385 - accuracy: 0.7967

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5405 - accuracy: 0.7954

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5404 - accuracy: 0.7946

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5409 - accuracy: 0.7941

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5409 - accuracy: 0.7946

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.5409 - accuracy: 0.7946 - val_loss: 0.6885 - val_accuracy: 0.7561


.. parsed-literal::

    Epoch 13/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.3683 - accuracy: 0.8438

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5108 - accuracy: 0.7656

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5085 - accuracy: 0.7708

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.4920 - accuracy: 0.7891

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.4479 - accuracy: 0.8125

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.4437 - accuracy: 0.8177

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.4623 - accuracy: 0.8170

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.4931 - accuracy: 0.8008

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.4960 - accuracy: 0.8021

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5029 - accuracy: 0.7937

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.4888 - accuracy: 0.8005

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.4984 - accuracy: 0.7966

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.4824 - accuracy: 0.8000

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.4826 - accuracy: 0.7966

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.4938 - accuracy: 0.7956

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.4920 - accuracy: 0.7948

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.4952 - accuracy: 0.7940

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.4997 - accuracy: 0.7900

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5000 - accuracy: 0.7927

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.4947 - accuracy: 0.7937

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.4949 - accuracy: 0.7945

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.4943 - accuracy: 0.7926

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5002 - accuracy: 0.7895

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.4972 - accuracy: 0.7891

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.4890 - accuracy: 0.7937

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.4905 - accuracy: 0.7921

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.4859 - accuracy: 0.7950

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.4889 - accuracy: 0.7967

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.4939 - accuracy: 0.7962

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.4913 - accuracy: 0.7967

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.4892 - accuracy: 0.7982

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.4870 - accuracy: 0.7977

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.4915 - accuracy: 0.7954

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.4921 - accuracy: 0.7950

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.4902 - accuracy: 0.7955

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.4886 - accuracy: 0.7968

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.4890 - accuracy: 0.7972

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.4947 - accuracy: 0.7968

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.4978 - accuracy: 0.7972

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.4980 - accuracy: 0.7975

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.4993 - accuracy: 0.7957

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.4973 - accuracy: 0.7968

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.4996 - accuracy: 0.7950

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.4975 - accuracy: 0.7968

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.4959 - accuracy: 0.7978

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.4961 - accuracy: 0.7961

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.4935 - accuracy: 0.7958

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.4940 - accuracy: 0.7955

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.4945 - accuracy: 0.7965

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.4996 - accuracy: 0.7950

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.4978 - accuracy: 0.7941

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.4966 - accuracy: 0.7950

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.4991 - accuracy: 0.7936

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.4965 - accuracy: 0.7951

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.4978 - accuracy: 0.7948

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5013 - accuracy: 0.7941

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5040 - accuracy: 0.7938

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5066 - accuracy: 0.7947

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5050 - accuracy: 0.7945

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5061 - accuracy: 0.7948

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5078 - accuracy: 0.7945

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5126 - accuracy: 0.7928

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5155 - accuracy: 0.7922

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5124 - accuracy: 0.7939

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5136 - accuracy: 0.7928

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5174 - accuracy: 0.7926

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5179 - accuracy: 0.7920

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5169 - accuracy: 0.7932

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5202 - accuracy: 0.7912

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5185 - accuracy: 0.7924

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5211 - accuracy: 0.7918

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5206 - accuracy: 0.7921

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5220 - accuracy: 0.7919

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5210 - accuracy: 0.7926

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5262 - accuracy: 0.7921

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5250 - accuracy: 0.7932

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5230 - accuracy: 0.7946

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5202 - accuracy: 0.7960

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5249 - accuracy: 0.7951

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5240 - accuracy: 0.7961

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5281 - accuracy: 0.7947

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5273 - accuracy: 0.7949

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5290 - accuracy: 0.7955

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5310 - accuracy: 0.7942

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5306 - accuracy: 0.7941

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5299 - accuracy: 0.7950

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5285 - accuracy: 0.7952

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5271 - accuracy: 0.7961

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5280 - accuracy: 0.7953

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5281 - accuracy: 0.7948

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5282 - accuracy: 0.7943

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.5282 - accuracy: 0.7943 - val_loss: 0.7076 - val_accuracy: 0.7289


.. parsed-literal::

    Epoch 14/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.6387 - accuracy: 0.8125

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.6002 - accuracy: 0.8438

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5437 - accuracy: 0.8438

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5291 - accuracy: 0.8281

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5776 - accuracy: 0.8250

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.5498 - accuracy: 0.8333

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.5494 - accuracy: 0.8348

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5663 - accuracy: 0.8242

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5545 - accuracy: 0.8264

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5389 - accuracy: 0.8281

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5345 - accuracy: 0.8295

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5414 - accuracy: 0.8229

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5402 - accuracy: 0.8149

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5431 - accuracy: 0.8103

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5376 - accuracy: 0.8125

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5240 - accuracy: 0.8184

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5158 - accuracy: 0.8199

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5021 - accuracy: 0.8229

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5021 - accuracy: 0.8240

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5007 - accuracy: 0.8281

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.4921 - accuracy: 0.8304

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.4914 - accuracy: 0.8310

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.4924 - accuracy: 0.8302

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.4931 - accuracy: 0.8281

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.4863 - accuracy: 0.8313

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.4891 - accuracy: 0.8281

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.4933 - accuracy: 0.8241

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.4928 - accuracy: 0.8237

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.4911 - accuracy: 0.8244

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.4926 - accuracy: 0.8219

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.4898 - accuracy: 0.8226

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.4861 - accuracy: 0.8232

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.4873 - accuracy: 0.8248

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.4906 - accuracy: 0.8226

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.4893 - accuracy: 0.8232

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.4873 - accuracy: 0.8238

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.4842 - accuracy: 0.8243

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.4896 - accuracy: 0.8224

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.4949 - accuracy: 0.8213

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.4961 - accuracy: 0.8203

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.4963 - accuracy: 0.8186

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.4912 - accuracy: 0.8199

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.4910 - accuracy: 0.8190

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.4977 - accuracy: 0.8168

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.4948 - accuracy: 0.8181

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.4945 - accuracy: 0.8186

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.4922 - accuracy: 0.8191

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.4927 - accuracy: 0.8197

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.4932 - accuracy: 0.8182

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.4982 - accuracy: 0.8150

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.4999 - accuracy: 0.8143

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.4994 - accuracy: 0.8143

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.4985 - accuracy: 0.8143

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.4949 - accuracy: 0.8166

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.4949 - accuracy: 0.8159

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.4982 - accuracy: 0.8147

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.4960 - accuracy: 0.8163

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.4979 - accuracy: 0.8157

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.4970 - accuracy: 0.8146

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.4978 - accuracy: 0.8135

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.4996 - accuracy: 0.8140

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5017 - accuracy: 0.8125

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5063 - accuracy: 0.8118

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5084 - accuracy: 0.8103

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5068 - accuracy: 0.8113

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5079 - accuracy: 0.8109

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5090 - accuracy: 0.8100

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5133 - accuracy: 0.8073

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5160 - accuracy: 0.8065

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5148 - accuracy: 0.8065

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5146 - accuracy: 0.8062

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5126 - accuracy: 0.8076

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5155 - accuracy: 0.8051

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5143 - accuracy: 0.8060

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5148 - accuracy: 0.8045

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5138 - accuracy: 0.8046

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5131 - accuracy: 0.8051

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5123 - accuracy: 0.8060

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5121 - accuracy: 0.8060

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5118 - accuracy: 0.8061

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5117 - accuracy: 0.8054

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5137 - accuracy: 0.8063

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5165 - accuracy: 0.8049

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5150 - accuracy: 0.8053

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5142 - accuracy: 0.8054

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5151 - accuracy: 0.8058

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5120 - accuracy: 0.8080

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5143 - accuracy: 0.8077

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5141 - accuracy: 0.8075

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5126 - accuracy: 0.8075

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5103 - accuracy: 0.8089

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.5103 - accuracy: 0.8089 - val_loss: 0.6848 - val_accuracy: 0.7507


.. parsed-literal::

    Epoch 15/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.7010 - accuracy: 0.7500

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5184 - accuracy: 0.8281

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.4646 - accuracy: 0.8542

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.4931 - accuracy: 0.8203

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.4735 - accuracy: 0.8250

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.4644 - accuracy: 0.8281

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.4866 - accuracy: 0.8214

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.4726 - accuracy: 0.8320

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.4768 - accuracy: 0.8333

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.4843 - accuracy: 0.8250

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.4768 - accuracy: 0.8295

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.4784 - accuracy: 0.8281

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.4701 - accuracy: 0.8317

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.4757 - accuracy: 0.8326

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5012 - accuracy: 0.8250

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5148 - accuracy: 0.8242

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5074 - accuracy: 0.8272

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5014 - accuracy: 0.8316

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.4970 - accuracy: 0.8355

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5039 - accuracy: 0.8344

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5050 - accuracy: 0.8318

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5021 - accuracy: 0.8310

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.5047 - accuracy: 0.8274

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5070 - accuracy: 0.8242

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5049 - accuracy: 0.8250

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5015 - accuracy: 0.8257

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5006 - accuracy: 0.8252

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5054 - accuracy: 0.8237

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.4996 - accuracy: 0.8265

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5036 - accuracy: 0.8260

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5007 - accuracy: 0.8266

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.4991 - accuracy: 0.8262

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.4956 - accuracy: 0.8267

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.4931 - accuracy: 0.8281

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.4939 - accuracy: 0.8277

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.4956 - accuracy: 0.8255

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.4925 - accuracy: 0.8260

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.4896 - accuracy: 0.8273

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.4900 - accuracy: 0.8253

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.4936 - accuracy: 0.8234

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.4889 - accuracy: 0.8247

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.4894 - accuracy: 0.8237

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.4867 - accuracy: 0.8234

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.4905 - accuracy: 0.8239

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.4889 - accuracy: 0.8231

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.4903 - accuracy: 0.8222

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.4863 - accuracy: 0.8240

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.4882 - accuracy: 0.8244

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.4976 - accuracy: 0.8222

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.4954 - accuracy: 0.8233

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.4954 - accuracy: 0.8219

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.4902 - accuracy: 0.8252

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.4923 - accuracy: 0.8221

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.4875 - accuracy: 0.8236

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.4829 - accuracy: 0.8257

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.4817 - accuracy: 0.8254

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.4882 - accuracy: 0.8236

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.4919 - accuracy: 0.8223

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.4974 - accuracy: 0.8201

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.4968 - accuracy: 0.8205

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5021 - accuracy: 0.8188

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5008 - accuracy: 0.8182

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.4989 - accuracy: 0.8186

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5016 - accuracy: 0.8181

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5053 - accuracy: 0.8175

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5099 - accuracy: 0.8155

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5115 - accuracy: 0.8155

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5109 - accuracy: 0.8155

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5154 - accuracy: 0.8127

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5164 - accuracy: 0.8127

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5157 - accuracy: 0.8140

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5155 - accuracy: 0.8149

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5127 - accuracy: 0.8165

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5144 - accuracy: 0.8165

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5154 - accuracy: 0.8172

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5132 - accuracy: 0.8184

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5121 - accuracy: 0.8191

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5109 - accuracy: 0.8198

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5092 - accuracy: 0.8209

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5093 - accuracy: 0.8212

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5106 - accuracy: 0.8203

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5089 - accuracy: 0.8206

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5078 - accuracy: 0.8209

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5081 - accuracy: 0.8208

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5088 - accuracy: 0.8200

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5079 - accuracy: 0.8192

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5072 - accuracy: 0.8202

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5058 - accuracy: 0.8201

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5062 - accuracy: 0.8193

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5076 - accuracy: 0.8189

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5075 - accuracy: 0.8185

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.5075 - accuracy: 0.8185 - val_loss: 0.6563 - val_accuracy: 0.7425



.. image:: 301-tensorflow-training-openvino-nncf-with-output_files/301-tensorflow-training-openvino-nncf-with-output_3_1452.png


.. parsed-literal::


   1/1 [==============================] - ETA: 0s

.. parsed-literal::

    
    1/1 [==============================] - 0s 78ms/step


.. parsed-literal::

    This image most likely belongs to sunflowers with a 99.74 percent confidence.


.. parsed-literal::

    2024-03-13 01:01:33.965755: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'random_flip_input' with dtype float and shape [?,180,180,3]
    	 [[{{node random_flip_input}}]]
    2024-03-13 01:01:34.051093: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-13 01:01:34.061726: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'random_flip_input' with dtype float and shape [?,180,180,3]
    	 [[{{node random_flip_input}}]]
    2024-03-13 01:01:34.073459: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-13 01:01:34.080567: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-13 01:01:34.087476: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-13 01:01:34.098574: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-13 01:01:34.137415: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'sequential_1_input' with dtype float and shape [?,180,180,3]
    	 [[{{node sequential_1_input}}]]


.. parsed-literal::

    2024-03-13 01:01:34.205352: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-13 01:01:34.225505: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'sequential_1_input' with dtype float and shape [?,180,180,3]
    	 [[{{node sequential_1_input}}]]
    2024-03-13 01:01:34.263910: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,22,22,64]
    	 [[{{node inputs}}]]
    2024-03-13 01:01:34.288270: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]


.. parsed-literal::

    2024-03-13 01:01:34.526623: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-13 01:01:34.665874: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]


.. parsed-literal::

    2024-03-13 01:01:34.801730: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,22,22,64]
    	 [[{{node inputs}}]]
    2024-03-13 01:01:34.835860: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-13 01:01:34.863528: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-03-13 01:01:34.909392: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
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
    This image most likely belongs to dandelion with a 99.50 percent confidence.



.. image:: 301-tensorflow-training-openvino-nncf-with-output_files/301-tensorflow-training-openvino-nncf-with-output_3_1464.png


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

    2024-03-13 01:01:37.480859: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [734]
    	 [[{{node Placeholder/_4}}]]
    2024-03-13 01:01:37.481108: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [734]
    	 [[{{node Placeholder/_4}}]]


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
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/live.py", line 32, in run
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    self.live.refresh()
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/live.py", line 223, in refresh
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    self._live_render.set_renderable(self.renderable)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/live.py", line 203, in renderable
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    renderable = self.get_renderable()
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/live.py", line 98, in get_renderable
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    self._get_renderable()
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1537, in get_renderable
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    renderable = Group(*self.get_renderables())
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1542, in get_renderables
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    table = self.make_tasks_table(self.tasks)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1566, in make_tasks_table
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    table.add_row(
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1571, in &lt;genexpr&gt;
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    else column(task)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 528, in __call__
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    renderable = self.render(task)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/nncf/common/logging/track_progress.py", line 58, in render
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    text = super().render(task)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 787, in render
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    task_time = task.time_remaining
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1039, in time_remaining
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    estimate = ceil(remaining / speed)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/tensorflow/python/util/traceback_utils.py", line 153, in error_handler
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    raise e.with_traceback(filtered_tb) from None
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
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

    Accuracy of the original model: 0.743
    Accuracy of the quantized model: 0.749


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


.. parsed-literal::

    This image most likely belongs to dandelion with a 99.53 percent confidence.



.. image:: 301-tensorflow-training-openvino-nncf-with-output_files/301-tensorflow-training-openvino-nncf-with-output_27_2.png


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
    [ INFO ] Read model took 4.25 ms
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

    [ INFO ] Compile model took 105.22 ms
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


.. parsed-literal::

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
    [ INFO ] First inference took 3.84 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            55932 iterations
    [ INFO ] Duration:         15004.24 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        3.03 ms
    [ INFO ]    Average:       3.03 ms
    [ INFO ]    Min:           1.72 ms
    [ INFO ]    Max:           11.92 ms
    [ INFO ] Throughput:   3727.75 FPS


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
    [ INFO ] Read model took 4.62 ms
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

    [ INFO ] Compile model took 113.92 ms
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
    [ INFO ] First inference took 1.72 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            178524 iterations
    [ INFO ] Duration:         15001.12 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        0.94 ms
    [ INFO ]    Average:       0.97 ms
    [ INFO ]    Min:           0.61 ms
    [ INFO ]    Max:           13.42 ms
    [ INFO ] Throughput:   11900.71 FPS

