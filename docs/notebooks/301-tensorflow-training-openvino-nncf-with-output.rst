Post-Training Quantization with TensorFlow Classification Model
===============================================================

This example demonstrates how to quantize the OpenVINO model that was
created in `301-tensorflow-training-openvino
notebook <301-tensorflow-training-openvino-with-output.html>`__, to improve
inference speed. Quantization is performed with `Post-training
Quantization with
NNCF <https://docs.openvino.ai/nightly/basic_quantization_flow.html>`__.
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

    %pip install -q tensorflow Pillow matplotlib numpy tqdm nncf


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

    2024-02-10 01:09:00.730910: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-02-10 01:09:00.766002: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


.. parsed-literal::

    2024-02-10 01:09:01.406366: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


.. parsed-literal::

    Executing training notebook. This will take a while...


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    3670
    Found 3670 files belonging to 5 classes.


.. parsed-literal::

    Using 2936 files for training.


.. parsed-literal::

    2024-02-10 01:09:08.525687: E tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:266] failed call to cuInit: CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: forward compatibility was attempted on non supported HW
    2024-02-10 01:09:08.525725: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:168] retrieving CUDA diagnostic information for host: iotg-dev-workstation-07
    2024-02-10 01:09:08.525729: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:175] hostname: iotg-dev-workstation-07
    2024-02-10 01:09:08.525856: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:199] libcuda reported version is: 470.223.2
    2024-02-10 01:09:08.525872: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:203] kernel reported version is: 470.182.3
    2024-02-10 01:09:08.525876: E tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:312] kernel version 470.182.3 does not match DSO version 470.223.2 -- cannot find working devices in this configuration


.. parsed-literal::

    Found 3670 files belonging to 5 classes.


.. parsed-literal::

    Using 734 files for validation.
    ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']


.. parsed-literal::

    2024-02-10 01:09:08.855253: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]
    2024-02-10 01:09:08.855534: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]



.. image:: 301-tensorflow-training-openvino-nncf-with-output_files/301-tensorflow-training-openvino-nncf-with-output_3_11.png


.. parsed-literal::

    2024-02-10 01:09:09.711519: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-02-10 01:09:09.711766: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]


.. parsed-literal::

    (32, 180, 180, 3)
    (32,)


.. parsed-literal::

    2024-02-10 01:09:10.063734: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-02-10 01:09:10.064340: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]


.. parsed-literal::

    0.0 0.9970461


.. parsed-literal::

    2024-02-10 01:09:10.875056: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]
    2024-02-10 01:09:10.875365: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
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

    2024-02-10 01:09:11.882327: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]
    2024-02-10 01:09:11.882802: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]


.. parsed-literal::


 1/92 [..............................] - ETA: 1:32 - loss: 1.6315 - accuracy: 0.1562

.. parsed-literal::

    
 2/92 [..............................] - ETA: 6s - loss: 1.7632 - accuracy: 0.2812

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 1.7516 - accuracy: 0.2708

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 1.7249 - accuracy: 0.2578

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 1.6968 - accuracy: 0.2750

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 1.6729 - accuracy: 0.2917

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 5s - loss: 1.6459 - accuracy: 0.3080

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 5s - loss: 1.6410 - accuracy: 0.3008

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 1.6246 - accuracy: 0.3125

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 1.6151 - accuracy: 0.3000

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 1.6065 - accuracy: 0.3011

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 1.5947 - accuracy: 0.3047

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 1.5839 - accuracy: 0.3077

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 1.5719 - accuracy: 0.3125

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 1.5604 - accuracy: 0.3187

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 1.5477 - accuracy: 0.3203

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 1.5317 - accuracy: 0.3272

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 1.5153 - accuracy: 0.3368

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 1.5118 - accuracy: 0.3355

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 1.4901 - accuracy: 0.3484

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 1.4818 - accuracy: 0.3569

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 1.4839 - accuracy: 0.3563

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 1.4731 - accuracy: 0.3599

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 1.4556 - accuracy: 0.3724

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 1.4413 - accuracy: 0.3788

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 1.4353 - accuracy: 0.3774

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 1.4367 - accuracy: 0.3762

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 1.4293 - accuracy: 0.3750

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 1.4196 - accuracy: 0.3793

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 1.4177 - accuracy: 0.3813

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 1.4057 - accuracy: 0.3872

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 1.4028 - accuracy: 0.3868

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 1.3896 - accuracy: 0.3950

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 1.3879 - accuracy: 0.3963

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 1.3886 - accuracy: 0.3966

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 1.3839 - accuracy: 0.3969

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 1.3853 - accuracy: 0.4022

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 1.3812 - accuracy: 0.4023

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 1.3746 - accuracy: 0.4065

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 1.3733 - accuracy: 0.4049

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 1.3684 - accuracy: 0.4064

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 1.3665 - accuracy: 0.4064

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 1.3624 - accuracy: 0.4108

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 1.3590 - accuracy: 0.4121

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 1.3533 - accuracy: 0.4148

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 1.3472 - accuracy: 0.4167

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 1.3448 - accuracy: 0.4164

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 1.3409 - accuracy: 0.4162

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 1.3383 - accuracy: 0.4186

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 1.3381 - accuracy: 0.4190

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 1.3341 - accuracy: 0.4212

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 1.3292 - accuracy: 0.4245

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 1.3286 - accuracy: 0.4277

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 1.3246 - accuracy: 0.4302

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 1.3228 - accuracy: 0.4309

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 1.3231 - accuracy: 0.4355

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 1.3221 - accuracy: 0.4350

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 1.3200 - accuracy: 0.4378

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 1.3177 - accuracy: 0.4394

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 1.3148 - accuracy: 0.4409

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 1.3140 - accuracy: 0.4408

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 1.3080 - accuracy: 0.4443

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 1.3096 - accuracy: 0.4447

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 1.3068 - accuracy: 0.4451

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 1.3014 - accuracy: 0.4469

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 1.3013 - accuracy: 0.4468

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 1.2977 - accuracy: 0.4480

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 1.2948 - accuracy: 0.4493

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 1.2914 - accuracy: 0.4500

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 1.2929 - accuracy: 0.4476

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 1.2929 - accuracy: 0.4479

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 1.2902 - accuracy: 0.4495

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 1.2864 - accuracy: 0.4506

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 1.2854 - accuracy: 0.4504

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 1.2853 - accuracy: 0.4494

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 1.2809 - accuracy: 0.4513

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 1.2779 - accuracy: 0.4532

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 1.2774 - accuracy: 0.4538

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 1.2739 - accuracy: 0.4540

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 1.2722 - accuracy: 0.4542

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 1.2669 - accuracy: 0.4582

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 1.2654 - accuracy: 0.4599

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 1.2625 - accuracy: 0.4622

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 1.2580 - accuracy: 0.4642

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 1.2573 - accuracy: 0.4653

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 1.2566 - accuracy: 0.4665

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 1.2564 - accuracy: 0.4665

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 1.2530 - accuracy: 0.4679

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 1.2492 - accuracy: 0.4690

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 1.2445 - accuracy: 0.4714

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 1.2402 - accuracy: 0.4735

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 1.2400 - accuracy: 0.4741

.. parsed-literal::

    2024-02-10 01:09:18.229567: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [734]
    	 [[{{node Placeholder/_0}}]]
    2024-02-10 01:09:18.229847: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [734]
    	 [[{{node Placeholder/_4}}]]


.. parsed-literal::

    
92/92 [==============================] - 7s 66ms/step - loss: 1.2400 - accuracy: 0.4741 - val_loss: 1.3762 - val_accuracy: 0.5014


.. parsed-literal::

    Epoch 2/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 1.2018 - accuracy: 0.5625

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 1.0597 - accuracy: 0.5469

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 1.0781 - accuracy: 0.5521

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.9952 - accuracy: 0.5938

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.9872 - accuracy: 0.6187

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.9615 - accuracy: 0.6094

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.9715 - accuracy: 0.6205

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.9608 - accuracy: 0.6211

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.9538 - accuracy: 0.6250

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.9487 - accuracy: 0.6250

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.9616 - accuracy: 0.6307

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.9475 - accuracy: 0.6302

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.9476 - accuracy: 0.6346

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.9426 - accuracy: 0.6295

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.9547 - accuracy: 0.6250

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.9611 - accuracy: 0.6211

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.9566 - accuracy: 0.6176

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.9654 - accuracy: 0.6111

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.9686 - accuracy: 0.6118

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.9718 - accuracy: 0.6078

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.9820 - accuracy: 0.6012

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.9866 - accuracy: 0.5994

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.9839 - accuracy: 0.6005

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.9903 - accuracy: 0.5964

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.9938 - accuracy: 0.5950

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 1.0051 - accuracy: 0.5901

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.9990 - accuracy: 0.5938

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.9965 - accuracy: 0.5949

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.9986 - accuracy: 0.5948

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.9992 - accuracy: 0.5917

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.9976 - accuracy: 0.5927

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.9922 - accuracy: 0.5964

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.9964 - accuracy: 0.5954

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 1.0031 - accuracy: 0.5935

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 1.0041 - accuracy: 0.5909

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 1.0075 - accuracy: 0.5884

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 1.0049 - accuracy: 0.5886

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 1.0093 - accuracy: 0.5879

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 1.0040 - accuracy: 0.5912

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 1.0038 - accuracy: 0.5920

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 1.0080 - accuracy: 0.5898

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 1.0082 - accuracy: 0.5914

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 1.0021 - accuracy: 0.5936

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 1.0045 - accuracy: 0.5936

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 1.0064 - accuracy: 0.5922

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 1.0063 - accuracy: 0.5936

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 1.0006 - accuracy: 0.5975

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.9997 - accuracy: 0.5981

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.9995 - accuracy: 0.5980

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.9969 - accuracy: 0.5998

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.9952 - accuracy: 0.5996

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.9974 - accuracy: 0.5983

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 1.0079 - accuracy: 0.5913

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 1.0106 - accuracy: 0.5908

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 1.0121 - accuracy: 0.5880

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 1.0146 - accuracy: 0.5887

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 1.0174 - accuracy: 0.5882

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 1.0150 - accuracy: 0.5883

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 1.0120 - accuracy: 0.5884

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 1.0103 - accuracy: 0.5890

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 1.0095 - accuracy: 0.5901

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 1.0114 - accuracy: 0.5886

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 1.0114 - accuracy: 0.5897

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 1.0108 - accuracy: 0.5907

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 1.0115 - accuracy: 0.5903

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 1.0117 - accuracy: 0.5894

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 1.0125 - accuracy: 0.5881

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 1.0151 - accuracy: 0.5850

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 1.0146 - accuracy: 0.5865

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 1.0158 - accuracy: 0.5866

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 1.0140 - accuracy: 0.5871

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 1.0130 - accuracy: 0.5872

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 1.0127 - accuracy: 0.5873

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 1.0109 - accuracy: 0.5886

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 1.0074 - accuracy: 0.5895

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 1.0038 - accuracy: 0.5916

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 1.0003 - accuracy: 0.5936

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.9982 - accuracy: 0.5944

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.9978 - accuracy: 0.5929

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 1.0002 - accuracy: 0.5940

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.9975 - accuracy: 0.5944

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.9982 - accuracy: 0.5948

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.9979 - accuracy: 0.5963

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.9960 - accuracy: 0.5970

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.9941 - accuracy: 0.5977

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.9931 - accuracy: 0.5973

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.9948 - accuracy: 0.5954

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.9953 - accuracy: 0.5951

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.9947 - accuracy: 0.5961

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.9932 - accuracy: 0.5964

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.9956 - accuracy: 0.5974

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.9956 - accuracy: 0.5974 - val_loss: 0.9920 - val_accuracy: 0.6090


.. parsed-literal::

    Epoch 3/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 1.2602 - accuracy: 0.4688

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 1.1814 - accuracy: 0.5781

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 1.1491 - accuracy: 0.5625

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 1.0875 - accuracy: 0.5781

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 1.0316 - accuracy: 0.5875

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 1.0206 - accuracy: 0.5833

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.9818 - accuracy: 0.5938

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 1.0018 - accuracy: 0.5859

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.9855 - accuracy: 0.5938

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.9760 - accuracy: 0.5969

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.9811 - accuracy: 0.5881

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.9836 - accuracy: 0.5859

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.9757 - accuracy: 0.5889

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.9660 - accuracy: 0.5893

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.9619 - accuracy: 0.5938

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.9688 - accuracy: 0.5898

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.9691 - accuracy: 0.5919

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.9729 - accuracy: 0.5938

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.9704 - accuracy: 0.5934

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.9652 - accuracy: 0.5934

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.9528 - accuracy: 0.6006

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.9511 - accuracy: 0.6044

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.9597 - accuracy: 0.6026

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.9707 - accuracy: 0.5972

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.9649 - accuracy: 0.5971

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.9528 - accuracy: 0.6016

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.9453 - accuracy: 0.6059

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.9458 - accuracy: 0.6065

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.9467 - accuracy: 0.6061

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.9432 - accuracy: 0.6098

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.9430 - accuracy: 0.6093

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.9351 - accuracy: 0.6164

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.9405 - accuracy: 0.6139

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.9356 - accuracy: 0.6169

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.9322 - accuracy: 0.6215

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.9363 - accuracy: 0.6199

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.9325 - accuracy: 0.6217

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.9322 - accuracy: 0.6234

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.9291 - accuracy: 0.6250

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.9252 - accuracy: 0.6242

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.9226 - accuracy: 0.6257

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.9213 - accuracy: 0.6250

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.9232 - accuracy: 0.6243

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.9261 - accuracy: 0.6222

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.9277 - accuracy: 0.6209

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.9308 - accuracy: 0.6197

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.9392 - accuracy: 0.6145

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.9396 - accuracy: 0.6122

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.9369 - accuracy: 0.6137

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.9361 - accuracy: 0.6139

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.9326 - accuracy: 0.6147

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.9335 - accuracy: 0.6149

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.9319 - accuracy: 0.6169

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.9303 - accuracy: 0.6170

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.9289 - accuracy: 0.6183

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.9316 - accuracy: 0.6167

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.9314 - accuracy: 0.6163

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.9313 - accuracy: 0.6165

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.9282 - accuracy: 0.6182

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.9264 - accuracy: 0.6188

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.9249 - accuracy: 0.6194

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.9295 - accuracy: 0.6165

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.9288 - accuracy: 0.6157

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.9234 - accuracy: 0.6178

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.9228 - accuracy: 0.6179

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.9232 - accuracy: 0.6170

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.9240 - accuracy: 0.6172

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.9200 - accuracy: 0.6205

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.9244 - accuracy: 0.6196

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.9237 - accuracy: 0.6197

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.9253 - accuracy: 0.6193

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.9243 - accuracy: 0.6203

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.9259 - accuracy: 0.6195

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.9242 - accuracy: 0.6204

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.9216 - accuracy: 0.6213

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.9203 - accuracy: 0.6209

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.9200 - accuracy: 0.6214

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.9181 - accuracy: 0.6226

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.9185 - accuracy: 0.6226

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.9160 - accuracy: 0.6250

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.9186 - accuracy: 0.6250

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.9164 - accuracy: 0.6258

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.9167 - accuracy: 0.6269

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.9177 - accuracy: 0.6265

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.9183 - accuracy: 0.6276

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.9182 - accuracy: 0.6275

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.9156 - accuracy: 0.6278

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.9135 - accuracy: 0.6292

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.9121 - accuracy: 0.6302

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.9129 - accuracy: 0.6298

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.9155 - accuracy: 0.6298

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.9155 - accuracy: 0.6298 - val_loss: 0.8959 - val_accuracy: 0.6621


.. parsed-literal::

    Epoch 4/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.7704 - accuracy: 0.7812

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.8739 - accuracy: 0.6562

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.9644 - accuracy: 0.6146

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.9070 - accuracy: 0.6484

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 4s - loss: 0.8696 - accuracy: 0.6625

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.8536 - accuracy: 0.6562

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.8587 - accuracy: 0.6473

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.8727 - accuracy: 0.6523

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.8413 - accuracy: 0.6701

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.8577 - accuracy: 0.6594

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.8386 - accuracy: 0.6733

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.8637 - accuracy: 0.6589

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.8819 - accuracy: 0.6659

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.8783 - accuracy: 0.6674

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.8797 - accuracy: 0.6667

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.8644 - accuracy: 0.6777

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.8715 - accuracy: 0.6746

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.8544 - accuracy: 0.6788

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.8484 - accuracy: 0.6809

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.8429 - accuracy: 0.6828

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.8352 - accuracy: 0.6860

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.8293 - accuracy: 0.6875

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.8324 - accuracy: 0.6861

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.8321 - accuracy: 0.6875

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.8377 - accuracy: 0.6913

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.8366 - accuracy: 0.6923

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.8278 - accuracy: 0.6933

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.8303 - accuracy: 0.6942

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.8305 - accuracy: 0.6950

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.8342 - accuracy: 0.6958

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.8350 - accuracy: 0.6956

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.8386 - accuracy: 0.6914

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.8354 - accuracy: 0.6922

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.8424 - accuracy: 0.6921

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.8367 - accuracy: 0.6920

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.8349 - accuracy: 0.6936

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.8365 - accuracy: 0.6926

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.8451 - accuracy: 0.6891

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.8401 - accuracy: 0.6899

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.8397 - accuracy: 0.6891

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.8379 - accuracy: 0.6913

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.8459 - accuracy: 0.6868

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.8410 - accuracy: 0.6860

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.8352 - accuracy: 0.6889

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.8372 - accuracy: 0.6872

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.8341 - accuracy: 0.6885

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.8279 - accuracy: 0.6918

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.8294 - accuracy: 0.6917

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.8300 - accuracy: 0.6928

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.8275 - accuracy: 0.6940

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.8267 - accuracy: 0.6944

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.8255 - accuracy: 0.6961

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.8220 - accuracy: 0.6977

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.8198 - accuracy: 0.6975

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.8172 - accuracy: 0.6979

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.8155 - accuracy: 0.6982

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.8128 - accuracy: 0.6997

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.8129 - accuracy: 0.7000

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.8170 - accuracy: 0.6987

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.8182 - accuracy: 0.6980

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.8189 - accuracy: 0.6964

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.8193 - accuracy: 0.6952

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.8194 - accuracy: 0.6961

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.8215 - accuracy: 0.6955

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.8210 - accuracy: 0.6953

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.8223 - accuracy: 0.6952

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.8207 - accuracy: 0.6960

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.8182 - accuracy: 0.6977

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.8195 - accuracy: 0.6962

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.8194 - accuracy: 0.6961

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.8208 - accuracy: 0.6956

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.8190 - accuracy: 0.6963

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.8170 - accuracy: 0.6970

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.8158 - accuracy: 0.6977

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.8179 - accuracy: 0.6955

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.8176 - accuracy: 0.6958

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.8175 - accuracy: 0.6957

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.8188 - accuracy: 0.6952

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.8172 - accuracy: 0.6959

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.8217 - accuracy: 0.6939

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.8197 - accuracy: 0.6950

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.8180 - accuracy: 0.6952

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.8183 - accuracy: 0.6951

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.8190 - accuracy: 0.6947

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.8208 - accuracy: 0.6928

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.8207 - accuracy: 0.6931

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.8205 - accuracy: 0.6930

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.8188 - accuracy: 0.6937

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.8180 - accuracy: 0.6939

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.8167 - accuracy: 0.6946

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.8158 - accuracy: 0.6945

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.8158 - accuracy: 0.6945 - val_loss: 0.8530 - val_accuracy: 0.6757


.. parsed-literal::

    Epoch 5/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.8907 - accuracy: 0.7188

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.8773 - accuracy: 0.6875

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.8330 - accuracy: 0.6771

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.7960 - accuracy: 0.6953

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.8390 - accuracy: 0.6812

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.8144 - accuracy: 0.6771

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 5s - loss: 0.8024 - accuracy: 0.6920

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.8119 - accuracy: 0.6914

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.8164 - accuracy: 0.6875

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.7930 - accuracy: 0.7000

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.7694 - accuracy: 0.7102

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.7519 - accuracy: 0.7161

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.7302 - accuracy: 0.7260

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.7293 - accuracy: 0.7210

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.7256 - accuracy: 0.7208

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.7320 - accuracy: 0.7207

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.7327 - accuracy: 0.7243

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.7351 - accuracy: 0.7205

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.7352 - accuracy: 0.7204

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.7393 - accuracy: 0.7203

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.7438 - accuracy: 0.7217

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.7462 - accuracy: 0.7216

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.7597 - accuracy: 0.7147

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.7511 - accuracy: 0.7188

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.7569 - accuracy: 0.7150

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.7474 - accuracy: 0.7175

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.7583 - accuracy: 0.7141

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.7548 - accuracy: 0.7154

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.7512 - accuracy: 0.7155

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.7478 - accuracy: 0.7156

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.7464 - accuracy: 0.7157

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.7567 - accuracy: 0.7129

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.7518 - accuracy: 0.7159

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.7569 - accuracy: 0.7160

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.7563 - accuracy: 0.7152

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.7563 - accuracy: 0.7153

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.7607 - accuracy: 0.7111

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.7650 - accuracy: 0.7072

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.7642 - accuracy: 0.7083

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.7700 - accuracy: 0.7078

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.7711 - accuracy: 0.7096

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.7661 - accuracy: 0.7128

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.7633 - accuracy: 0.7129

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.7632 - accuracy: 0.7124

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.7625 - accuracy: 0.7125

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.7612 - accuracy: 0.7120

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.7590 - accuracy: 0.7108

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.7586 - accuracy: 0.7103

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.7561 - accuracy: 0.7111

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.7645 - accuracy: 0.7050

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.7645 - accuracy: 0.7047

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.7692 - accuracy: 0.7038

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.7705 - accuracy: 0.7035

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.7802 - accuracy: 0.6986

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.7780 - accuracy: 0.6990

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.7770 - accuracy: 0.6982

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.7749 - accuracy: 0.6997

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.7784 - accuracy: 0.6984

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.7787 - accuracy: 0.6967

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.7795 - accuracy: 0.6955

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.7780 - accuracy: 0.6964

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.7771 - accuracy: 0.6977

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.7814 - accuracy: 0.6961

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.7836 - accuracy: 0.6950

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.7805 - accuracy: 0.6968

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.7795 - accuracy: 0.6966

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.7841 - accuracy: 0.6933

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.7849 - accuracy: 0.6932

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.7907 - accuracy: 0.6927

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.7915 - accuracy: 0.6917

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.7909 - accuracy: 0.6916

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.7939 - accuracy: 0.6903

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.7967 - accuracy: 0.6890

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.7993 - accuracy: 0.6873

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.8003 - accuracy: 0.6861

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.8013 - accuracy: 0.6853

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.8004 - accuracy: 0.6853

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.7983 - accuracy: 0.6861

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.8004 - accuracy: 0.6846

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.8001 - accuracy: 0.6854

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.7997 - accuracy: 0.6858

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.7988 - accuracy: 0.6869

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.7983 - accuracy: 0.6869

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.7975 - accuracy: 0.6881

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.7958 - accuracy: 0.6891

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.7945 - accuracy: 0.6902

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.7928 - accuracy: 0.6909

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.7910 - accuracy: 0.6919

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.7917 - accuracy: 0.6915

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.7883 - accuracy: 0.6932

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.7896 - accuracy: 0.6931

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.7896 - accuracy: 0.6931 - val_loss: 0.8867 - val_accuracy: 0.6798


.. parsed-literal::

    Epoch 6/15


.. parsed-literal::


 1/92 [..............................] - ETA: 6s - loss: 0.5518 - accuracy: 0.7812

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.7630 - accuracy: 0.7500

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.7584 - accuracy: 0.7604

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.7502 - accuracy: 0.7266

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.7661 - accuracy: 0.7125

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.7633 - accuracy: 0.7083

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.7836 - accuracy: 0.7054

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.7776 - accuracy: 0.7070

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.7611 - accuracy: 0.7153

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.7587 - accuracy: 0.7125

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.7458 - accuracy: 0.7244

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.7555 - accuracy: 0.7266

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.7522 - accuracy: 0.7308

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.7398 - accuracy: 0.7277

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.7376 - accuracy: 0.7312

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.7344 - accuracy: 0.7285

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.7325 - accuracy: 0.7298

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.7301 - accuracy: 0.7292

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.7525 - accuracy: 0.7237

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.7644 - accuracy: 0.7188

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.7723 - accuracy: 0.7158

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.7636 - accuracy: 0.7202

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.7545 - accuracy: 0.7228

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.7522 - accuracy: 0.7227

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.7536 - accuracy: 0.7200

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.7700 - accuracy: 0.7103

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.7697 - accuracy: 0.7106

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.7738 - accuracy: 0.7121

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.7679 - accuracy: 0.7155

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.7772 - accuracy: 0.7104

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.7790 - accuracy: 0.7107

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.7822 - accuracy: 0.7100

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.7819 - accuracy: 0.7112

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.7831 - accuracy: 0.7086

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.7782 - accuracy: 0.7089

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.7777 - accuracy: 0.7083

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.7794 - accuracy: 0.7078

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.7856 - accuracy: 0.7064

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.7847 - accuracy: 0.7067

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.7862 - accuracy: 0.7047

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.7830 - accuracy: 0.7058

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.7793 - accuracy: 0.7068

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.7756 - accuracy: 0.7086

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.7727 - accuracy: 0.7095

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.7714 - accuracy: 0.7076

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.7723 - accuracy: 0.7079

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.7696 - accuracy: 0.7094

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.7686 - accuracy: 0.7109

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.7657 - accuracy: 0.7136

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.7702 - accuracy: 0.7119

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.7732 - accuracy: 0.7126

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.7772 - accuracy: 0.7109

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.7780 - accuracy: 0.7099

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.7742 - accuracy: 0.7124

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.7721 - accuracy: 0.7136

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.7717 - accuracy: 0.7132

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.7688 - accuracy: 0.7138

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.7733 - accuracy: 0.7101

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.7728 - accuracy: 0.7119

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.7746 - accuracy: 0.7109

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.7697 - accuracy: 0.7126

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.7746 - accuracy: 0.7117

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.7725 - accuracy: 0.7123

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.7711 - accuracy: 0.7134

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.7713 - accuracy: 0.7130

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.7690 - accuracy: 0.7135

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.7682 - accuracy: 0.7141

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.7670 - accuracy: 0.7146

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.7655 - accuracy: 0.7156

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.7639 - accuracy: 0.7161

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.7664 - accuracy: 0.7139

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.7650 - accuracy: 0.7153

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.7669 - accuracy: 0.7132

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.7677 - accuracy: 0.7124

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.7683 - accuracy: 0.7120

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.7676 - accuracy: 0.7113

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.7679 - accuracy: 0.7122

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.7655 - accuracy: 0.7127

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.7639 - accuracy: 0.7132

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.7643 - accuracy: 0.7136

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.7672 - accuracy: 0.7122

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.7681 - accuracy: 0.7115

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.7652 - accuracy: 0.7131

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.7664 - accuracy: 0.7124

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.7678 - accuracy: 0.7114

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.7660 - accuracy: 0.7118

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.7654 - accuracy: 0.7115

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.7658 - accuracy: 0.7109

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.7648 - accuracy: 0.7107

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.7648 - accuracy: 0.7107

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.7647 - accuracy: 0.7115

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.7647 - accuracy: 0.7115 - val_loss: 0.7599 - val_accuracy: 0.7016


.. parsed-literal::

    Epoch 7/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.4912 - accuracy: 0.8438

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5197 - accuracy: 0.7812

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6350 - accuracy: 0.7396

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6448 - accuracy: 0.7500

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.6741 - accuracy: 0.7375

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.7069 - accuracy: 0.7344

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 5s - loss: 0.7105 - accuracy: 0.7321

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 5s - loss: 0.7082 - accuracy: 0.7344

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.7131 - accuracy: 0.7326

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.7040 - accuracy: 0.7312

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.7117 - accuracy: 0.7244

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.7376 - accuracy: 0.7161

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.7223 - accuracy: 0.7236

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.7167 - accuracy: 0.7210

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.7110 - accuracy: 0.7250

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6943 - accuracy: 0.7324

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6881 - accuracy: 0.7335

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6882 - accuracy: 0.7326

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6898 - accuracy: 0.7319

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6850 - accuracy: 0.7328

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6983 - accuracy: 0.7292

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6962 - accuracy: 0.7301

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.6905 - accuracy: 0.7323

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 4s - loss: 0.6827 - accuracy: 0.7370

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6814 - accuracy: 0.7350

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6826 - accuracy: 0.7332

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6718 - accuracy: 0.7396

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6691 - accuracy: 0.7388

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6769 - accuracy: 0.7349

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6747 - accuracy: 0.7365

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6848 - accuracy: 0.7339

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6793 - accuracy: 0.7383

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6826 - accuracy: 0.7377

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6777 - accuracy: 0.7408

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6865 - accuracy: 0.7357

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6899 - accuracy: 0.7352

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6925 - accuracy: 0.7340

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6955 - accuracy: 0.7327

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6942 - accuracy: 0.7324

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.7007 - accuracy: 0.7297

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.7064 - accuracy: 0.7264

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.7118 - accuracy: 0.7262

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.7098 - accuracy: 0.7267

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.7088 - accuracy: 0.7294

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.7112 - accuracy: 0.7292

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.7105 - accuracy: 0.7283

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.7076 - accuracy: 0.7294

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.7085 - accuracy: 0.7272

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.7099 - accuracy: 0.7277

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.7086 - accuracy: 0.7287

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.7087 - accuracy: 0.7298

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.7098 - accuracy: 0.7302

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.7106 - accuracy: 0.7300

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.7093 - accuracy: 0.7315

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.7085 - accuracy: 0.7318

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.7107 - accuracy: 0.7299

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.7082 - accuracy: 0.7308

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.7098 - accuracy: 0.7295

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.7162 - accuracy: 0.7251

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.7135 - accuracy: 0.7260

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.7132 - accuracy: 0.7259

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.7137 - accuracy: 0.7263

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.7139 - accuracy: 0.7257

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.7139 - accuracy: 0.7251

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.7107 - accuracy: 0.7264

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.7091 - accuracy: 0.7268

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.7080 - accuracy: 0.7276

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.7085 - accuracy: 0.7275

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.7085 - accuracy: 0.7274

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.7086 - accuracy: 0.7277

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.7068 - accuracy: 0.7293

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.7058 - accuracy: 0.7300

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.7033 - accuracy: 0.7292

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.7020 - accuracy: 0.7304

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.7010 - accuracy: 0.7306

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6969 - accuracy: 0.7325

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6963 - accuracy: 0.7327

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6964 - accuracy: 0.7329

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6943 - accuracy: 0.7343

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6968 - accuracy: 0.7345

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6975 - accuracy: 0.7343

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.7000 - accuracy: 0.7330

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.7012 - accuracy: 0.7328

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6997 - accuracy: 0.7334

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6986 - accuracy: 0.7336

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6983 - accuracy: 0.7338

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6983 - accuracy: 0.7340

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6967 - accuracy: 0.7338

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6940 - accuracy: 0.7357

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6932 - accuracy: 0.7362

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6932 - accuracy: 0.7360

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.6932 - accuracy: 0.7360 - val_loss: 0.7731 - val_accuracy: 0.6853


.. parsed-literal::

    Epoch 8/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.5886 - accuracy: 0.7500

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.6209 - accuracy: 0.7500

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6834 - accuracy: 0.7396

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6812 - accuracy: 0.7266

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.6540 - accuracy: 0.7312

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.6633 - accuracy: 0.7240

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6415 - accuracy: 0.7411

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6256 - accuracy: 0.7539

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5989 - accuracy: 0.7604

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6047 - accuracy: 0.7500

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6068 - accuracy: 0.7500

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6024 - accuracy: 0.7500

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6013 - accuracy: 0.7548

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5978 - accuracy: 0.7545

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6020 - accuracy: 0.7479

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5969 - accuracy: 0.7500

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6147 - accuracy: 0.7408

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6109 - accuracy: 0.7448

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6156 - accuracy: 0.7467

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6125 - accuracy: 0.7516

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6098 - accuracy: 0.7530

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6131 - accuracy: 0.7528

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.6149 - accuracy: 0.7527

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6234 - accuracy: 0.7474

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6214 - accuracy: 0.7487

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6183 - accuracy: 0.7512

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6153 - accuracy: 0.7535

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6128 - accuracy: 0.7556

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6245 - accuracy: 0.7522

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6246 - accuracy: 0.7510

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6228 - accuracy: 0.7530

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6247 - accuracy: 0.7539

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6282 - accuracy: 0.7509

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6380 - accuracy: 0.7491

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6360 - accuracy: 0.7500

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6347 - accuracy: 0.7526

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6365 - accuracy: 0.7525

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6336 - accuracy: 0.7549

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6360 - accuracy: 0.7532

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.6311 - accuracy: 0.7547

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6326 - accuracy: 0.7553

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6372 - accuracy: 0.7545

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6439 - accuracy: 0.7536

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6455 - accuracy: 0.7521

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6455 - accuracy: 0.7514

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6464 - accuracy: 0.7520

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6517 - accuracy: 0.7513

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6550 - accuracy: 0.7513

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6535 - accuracy: 0.7519

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6543 - accuracy: 0.7525

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6592 - accuracy: 0.7482

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6656 - accuracy: 0.7447

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6636 - accuracy: 0.7459

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6646 - accuracy: 0.7454

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6710 - accuracy: 0.7444

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6695 - accuracy: 0.7450

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6672 - accuracy: 0.7462

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6702 - accuracy: 0.7452

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6684 - accuracy: 0.7453

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6693 - accuracy: 0.7454

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6713 - accuracy: 0.7449

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6722 - accuracy: 0.7440

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6705 - accuracy: 0.7451

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6720 - accuracy: 0.7452

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6696 - accuracy: 0.7462

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6726 - accuracy: 0.7458

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6737 - accuracy: 0.7454

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6717 - accuracy: 0.7459

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6693 - accuracy: 0.7460

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6691 - accuracy: 0.7460

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6727 - accuracy: 0.7443

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6732 - accuracy: 0.7440

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6720 - accuracy: 0.7445

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6715 - accuracy: 0.7458

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6703 - accuracy: 0.7459

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6712 - accuracy: 0.7455

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6727 - accuracy: 0.7456

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6713 - accuracy: 0.7460

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6690 - accuracy: 0.7473

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6698 - accuracy: 0.7457

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6702 - accuracy: 0.7450

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6702 - accuracy: 0.7455

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6699 - accuracy: 0.7451

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6690 - accuracy: 0.7463

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6715 - accuracy: 0.7449

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6732 - accuracy: 0.7439

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6717 - accuracy: 0.7439

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6743 - accuracy: 0.7433

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6769 - accuracy: 0.7427

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6763 - accuracy: 0.7428

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6821 - accuracy: 0.7398

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.6821 - accuracy: 0.7398 - val_loss: 0.7942 - val_accuracy: 0.6812


.. parsed-literal::

    Epoch 9/15


.. parsed-literal::


 1/92 [..............................] - ETA: 6s - loss: 0.5542 - accuracy: 0.7812

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.6572 - accuracy: 0.7500

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6516 - accuracy: 0.7604

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6615 - accuracy: 0.7344

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.6472 - accuracy: 0.7625

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.6448 - accuracy: 0.7604

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6265 - accuracy: 0.7545

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6567 - accuracy: 0.7422

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6387 - accuracy: 0.7465

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6596 - accuracy: 0.7406

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6548 - accuracy: 0.7415

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6587 - accuracy: 0.7370

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6604 - accuracy: 0.7356

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6458 - accuracy: 0.7433

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6432 - accuracy: 0.7479

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6368 - accuracy: 0.7500

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6318 - accuracy: 0.7574

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6263 - accuracy: 0.7604

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6300 - accuracy: 0.7599

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6349 - accuracy: 0.7578

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6289 - accuracy: 0.7589

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6234 - accuracy: 0.7614

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.6200 - accuracy: 0.7609

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6159 - accuracy: 0.7630

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6139 - accuracy: 0.7625

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6118 - accuracy: 0.7620

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6134 - accuracy: 0.7581

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6103 - accuracy: 0.7567

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6040 - accuracy: 0.7586

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6029 - accuracy: 0.7604

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6022 - accuracy: 0.7621

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6037 - accuracy: 0.7607

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6135 - accuracy: 0.7566

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6141 - accuracy: 0.7574

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6172 - accuracy: 0.7554

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6220 - accuracy: 0.7552

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6279 - accuracy: 0.7525

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6295 - accuracy: 0.7500

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6328 - accuracy: 0.7492

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.6366 - accuracy: 0.7477

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6435 - accuracy: 0.7470

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6465 - accuracy: 0.7448

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6493 - accuracy: 0.7442

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6462 - accuracy: 0.7450

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6506 - accuracy: 0.7437

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6490 - accuracy: 0.7432

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6511 - accuracy: 0.7414

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6528 - accuracy: 0.7415

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6529 - accuracy: 0.7411

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6542 - accuracy: 0.7419

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6546 - accuracy: 0.7408

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6496 - accuracy: 0.7434

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6550 - accuracy: 0.7417

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6544 - accuracy: 0.7413

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6563 - accuracy: 0.7420

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6577 - accuracy: 0.7416

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6562 - accuracy: 0.7429

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6543 - accuracy: 0.7441

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6554 - accuracy: 0.7431

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6605 - accuracy: 0.7401

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6579 - accuracy: 0.7423

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6568 - accuracy: 0.7414

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6570 - accuracy: 0.7412

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6617 - accuracy: 0.7384

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6606 - accuracy: 0.7395

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6609 - accuracy: 0.7392

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6580 - accuracy: 0.7408

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6592 - accuracy: 0.7409

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6564 - accuracy: 0.7424

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6546 - accuracy: 0.7434

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6555 - accuracy: 0.7435

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6536 - accuracy: 0.7444

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6532 - accuracy: 0.7449

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6585 - accuracy: 0.7425

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6595 - accuracy: 0.7426

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6558 - accuracy: 0.7447

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6535 - accuracy: 0.7456

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6509 - accuracy: 0.7472

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6487 - accuracy: 0.7480

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6491 - accuracy: 0.7473

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6480 - accuracy: 0.7477

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6493 - accuracy: 0.7481

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6504 - accuracy: 0.7466

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6491 - accuracy: 0.7474

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6478 - accuracy: 0.7482

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6478 - accuracy: 0.7486

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6470 - accuracy: 0.7504

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6463 - accuracy: 0.7511

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6476 - accuracy: 0.7510

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6475 - accuracy: 0.7507

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6469 - accuracy: 0.7510

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.6469 - accuracy: 0.7510 - val_loss: 0.7705 - val_accuracy: 0.6921


.. parsed-literal::

    Epoch 10/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.5019 - accuracy: 0.8438

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5805 - accuracy: 0.7969

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6209 - accuracy: 0.7604

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6745 - accuracy: 0.7109

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.6841 - accuracy: 0.7125

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.6510 - accuracy: 0.7188

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6254 - accuracy: 0.7411

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6364 - accuracy: 0.7383

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6488 - accuracy: 0.7292

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6263 - accuracy: 0.7406

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6172 - accuracy: 0.7443

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6176 - accuracy: 0.7422

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6043 - accuracy: 0.7452

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6265 - accuracy: 0.7433

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6181 - accuracy: 0.7479

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6257 - accuracy: 0.7520

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6240 - accuracy: 0.7574

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6256 - accuracy: 0.7535

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6189 - accuracy: 0.7566

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6213 - accuracy: 0.7578

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6196 - accuracy: 0.7589

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6144 - accuracy: 0.7642

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.6133 - accuracy: 0.7649

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6115 - accuracy: 0.7669

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6141 - accuracy: 0.7638

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6078 - accuracy: 0.7656

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6107 - accuracy: 0.7639

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6194 - accuracy: 0.7578

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6195 - accuracy: 0.7575

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6170 - accuracy: 0.7604

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6153 - accuracy: 0.7601

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6169 - accuracy: 0.7588

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6183 - accuracy: 0.7576

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6117 - accuracy: 0.7610

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6171 - accuracy: 0.7607

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6148 - accuracy: 0.7613

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6160 - accuracy: 0.7601

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6141 - accuracy: 0.7615

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6149 - accuracy: 0.7612

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.6139 - accuracy: 0.7625

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6141 - accuracy: 0.7630

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6113 - accuracy: 0.7634

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6049 - accuracy: 0.7667

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6052 - accuracy: 0.7670

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6061 - accuracy: 0.7678

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6016 - accuracy: 0.7701

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6008 - accuracy: 0.7703

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6031 - accuracy: 0.7705

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6029 - accuracy: 0.7714

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6044 - accuracy: 0.7691

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6106 - accuracy: 0.7645

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6055 - accuracy: 0.7660

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6085 - accuracy: 0.7645

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6119 - accuracy: 0.7637

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6117 - accuracy: 0.7640

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6108 - accuracy: 0.7649

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6119 - accuracy: 0.7641

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6111 - accuracy: 0.7660

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6074 - accuracy: 0.7667

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6087 - accuracy: 0.7665

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6106 - accuracy: 0.7672

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6161 - accuracy: 0.7659

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6146 - accuracy: 0.7672

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6118 - accuracy: 0.7688

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6131 - accuracy: 0.7681

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6130 - accuracy: 0.7678

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6126 - accuracy: 0.7680

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6167 - accuracy: 0.7655

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6165 - accuracy: 0.7652

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6148 - accuracy: 0.7655

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6191 - accuracy: 0.7635

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6240 - accuracy: 0.7625

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6252 - accuracy: 0.7614

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6270 - accuracy: 0.7596

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6247 - accuracy: 0.7603

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6228 - accuracy: 0.7610

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6224 - accuracy: 0.7609

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6223 - accuracy: 0.7615

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6221 - accuracy: 0.7614

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6233 - accuracy: 0.7608

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6214 - accuracy: 0.7615

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6221 - accuracy: 0.7610

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6238 - accuracy: 0.7616

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6236 - accuracy: 0.7622

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6234 - accuracy: 0.7624

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6231 - accuracy: 0.7630

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6217 - accuracy: 0.7632

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6214 - accuracy: 0.7641

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6248 - accuracy: 0.7639

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6238 - accuracy: 0.7645

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6230 - accuracy: 0.7646

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.6230 - accuracy: 0.7646 - val_loss: 0.7725 - val_accuracy: 0.7153


.. parsed-literal::

    Epoch 11/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.6668 - accuracy: 0.6875

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5528 - accuracy: 0.7656

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5535 - accuracy: 0.7917

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5296 - accuracy: 0.7969

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5133 - accuracy: 0.8062

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.5098 - accuracy: 0.8073

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.5332 - accuracy: 0.8036

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5426 - accuracy: 0.7969

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5770 - accuracy: 0.7778

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6016 - accuracy: 0.7719

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5975 - accuracy: 0.7699

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5897 - accuracy: 0.7734

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5994 - accuracy: 0.7716

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5986 - accuracy: 0.7746

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5995 - accuracy: 0.7708

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6063 - accuracy: 0.7656

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6023 - accuracy: 0.7665

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6027 - accuracy: 0.7656

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5956 - accuracy: 0.7664

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5947 - accuracy: 0.7666

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5901 - accuracy: 0.7672

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.5811 - accuracy: 0.7720

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5810 - accuracy: 0.7737

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5865 - accuracy: 0.7715

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5784 - accuracy: 0.7767

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5786 - accuracy: 0.7745

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5726 - accuracy: 0.7770

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5660 - accuracy: 0.7793

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5637 - accuracy: 0.7805

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5661 - accuracy: 0.7785

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5745 - accuracy: 0.7766

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5816 - accuracy: 0.7748

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5781 - accuracy: 0.7750

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5772 - accuracy: 0.7734

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5729 - accuracy: 0.7753

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5776 - accuracy: 0.7747

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5781 - accuracy: 0.7781

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5773 - accuracy: 0.7790

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.5783 - accuracy: 0.7799

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5778 - accuracy: 0.7799

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5755 - accuracy: 0.7792

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5769 - accuracy: 0.7792

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5776 - accuracy: 0.7786

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5752 - accuracy: 0.7793

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5734 - accuracy: 0.7801

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5765 - accuracy: 0.7787

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5780 - accuracy: 0.7788

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5799 - accuracy: 0.7776

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5836 - accuracy: 0.7770

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5823 - accuracy: 0.7777

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5808 - accuracy: 0.7790

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5838 - accuracy: 0.7778

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5808 - accuracy: 0.7791

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5803 - accuracy: 0.7780

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5826 - accuracy: 0.7775

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5896 - accuracy: 0.7742

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5896 - accuracy: 0.7738

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5893 - accuracy: 0.7739

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5915 - accuracy: 0.7735

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5931 - accuracy: 0.7726

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5908 - accuracy: 0.7733

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5886 - accuracy: 0.7744

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5892 - accuracy: 0.7745

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5897 - accuracy: 0.7736

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5921 - accuracy: 0.7728

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5900 - accuracy: 0.7748

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5871 - accuracy: 0.7763

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5849 - accuracy: 0.7759

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5883 - accuracy: 0.7742

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5860 - accuracy: 0.7752

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5845 - accuracy: 0.7757

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5850 - accuracy: 0.7762

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5829 - accuracy: 0.7771

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5818 - accuracy: 0.7780

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5827 - accuracy: 0.7772

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5818 - accuracy: 0.7761

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5856 - accuracy: 0.7761

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5870 - accuracy: 0.7758

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5876 - accuracy: 0.7743

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5868 - accuracy: 0.7748

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5895 - accuracy: 0.7737

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5908 - accuracy: 0.7727

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5928 - accuracy: 0.7720

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5920 - accuracy: 0.7718

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5914 - accuracy: 0.7726

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5921 - accuracy: 0.7723

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5911 - accuracy: 0.7724

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5913 - accuracy: 0.7725

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5910 - accuracy: 0.7726

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5901 - accuracy: 0.7724

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5883 - accuracy: 0.7725

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.5883 - accuracy: 0.7725 - val_loss: 0.7175 - val_accuracy: 0.7234


.. parsed-literal::

    Epoch 12/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.5461 - accuracy: 0.7812

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.6078 - accuracy: 0.7500

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5296 - accuracy: 0.7917

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5025 - accuracy: 0.8047

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5195 - accuracy: 0.7812

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.4948 - accuracy: 0.7917

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.4886 - accuracy: 0.7857

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5058 - accuracy: 0.7773

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.4985 - accuracy: 0.7882

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.4993 - accuracy: 0.7969

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.4915 - accuracy: 0.8011

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5063 - accuracy: 0.8047

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5257 - accuracy: 0.7981

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5310 - accuracy: 0.7969

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5420 - accuracy: 0.7875

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5323 - accuracy: 0.7910

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5475 - accuracy: 0.7831

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5417 - accuracy: 0.7830

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5430 - accuracy: 0.7829

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5497 - accuracy: 0.7766

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5450 - accuracy: 0.7783

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5501 - accuracy: 0.7784

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.5590 - accuracy: 0.7799

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5482 - accuracy: 0.7865

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5465 - accuracy: 0.7875

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5421 - accuracy: 0.7885

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5454 - accuracy: 0.7882

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5488 - accuracy: 0.7868

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5551 - accuracy: 0.7812

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5555 - accuracy: 0.7802

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5554 - accuracy: 0.7782

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5517 - accuracy: 0.7773

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5521 - accuracy: 0.7794

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5585 - accuracy: 0.7776

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5562 - accuracy: 0.7777

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5507 - accuracy: 0.7812

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5586 - accuracy: 0.7779

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5564 - accuracy: 0.7796

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5534 - accuracy: 0.7812

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.5536 - accuracy: 0.7820

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5572 - accuracy: 0.7805

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5569 - accuracy: 0.7805

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5613 - accuracy: 0.7776

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5597 - accuracy: 0.7784

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5597 - accuracy: 0.7799

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5566 - accuracy: 0.7812

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5596 - accuracy: 0.7793

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5574 - accuracy: 0.7806

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5618 - accuracy: 0.7793

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5593 - accuracy: 0.7806

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5628 - accuracy: 0.7794

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5675 - accuracy: 0.7776

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5661 - accuracy: 0.7783

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5638 - accuracy: 0.7795

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5641 - accuracy: 0.7790

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5638 - accuracy: 0.7785

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5621 - accuracy: 0.7785

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5647 - accuracy: 0.7775

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5642 - accuracy: 0.7770

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5652 - accuracy: 0.7760

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5647 - accuracy: 0.7772

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5618 - accuracy: 0.7792

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5621 - accuracy: 0.7783

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5602 - accuracy: 0.7788

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5594 - accuracy: 0.7793

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5603 - accuracy: 0.7784

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5598 - accuracy: 0.7789

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5633 - accuracy: 0.7776

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5660 - accuracy: 0.7767

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5666 - accuracy: 0.7763

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5642 - accuracy: 0.7777

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5633 - accuracy: 0.7778

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5618 - accuracy: 0.7787

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5634 - accuracy: 0.7779

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5613 - accuracy: 0.7783

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5630 - accuracy: 0.7769

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5655 - accuracy: 0.7761

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5639 - accuracy: 0.7770

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5662 - accuracy: 0.7766

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5626 - accuracy: 0.7783

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5621 - accuracy: 0.7794

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5608 - accuracy: 0.7806

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5608 - accuracy: 0.7806

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5613 - accuracy: 0.7802

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5608 - accuracy: 0.7806

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5642 - accuracy: 0.7803

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5625 - accuracy: 0.7810

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5648 - accuracy: 0.7806

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5615 - accuracy: 0.7824

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5611 - accuracy: 0.7820

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5609 - accuracy: 0.7827

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.5609 - accuracy: 0.7827 - val_loss: 0.6652 - val_accuracy: 0.7357


.. parsed-literal::

    Epoch 13/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.5252 - accuracy: 0.8438

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5595 - accuracy: 0.7969

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5306 - accuracy: 0.8125

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5318 - accuracy: 0.8125

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.4936 - accuracy: 0.8313

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.4675 - accuracy: 0.8438

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.4796 - accuracy: 0.8348

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5024 - accuracy: 0.8164

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.4919 - accuracy: 0.8264

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5071 - accuracy: 0.8219

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5112 - accuracy: 0.8182

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5037 - accuracy: 0.8203

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.4893 - accuracy: 0.8245

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.4904 - accuracy: 0.8281

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.4893 - accuracy: 0.8271

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.4908 - accuracy: 0.8242

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.4963 - accuracy: 0.8180

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.4981 - accuracy: 0.8177

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5066 - accuracy: 0.8141

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5055 - accuracy: 0.8125

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5156 - accuracy: 0.8065

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5282 - accuracy: 0.8054

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.5264 - accuracy: 0.8084

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5195 - accuracy: 0.8099

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5105 - accuracy: 0.8138

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5050 - accuracy: 0.8149

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5063 - accuracy: 0.8171

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5090 - accuracy: 0.8147

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5016 - accuracy: 0.8179

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.4980 - accuracy: 0.8188

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5032 - accuracy: 0.8155

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.4998 - accuracy: 0.8164

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.4974 - accuracy: 0.8163

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5017 - accuracy: 0.8134

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5067 - accuracy: 0.8116

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5030 - accuracy: 0.8134

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.4970 - accuracy: 0.8159

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.4961 - accuracy: 0.8158

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.4938 - accuracy: 0.8165

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.4929 - accuracy: 0.8164

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.4964 - accuracy: 0.8171

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.4964 - accuracy: 0.8155

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.4995 - accuracy: 0.8147

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5069 - accuracy: 0.8118

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5136 - accuracy: 0.8083

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5124 - accuracy: 0.8084

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5139 - accuracy: 0.8072

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5154 - accuracy: 0.8073

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5157 - accuracy: 0.8068

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5195 - accuracy: 0.8062

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5191 - accuracy: 0.8064

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5174 - accuracy: 0.8077

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5186 - accuracy: 0.8078

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5211 - accuracy: 0.8079

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5175 - accuracy: 0.8091

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5185 - accuracy: 0.8092

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5193 - accuracy: 0.8087

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5222 - accuracy: 0.8071

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5193 - accuracy: 0.8088

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5179 - accuracy: 0.8094

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5190 - accuracy: 0.8094

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5203 - accuracy: 0.8095

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5183 - accuracy: 0.8100

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5221 - accuracy: 0.8091

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5257 - accuracy: 0.8077

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5256 - accuracy: 0.8078

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5246 - accuracy: 0.8083

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5257 - accuracy: 0.8065

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5277 - accuracy: 0.8057

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5288 - accuracy: 0.8049

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5315 - accuracy: 0.8041

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5324 - accuracy: 0.8030

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5341 - accuracy: 0.8031

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5340 - accuracy: 0.8032

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5340 - accuracy: 0.8029

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5359 - accuracy: 0.8014

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5358 - accuracy: 0.8019

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5377 - accuracy: 0.8021

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5358 - accuracy: 0.8030

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5352 - accuracy: 0.8035

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5352 - accuracy: 0.8025

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5368 - accuracy: 0.8018

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5345 - accuracy: 0.8027

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5322 - accuracy: 0.8036

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5315 - accuracy: 0.8043

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5294 - accuracy: 0.8044

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5280 - accuracy: 0.8052

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5288 - accuracy: 0.8042

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5304 - accuracy: 0.8043

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5279 - accuracy: 0.8054

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5255 - accuracy: 0.8072

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.5255 - accuracy: 0.8072 - val_loss: 0.7346 - val_accuracy: 0.7384


.. parsed-literal::

    Epoch 14/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.3946 - accuracy: 0.8125

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.4186 - accuracy: 0.8438

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5825 - accuracy: 0.7708

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5210 - accuracy: 0.8047

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5927 - accuracy: 0.7750

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.5649 - accuracy: 0.7865

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.5574 - accuracy: 0.7902

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5343 - accuracy: 0.7969

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5348 - accuracy: 0.7951

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5328 - accuracy: 0.7937

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5417 - accuracy: 0.7898

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5423 - accuracy: 0.7839

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5293 - accuracy: 0.7957

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5237 - accuracy: 0.8013

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5192 - accuracy: 0.7958

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5227 - accuracy: 0.7969

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5219 - accuracy: 0.7996

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5389 - accuracy: 0.7951

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5362 - accuracy: 0.7927

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5310 - accuracy: 0.7937

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5365 - accuracy: 0.7945

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.5421 - accuracy: 0.7940

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5364 - accuracy: 0.7947

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5400 - accuracy: 0.7942

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5380 - accuracy: 0.7949

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5330 - accuracy: 0.7967

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5423 - accuracy: 0.7928

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5431 - accuracy: 0.7935

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5458 - accuracy: 0.7920

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5496 - accuracy: 0.7907

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5505 - accuracy: 0.7884

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5535 - accuracy: 0.7872

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5620 - accuracy: 0.7870

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5640 - accuracy: 0.7878

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5689 - accuracy: 0.7858

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5683 - accuracy: 0.7849

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5653 - accuracy: 0.7848

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5606 - accuracy: 0.7863

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.5612 - accuracy: 0.7862

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5642 - accuracy: 0.7860

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5674 - accuracy: 0.7844

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5612 - accuracy: 0.7880

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5614 - accuracy: 0.7879

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5615 - accuracy: 0.7863

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5615 - accuracy: 0.7855

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5600 - accuracy: 0.7861

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5585 - accuracy: 0.7880

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5641 - accuracy: 0.7846

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5635 - accuracy: 0.7864

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5630 - accuracy: 0.7857

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5604 - accuracy: 0.7874

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5621 - accuracy: 0.7855

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5613 - accuracy: 0.7855

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5630 - accuracy: 0.7842

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5613 - accuracy: 0.7853

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5607 - accuracy: 0.7858

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5580 - accuracy: 0.7873

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5560 - accuracy: 0.7872

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5529 - accuracy: 0.7887

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5507 - accuracy: 0.7896

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5497 - accuracy: 0.7900

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5490 - accuracy: 0.7903

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5505 - accuracy: 0.7887

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5467 - accuracy: 0.7901

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5484 - accuracy: 0.7899

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5478 - accuracy: 0.7903

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5458 - accuracy: 0.7911

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5454 - accuracy: 0.7914

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5482 - accuracy: 0.7899

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5472 - accuracy: 0.7902

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5475 - accuracy: 0.7888

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5450 - accuracy: 0.7904

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5426 - accuracy: 0.7911

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5456 - accuracy: 0.7901

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5460 - accuracy: 0.7892

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5432 - accuracy: 0.7911

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5412 - accuracy: 0.7914

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5420 - accuracy: 0.7909

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5455 - accuracy: 0.7888

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5470 - accuracy: 0.7891

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5454 - accuracy: 0.7898

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5431 - accuracy: 0.7908

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5453 - accuracy: 0.7907

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5463 - accuracy: 0.7906

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5460 - accuracy: 0.7905

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5441 - accuracy: 0.7903

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5444 - accuracy: 0.7899

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5429 - accuracy: 0.7901

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5423 - accuracy: 0.7904

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5448 - accuracy: 0.7893

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5438 - accuracy: 0.7895

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.5438 - accuracy: 0.7895 - val_loss: 0.7761 - val_accuracy: 0.7275


.. parsed-literal::

    Epoch 15/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.5374 - accuracy: 0.8750

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.4638 - accuracy: 0.8906

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.4361 - accuracy: 0.8750

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.4714 - accuracy: 0.8281

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 4s - loss: 0.4472 - accuracy: 0.8375

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.4562 - accuracy: 0.8281

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.4228 - accuracy: 0.8438

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.4377 - accuracy: 0.8359

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.4744 - accuracy: 0.8264

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.4706 - accuracy: 0.8313

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.4714 - accuracy: 0.8324

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.4935 - accuracy: 0.8255

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.4925 - accuracy: 0.8245

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.4784 - accuracy: 0.8281

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.4763 - accuracy: 0.8271

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.4744 - accuracy: 0.8301

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.4797 - accuracy: 0.8254

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.4824 - accuracy: 0.8264

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.4777 - accuracy: 0.8273

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.4756 - accuracy: 0.8297

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.4717 - accuracy: 0.8304

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.4740 - accuracy: 0.8281

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.4742 - accuracy: 0.8261

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.4757 - accuracy: 0.8229

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.4786 - accuracy: 0.8213

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.4786 - accuracy: 0.8209

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.4836 - accuracy: 0.8183

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.4816 - accuracy: 0.8181

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.4780 - accuracy: 0.8190

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.4789 - accuracy: 0.8177

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.4751 - accuracy: 0.8175

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.4703 - accuracy: 0.8184

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.4701 - accuracy: 0.8191

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.4664 - accuracy: 0.8199

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.4676 - accuracy: 0.8196

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.4686 - accuracy: 0.8203

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.4684 - accuracy: 0.8209

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.4648 - accuracy: 0.8199

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.4673 - accuracy: 0.8181

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.4649 - accuracy: 0.8188

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.4656 - accuracy: 0.8186

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.4694 - accuracy: 0.8177

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.4750 - accuracy: 0.8154

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.4788 - accuracy: 0.8118

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.4772 - accuracy: 0.8132

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.4774 - accuracy: 0.8132

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.4781 - accuracy: 0.8145

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.4757 - accuracy: 0.8158

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.4758 - accuracy: 0.8157

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.4741 - accuracy: 0.8169

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.4775 - accuracy: 0.8168

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.4794 - accuracy: 0.8149

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.4896 - accuracy: 0.8125

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.4885 - accuracy: 0.8137

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.4864 - accuracy: 0.8136

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.4861 - accuracy: 0.8136

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.4869 - accuracy: 0.8130

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.4843 - accuracy: 0.8141

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.4882 - accuracy: 0.8130

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.4926 - accuracy: 0.8109

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.4920 - accuracy: 0.8115

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.4945 - accuracy: 0.8095

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.4951 - accuracy: 0.8095

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.4949 - accuracy: 0.8101

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.4933 - accuracy: 0.8096

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.4981 - accuracy: 0.8073

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.4964 - accuracy: 0.8083

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.4934 - accuracy: 0.8093

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.4972 - accuracy: 0.8093

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.4996 - accuracy: 0.8080

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5009 - accuracy: 0.8072

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5067 - accuracy: 0.8064

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5055 - accuracy: 0.8061

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5039 - accuracy: 0.8066

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5070 - accuracy: 0.8046

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5080 - accuracy: 0.8047

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5113 - accuracy: 0.8040

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5107 - accuracy: 0.8045

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5141 - accuracy: 0.8026

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5121 - accuracy: 0.8035

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5107 - accuracy: 0.8044

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5102 - accuracy: 0.8041

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5116 - accuracy: 0.8035

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5117 - accuracy: 0.8032

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5099 - accuracy: 0.8037

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5129 - accuracy: 0.8016

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5155 - accuracy: 0.8006

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5151 - accuracy: 0.8011

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5154 - accuracy: 0.8012

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5143 - accuracy: 0.8017

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5165 - accuracy: 0.8004

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.5165 - accuracy: 0.8004 - val_loss: 0.7822 - val_accuracy: 0.7289



.. image:: 301-tensorflow-training-openvino-nncf-with-output_files/301-tensorflow-training-openvino-nncf-with-output_3_1452.png


.. parsed-literal::


   1/1 [==============================] - ETA: 0s

.. parsed-literal::

    
    1/1 [==============================] - 0s 74ms/step


.. parsed-literal::

    This image most likely belongs to sunflowers with a 99.24 percent confidence.


.. parsed-literal::

    2024-02-10 01:10:41.607321: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'random_flip_input' with dtype float and shape [?,180,180,3]
    	 [[{{node random_flip_input}}]]
    2024-02-10 01:10:41.692936: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-02-10 01:10:41.703478: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'random_flip_input' with dtype float and shape [?,180,180,3]
    	 [[{{node random_flip_input}}]]
    2024-02-10 01:10:41.714441: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-02-10 01:10:41.722136: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-02-10 01:10:41.728943: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-02-10 01:10:41.739850: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-02-10 01:10:41.778944: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'sequential_1_input' with dtype float and shape [?,180,180,3]
    	 [[{{node sequential_1_input}}]]


.. parsed-literal::

    2024-02-10 01:10:41.847949: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-02-10 01:10:41.868730: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'sequential_1_input' with dtype float and shape [?,180,180,3]
    	 [[{{node sequential_1_input}}]]
    2024-02-10 01:10:41.907533: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,22,22,64]
    	 [[{{node inputs}}]]
    2024-02-10 01:10:41.933427: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-02-10 01:10:42.007195: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]


.. parsed-literal::

    2024-02-10 01:10:42.149634: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-02-10 01:10:42.286901: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,22,22,64]
    	 [[{{node inputs}}]]


.. parsed-literal::

    2024-02-10 01:10:42.489392: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-02-10 01:10:42.517820: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-02-10 01:10:42.563861: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
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
    This image most likely belongs to dandelion with a 97.96 percent confidence.



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

    2024-02-10 01:10:45.668839: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [734]
    	 [[{{node Placeholder/_0}}]]
    2024-02-10 01:10:45.669302: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [734]
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
Flow <https://docs.openvino.ai/2023.3/basic_quantization_flow.html>`__.
To use the most advanced quantization flow that allows to apply 8-bit
quantization to the model with accuracy control see `Quantizing with
accuracy
control <https://docs.openvino.ai/2023.3/quantization_w_accuracy_control.html>`__.

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
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/live.py", line 32, in run
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    self.live.refresh()
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/live.py", line 223, in refresh
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    self._live_render.set_renderable(self.renderable)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/live.py", line 203, in renderable
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    renderable = self.get_renderable()
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/live.py", line 98, in get_renderable
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    self._get_renderable()
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1537, in get_renderable
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    renderable = Group(*self.get_renderables())
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1542, in get_renderables
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    table = self.make_tasks_table(self.tasks)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1566, in make_tasks_table
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    table.add_row(
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1571, in &lt;genexpr&gt;
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    else column(task)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 528, in __call__
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    renderable = self.render(task)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/nncf/common/logging/track_progress.py", line 58, in render
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    text = super().render(task)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 787, in render
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    task_time = task.time_remaining
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1039, in time_remaining
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    estimate = ceil(remaining / speed)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/tensorflow/python/util/traceback_utils.py", line 153, in error_handler
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    raise e.with_traceback(filtered_tb) from None
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/tensorflow/python/ops/math_ops.py", line 1569, in _truediv_python3
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    raise TypeError(f"`x` and `y` must have the same dtype, "
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">TypeError: `x` and `y` must have the same dtype, got tf.int64 != tf.float32.
    </pre>









.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>




.. parsed-literal::

    Output()








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
        options=core.available_devices + ["AUTO"],
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

    Accuracy of the original model: 0.729
    Accuracy of the quantized model: 0.729


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
inference on the quantized model with OpenVINO. See the
`OpenVINO API tutorial <002-openvino-api-with-output.html>`__
for more information about running inference with OpenVINO
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
    # model_pot = ie.read_model(model="model/optimized/flower_ir.xml")
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
    This image most likely belongs to dandelion with a 98.03 percent confidence.



.. image:: 301-tensorflow-training-openvino-nncf-with-output_files/301-tensorflow-training-openvino-nncf-with-output_27_1.png


Compare Inference Speed
-----------------------



Measure inference speed with the `OpenVINO Benchmark
App <https://docs.openvino.ai/2023.3/openvino_sample_benchmark_tool.html>`__.

Benchmark App is a command line tool that measures raw inference
performance for a specified OpenVINO IR model. Run
``benchmark_app --help`` to see a list of available parameters. By
default, Benchmark App tests the performance of the model specified with
the ``-m`` parameter with asynchronous inference on CPU, for one minute.
Use the ``-d`` parameter to test performance on a different device, for
example an Intel integrated Graphics (iGPU), and ``-t`` to set the
number of seconds to run inference. See the
`documentation <https://docs.openvino.ai/2023.3/openvino_sample_benchmark_tool.html>`__
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
    print(core.get_property("CPU", "FULL_DEVICE_NAME"))
    if "GPU" in core.available_devices:
        print(core.get_property("GPU", "FULL_DEVICE_NAME"))


.. parsed-literal::

    Device information:
    Intel(R) Core(TM) i9-10920X CPU @ 3.50GHz


.. code:: ipython3

    # Original model - CPU
    ! benchmark_app -m $model_xml -d CPU -t 15 -api async


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2023.3.0-13775-ceeafaf64f3-releases/2023/3
    [ INFO ]
    [ INFO ] Device info:


.. parsed-literal::

    [ INFO ] CPU
    [ INFO ] Build ................................. 2023.3.0-13775-ceeafaf64f3-releases/2023/3
    [ INFO ]
    [ INFO ]
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(CPU) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 12.98 ms
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

    [ INFO ] Compile model took 71.95 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: TensorFlow_Frontend_IR
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 12
    [ INFO ]   NUM_STREAMS: 12
    [ INFO ]   AFFINITY: Affinity.CORE
    [ INFO ]   INFERENCE_NUM_THREADS: 24
    [ INFO ]   PERF_COUNT: NO
    [ INFO ]   INFERENCE_PRECISION_HINT: <Type: 'float32'>
    [ INFO ]   PERFORMANCE_HINT: THROUGHPUT
    [ INFO ]   EXECUTION_MODE_HINT: ExecutionMode.PERFORMANCE
    [ INFO ]   PERFORMANCE_HINT_NUM_REQUESTS: 0
    [ INFO ]   ENABLE_CPU_PINNING: True
    [ INFO ]   SCHEDULING_CORE_TYPE: SchedulingCoreType.ANY_CORE
    [ INFO ]   ENABLE_HYPER_THREADING: True
    [ INFO ]   EXECUTION_DEVICES: ['CPU']
    [ INFO ]   CPU_DENORMALS_OPTIMIZATION: False
    [ INFO ]   CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE: 1.0
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'sequential_1_input'!. This input will be filled with random values!
    [ INFO ] Fill input 'sequential_1_input' with random values
    [Step 10/11] Measuring performance (Start inference asynchronously, 12 inference requests, limits: 15000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 7.48 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            57660 iterations
    [ INFO ] Duration:         15004.59 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        2.95 ms
    [ INFO ]    Average:       2.95 ms
    [ INFO ]    Min:           1.69 ms
    [ INFO ]    Max:           12.80 ms
    [ INFO ] Throughput:   3842.82 FPS


.. code:: ipython3

    # Quantized model - CPU
    ! benchmark_app -m $compressed_model_xml -d CPU -t 15 -api async


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2023.3.0-13775-ceeafaf64f3-releases/2023/3
    [ INFO ]
    [ INFO ] Device info:


.. parsed-literal::

    [ INFO ] CPU
    [ INFO ] Build ................................. 2023.3.0-13775-ceeafaf64f3-releases/2023/3
    [ INFO ]
    [ INFO ]
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(CPU) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 15.15 ms
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

    [ INFO ] Compile model took 67.57 ms
    [Step 8/11] Querying optimal runtime parameters


.. parsed-literal::

    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: TensorFlow_Frontend_IR
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 12
    [ INFO ]   NUM_STREAMS: 12
    [ INFO ]   AFFINITY: Affinity.CORE
    [ INFO ]   INFERENCE_NUM_THREADS: 24
    [ INFO ]   PERF_COUNT: NO
    [ INFO ]   INFERENCE_PRECISION_HINT: <Type: 'float32'>
    [ INFO ]   PERFORMANCE_HINT: THROUGHPUT
    [ INFO ]   EXECUTION_MODE_HINT: ExecutionMode.PERFORMANCE
    [ INFO ]   PERFORMANCE_HINT_NUM_REQUESTS: 0
    [ INFO ]   ENABLE_CPU_PINNING: True
    [ INFO ]   SCHEDULING_CORE_TYPE: SchedulingCoreType.ANY_CORE
    [ INFO ]   ENABLE_HYPER_THREADING: True
    [ INFO ]   EXECUTION_DEVICES: ['CPU']
    [ INFO ]   CPU_DENORMALS_OPTIMIZATION: False
    [ INFO ]   CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE: 1.0
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'sequential_1_input'!. This input will be filled with random values!
    [ INFO ] Fill input 'sequential_1_input' with random values
    [Step 10/11] Measuring performance (Start inference asynchronously, 12 inference requests, limits: 15000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 1.99 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            178152 iterations
    [ INFO ] Duration:         15001.85 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        0.94 ms
    [ INFO ]    Average:       0.98 ms
    [ INFO ]    Min:           0.55 ms
    [ INFO ]    Max:           11.77 ms
    [ INFO ] Throughput:   11875.34 FPS


**Benchmark on MULTI:CPU,GPU**

With a recent Intel CPU, the best performance can often be achieved by
doing inference on both the CPU and the iGPU, with OpenVINO’s `Multi
Device
Plugin <https://docs.openvino.ai/2021.4/openvino_docs_IE_DG_supported_plugins_MULTI.html>`__.
It takes a bit longer to load a model on GPU than on CPU, so this
benchmark will take a bit longer to complete than the CPU benchmark,
when run for the first time. Benchmark App supports caching, by
specifying the ``--cdir`` parameter. In the cells below, the model will
cached to the ``model_cache`` directory.

.. code:: ipython3

    # Original model - MULTI:CPU,GPU
    if "GPU" in core.available_devices:
        ! benchmark_app -m $model_xml -d MULTI:CPU,GPU -t 15 -api async
    else:
        print("A supported integrated GPU is not available on this system.")


.. parsed-literal::

    A supported integrated GPU is not available on this system.


.. code:: ipython3

    # Quantized model - MULTI:CPU,GPU
    if "GPU" in core.available_devices:
        ! benchmark_app -m $compressed_model_xml -d MULTI:CPU,GPU -t 15 -api async
    else:
        print("A supported integrated GPU is not available on this system.")


.. parsed-literal::

    A supported integrated GPU is not available on this system.


.. code:: ipython3

    # print the available devices on this system
    print("Device information:")
    print(core.get_property("CPU", "FULL_DEVICE_NAME"))
    if "GPU" in core.available_devices:
        print(core.get_property("GPU", "FULL_DEVICE_NAME"))


.. parsed-literal::

    Device information:
    Intel(R) Core(TM) i9-10920X CPU @ 3.50GHz


**Original IR model - CPU**

.. code:: ipython3

    benchmark_output = %sx benchmark_app -m $model_xml -t 15 -api async
    # Remove logging info from benchmark_app output and show only the results
    benchmark_result = benchmark_output[-8:]
    print("\n".join(benchmark_result))


.. parsed-literal::

    [ INFO ] Count:            57840 iterations
    [ INFO ] Duration:         15004.24 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        2.94 ms
    [ INFO ]    Average:       2.94 ms
    [ INFO ]    Min:           1.98 ms
    [ INFO ]    Max:           12.12 ms
    [ INFO ] Throughput:   3854.91 FPS


**Quantized IR model - CPU**

.. code:: ipython3

    benchmark_output = %sx benchmark_app -m $compressed_model_xml -t 15 -api async
    # Remove logging info from benchmark_app output and show only the results
    benchmark_result = benchmark_output[-8:]
    print("\n".join(benchmark_result))


.. parsed-literal::

    [ INFO ] Count:            178836 iterations
    [ INFO ] Duration:         15001.19 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        0.94 ms
    [ INFO ]    Average:       0.97 ms
    [ INFO ]    Min:           0.58 ms
    [ INFO ]    Max:           6.85 ms
    [ INFO ] Throughput:   11921.45 FPS


**Original IR model - MULTI:CPU,GPU**

With a recent Intel CPU, the best performance can often be achieved by
doing inference on both the CPU and the iGPU, with OpenVINO’s `Multi
Device
Plugin <https://docs.openvino.ai/2023.3/openvino_docs_OV_UG_Running_on_multiple_devices.html>`__.
It takes a bit longer to load a model on GPU than on CPU, so this
benchmark will take a bit longer to complete than the CPU benchmark.

.. code:: ipython3

    if "GPU" in core.available_devices:
        benchmark_output = %sx benchmark_app -m $model_xml -d MULTI:CPU,GPU -t 15 -api async
        # Remove logging info from benchmark_app output and show only the results
        benchmark_result = benchmark_output[-8:]
        print("\n".join(benchmark_result))
    else:
        print("An GPU is not available on this system.")


.. parsed-literal::

    An GPU is not available on this system.


**Quantized IR model - MULTI:CPU,GPU**

.. code:: ipython3

    if "GPU" in core.available_devices:
        benchmark_output = %sx benchmark_app -m $compressed_model_xml -d MULTI:CPU,GPU -t 15 -api async
        # Remove logging info from benchmark_app output and show only the results
        benchmark_result = benchmark_output[-8:]
        print("\n".join(benchmark_result))
    else:
        print("An GPU is not available on this system.")


.. parsed-literal::

    An GPU is not available on this system.

