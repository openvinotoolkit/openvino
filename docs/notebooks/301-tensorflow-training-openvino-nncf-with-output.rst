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

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.0 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063


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

    2024-01-19 00:28:23.092118: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-01-19 00:28:23.125973: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


.. parsed-literal::

    2024-01-19 00:28:23.720607: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


.. parsed-literal::

    Executing training notebook. This will take a while...


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.0 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    3670


.. parsed-literal::

    Found 3670 files belonging to 5 classes.


.. parsed-literal::

    Using 2936 files for training.


.. parsed-literal::

    2024-01-19 00:28:30.598646: E tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:266] failed call to cuInit: CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: forward compatibility was attempted on non supported HW
    2024-01-19 00:28:30.598682: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:168] retrieving CUDA diagnostic information for host: iotg-dev-workstation-07
    2024-01-19 00:28:30.598688: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:175] hostname: iotg-dev-workstation-07
    2024-01-19 00:28:30.598816: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:199] libcuda reported version is: 470.223.2
    2024-01-19 00:28:30.598833: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:203] kernel reported version is: 470.182.3
    2024-01-19 00:28:30.598837: E tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:312] kernel version 470.182.3 does not match DSO version 470.223.2 -- cannot find working devices in this configuration


.. parsed-literal::

    Found 3670 files belonging to 5 classes.


.. parsed-literal::

    Using 734 files for validation.
    ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']


.. parsed-literal::

    2024-01-19 00:28:30.904530: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-01-19 00:28:30.904799: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]



.. image:: 301-tensorflow-training-openvino-nncf-with-output_files/301-tensorflow-training-openvino-nncf-with-output_3_12.png


.. parsed-literal::

    2024-01-19 00:28:31.769096: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-01-19 00:28:31.769340: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-01-19 00:28:31.941939: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-01-19 00:28:31.942303: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]


.. parsed-literal::

    (32, 180, 180, 3)
    (32,)


.. parsed-literal::

    0.0 1.0


.. parsed-literal::

    2024-01-19 00:28:32.827640: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-01-19 00:28:32.827943: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]



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

    2024-01-19 00:28:33.838211: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]
    2024-01-19 00:28:33.838608: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]


.. parsed-literal::


 1/92 [..............................] - ETA: 1:25 - loss: 1.6054 - accuracy: 0.2188

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 3.3172 - accuracy: 0.1875

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 2.7950 - accuracy: 0.1771

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 2.5016 - accuracy: 0.2188

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 2.3072 - accuracy: 0.2500

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 2.2015 - accuracy: 0.2448

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 5s - loss: 2.1411 - accuracy: 0.2411

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 5s - loss: 2.0920 - accuracy: 0.2422

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 2.0540 - accuracy: 0.2326

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 2.0176 - accuracy: 0.2188

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 1.9728 - accuracy: 0.2273

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 1.9401 - accuracy: 0.2266

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 1.9122 - accuracy: 0.2236

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 1.8887 - accuracy: 0.2188

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 1.8683 - accuracy: 0.2229

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 1.8486 - accuracy: 0.2246

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 1.8308 - accuracy: 0.2243

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 1.8171 - accuracy: 0.2257

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 1.8070 - accuracy: 0.2300

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 1.7944 - accuracy: 0.2294

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 1.7822 - accuracy: 0.2364

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 1.7719 - accuracy: 0.2371

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 1.7654 - accuracy: 0.2376

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 4s - loss: 1.7560 - accuracy: 0.2382

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 1.7467 - accuracy: 0.2399

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 1.7397 - accuracy: 0.2427

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 1.7338 - accuracy: 0.2442

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 1.7247 - accuracy: 0.2466

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 1.7169 - accuracy: 0.2533

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 1.7096 - accuracy: 0.2542

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 1.7025 - accuracy: 0.2581

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 1.6932 - accuracy: 0.2638

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 1.6825 - accuracy: 0.2672

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 1.6768 - accuracy: 0.2694

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 1.6696 - accuracy: 0.2779

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 1.6661 - accuracy: 0.2780

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 1.6552 - accuracy: 0.2866

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 1.6467 - accuracy: 0.2930

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 1.6407 - accuracy: 0.2960

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 1.6311 - accuracy: 0.3019

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 1.6221 - accuracy: 0.3067

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 1.6141 - accuracy: 0.3106

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 1.6076 - accuracy: 0.3136

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 1.6031 - accuracy: 0.3129

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 1.5925 - accuracy: 0.3177

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 1.5834 - accuracy: 0.3231

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 1.5841 - accuracy: 0.3249

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 1.5770 - accuracy: 0.3272

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 1.5696 - accuracy: 0.3295

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 1.5619 - accuracy: 0.3354

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 1.5526 - accuracy: 0.3381

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 1.5446 - accuracy: 0.3400

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 1.5373 - accuracy: 0.3424

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 1.5326 - accuracy: 0.3424

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 1.5286 - accuracy: 0.3425

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 1.5218 - accuracy: 0.3436

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 1.5158 - accuracy: 0.3447

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 1.5096 - accuracy: 0.3469

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 1.5026 - accuracy: 0.3500

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 1.4953 - accuracy: 0.3541

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 1.4881 - accuracy: 0.3575

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 1.4845 - accuracy: 0.3568

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 1.4835 - accuracy: 0.3581

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 1.4787 - accuracy: 0.3613

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 1.4736 - accuracy: 0.3629

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 1.4668 - accuracy: 0.3664

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 1.4615 - accuracy: 0.3689

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 1.4528 - accuracy: 0.3741

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 1.4467 - accuracy: 0.3777

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 1.4439 - accuracy: 0.3781

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 1.4405 - accuracy: 0.3794

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 1.4375 - accuracy: 0.3807

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 1.4333 - accuracy: 0.3845

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 1.4285 - accuracy: 0.3869

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 1.4240 - accuracy: 0.3892

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 1.4216 - accuracy: 0.3894

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 1.4206 - accuracy: 0.3921

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 1.4234 - accuracy: 0.3907

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 1.4240 - accuracy: 0.3913

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 1.4200 - accuracy: 0.3942

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 1.4154 - accuracy: 0.3967

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 1.4134 - accuracy: 0.3960

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 1.4126 - accuracy: 0.3980

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 1.4086 - accuracy: 0.4004

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 1.4042 - accuracy: 0.4034

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 1.4001 - accuracy: 0.4052

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 1.3971 - accuracy: 0.4056

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 1.3935 - accuracy: 0.4060

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 1.3913 - accuracy: 0.4070

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 1.3901 - accuracy: 0.4074

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 1.3887 - accuracy: 0.4084

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 1.3861 - accuracy: 0.4101

.. parsed-literal::

    2024-01-19 00:28:40.056852: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [734]
    	 [[{{node Placeholder/_4}}]]
    2024-01-19 00:28:40.057128: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [734]
    	 [[{{node Placeholder/_4}}]]


.. parsed-literal::

    
92/92 [==============================] - 7s 66ms/step - loss: 1.3861 - accuracy: 0.4101 - val_loss: 1.0463 - val_accuracy: 0.5954


.. parsed-literal::

    Epoch 2/15


.. parsed-literal::


 1/92 [..............................] - ETA: 8s - loss: 1.4412 - accuracy: 0.3438

.. parsed-literal::

    
 2/92 [..............................] - ETA: 6s - loss: 1.2551 - accuracy: 0.4844

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 1.2595 - accuracy: 0.5104

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 1.2418 - accuracy: 0.5234

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 1.2431 - accuracy: 0.5188

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 1.1879 - accuracy: 0.5469

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 1.1770 - accuracy: 0.5524

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 1.1701 - accuracy: 0.5429

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 1.1640 - accuracy: 0.5417

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 1.1399 - accuracy: 0.5581

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 1.1471 - accuracy: 0.5559

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 1.1395 - accuracy: 0.5588

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 1.1443 - accuracy: 0.5591

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 1.1549 - accuracy: 0.5572

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 1.1455 - accuracy: 0.5635

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 1.1393 - accuracy: 0.5653

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 1.1324 - accuracy: 0.5687

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 1.1285 - accuracy: 0.5667

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 1.1182 - accuracy: 0.5680

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 1.1052 - accuracy: 0.5678

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 1.0954 - accuracy: 0.5761

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 1.0930 - accuracy: 0.5769

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 1.1001 - accuracy: 0.5724

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 1.0994 - accuracy: 0.5720

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 1.0992 - accuracy: 0.5716

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 1.0997 - accuracy: 0.5724

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 1.0972 - accuracy: 0.5732

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 1.0901 - accuracy: 0.5761

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 1.0902 - accuracy: 0.5735

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 1.0933 - accuracy: 0.5722

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 1.0873 - accuracy: 0.5768

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 1.0859 - accuracy: 0.5754

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 1.0893 - accuracy: 0.5713

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 1.0849 - accuracy: 0.5710

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 1.0791 - accuracy: 0.5760

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 1.0745 - accuracy: 0.5782

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 1.0710 - accuracy: 0.5795

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 1.0641 - accuracy: 0.5831

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 1.0660 - accuracy: 0.5818

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 1.0600 - accuracy: 0.5821

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 1.0589 - accuracy: 0.5823

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 1.0586 - accuracy: 0.5855

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 1.0549 - accuracy: 0.5893

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 1.0585 - accuracy: 0.5873

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 1.0568 - accuracy: 0.5861

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 1.0531 - accuracy: 0.5889

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 1.0566 - accuracy: 0.5890

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 1.0538 - accuracy: 0.5878

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 1.0568 - accuracy: 0.5854

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 1.0535 - accuracy: 0.5862

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 1.0524 - accuracy: 0.5845

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 1.0498 - accuracy: 0.5841

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 1.0541 - accuracy: 0.5843

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 1.0476 - accuracy: 0.5868

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 1.0463 - accuracy: 0.5852

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 1.0393 - accuracy: 0.5887

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 1.0358 - accuracy: 0.5898

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 1.0343 - accuracy: 0.5904

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 1.0278 - accuracy: 0.5931

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 1.0283 - accuracy: 0.5931

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 1.0271 - accuracy: 0.5931

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 1.0279 - accuracy: 0.5941

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 1.0269 - accuracy: 0.5936

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 1.0260 - accuracy: 0.5956

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 1.0285 - accuracy: 0.5941

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 1.0257 - accuracy: 0.5960

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 1.0232 - accuracy: 0.5978

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 1.0207 - accuracy: 0.5991

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 1.0209 - accuracy: 0.5990

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 1.0214 - accuracy: 0.5985

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 1.0238 - accuracy: 0.5976

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 1.0244 - accuracy: 0.5975

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 1.0208 - accuracy: 0.6000

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 1.0217 - accuracy: 0.5987

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 1.0217 - accuracy: 0.5990

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 1.0189 - accuracy: 0.5998

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 1.0201 - accuracy: 0.5989

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 1.0227 - accuracy: 0.5972

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 1.0223 - accuracy: 0.5976

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 1.0205 - accuracy: 0.6002

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 1.0216 - accuracy: 0.6002

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 1.0258 - accuracy: 0.5974

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 1.0259 - accuracy: 0.5970

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 1.0235 - accuracy: 0.5973

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 1.0242 - accuracy: 0.5977

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 1.0248 - accuracy: 0.5973

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 1.0254 - accuracy: 0.5972

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 1.0293 - accuracy: 0.5951

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 1.0295 - accuracy: 0.5947

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 1.0285 - accuracy: 0.5947

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 1.0278 - accuracy: 0.5954

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 1.0278 - accuracy: 0.5954 - val_loss: 1.0710 - val_accuracy: 0.5804


.. parsed-literal::

    Epoch 3/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 1.0578 - accuracy: 0.6562

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.9947 - accuracy: 0.5938

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.9343 - accuracy: 0.6250

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.9289 - accuracy: 0.6250

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.9037 - accuracy: 0.6313

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.9172 - accuracy: 0.6354

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.8735 - accuracy: 0.6607

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.8584 - accuracy: 0.6680

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.8784 - accuracy: 0.6597

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.8901 - accuracy: 0.6594

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.8971 - accuracy: 0.6562

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.8893 - accuracy: 0.6641

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.9167 - accuracy: 0.6466

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.9154 - accuracy: 0.6518

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.9394 - accuracy: 0.6396

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.9537 - accuracy: 0.6406

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.9384 - accuracy: 0.6434

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.9308 - accuracy: 0.6424

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.9313 - accuracy: 0.6365

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.9421 - accuracy: 0.6328

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.9430 - accuracy: 0.6295

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.9394 - accuracy: 0.6321

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.9454 - accuracy: 0.6304

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 4s - loss: 0.9451 - accuracy: 0.6289

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.9393 - accuracy: 0.6325

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.9342 - accuracy: 0.6334

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.9283 - accuracy: 0.6377

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.9306 - accuracy: 0.6384

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.9408 - accuracy: 0.6336

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.9388 - accuracy: 0.6302

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.9403 - accuracy: 0.6270

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.9390 - accuracy: 0.6279

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.9407 - accuracy: 0.6278

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.9462 - accuracy: 0.6250

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.9458 - accuracy: 0.6277

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.9477 - accuracy: 0.6285

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.9553 - accuracy: 0.6242

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.9545 - accuracy: 0.6234

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.9577 - accuracy: 0.6234

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.9563 - accuracy: 0.6242

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.9553 - accuracy: 0.6265

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.9585 - accuracy: 0.6228

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.9537 - accuracy: 0.6235

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.9524 - accuracy: 0.6250

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.9532 - accuracy: 0.6222

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.9534 - accuracy: 0.6236

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.9541 - accuracy: 0.6250

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.9590 - accuracy: 0.6211

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.9570 - accuracy: 0.6237

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.9549 - accuracy: 0.6231

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.9554 - accuracy: 0.6232

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.9510 - accuracy: 0.6256

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.9490 - accuracy: 0.6262

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.9454 - accuracy: 0.6279

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.9438 - accuracy: 0.6284

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.9468 - accuracy: 0.6272

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.9425 - accuracy: 0.6288

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.9384 - accuracy: 0.6304

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.9416 - accuracy: 0.6298

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.9417 - accuracy: 0.6297

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.9384 - accuracy: 0.6306

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.9486 - accuracy: 0.6270

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.9508 - accuracy: 0.6260

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.9489 - accuracy: 0.6274

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.9505 - accuracy: 0.6274

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.9553 - accuracy: 0.6245

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.9539 - accuracy: 0.6250

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.9520 - accuracy: 0.6255

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.9524 - accuracy: 0.6255

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.9523 - accuracy: 0.6259

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.9585 - accuracy: 0.6237

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.9570 - accuracy: 0.6241

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.9561 - accuracy: 0.6259

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.9563 - accuracy: 0.6263

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.9552 - accuracy: 0.6279

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.9535 - accuracy: 0.6287

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.9552 - accuracy: 0.6278

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.9535 - accuracy: 0.6286

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.9506 - accuracy: 0.6297

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.9490 - accuracy: 0.6301

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.9473 - accuracy: 0.6307

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.9458 - accuracy: 0.6318

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.9454 - accuracy: 0.6317

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.9467 - accuracy: 0.6309

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.9506 - accuracy: 0.6297

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.9498 - accuracy: 0.6304

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.9476 - accuracy: 0.6311

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.9448 - accuracy: 0.6331

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.9478 - accuracy: 0.6323

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.9460 - accuracy: 0.6322

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.9444 - accuracy: 0.6339

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.9444 - accuracy: 0.6339 - val_loss: 0.9729 - val_accuracy: 0.6144


.. parsed-literal::

    Epoch 4/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 1.0013 - accuracy: 0.5938

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.9012 - accuracy: 0.6250

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.8791 - accuracy: 0.6458

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 1.0399 - accuracy: 0.5859

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.9726 - accuracy: 0.6125

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.9339 - accuracy: 0.6250

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.9365 - accuracy: 0.6295

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.9090 - accuracy: 0.6445

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.8858 - accuracy: 0.6632

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.8865 - accuracy: 0.6687

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.8895 - accuracy: 0.6648

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.8848 - accuracy: 0.6667

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.8808 - accuracy: 0.6587

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.8722 - accuracy: 0.6585

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.8711 - accuracy: 0.6646

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.8847 - accuracy: 0.6641

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.8829 - accuracy: 0.6654

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.8798 - accuracy: 0.6684

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.8716 - accuracy: 0.6711

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.8578 - accuracy: 0.6777

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 3s - loss: 0.8572 - accuracy: 0.6810

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.8579 - accuracy: 0.6799

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.8509 - accuracy: 0.6816

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.8493 - accuracy: 0.6818

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.8535 - accuracy: 0.6796

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.8527 - accuracy: 0.6811

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.8520 - accuracy: 0.6813

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.8582 - accuracy: 0.6750

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.8548 - accuracy: 0.6775

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.8514 - accuracy: 0.6778

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.8546 - accuracy: 0.6762

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.8591 - accuracy: 0.6708

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.8568 - accuracy: 0.6722

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.8536 - accuracy: 0.6727

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.8552 - accuracy: 0.6713

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.8623 - accuracy: 0.6692

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.8603 - accuracy: 0.6689

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.8566 - accuracy: 0.6702

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.8547 - accuracy: 0.6714

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.8599 - accuracy: 0.6710

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.8669 - accuracy: 0.6692

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.8689 - accuracy: 0.6703

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.8718 - accuracy: 0.6679

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.8739 - accuracy: 0.6669

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.8745 - accuracy: 0.6673

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.8740 - accuracy: 0.6678

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.8726 - accuracy: 0.6682

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.8740 - accuracy: 0.6679

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.8737 - accuracy: 0.6683

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.8746 - accuracy: 0.6687

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.8760 - accuracy: 0.6667

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.8718 - accuracy: 0.6677

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.8713 - accuracy: 0.6686

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.8754 - accuracy: 0.6661

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.8792 - accuracy: 0.6642

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.8790 - accuracy: 0.6630

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.8766 - accuracy: 0.6623

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.8772 - accuracy: 0.6622

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.8795 - accuracy: 0.6611

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.8835 - accuracy: 0.6600

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.8821 - accuracy: 0.6589

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.8825 - accuracy: 0.6589

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.8814 - accuracy: 0.6598

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.8852 - accuracy: 0.6569

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.8856 - accuracy: 0.6568

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.8879 - accuracy: 0.6559

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.8852 - accuracy: 0.6582

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.8806 - accuracy: 0.6600

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.8794 - accuracy: 0.6613

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.8806 - accuracy: 0.6621

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.8834 - accuracy: 0.6616

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.8814 - accuracy: 0.6624

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.8792 - accuracy: 0.6636

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.8794 - accuracy: 0.6630

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.8829 - accuracy: 0.6617

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.8875 - accuracy: 0.6592

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.8912 - accuracy: 0.6572

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.8899 - accuracy: 0.6575

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.8868 - accuracy: 0.6579

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.8857 - accuracy: 0.6591

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.8853 - accuracy: 0.6586

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.8877 - accuracy: 0.6582

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.8869 - accuracy: 0.6593

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.8892 - accuracy: 0.6578

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.8894 - accuracy: 0.6578

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.8906 - accuracy: 0.6563

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.8925 - accuracy: 0.6560

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.8928 - accuracy: 0.6556

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.8919 - accuracy: 0.6570

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.8912 - accuracy: 0.6574

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.8903 - accuracy: 0.6570

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.8903 - accuracy: 0.6570 - val_loss: 0.8825 - val_accuracy: 0.6553


.. parsed-literal::

    Epoch 5/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.7030 - accuracy: 0.7500

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.8057 - accuracy: 0.6719

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.8193 - accuracy: 0.6875

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.8461 - accuracy: 0.6797

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.8406 - accuracy: 0.6812

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.8467 - accuracy: 0.6823

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.8486 - accuracy: 0.6786

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.8296 - accuracy: 0.6953

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.8424 - accuracy: 0.6840

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.8555 - accuracy: 0.6719

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.8475 - accuracy: 0.6847

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.8806 - accuracy: 0.6719

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.8643 - accuracy: 0.6803

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.8647 - accuracy: 0.6830

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.8604 - accuracy: 0.6771

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.8383 - accuracy: 0.6855

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.8208 - accuracy: 0.6967

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.8334 - accuracy: 0.6892

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.8309 - accuracy: 0.6924

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.8258 - accuracy: 0.6891

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.8342 - accuracy: 0.6905

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.8279 - accuracy: 0.6918

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.8269 - accuracy: 0.6875

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.8319 - accuracy: 0.6862

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.8271 - accuracy: 0.6875

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.8244 - accuracy: 0.6863

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.8199 - accuracy: 0.6910

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.8250 - accuracy: 0.6886

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.8302 - accuracy: 0.6853

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.8322 - accuracy: 0.6812

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.8357 - accuracy: 0.6784

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.8338 - accuracy: 0.6777

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.8311 - accuracy: 0.6818

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.8310 - accuracy: 0.6811

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.8323 - accuracy: 0.6821

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.8293 - accuracy: 0.6858

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.8362 - accuracy: 0.6833

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.8372 - accuracy: 0.6817

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.8403 - accuracy: 0.6827

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.8362 - accuracy: 0.6859

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.8390 - accuracy: 0.6829

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.8428 - accuracy: 0.6830

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.8403 - accuracy: 0.6839

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.8401 - accuracy: 0.6839

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.8363 - accuracy: 0.6840

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.8427 - accuracy: 0.6821

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.8442 - accuracy: 0.6795

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.8466 - accuracy: 0.6797

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.8428 - accuracy: 0.6798

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.8458 - accuracy: 0.6775

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.8480 - accuracy: 0.6777

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.8463 - accuracy: 0.6773

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.8460 - accuracy: 0.6775

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.8478 - accuracy: 0.6759

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.8467 - accuracy: 0.6767

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.8474 - accuracy: 0.6741

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.8447 - accuracy: 0.6749

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.8490 - accuracy: 0.6735

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.8484 - accuracy: 0.6753

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.8483 - accuracy: 0.6755

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.8515 - accuracy: 0.6737

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.8504 - accuracy: 0.6729

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.8496 - accuracy: 0.6716

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.8507 - accuracy: 0.6719

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.8476 - accuracy: 0.6736

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.8460 - accuracy: 0.6752

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.8512 - accuracy: 0.6707

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.8499 - accuracy: 0.6733

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.8522 - accuracy: 0.6726

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.8511 - accuracy: 0.6728

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.8511 - accuracy: 0.6730

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.8492 - accuracy: 0.6736

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.8483 - accuracy: 0.6725

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.8481 - accuracy: 0.6736

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.8479 - accuracy: 0.6746

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.8458 - accuracy: 0.6764

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.8480 - accuracy: 0.6757

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.8449 - accuracy: 0.6767

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.8455 - accuracy: 0.6768

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.8459 - accuracy: 0.6766

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.8452 - accuracy: 0.6767

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.8430 - accuracy: 0.6784

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.8432 - accuracy: 0.6787

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.8434 - accuracy: 0.6777

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.8406 - accuracy: 0.6782

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.8425 - accuracy: 0.6790

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.8386 - accuracy: 0.6809

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.8394 - accuracy: 0.6803

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.8376 - accuracy: 0.6807

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.8399 - accuracy: 0.6804

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.8388 - accuracy: 0.6805

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.8388 - accuracy: 0.6805 - val_loss: 0.7987 - val_accuracy: 0.6771


.. parsed-literal::

    Epoch 6/15


.. parsed-literal::


 1/92 [..............................] - ETA: 6s - loss: 0.7664 - accuracy: 0.7812

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.7765 - accuracy: 0.7812

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.9823 - accuracy: 0.6667

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.9155 - accuracy: 0.7109

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.8833 - accuracy: 0.7174

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.8623 - accuracy: 0.7130

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.8366 - accuracy: 0.7177

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.8058 - accuracy: 0.7321

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.8144 - accuracy: 0.7179

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.8255 - accuracy: 0.7151

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.8285 - accuracy: 0.7074

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.8371 - accuracy: 0.7034

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.8416 - accuracy: 0.6932

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.8253 - accuracy: 0.7013

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.8252 - accuracy: 0.6984

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.8357 - accuracy: 0.6847

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.8329 - accuracy: 0.6849

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.8406 - accuracy: 0.6817

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.8493 - accuracy: 0.6772

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.8449 - accuracy: 0.6762

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.8292 - accuracy: 0.6825

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.8183 - accuracy: 0.6882

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.8152 - accuracy: 0.6908

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.8105 - accuracy: 0.6919

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.8104 - accuracy: 0.6893

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.8009 - accuracy: 0.6916

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.7928 - accuracy: 0.6959

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.8022 - accuracy: 0.6924

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.8011 - accuracy: 0.6922

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.8001 - accuracy: 0.6951

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.7984 - accuracy: 0.6959

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.8026 - accuracy: 0.6956

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.8044 - accuracy: 0.6944

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.8014 - accuracy: 0.6942

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.7971 - accuracy: 0.6958

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.7997 - accuracy: 0.6956

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.7963 - accuracy: 0.6970

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.8029 - accuracy: 0.6952

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.8103 - accuracy: 0.6926

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.8079 - accuracy: 0.6910

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.8065 - accuracy: 0.6931

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.8051 - accuracy: 0.6937

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.8045 - accuracy: 0.6936

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.8108 - accuracy: 0.6913

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.8087 - accuracy: 0.6919

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.8126 - accuracy: 0.6898

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.8150 - accuracy: 0.6898

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.8101 - accuracy: 0.6917

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.8104 - accuracy: 0.6916

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.8141 - accuracy: 0.6903

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.8093 - accuracy: 0.6914

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.8124 - accuracy: 0.6896

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.8124 - accuracy: 0.6890

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.8130 - accuracy: 0.6884

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.8122 - accuracy: 0.6895

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.8142 - accuracy: 0.6889

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.8122 - accuracy: 0.6910

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.8122 - accuracy: 0.6920

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.8121 - accuracy: 0.6925

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.8119 - accuracy: 0.6919

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.8106 - accuracy: 0.6928

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.8142 - accuracy: 0.6912

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.8153 - accuracy: 0.6922

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.8123 - accuracy: 0.6935

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.8094 - accuracy: 0.6953

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.8128 - accuracy: 0.6948

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.8113 - accuracy: 0.6960

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.8135 - accuracy: 0.6959

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.8119 - accuracy: 0.6962

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.8123 - accuracy: 0.6957

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.8095 - accuracy: 0.6964

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.8104 - accuracy: 0.6967

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.8094 - accuracy: 0.6970

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.8094 - accuracy: 0.6982

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.8087 - accuracy: 0.6980

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.8083 - accuracy: 0.6983

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.8051 - accuracy: 0.6994

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.8039 - accuracy: 0.7000

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.8034 - accuracy: 0.6995

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.8049 - accuracy: 0.6985

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.8078 - accuracy: 0.6965

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.8059 - accuracy: 0.6971

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.8051 - accuracy: 0.6970

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.8043 - accuracy: 0.6976

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.8021 - accuracy: 0.6986

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.8020 - accuracy: 0.6985

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.8038 - accuracy: 0.6984

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.8033 - accuracy: 0.6993

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.8022 - accuracy: 0.6988

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.8013 - accuracy: 0.6983

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.8007 - accuracy: 0.6986

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.8007 - accuracy: 0.6986 - val_loss: 0.8190 - val_accuracy: 0.6703


.. parsed-literal::

    Epoch 7/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.6783 - accuracy: 0.7812

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.7544 - accuracy: 0.7031

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.7007 - accuracy: 0.7188

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6842 - accuracy: 0.7188

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.7597 - accuracy: 0.6938

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.7500 - accuracy: 0.7031

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.7762 - accuracy: 0.7054

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.7741 - accuracy: 0.6992

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.7690 - accuracy: 0.7014

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.7460 - accuracy: 0.7094

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.7457 - accuracy: 0.7045

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.7533 - accuracy: 0.7083

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.7426 - accuracy: 0.7139

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.7335 - accuracy: 0.7188

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.7309 - accuracy: 0.7250

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.7295 - accuracy: 0.7266

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.7245 - accuracy: 0.7279

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.7198 - accuracy: 0.7326

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.7226 - accuracy: 0.7368

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.7176 - accuracy: 0.7406

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.7141 - accuracy: 0.7396

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.7028 - accuracy: 0.7443

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.7039 - accuracy: 0.7378

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.7064 - accuracy: 0.7344

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.7013 - accuracy: 0.7362

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.7040 - accuracy: 0.7356

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.7051 - accuracy: 0.7350

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.7044 - accuracy: 0.7366

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.7008 - accuracy: 0.7349

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.7057 - accuracy: 0.7344

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.7046 - accuracy: 0.7329

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.7049 - accuracy: 0.7305

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.7119 - accuracy: 0.7282

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.7168 - accuracy: 0.7270

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.7256 - accuracy: 0.7223

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.7264 - accuracy: 0.7222

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.7246 - accuracy: 0.7247

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.7235 - accuracy: 0.7237

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.7288 - accuracy: 0.7212

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.7285 - accuracy: 0.7224

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.7265 - accuracy: 0.7223

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.7338 - accuracy: 0.7193

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.7284 - accuracy: 0.7207

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.7287 - accuracy: 0.7193

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.7331 - accuracy: 0.7172

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.7320 - accuracy: 0.7186

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.7315 - accuracy: 0.7192

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.7325 - accuracy: 0.7205

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.7304 - accuracy: 0.7224

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.7301 - accuracy: 0.7217

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.7344 - accuracy: 0.7222

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.7372 - accuracy: 0.7204

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.7399 - accuracy: 0.7186

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.7366 - accuracy: 0.7203

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.7358 - accuracy: 0.7209

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.7356 - accuracy: 0.7203

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.7371 - accuracy: 0.7192

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.7338 - accuracy: 0.7197

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.7353 - accuracy: 0.7191

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.7352 - accuracy: 0.7186

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.7338 - accuracy: 0.7196

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.7355 - accuracy: 0.7181

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.7379 - accuracy: 0.7186

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.7381 - accuracy: 0.7191

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.7398 - accuracy: 0.7191

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.7385 - accuracy: 0.7205

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.7415 - accuracy: 0.7186

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.7385 - accuracy: 0.7205

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.7407 - accuracy: 0.7200

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.7412 - accuracy: 0.7204

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.7424 - accuracy: 0.7199

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.7462 - accuracy: 0.7191

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.7441 - accuracy: 0.7195

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.7463 - accuracy: 0.7182

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.7483 - accuracy: 0.7174

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.7476 - accuracy: 0.7182

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.7501 - accuracy: 0.7170

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.7513 - accuracy: 0.7167

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.7517 - accuracy: 0.7167

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.7509 - accuracy: 0.7175

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.7501 - accuracy: 0.7179

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.7490 - accuracy: 0.7183

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.7498 - accuracy: 0.7172

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.7502 - accuracy: 0.7161

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.7505 - accuracy: 0.7161

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.7525 - accuracy: 0.7154

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.7504 - accuracy: 0.7165

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.7490 - accuracy: 0.7169

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.7505 - accuracy: 0.7159

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.7514 - accuracy: 0.7149

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.7523 - accuracy: 0.7142

.. parsed-literal::

    
92/92 [==============================] - 6s 64ms/step - loss: 0.7523 - accuracy: 0.7142 - val_loss: 0.8176 - val_accuracy: 0.6703


.. parsed-literal::

    Epoch 8/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.5886 - accuracy: 0.7812

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.6825 - accuracy: 0.7500

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6810 - accuracy: 0.7396

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.6483 - accuracy: 0.7734

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.6525 - accuracy: 0.7563

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.6596 - accuracy: 0.7552

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6899 - accuracy: 0.7545

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.7033 - accuracy: 0.7422

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.7033 - accuracy: 0.7361

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.7208 - accuracy: 0.7312

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.7256 - accuracy: 0.7216

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.7052 - accuracy: 0.7318

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6961 - accuracy: 0.7356

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6857 - accuracy: 0.7366

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6921 - accuracy: 0.7396

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6866 - accuracy: 0.7422

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6742 - accuracy: 0.7445

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6683 - accuracy: 0.7465

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6656 - accuracy: 0.7500

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6552 - accuracy: 0.7516

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6597 - accuracy: 0.7485

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6686 - accuracy: 0.7457

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.6603 - accuracy: 0.7486

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6616 - accuracy: 0.7500

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6651 - accuracy: 0.7450

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6690 - accuracy: 0.7428

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6708 - accuracy: 0.7454

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6721 - accuracy: 0.7411

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6832 - accuracy: 0.7360

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6864 - accuracy: 0.7333

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6866 - accuracy: 0.7329

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6955 - accuracy: 0.7285

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6960 - accuracy: 0.7282

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6936 - accuracy: 0.7307

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6930 - accuracy: 0.7312

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6893 - accuracy: 0.7309

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6934 - accuracy: 0.7297

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6890 - accuracy: 0.7319

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6888 - accuracy: 0.7308

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.6942 - accuracy: 0.7297

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6933 - accuracy: 0.7294

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.7023 - accuracy: 0.7254

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.7015 - accuracy: 0.7253

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6983 - accuracy: 0.7280

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6961 - accuracy: 0.7285

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.7007 - accuracy: 0.7249

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.7001 - accuracy: 0.7247

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6989 - accuracy: 0.7266

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6998 - accuracy: 0.7264

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.7025 - accuracy: 0.7237

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.7032 - accuracy: 0.7212

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.7041 - accuracy: 0.7218

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.7039 - accuracy: 0.7211

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.7034 - accuracy: 0.7222

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.7032 - accuracy: 0.7239

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.7019 - accuracy: 0.7249

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.7008 - accuracy: 0.7259

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.7047 - accuracy: 0.7247

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.7068 - accuracy: 0.7235

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.7056 - accuracy: 0.7240

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.7045 - accuracy: 0.7234

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.7076 - accuracy: 0.7243

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.7057 - accuracy: 0.7247

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.7042 - accuracy: 0.7251

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.7029 - accuracy: 0.7250

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.7001 - accuracy: 0.7254

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.7012 - accuracy: 0.7243

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.7035 - accuracy: 0.7238

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.7034 - accuracy: 0.7228

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.7034 - accuracy: 0.7232

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.7107 - accuracy: 0.7205

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.7186 - accuracy: 0.7188

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.7184 - accuracy: 0.7192

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.7201 - accuracy: 0.7183

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.7181 - accuracy: 0.7192

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.7170 - accuracy: 0.7200

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.7181 - accuracy: 0.7188

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.7185 - accuracy: 0.7192

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.7200 - accuracy: 0.7184

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.7210 - accuracy: 0.7180

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.7216 - accuracy: 0.7176

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.7221 - accuracy: 0.7176

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.7222 - accuracy: 0.7176

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.7239 - accuracy: 0.7173

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.7237 - accuracy: 0.7173

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.7238 - accuracy: 0.7161

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.7247 - accuracy: 0.7158

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.7246 - accuracy: 0.7158

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.7222 - accuracy: 0.7173

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.7197 - accuracy: 0.7187

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.7191 - accuracy: 0.7187

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.7191 - accuracy: 0.7187 - val_loss: 0.7469 - val_accuracy: 0.6880


.. parsed-literal::

    Epoch 9/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.8247 - accuracy: 0.7188

.. parsed-literal::

    
 3/92 [..............................] - ETA: 4s - loss: 0.6887 - accuracy: 0.7614

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 4s - loss: 0.6661 - accuracy: 0.7750

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 4s - loss: 0.6376 - accuracy: 0.7632

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.7175 - accuracy: 0.7174

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6905 - accuracy: 0.7407

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.7124 - accuracy: 0.7339

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.7232 - accuracy: 0.7357

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.7313 - accuracy: 0.7340

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.7176 - accuracy: 0.7384

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6941 - accuracy: 0.7500

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6920 - accuracy: 0.7500

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6805 - accuracy: 0.7545

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6956 - accuracy: 0.7458

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6868 - accuracy: 0.7500

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6855 - accuracy: 0.7500

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6954 - accuracy: 0.7430

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.7045 - accuracy: 0.7433

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.7004 - accuracy: 0.7453

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6943 - accuracy: 0.7470

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6969 - accuracy: 0.7385

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.6873 - accuracy: 0.7418

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6786 - accuracy: 0.7474

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6782 - accuracy: 0.7462

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6772 - accuracy: 0.7500

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6733 - accuracy: 0.7512

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6739 - accuracy: 0.7511

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6716 - accuracy: 0.7511

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6759 - accuracy: 0.7500

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6733 - accuracy: 0.7510

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6744 - accuracy: 0.7500

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6746 - accuracy: 0.7490

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6801 - accuracy: 0.7454

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6710 - accuracy: 0.7482

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6647 - accuracy: 0.7517

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6641 - accuracy: 0.7526

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6628 - accuracy: 0.7525

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6702 - accuracy: 0.7508

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.6705 - accuracy: 0.7516

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6785 - accuracy: 0.7492

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6807 - accuracy: 0.7478

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6804 - accuracy: 0.7478

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6783 - accuracy: 0.7486

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6839 - accuracy: 0.7451

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6870 - accuracy: 0.7445

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6840 - accuracy: 0.7447

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6841 - accuracy: 0.7435

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6832 - accuracy: 0.7442

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6854 - accuracy: 0.7443

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6821 - accuracy: 0.7457

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6806 - accuracy: 0.7452

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6763 - accuracy: 0.7476

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6801 - accuracy: 0.7459

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6758 - accuracy: 0.7466

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6764 - accuracy: 0.7455

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6786 - accuracy: 0.7439

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6795 - accuracy: 0.7440

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6796 - accuracy: 0.7431

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6804 - accuracy: 0.7427

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6817 - accuracy: 0.7428

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6783 - accuracy: 0.7434

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6794 - accuracy: 0.7425

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6787 - accuracy: 0.7431

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6772 - accuracy: 0.7432

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6757 - accuracy: 0.7443

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6795 - accuracy: 0.7430

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6772 - accuracy: 0.7422

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6752 - accuracy: 0.7427

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6770 - accuracy: 0.7419

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6766 - accuracy: 0.7425

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6784 - accuracy: 0.7417

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6764 - accuracy: 0.7423

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6787 - accuracy: 0.7419

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6823 - accuracy: 0.7408

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6814 - accuracy: 0.7413

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6795 - accuracy: 0.7427

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6780 - accuracy: 0.7428

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6756 - accuracy: 0.7437

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6736 - accuracy: 0.7445

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6741 - accuracy: 0.7446

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6741 - accuracy: 0.7454

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6755 - accuracy: 0.7443

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6730 - accuracy: 0.7451

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6714 - accuracy: 0.7459

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6693 - accuracy: 0.7464

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6700 - accuracy: 0.7457

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6704 - accuracy: 0.7454

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6694 - accuracy: 0.7454

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6701 - accuracy: 0.7451

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6691 - accuracy: 0.7459

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6698 - accuracy: 0.7456

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.6698 - accuracy: 0.7456 - val_loss: 0.7524 - val_accuracy: 0.6989


.. parsed-literal::

    Epoch 10/15


.. parsed-literal::


 1/92 [..............................] - ETA: 6s - loss: 0.8125 - accuracy: 0.6562

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.7363 - accuracy: 0.7031

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.8460 - accuracy: 0.6667

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 4s - loss: 0.7945 - accuracy: 0.6875

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 4s - loss: 0.7304 - accuracy: 0.7125

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.7014 - accuracy: 0.7240

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6732 - accuracy: 0.7366

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6559 - accuracy: 0.7422

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6480 - accuracy: 0.7361

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6854 - accuracy: 0.7281

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6830 - accuracy: 0.7330

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6926 - accuracy: 0.7266

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6845 - accuracy: 0.7260

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6801 - accuracy: 0.7299

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6811 - accuracy: 0.7312

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6820 - accuracy: 0.7305

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6813 - accuracy: 0.7353

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6727 - accuracy: 0.7378

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6655 - accuracy: 0.7434

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6663 - accuracy: 0.7422

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6672 - accuracy: 0.7366

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6647 - accuracy: 0.7358

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.6707 - accuracy: 0.7364

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6666 - accuracy: 0.7396

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6600 - accuracy: 0.7425

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6557 - accuracy: 0.7416

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6561 - accuracy: 0.7419

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6520 - accuracy: 0.7433

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6456 - accuracy: 0.7435

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6430 - accuracy: 0.7437

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6417 - accuracy: 0.7409

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6436 - accuracy: 0.7383

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6549 - accuracy: 0.7301

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6611 - accuracy: 0.7279

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6598 - accuracy: 0.7304

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6655 - accuracy: 0.7274

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6598 - accuracy: 0.7297

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6558 - accuracy: 0.7311

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6571 - accuracy: 0.7276

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.6566 - accuracy: 0.7281

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6548 - accuracy: 0.7294

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6571 - accuracy: 0.7277

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6530 - accuracy: 0.7289

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6557 - accuracy: 0.7294

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6595 - accuracy: 0.7292

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6544 - accuracy: 0.7310

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6508 - accuracy: 0.7327

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6515 - accuracy: 0.7318

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6521 - accuracy: 0.7328

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6485 - accuracy: 0.7356

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6529 - accuracy: 0.7335

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6503 - accuracy: 0.7344

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6524 - accuracy: 0.7335

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6508 - accuracy: 0.7338

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6493 - accuracy: 0.7347

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6476 - accuracy: 0.7360

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6491 - accuracy: 0.7346

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6464 - accuracy: 0.7365

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6467 - accuracy: 0.7378

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6497 - accuracy: 0.7370

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6508 - accuracy: 0.7367

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6534 - accuracy: 0.7369

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6532 - accuracy: 0.7376

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6573 - accuracy: 0.7373

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6562 - accuracy: 0.7370

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6545 - accuracy: 0.7367

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6525 - accuracy: 0.7374

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6516 - accuracy: 0.7367

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6504 - accuracy: 0.7387

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6509 - accuracy: 0.7388

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6518 - accuracy: 0.7381

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6491 - accuracy: 0.7391

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6481 - accuracy: 0.7402

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6490 - accuracy: 0.7403

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6488 - accuracy: 0.7408

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6474 - accuracy: 0.7418

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6501 - accuracy: 0.7411

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6512 - accuracy: 0.7412

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6500 - accuracy: 0.7417

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6508 - accuracy: 0.7406

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6525 - accuracy: 0.7407

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6545 - accuracy: 0.7401

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6554 - accuracy: 0.7402

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6549 - accuracy: 0.7403

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6561 - accuracy: 0.7397

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6547 - accuracy: 0.7409

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6541 - accuracy: 0.7407

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6563 - accuracy: 0.7405

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6592 - accuracy: 0.7406

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6590 - accuracy: 0.7417

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6613 - accuracy: 0.7408

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.6613 - accuracy: 0.7408 - val_loss: 0.7715 - val_accuracy: 0.6975


.. parsed-literal::

    Epoch 11/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.5333 - accuracy: 0.7500

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.6506 - accuracy: 0.7500

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6125 - accuracy: 0.7708

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5779 - accuracy: 0.7891

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 4s - loss: 0.5853 - accuracy: 0.7750

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.6210 - accuracy: 0.7656

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6050 - accuracy: 0.7679

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5874 - accuracy: 0.7734

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6112 - accuracy: 0.7674

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6089 - accuracy: 0.7656

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6000 - accuracy: 0.7727

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5996 - accuracy: 0.7708

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5828 - accuracy: 0.7788

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5818 - accuracy: 0.7746

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5858 - accuracy: 0.7729

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5970 - accuracy: 0.7676

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.6000 - accuracy: 0.7665

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.6140 - accuracy: 0.7639

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.6108 - accuracy: 0.7615

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.6206 - accuracy: 0.7641

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.6236 - accuracy: 0.7649

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6153 - accuracy: 0.7685

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.6121 - accuracy: 0.7690

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6128 - accuracy: 0.7677

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6174 - accuracy: 0.7609

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6167 - accuracy: 0.7617

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6176 - accuracy: 0.7601

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6309 - accuracy: 0.7576

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6300 - accuracy: 0.7584

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6305 - accuracy: 0.7571

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6354 - accuracy: 0.7520

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6398 - accuracy: 0.7471

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6430 - accuracy: 0.7463

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6424 - accuracy: 0.7473

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.6445 - accuracy: 0.7439

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.6478 - accuracy: 0.7423

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.6511 - accuracy: 0.7417

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.6522 - accuracy: 0.7419

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.6550 - accuracy: 0.7414

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.6634 - accuracy: 0.7362

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.6657 - accuracy: 0.7350

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.6644 - accuracy: 0.7361

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.6695 - accuracy: 0.7336

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.6654 - accuracy: 0.7353

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.6681 - accuracy: 0.7357

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.6624 - accuracy: 0.7393

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.6630 - accuracy: 0.7402

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.6620 - accuracy: 0.7397

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.6594 - accuracy: 0.7431

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.6582 - accuracy: 0.7445

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.6608 - accuracy: 0.7440

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.6629 - accuracy: 0.7417

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.6603 - accuracy: 0.7430

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.6584 - accuracy: 0.7443

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.6636 - accuracy: 0.7427

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.6637 - accuracy: 0.7428

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.6643 - accuracy: 0.7413

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.6644 - accuracy: 0.7410

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.6684 - accuracy: 0.7406

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.6707 - accuracy: 0.7392

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.6668 - accuracy: 0.7404

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.6637 - accuracy: 0.7410

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.6605 - accuracy: 0.7422

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.6581 - accuracy: 0.7432

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.6591 - accuracy: 0.7429

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.6564 - accuracy: 0.7439

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.6587 - accuracy: 0.7431

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.6575 - accuracy: 0.7432

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.6550 - accuracy: 0.7451

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6556 - accuracy: 0.7447

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6548 - accuracy: 0.7456

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6574 - accuracy: 0.7440

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6549 - accuracy: 0.7445

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6546 - accuracy: 0.7437

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6552 - accuracy: 0.7442

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6550 - accuracy: 0.7435

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6551 - accuracy: 0.7436

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6615 - accuracy: 0.7409

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6614 - accuracy: 0.7402

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6611 - accuracy: 0.7407

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6609 - accuracy: 0.7401

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6618 - accuracy: 0.7394

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6632 - accuracy: 0.7388

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6624 - accuracy: 0.7389

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6621 - accuracy: 0.7391

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6608 - accuracy: 0.7392

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6578 - accuracy: 0.7407

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6602 - accuracy: 0.7405

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6618 - accuracy: 0.7403

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6610 - accuracy: 0.7400

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6619 - accuracy: 0.7394

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.6619 - accuracy: 0.7394 - val_loss: 0.7529 - val_accuracy: 0.7044


.. parsed-literal::

    Epoch 12/15


.. parsed-literal::


 1/92 [..............................] - ETA: 6s - loss: 0.5576 - accuracy: 0.7188

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5416 - accuracy: 0.7344

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.6216 - accuracy: 0.7396

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5764 - accuracy: 0.7578

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5470 - accuracy: 0.7812

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.5338 - accuracy: 0.7812

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.5280 - accuracy: 0.7812

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5390 - accuracy: 0.7812

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5493 - accuracy: 0.7847

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5942 - accuracy: 0.7625

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5868 - accuracy: 0.7699

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5710 - accuracy: 0.7760

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5644 - accuracy: 0.7740

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5788 - accuracy: 0.7656

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5938 - accuracy: 0.7604

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5888 - accuracy: 0.7676

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5909 - accuracy: 0.7721

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5880 - accuracy: 0.7778

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5936 - accuracy: 0.7780

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5898 - accuracy: 0.7797

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5891 - accuracy: 0.7798

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.6081 - accuracy: 0.7699

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.6133 - accuracy: 0.7677

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.6186 - accuracy: 0.7656

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.6115 - accuracy: 0.7688

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.6163 - accuracy: 0.7656

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6107 - accuracy: 0.7635

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.6059 - accuracy: 0.7652

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6078 - accuracy: 0.7637

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6018 - accuracy: 0.7673

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5994 - accuracy: 0.7657

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6009 - accuracy: 0.7653

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5994 - accuracy: 0.7667

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5976 - accuracy: 0.7671

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5939 - accuracy: 0.7675

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5960 - accuracy: 0.7662

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5965 - accuracy: 0.7682

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5953 - accuracy: 0.7677

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.5905 - accuracy: 0.7689

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5882 - accuracy: 0.7692

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5853 - accuracy: 0.7680

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5848 - accuracy: 0.7675

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5848 - accuracy: 0.7664

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5827 - accuracy: 0.7668

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5871 - accuracy: 0.7637

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5910 - accuracy: 0.7647

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5897 - accuracy: 0.7670

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5913 - accuracy: 0.7667

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5903 - accuracy: 0.7670

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5841 - accuracy: 0.7697

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5799 - accuracy: 0.7717

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5871 - accuracy: 0.7695

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5914 - accuracy: 0.7669

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5954 - accuracy: 0.7648

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5967 - accuracy: 0.7640

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5950 - accuracy: 0.7654

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5950 - accuracy: 0.7662

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5928 - accuracy: 0.7665

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5908 - accuracy: 0.7667

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5892 - accuracy: 0.7675

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5924 - accuracy: 0.7662

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5900 - accuracy: 0.7674

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5918 - accuracy: 0.7672

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5941 - accuracy: 0.7674

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5946 - accuracy: 0.7671

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5961 - accuracy: 0.7659

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5970 - accuracy: 0.7652

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5958 - accuracy: 0.7655

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5988 - accuracy: 0.7652

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.6015 - accuracy: 0.7655

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.6037 - accuracy: 0.7657

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.6031 - accuracy: 0.7663

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.6061 - accuracy: 0.7648

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.6101 - accuracy: 0.7630

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.6100 - accuracy: 0.7624

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.6106 - accuracy: 0.7614

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.6131 - accuracy: 0.7609

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.6123 - accuracy: 0.7615

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.6143 - accuracy: 0.7610

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.6143 - accuracy: 0.7604

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.6134 - accuracy: 0.7603

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.6123 - accuracy: 0.7610

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.6109 - accuracy: 0.7608

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.6103 - accuracy: 0.7614

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.6090 - accuracy: 0.7624

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.6087 - accuracy: 0.7626

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.6061 - accuracy: 0.7642

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.6075 - accuracy: 0.7648

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.6068 - accuracy: 0.7650

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.6083 - accuracy: 0.7658

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.6096 - accuracy: 0.7660

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.6096 - accuracy: 0.7660 - val_loss: 0.7736 - val_accuracy: 0.7098


.. parsed-literal::

    Epoch 13/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.6388 - accuracy: 0.7188

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5525 - accuracy: 0.8125

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.5307 - accuracy: 0.8125

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5892 - accuracy: 0.7812

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 4s - loss: 0.6239 - accuracy: 0.7688

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 4s - loss: 0.6347 - accuracy: 0.7656

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.6207 - accuracy: 0.7679

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.6102 - accuracy: 0.7773

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.6046 - accuracy: 0.7812

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.6184 - accuracy: 0.7812

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.6151 - accuracy: 0.7812

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.6196 - accuracy: 0.7812

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.6237 - accuracy: 0.7788

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.6062 - accuracy: 0.7835

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.6051 - accuracy: 0.7875

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.6059 - accuracy: 0.7832

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5930 - accuracy: 0.7849

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5822 - accuracy: 0.7882

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5805 - accuracy: 0.7878

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5782 - accuracy: 0.7891

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5738 - accuracy: 0.7887

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5908 - accuracy: 0.7841

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.6003 - accuracy: 0.7812

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5931 - accuracy: 0.7812

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5951 - accuracy: 0.7800

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5945 - accuracy: 0.7800

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.6022 - accuracy: 0.7766

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.6061 - accuracy: 0.7768

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.6209 - accuracy: 0.7647

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.6179 - accuracy: 0.7663

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.6138 - accuracy: 0.7657

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.6187 - accuracy: 0.7672

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.6135 - accuracy: 0.7704

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.6053 - accuracy: 0.7752

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5997 - accuracy: 0.7771

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5951 - accuracy: 0.7781

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5918 - accuracy: 0.7790

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5893 - accuracy: 0.7806

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.5918 - accuracy: 0.7799

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5865 - accuracy: 0.7807

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5917 - accuracy: 0.7792

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5891 - accuracy: 0.7800

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5859 - accuracy: 0.7814

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5871 - accuracy: 0.7807

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5870 - accuracy: 0.7807

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5863 - accuracy: 0.7807

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5858 - accuracy: 0.7801

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5882 - accuracy: 0.7801

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5925 - accuracy: 0.7783

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5916 - accuracy: 0.7783

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5879 - accuracy: 0.7802

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5860 - accuracy: 0.7808

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5846 - accuracy: 0.7808

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5813 - accuracy: 0.7825

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5845 - accuracy: 0.7814

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5829 - accuracy: 0.7825

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5808 - accuracy: 0.7835

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5787 - accuracy: 0.7846

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5762 - accuracy: 0.7850

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5756 - accuracy: 0.7850

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5722 - accuracy: 0.7859

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5729 - accuracy: 0.7854

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5751 - accuracy: 0.7853

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5741 - accuracy: 0.7852

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5769 - accuracy: 0.7837

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5771 - accuracy: 0.7837

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5793 - accuracy: 0.7841

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5784 - accuracy: 0.7832

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5787 - accuracy: 0.7841

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5802 - accuracy: 0.7827

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5812 - accuracy: 0.7814

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5852 - accuracy: 0.7801

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5839 - accuracy: 0.7809

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5852 - accuracy: 0.7801

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5913 - accuracy: 0.7764

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5914 - accuracy: 0.7757

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5887 - accuracy: 0.7769

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5878 - accuracy: 0.7774

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5858 - accuracy: 0.7786

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5893 - accuracy: 0.7771

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5886 - accuracy: 0.7768

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5898 - accuracy: 0.7749

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5900 - accuracy: 0.7750

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5916 - accuracy: 0.7747

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5905 - accuracy: 0.7755

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5945 - accuracy: 0.7741

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5925 - accuracy: 0.7753

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5946 - accuracy: 0.7746

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5962 - accuracy: 0.7730

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5961 - accuracy: 0.7731

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5991 - accuracy: 0.7725

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.5991 - accuracy: 0.7725 - val_loss: 0.7365 - val_accuracy: 0.7316


.. parsed-literal::

    Epoch 14/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.3096 - accuracy: 0.9375

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.3655 - accuracy: 0.9062

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.4399 - accuracy: 0.8646

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.4835 - accuracy: 0.8359

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5127 - accuracy: 0.8250

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.5162 - accuracy: 0.8229

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.5348 - accuracy: 0.8125

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5471 - accuracy: 0.8047

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5487 - accuracy: 0.8056

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5446 - accuracy: 0.8031

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5445 - accuracy: 0.8040

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5417 - accuracy: 0.7995

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5406 - accuracy: 0.7981

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5389 - accuracy: 0.8013

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5604 - accuracy: 0.7896

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5539 - accuracy: 0.7910

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5519 - accuracy: 0.7941

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5561 - accuracy: 0.7917

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5581 - accuracy: 0.7928

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5606 - accuracy: 0.7922

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5721 - accuracy: 0.7827

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5756 - accuracy: 0.7798

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 4s - loss: 0.5881 - accuracy: 0.7731

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 4s - loss: 0.5822 - accuracy: 0.7747

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5797 - accuracy: 0.7763

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5735 - accuracy: 0.7800

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5707 - accuracy: 0.7812

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5749 - accuracy: 0.7846

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5708 - accuracy: 0.7877

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5714 - accuracy: 0.7885

.. parsed-literal::

    
31/92 [=========>....................] - ETA: 3s - loss: 0.5716 - accuracy: 0.7883

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5691 - accuracy: 0.7891

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5686 - accuracy: 0.7888

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5678 - accuracy: 0.7904

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5655 - accuracy: 0.7911

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5612 - accuracy: 0.7925

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5646 - accuracy: 0.7947

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5650 - accuracy: 0.7952

.. parsed-literal::

    
40/92 [============>.................] - ETA: 3s - loss: 0.5624 - accuracy: 0.7948

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5672 - accuracy: 0.7952

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5663 - accuracy: 0.7942

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5704 - accuracy: 0.7931

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5664 - accuracy: 0.7950

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5674 - accuracy: 0.7940

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5646 - accuracy: 0.7964

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5654 - accuracy: 0.7961

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5644 - accuracy: 0.7971

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5726 - accuracy: 0.7955

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5717 - accuracy: 0.7965

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5747 - accuracy: 0.7950

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5726 - accuracy: 0.7959

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5722 - accuracy: 0.7962

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5728 - accuracy: 0.7953

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5721 - accuracy: 0.7962

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5716 - accuracy: 0.7971

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5718 - accuracy: 0.7963

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5748 - accuracy: 0.7965

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5753 - accuracy: 0.7963

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5761 - accuracy: 0.7965

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5724 - accuracy: 0.7973

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5733 - accuracy: 0.7966

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5773 - accuracy: 0.7938

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5865 - accuracy: 0.7912

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5862 - accuracy: 0.7915

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5859 - accuracy: 0.7909

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5870 - accuracy: 0.7907

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5876 - accuracy: 0.7906

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5891 - accuracy: 0.7895

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5866 - accuracy: 0.7903

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5852 - accuracy: 0.7911

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5819 - accuracy: 0.7927

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5803 - accuracy: 0.7930

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5800 - accuracy: 0.7919

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5811 - accuracy: 0.7910

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5805 - accuracy: 0.7908

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5809 - accuracy: 0.7899

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5854 - accuracy: 0.7882

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5852 - accuracy: 0.7881

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5856 - accuracy: 0.7880

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5859 - accuracy: 0.7868

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5883 - accuracy: 0.7852

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5874 - accuracy: 0.7855

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5868 - accuracy: 0.7862

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5867 - accuracy: 0.7865

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5842 - accuracy: 0.7875

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5818 - accuracy: 0.7889

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5835 - accuracy: 0.7881

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5856 - accuracy: 0.7866

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5865 - accuracy: 0.7859

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5887 - accuracy: 0.7837

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5859 - accuracy: 0.7847

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.5859 - accuracy: 0.7847 - val_loss: 0.7476 - val_accuracy: 0.7030


.. parsed-literal::

    Epoch 15/15


.. parsed-literal::


 1/92 [..............................] - ETA: 7s - loss: 0.6296 - accuracy: 0.7188

.. parsed-literal::

    
 2/92 [..............................] - ETA: 5s - loss: 0.5183 - accuracy: 0.7969

.. parsed-literal::

    
 3/92 [..............................] - ETA: 5s - loss: 0.4555 - accuracy: 0.8438

.. parsed-literal::

    
 4/92 [>.............................] - ETA: 5s - loss: 0.5554 - accuracy: 0.8125

.. parsed-literal::

    
 5/92 [>.............................] - ETA: 5s - loss: 0.5277 - accuracy: 0.8250

.. parsed-literal::

    
 6/92 [>.............................] - ETA: 5s - loss: 0.5225 - accuracy: 0.8229

.. parsed-literal::

    
 7/92 [=>............................] - ETA: 4s - loss: 0.5193 - accuracy: 0.8214

.. parsed-literal::

    
 8/92 [=>............................] - ETA: 4s - loss: 0.5104 - accuracy: 0.8203

.. parsed-literal::

    
 9/92 [=>............................] - ETA: 4s - loss: 0.5158 - accuracy: 0.8056

.. parsed-literal::

    
10/92 [==>...........................] - ETA: 4s - loss: 0.5211 - accuracy: 0.8062

.. parsed-literal::

    
11/92 [==>...........................] - ETA: 4s - loss: 0.5356 - accuracy: 0.8011

.. parsed-literal::

    
12/92 [==>...........................] - ETA: 4s - loss: 0.5292 - accuracy: 0.8047

.. parsed-literal::

    
13/92 [===>..........................] - ETA: 4s - loss: 0.5390 - accuracy: 0.8029

.. parsed-literal::

    
14/92 [===>..........................] - ETA: 4s - loss: 0.5287 - accuracy: 0.8103

.. parsed-literal::

    
15/92 [===>..........................] - ETA: 4s - loss: 0.5260 - accuracy: 0.8125

.. parsed-literal::

    
16/92 [====>.........................] - ETA: 4s - loss: 0.5135 - accuracy: 0.8184

.. parsed-literal::

    
17/92 [====>.........................] - ETA: 4s - loss: 0.5157 - accuracy: 0.8125

.. parsed-literal::

    
18/92 [====>.........................] - ETA: 4s - loss: 0.5152 - accuracy: 0.8108

.. parsed-literal::

    
19/92 [=====>........................] - ETA: 4s - loss: 0.5102 - accuracy: 0.8125

.. parsed-literal::

    
20/92 [=====>........................] - ETA: 4s - loss: 0.5124 - accuracy: 0.8109

.. parsed-literal::

    
21/92 [=====>........................] - ETA: 4s - loss: 0.5310 - accuracy: 0.8065

.. parsed-literal::

    
22/92 [======>.......................] - ETA: 4s - loss: 0.5315 - accuracy: 0.8082

.. parsed-literal::

    
23/92 [======>.......................] - ETA: 3s - loss: 0.5295 - accuracy: 0.8071

.. parsed-literal::

    
24/92 [======>.......................] - ETA: 3s - loss: 0.5276 - accuracy: 0.8060

.. parsed-literal::

    
25/92 [=======>......................] - ETA: 3s - loss: 0.5272 - accuracy: 0.8075

.. parsed-literal::

    
26/92 [=======>......................] - ETA: 3s - loss: 0.5415 - accuracy: 0.7993

.. parsed-literal::

    
27/92 [=======>......................] - ETA: 3s - loss: 0.5368 - accuracy: 0.7998

.. parsed-literal::

    
28/92 [========>.....................] - ETA: 3s - loss: 0.5393 - accuracy: 0.8013

.. parsed-literal::

    
29/92 [========>.....................] - ETA: 3s - loss: 0.5347 - accuracy: 0.8017

.. parsed-literal::

    
30/92 [========>.....................] - ETA: 3s - loss: 0.5345 - accuracy: 0.8021

.. parsed-literal::

    
32/92 [=========>....................] - ETA: 3s - loss: 0.5380 - accuracy: 0.8022

.. parsed-literal::

    
33/92 [=========>....................] - ETA: 3s - loss: 0.5313 - accuracy: 0.8053

.. parsed-literal::

    
34/92 [==========>...................] - ETA: 3s - loss: 0.5316 - accuracy: 0.8037

.. parsed-literal::

    
35/92 [==========>...................] - ETA: 3s - loss: 0.5290 - accuracy: 0.8040

.. parsed-literal::

    
36/92 [==========>...................] - ETA: 3s - loss: 0.5294 - accuracy: 0.8024

.. parsed-literal::

    
37/92 [===========>..................] - ETA: 3s - loss: 0.5340 - accuracy: 0.7985

.. parsed-literal::

    
38/92 [===========>..................] - ETA: 3s - loss: 0.5360 - accuracy: 0.7988

.. parsed-literal::

    
39/92 [===========>..................] - ETA: 3s - loss: 0.5379 - accuracy: 0.7976

.. parsed-literal::

    
40/92 [============>.................] - ETA: 2s - loss: 0.5363 - accuracy: 0.7972

.. parsed-literal::

    
41/92 [============>.................] - ETA: 2s - loss: 0.5358 - accuracy: 0.7968

.. parsed-literal::

    
42/92 [============>.................] - ETA: 2s - loss: 0.5333 - accuracy: 0.8001

.. parsed-literal::

    
43/92 [=============>................] - ETA: 2s - loss: 0.5359 - accuracy: 0.8004

.. parsed-literal::

    
44/92 [=============>................] - ETA: 2s - loss: 0.5386 - accuracy: 0.7986

.. parsed-literal::

    
45/92 [=============>................] - ETA: 2s - loss: 0.5362 - accuracy: 0.8003

.. parsed-literal::

    
46/92 [==============>...............] - ETA: 2s - loss: 0.5376 - accuracy: 0.7985

.. parsed-literal::

    
47/92 [==============>...............] - ETA: 2s - loss: 0.5478 - accuracy: 0.7955

.. parsed-literal::

    
48/92 [==============>...............] - ETA: 2s - loss: 0.5489 - accuracy: 0.7965

.. parsed-literal::

    
49/92 [==============>...............] - ETA: 2s - loss: 0.5485 - accuracy: 0.7974

.. parsed-literal::

    
50/92 [===============>..............] - ETA: 2s - loss: 0.5489 - accuracy: 0.7977

.. parsed-literal::

    
51/92 [===============>..............] - ETA: 2s - loss: 0.5501 - accuracy: 0.7974

.. parsed-literal::

    
52/92 [===============>..............] - ETA: 2s - loss: 0.5439 - accuracy: 0.8007

.. parsed-literal::

    
53/92 [================>.............] - ETA: 2s - loss: 0.5419 - accuracy: 0.8015

.. parsed-literal::

    
54/92 [================>.............] - ETA: 2s - loss: 0.5443 - accuracy: 0.7994

.. parsed-literal::

    
55/92 [================>.............] - ETA: 2s - loss: 0.5414 - accuracy: 0.8014

.. parsed-literal::

    
56/92 [=================>............] - ETA: 2s - loss: 0.5405 - accuracy: 0.8004

.. parsed-literal::

    
57/92 [=================>............] - ETA: 2s - loss: 0.5417 - accuracy: 0.8012

.. parsed-literal::

    
58/92 [=================>............] - ETA: 1s - loss: 0.5396 - accuracy: 0.8019

.. parsed-literal::

    
59/92 [==================>...........] - ETA: 1s - loss: 0.5418 - accuracy: 0.8005

.. parsed-literal::

    
60/92 [==================>...........] - ETA: 1s - loss: 0.5395 - accuracy: 0.8018

.. parsed-literal::

    
61/92 [==================>...........] - ETA: 1s - loss: 0.5413 - accuracy: 0.8009

.. parsed-literal::

    
62/92 [===================>..........] - ETA: 1s - loss: 0.5404 - accuracy: 0.8016

.. parsed-literal::

    
63/92 [===================>..........] - ETA: 1s - loss: 0.5386 - accuracy: 0.8023

.. parsed-literal::

    
64/92 [===================>..........] - ETA: 1s - loss: 0.5376 - accuracy: 0.8010

.. parsed-literal::

    
65/92 [====================>.........] - ETA: 1s - loss: 0.5361 - accuracy: 0.8012

.. parsed-literal::

    
66/92 [====================>.........] - ETA: 1s - loss: 0.5413 - accuracy: 0.7999

.. parsed-literal::

    
67/92 [====================>.........] - ETA: 1s - loss: 0.5417 - accuracy: 0.7992

.. parsed-literal::

    
68/92 [=====================>........] - ETA: 1s - loss: 0.5408 - accuracy: 0.7989

.. parsed-literal::

    
69/92 [=====================>........] - ETA: 1s - loss: 0.5430 - accuracy: 0.7982

.. parsed-literal::

    
70/92 [=====================>........] - ETA: 1s - loss: 0.5426 - accuracy: 0.7975

.. parsed-literal::

    
71/92 [======================>.......] - ETA: 1s - loss: 0.5426 - accuracy: 0.7964

.. parsed-literal::

    
72/92 [======================>.......] - ETA: 1s - loss: 0.5427 - accuracy: 0.7966

.. parsed-literal::

    
73/92 [======================>.......] - ETA: 1s - loss: 0.5448 - accuracy: 0.7960

.. parsed-literal::

    
74/92 [=======================>......] - ETA: 1s - loss: 0.5487 - accuracy: 0.7936

.. parsed-literal::

    
75/92 [=======================>......] - ETA: 0s - loss: 0.5475 - accuracy: 0.7943

.. parsed-literal::

    
76/92 [=======================>......] - ETA: 0s - loss: 0.5472 - accuracy: 0.7950

.. parsed-literal::

    
77/92 [========================>.....] - ETA: 0s - loss: 0.5493 - accuracy: 0.7936

.. parsed-literal::

    
78/92 [========================>.....] - ETA: 0s - loss: 0.5496 - accuracy: 0.7922

.. parsed-literal::

    
79/92 [========================>.....] - ETA: 0s - loss: 0.5508 - accuracy: 0.7909

.. parsed-literal::

    
80/92 [=========================>....] - ETA: 0s - loss: 0.5509 - accuracy: 0.7904

.. parsed-literal::

    
81/92 [=========================>....] - ETA: 0s - loss: 0.5510 - accuracy: 0.7906

.. parsed-literal::

    
82/92 [=========================>....] - ETA: 0s - loss: 0.5520 - accuracy: 0.7909

.. parsed-literal::

    
83/92 [==========================>...] - ETA: 0s - loss: 0.5527 - accuracy: 0.7897

.. parsed-literal::

    
84/92 [==========================>...] - ETA: 0s - loss: 0.5515 - accuracy: 0.7896

.. parsed-literal::

    
85/92 [==========================>...] - ETA: 0s - loss: 0.5503 - accuracy: 0.7902

.. parsed-literal::

    
86/92 [===========================>..] - ETA: 0s - loss: 0.5488 - accuracy: 0.7908

.. parsed-literal::

    
87/92 [===========================>..] - ETA: 0s - loss: 0.5493 - accuracy: 0.7903

.. parsed-literal::

    
88/92 [===========================>..] - ETA: 0s - loss: 0.5476 - accuracy: 0.7910

.. parsed-literal::

    
89/92 [============================>.] - ETA: 0s - loss: 0.5453 - accuracy: 0.7919

.. parsed-literal::

    
90/92 [============================>.] - ETA: 0s - loss: 0.5481 - accuracy: 0.7911

.. parsed-literal::

    
91/92 [============================>.] - ETA: 0s - loss: 0.5487 - accuracy: 0.7906

.. parsed-literal::

    
92/92 [==============================] - ETA: 0s - loss: 0.5490 - accuracy: 0.7902

.. parsed-literal::

    
92/92 [==============================] - 6s 63ms/step - loss: 0.5490 - accuracy: 0.7902 - val_loss: 0.7075 - val_accuracy: 0.7221



.. image:: 301-tensorflow-training-openvino-nncf-with-output_files/301-tensorflow-training-openvino-nncf-with-output_3_1452.png


.. parsed-literal::


   1/1 [==============================] - ETA: 0s

.. parsed-literal::

    
   1/1 [==============================] - 0s 77ms/step


.. parsed-literal::

    This image most likely belongs to sunflowers with a 91.84 percent confidence.


.. parsed-literal::

    2024-01-19 00:30:03.067538: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'random_flip_input' with dtype float and shape [?,180,180,3]
    	 [[{{node random_flip_input}}]]
    2024-01-19 00:30:03.156796: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-01-19 00:30:03.167179: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'random_flip_input' with dtype float and shape [?,180,180,3]
    	 [[{{node random_flip_input}}]]
    2024-01-19 00:30:03.197932: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-01-19 00:30:03.209218: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-01-19 00:30:03.217377: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-01-19 00:30:03.228606: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-01-19 00:30:03.269138: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'sequential_1_input' with dtype float and shape [?,180,180,3]
    	 [[{{node sequential_1_input}}]]


.. parsed-literal::

    2024-01-19 00:30:03.338860: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-01-19 00:30:03.360704: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'sequential_1_input' with dtype float and shape [?,180,180,3]
    	 [[{{node sequential_1_input}}]]
    2024-01-19 00:30:03.403281: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,22,22,64]
    	 [[{{node inputs}}]]
    2024-01-19 00:30:03.429369: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-01-19 00:30:03.504954: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]


.. parsed-literal::

    2024-01-19 00:30:03.677203: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-01-19 00:30:03.823587: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,22,22,64]
    	 [[{{node inputs}}]]
    2024-01-19 00:30:03.858869: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]


.. parsed-literal::

    2024-01-19 00:30:03.888843: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-01-19 00:30:03.936701: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]


.. parsed-literal::

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
    This image most likely belongs to dandelion with a 99.81 percent confidence.



.. image:: 301-tensorflow-training-openvino-nncf-with-output_files/301-tensorflow-training-openvino-nncf-with-output_3_1465.png


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

    2024-01-19 00:30:07.150353: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [734]
    	 [[{{node Placeholder/_4}}]]
    2024-01-19 00:30:07.150605: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [734]
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
Flow <https://docs.openvino.ai/2023.0/basic_qauntization_flow.html#doxid-basic-qauntization-flow>`__.
To use the most advanced quantization flow that allows to apply 8-bit
quantization to the model with accuracy control see `Quantizing with
accuracy
control <https://docs.openvino.ai/2023.0/quantization_w_accuracy_control.html#>`__.

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
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-593/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/live.py", line 32, in run
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    self.live.refresh()
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-593/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/live.py", line 223, in refresh
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    self._live_render.set_renderable(self.renderable)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-593/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/live.py", line 203, in renderable
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    renderable = self.get_renderable()
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-593/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/live.py", line 98, in get_renderable
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    self._get_renderable()
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-593/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1537, in get_renderable
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    renderable = Group(*self.get_renderables())
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-593/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1542, in get_renderables
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    table = self.make_tasks_table(self.tasks)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-593/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1566, in make_tasks_table
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    table.add_row(
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-593/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1571, in &lt;genexpr&gt;
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    else column(task)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-593/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 528, in __call__
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    renderable = self.render(task)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-593/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/nncf/common/logging/track_progress.py", line 58, in render
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    text = super().render(task)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-593/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 787, in render
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    task_time = task.time_remaining
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-593/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1039, in time_remaining
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    estimate = ceil(remaining / speed)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-593/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/tensorflow/python/util/traceback_utils.py", line 153, in error_handler
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    raise e.with_traceback(filtered_tb) from None
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-593/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
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

    Accuracy of the original model: 0.722
    Accuracy of the quantized model: 0.721


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
    This image most likely belongs to dandelion with a 99.80 percent confidence.



.. image:: 301-tensorflow-training-openvino-nncf-with-output_files/301-tensorflow-training-openvino-nncf-with-output_27_1.png


Compare Inference Speed
-----------------------



Measure inference speed with the `OpenVINO Benchmark
App <https://docs.openvino.ai/2023.0/openvino_inference_engine_tools_benchmark_tool_README.html>`__.

Benchmark App is a command line tool that measures raw inference
performance for a specified OpenVINO IR model. Run
``benchmark_app --help`` to see a list of available parameters. By
default, Benchmark App tests the performance of the model specified with
the ``-m`` parameter with asynchronous inference on CPU, for one minute.
Use the ``-d`` parameter to test performance on a different device, for
example an Intel integrated Graphics (iGPU), and ``-t`` to set the
number of seconds to run inference. See the
`documentation <https://docs.openvino.ai/2023.0/openvino_inference_engine_tools_benchmark_tool_README.html>`__
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
    [ INFO ] Build ................................. 2023.2.0-13089-cfd42bd2cb0-HEAD
    [ INFO ]
    [ INFO ] Device info:


.. parsed-literal::

    [ INFO ] CPU
    [ INFO ] Build ................................. 2023.2.0-13089-cfd42bd2cb0-HEAD
    [ INFO ]
    [ INFO ]
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(CPU) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 13.25 ms
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

    [ INFO ] Compile model took 65.42 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: TensorFlow_Frontend_IR
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 12
    [ INFO ]   NUM_STREAMS: 12
    [ INFO ]   AFFINITY: Affinity.CORE
    [ INFO ]   INFERENCE_NUM_THREADS: 24
    [ INFO ]   PERF_COUNT: False
    [ INFO ]   INFERENCE_PRECISION_HINT: <Type: 'float32'>
    [ INFO ]   PERFORMANCE_HINT: PerformanceMode.THROUGHPUT
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
    [ INFO ] Count:            57924 iterations
    [ INFO ] Duration:         15005.01 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        2.91 ms
    [ INFO ]    Average:       2.92 ms
    [ INFO ]    Min:           1.99 ms
    [ INFO ]    Max:           13.27 ms
    [ INFO ] Throughput:   3860.31 FPS


.. code:: ipython3

    # Quantized model - CPU
    ! benchmark_app -m $compressed_model_xml -d CPU -t 15 -api async


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2023.2.0-13089-cfd42bd2cb0-HEAD
    [ INFO ]
    [ INFO ] Device info:


.. parsed-literal::

    [ INFO ] CPU
    [ INFO ] Build ................................. 2023.2.0-13089-cfd42bd2cb0-HEAD
    [ INFO ]
    [ INFO ]
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(CPU) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 13.41 ms
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

    [ INFO ] Compile model took 70.90 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: TensorFlow_Frontend_IR
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 12
    [ INFO ]   NUM_STREAMS: 12
    [ INFO ]   AFFINITY: Affinity.CORE
    [ INFO ]   INFERENCE_NUM_THREADS: 24
    [ INFO ]   PERF_COUNT: False
    [ INFO ]   INFERENCE_PRECISION_HINT: <Type: 'float32'>
    [ INFO ]   PERFORMANCE_HINT: PerformanceMode.THROUGHPUT
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
    [ INFO ] First inference took 2.13 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            182760 iterations
    [ INFO ] Duration:         15001.29 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        0.92 ms
    [ INFO ]    Average:       0.95 ms
    [ INFO ]    Min:           0.61 ms
    [ INFO ]    Max:           7.15 ms
    [ INFO ] Throughput:   12182.95 FPS


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

    [ INFO ] Count:            58152 iterations
    [ INFO ] Duration:         15006.14 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        2.91 ms
    [ INFO ]    Average:       2.90 ms
    [ INFO ]    Min:           1.94 ms
    [ INFO ]    Max:           11.90 ms
    [ INFO ] Throughput:   3875.21 FPS


**Quantized IR model - CPU**

.. code:: ipython3

    benchmark_output = %sx benchmark_app -m $compressed_model_xml -t 15 -api async
    # Remove logging info from benchmark_app output and show only the results
    benchmark_result = benchmark_output[-8:]
    print("\n".join(benchmark_result))


.. parsed-literal::

    [ INFO ] Count:            182592 iterations
    [ INFO ] Duration:         15001.73 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        0.92 ms
    [ INFO ]    Average:       0.95 ms
    [ INFO ]    Min:           0.59 ms
    [ INFO ]    Max:           7.27 ms
    [ INFO ] Throughput:   12171.39 FPS


**Original IR model - MULTI:CPU,GPU**

With a recent Intel CPU, the best performance can often be achieved by
doing inference on both the CPU and the iGPU, with OpenVINO’s `Multi
Device
Plugin <https://docs.openvino.ai/2023.0/openvino_docs_OV_UG_Running_on_multiple_devices.html>`__.
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

