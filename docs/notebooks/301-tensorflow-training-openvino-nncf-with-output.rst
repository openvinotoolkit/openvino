Post-Training Quantization with TensorFlow Classification Model
===============================================================

This example demonstrates how to quantize the OpenVINO model that was
created in `301-tensorflow-training-openvino
notebook <301-tensorflow-training-openvino.ipynb>`__, to improve
inference speed. Quantization is performed with `Post-training
Quantization with
NNCF <https://docs.openvino.ai/nightly/basic_quantization_flow.html>`__.
A custom dataloader and metric will be defined, and accuracy and
performance will be computed for the original IR model and the quantized
model.

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Preparation <#Preparation>`__

   -  `Imports <#Imports>`__

-  `Post-training Quantization with
   NNCF <#Post-training-Quantization-with-NNCF>`__

   -  `Select inference device <#Select-inference-device>`__

-  `Compare Metrics <#Compare-Metrics>`__
-  `Run Inference on Quantized
   Model <#Run-Inference-on-Quantized-Model>`__
-  `Compare Inference Speed <#Compare-Inference-Speed>`__

Preparation
-----------

`back to top ⬆️ <#Table-of-contents:>`__

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

    2024-01-26 00:38:58.168511: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-01-26 00:38:58.203263: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


.. parsed-literal::

    2024-01-26 00:38:58.795644: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


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

    2024-01-26 00:39:04.673372: E tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:266] failed call to cuInit: CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: forward compatibility was attempted on non supported HW
    2024-01-26 00:39:04.673408: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:168] retrieving CUDA diagnostic information for host: iotg-dev-workstation-07
    2024-01-26 00:39:04.673412: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:175] hostname: iotg-dev-workstation-07
    2024-01-26 00:39:04.673543: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:199] libcuda reported version is: 470.223.2
    2024-01-26 00:39:04.673559: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:203] kernel reported version is: 470.182.3
    2024-01-26 00:39:04.673562: E tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:312] kernel version 470.182.3 does not match DSO version 470.223.2 -- cannot find working devices in this configuration


.. parsed-literal::

    Found 3670 files belonging to 5 classes.


.. parsed-literal::

    Using 734 files for validation.
    ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']


.. parsed-literal::

    2024-01-26 00:39:04.952983: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-01-26 00:39:04.953258: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]



.. image:: 301-tensorflow-training-openvino-nncf-with-output_files/301-tensorflow-training-openvino-nncf-with-output_3_12.png


.. parsed-literal::

    2024-01-26 00:39:05.819322: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-01-26 00:39:05.819561: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]


.. parsed-literal::

    (32, 180, 180, 3)
    (32,)


.. parsed-literal::

    2024-01-26 00:39:06.138784: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]
    2024-01-26 00:39:06.139071: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]


.. parsed-literal::

    0.005936881 0.9981924


.. parsed-literal::

    2024-01-26 00:39:06.854372: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2024-01-26 00:39:06.854685: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]



.. image:: 301-tensorflow-training-openvino-nncf-with-output_files/301-tensorflow-training-openvino-nncf-with-output_3_18.png


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

                                                                     


.. parsed-literal::

     rescaling_2 (Rescaling)     (None, 180, 180, 3)       0         


.. parsed-literal::

                                                                     


.. parsed-literal::

     conv2d_3 (Conv2D)           (None, 180, 180, 16)      448       


.. parsed-literal::

                                                                     


.. parsed-literal::

     max_pooling2d_3 (MaxPooling  (None, 90, 90, 16)       0         


.. parsed-literal::

     2D)                                                             


.. parsed-literal::

                                                                     


.. parsed-literal::

     conv2d_4 (Conv2D)           (None, 90, 90, 32)        4640      


.. parsed-literal::

                                                                     


.. parsed-literal::

     max_pooling2d_4 (MaxPooling  (None, 45, 45, 32)       0         


.. parsed-literal::

     2D)                                                             


.. parsed-literal::

                                                                     


.. parsed-literal::

     conv2d_5 (Conv2D)           (None, 45, 45, 64)        18496     


.. parsed-literal::

                                                                     


.. parsed-literal::

     max_pooling2d_5 (MaxPooling  (None, 22, 22, 64)       0         


.. parsed-literal::

     2D)                                                             


.. parsed-literal::

                                                                     


.. parsed-literal::

     dropout (Dropout)           (None, 22, 22, 64)        0         


.. parsed-literal::

                                                                     


.. parsed-literal::

     flatten_1 (Flatten)         (None, 30976)             0         


.. parsed-literal::

                                                                     


.. parsed-literal::

     dense_2 (Dense)             (None, 128)               3965056   


.. parsed-literal::

                                                                     


.. parsed-literal::

     outputs (Dense)             (None, 5)                 645       


.. parsed-literal::

                                                                     


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

    2024-01-26 00:39:07.843867: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]
    2024-01-26 00:39:07.844357: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]


.. parsed-literal::

     1/92 [..............................] - ETA: 1:28 - loss: 1.6071 - accuracy: 0.1875

.. parsed-literal::

     2/92 [..............................] - ETA: 6s - loss: 2.4656 - accuracy: 0.1406  

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 2.2200 - accuracy: 0.1875

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 2.0727 - accuracy: 0.2188

.. parsed-literal::

     5/92 [>.............................] - ETA: 5s - loss: 1.9932 - accuracy: 0.2125

.. parsed-literal::

     6/92 [>.............................] - ETA: 5s - loss: 1.9251 - accuracy: 0.2240

.. parsed-literal::

     7/92 [=>............................] - ETA: 5s - loss: 1.8756 - accuracy: 0.2277

.. parsed-literal::

     8/92 [=>............................] - ETA: 5s - loss: 1.8491 - accuracy: 0.2148

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 1.8206 - accuracy: 0.2257

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 1.7968 - accuracy: 0.2281

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 1.7823 - accuracy: 0.2188

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 1.7672 - accuracy: 0.2135

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 1.7513 - accuracy: 0.2091

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 1.7359 - accuracy: 0.2254

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 1.7240 - accuracy: 0.2313

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 1.7148 - accuracy: 0.2363

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 1.7015 - accuracy: 0.2463

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 1.6962 - accuracy: 0.2413

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 1.6902 - accuracy: 0.2418

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 1.6793 - accuracy: 0.2484

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 1.6708 - accuracy: 0.2560

.. parsed-literal::

    22/92 [======>.......................] - ETA: 4s - loss: 1.6617 - accuracy: 0.2557

.. parsed-literal::

    23/92 [======>.......................] - ETA: 4s - loss: 1.6494 - accuracy: 0.2609

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 1.6436 - accuracy: 0.2630

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 1.6359 - accuracy: 0.2637

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 1.6257 - accuracy: 0.2656

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 1.6188 - accuracy: 0.2662

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 1.6132 - accuracy: 0.2712

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 1.6083 - accuracy: 0.2748

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 1.6016 - accuracy: 0.2781

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 1.5897 - accuracy: 0.2782

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 1.5838 - accuracy: 0.2783

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 1.5722 - accuracy: 0.2860

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 1.5643 - accuracy: 0.2886

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 1.5592 - accuracy: 0.2920

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 1.5542 - accuracy: 0.2986

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 1.5451 - accuracy: 0.3041

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 1.5380 - accuracy: 0.3100

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 1.5306 - accuracy: 0.3133

.. parsed-literal::

    40/92 [============>.................] - ETA: 3s - loss: 1.5233 - accuracy: 0.3164

.. parsed-literal::

    41/92 [============>.................] - ETA: 3s - loss: 1.5140 - accuracy: 0.3209

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 1.5108 - accuracy: 0.3237

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 1.4992 - accuracy: 0.3321

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 1.4914 - accuracy: 0.3366

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 1.4807 - accuracy: 0.3431

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 1.4754 - accuracy: 0.3471

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 1.4703 - accuracy: 0.3491

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 1.4625 - accuracy: 0.3516

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 1.4525 - accuracy: 0.3559

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 1.4433 - accuracy: 0.3625

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 1.4391 - accuracy: 0.3652

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 1.4327 - accuracy: 0.3678

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 1.4274 - accuracy: 0.3691

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 1.4303 - accuracy: 0.3709

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 1.4226 - accuracy: 0.3756

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 1.4221 - accuracy: 0.3761

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 1.4182 - accuracy: 0.3777

.. parsed-literal::

    58/92 [=================>............] - ETA: 2s - loss: 1.4175 - accuracy: 0.3815

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 1.4123 - accuracy: 0.3819

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 1.4066 - accuracy: 0.3844

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 1.4016 - accuracy: 0.3868

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 1.3981 - accuracy: 0.3871

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 1.3937 - accuracy: 0.3889

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 1.3899 - accuracy: 0.3896

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 1.3851 - accuracy: 0.3928

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 1.3816 - accuracy: 0.3949

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 1.3789 - accuracy: 0.3951

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 1.3739 - accuracy: 0.3980

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 1.3693 - accuracy: 0.3995

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 1.3656 - accuracy: 0.4009

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 1.3617 - accuracy: 0.4018

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 1.3611 - accuracy: 0.4015

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 1.3605 - accuracy: 0.4024

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 1.3559 - accuracy: 0.4055

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 1.3569 - accuracy: 0.4055

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 1.3548 - accuracy: 0.4051

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 1.3527 - accuracy: 0.4072

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 1.3502 - accuracy: 0.4075

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 1.3502 - accuracy: 0.4071

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 1.3499 - accuracy: 0.4087

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 1.3477 - accuracy: 0.4094

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 1.3439 - accuracy: 0.4120

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 1.3435 - accuracy: 0.4134

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 1.3402 - accuracy: 0.4152

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 1.3362 - accuracy: 0.4169

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 1.3343 - accuracy: 0.4168

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 1.3312 - accuracy: 0.4188

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 1.3281 - accuracy: 0.4204

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 1.3265 - accuracy: 0.4206

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 1.3215 - accuracy: 0.4225

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 1.3202 - accuracy: 0.4230

.. parsed-literal::

    2024-01-26 00:39:14.155330: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [734]
    	 [[{{node Placeholder/_4}}]]
    2024-01-26 00:39:14.155580: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [734]
    	 [[{{node Placeholder/_0}}]]


.. parsed-literal::

    92/92 [==============================] - 7s 66ms/step - loss: 1.3202 - accuracy: 0.4230 - val_loss: 1.1764 - val_accuracy: 0.5014


.. parsed-literal::

    Epoch 2/15


.. parsed-literal::

     1/92 [..............................] - ETA: 7s - loss: 1.0054 - accuracy: 0.5625

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 1.0611 - accuracy: 0.5312

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 1.0458 - accuracy: 0.5312

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 1.0587 - accuracy: 0.5312

.. parsed-literal::

     5/92 [>.............................] - ETA: 5s - loss: 1.0272 - accuracy: 0.5375

.. parsed-literal::

     6/92 [>.............................] - ETA: 5s - loss: 1.0667 - accuracy: 0.5312

.. parsed-literal::

     7/92 [=>............................] - ETA: 4s - loss: 1.0400 - accuracy: 0.5491

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 1.0480 - accuracy: 0.5586

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 1.0781 - accuracy: 0.5556

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 1.0623 - accuracy: 0.5625

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 1.0856 - accuracy: 0.5511

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 1.0776 - accuracy: 0.5599

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 1.0801 - accuracy: 0.5577

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 1.0849 - accuracy: 0.5580

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 1.0799 - accuracy: 0.5583

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 1.0661 - accuracy: 0.5645

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 1.0665 - accuracy: 0.5662

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 1.0734 - accuracy: 0.5677

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 1.0674 - accuracy: 0.5691

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 1.0721 - accuracy: 0.5688

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 1.0732 - accuracy: 0.5685

.. parsed-literal::

    22/92 [======>.......................] - ETA: 4s - loss: 1.0699 - accuracy: 0.5710

.. parsed-literal::

    23/92 [======>.......................] - ETA: 4s - loss: 1.0660 - accuracy: 0.5734

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 1.0608 - accuracy: 0.5768

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 1.0727 - accuracy: 0.5775

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 1.0799 - accuracy: 0.5769

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 1.0765 - accuracy: 0.5787

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 1.0706 - accuracy: 0.5837

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 1.0610 - accuracy: 0.5924

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 1.0564 - accuracy: 0.5955

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 1.0487 - accuracy: 0.5984

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 1.0510 - accuracy: 0.5945

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 1.0492 - accuracy: 0.5935

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 1.0559 - accuracy: 0.5944

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 1.0565 - accuracy: 0.5970

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 1.0568 - accuracy: 0.5961

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 1.0561 - accuracy: 0.5977

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 1.0584 - accuracy: 0.5976

.. parsed-literal::

    40/92 [============>.................] - ETA: 3s - loss: 1.0575 - accuracy: 0.5967

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 1.0606 - accuracy: 0.5920

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 1.0603 - accuracy: 0.5913

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 1.0606 - accuracy: 0.5914

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 1.0575 - accuracy: 0.5921

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 1.0582 - accuracy: 0.5936

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 1.0554 - accuracy: 0.5922

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 1.0499 - accuracy: 0.5949

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 1.0453 - accuracy: 0.5975

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 1.0448 - accuracy: 0.5968

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 1.0421 - accuracy: 0.5980

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 1.0402 - accuracy: 0.6004

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 1.0379 - accuracy: 0.6014

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 1.0340 - accuracy: 0.6013

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 1.0378 - accuracy: 0.6023

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 1.0323 - accuracy: 0.6027

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 1.0333 - accuracy: 0.6015

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 1.0385 - accuracy: 0.5991

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 1.0366 - accuracy: 0.5990

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 1.0347 - accuracy: 0.5984

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 1.0343 - accuracy: 0.5973

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 1.0300 - accuracy: 0.5993

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 1.0270 - accuracy: 0.6012

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 1.0235 - accuracy: 0.6046

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 1.0229 - accuracy: 0.6054

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 1.0212 - accuracy: 0.6047

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 1.0248 - accuracy: 0.6036

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 1.0263 - accuracy: 0.6021

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 1.0267 - accuracy: 0.6010

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 1.0243 - accuracy: 0.6023

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 1.0252 - accuracy: 0.6022

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 1.0244 - accuracy: 0.6011

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 1.0257 - accuracy: 0.6006

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 1.0217 - accuracy: 0.6022

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 1.0199 - accuracy: 0.6030

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 1.0210 - accuracy: 0.6037

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 1.0193 - accuracy: 0.6044

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 1.0196 - accuracy: 0.6042

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 1.0203 - accuracy: 0.6025

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 1.0205 - accuracy: 0.6020

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 1.0210 - accuracy: 0.6007

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 1.0207 - accuracy: 0.6014

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 1.0200 - accuracy: 0.6017

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 1.0197 - accuracy: 0.6012

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 1.0267 - accuracy: 0.5989

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 1.0285 - accuracy: 0.5977

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 1.0307 - accuracy: 0.5958

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 1.0290 - accuracy: 0.5969

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 1.0284 - accuracy: 0.5958

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 1.0286 - accuracy: 0.5968

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 1.0287 - accuracy: 0.5964

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 1.0310 - accuracy: 0.5944

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 1.0321 - accuracy: 0.5926

.. parsed-literal::

    92/92 [==============================] - 6s 64ms/step - loss: 1.0321 - accuracy: 0.5926 - val_loss: 1.0165 - val_accuracy: 0.5736


.. parsed-literal::

    Epoch 3/15


.. parsed-literal::

     1/92 [..............................] - ETA: 7s - loss: 1.2481 - accuracy: 0.4375

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 1.1014 - accuracy: 0.5156

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 1.0788 - accuracy: 0.5104

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 1.0381 - accuracy: 0.5547

.. parsed-literal::

     5/92 [>.............................] - ETA: 5s - loss: 1.0067 - accuracy: 0.5813

.. parsed-literal::

     6/92 [>.............................] - ETA: 4s - loss: 0.9963 - accuracy: 0.5885

.. parsed-literal::

     7/92 [=>............................] - ETA: 4s - loss: 0.9834 - accuracy: 0.6027

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 0.9728 - accuracy: 0.6055

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 0.9647 - accuracy: 0.6076

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 0.9628 - accuracy: 0.6062

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 0.9781 - accuracy: 0.6080

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 0.9847 - accuracy: 0.6094

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 0.9826 - accuracy: 0.6034

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 0.9844 - accuracy: 0.6027

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 0.9890 - accuracy: 0.6000

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 0.9842 - accuracy: 0.6016

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 0.9776 - accuracy: 0.6048

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 0.9841 - accuracy: 0.6042

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 0.9820 - accuracy: 0.6053

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 0.9797 - accuracy: 0.6031

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 0.9670 - accuracy: 0.6116

.. parsed-literal::

    22/92 [======>.......................] - ETA: 4s - loss: 0.9605 - accuracy: 0.6151

.. parsed-literal::

    23/92 [======>.......................] - ETA: 4s - loss: 0.9640 - accuracy: 0.6209

.. parsed-literal::

    24/92 [======>.......................] - ETA: 4s - loss: 0.9640 - accuracy: 0.6250

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 0.9682 - accuracy: 0.6225

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 0.9630 - accuracy: 0.6238

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 0.9672 - accuracy: 0.6238

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 0.9672 - accuracy: 0.6239

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 0.9636 - accuracy: 0.6293

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 0.9603 - accuracy: 0.6271

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 0.9647 - accuracy: 0.6230

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 0.9598 - accuracy: 0.6279

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 0.9598 - accuracy: 0.6297

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 0.9581 - accuracy: 0.6305

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 0.9498 - accuracy: 0.6366

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 0.9522 - accuracy: 0.6345

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 0.9531 - accuracy: 0.6334

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 0.9606 - accuracy: 0.6291

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 0.9587 - accuracy: 0.6290

.. parsed-literal::

    40/92 [============>.................] - ETA: 3s - loss: 0.9551 - accuracy: 0.6305

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 0.9482 - accuracy: 0.6341

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 0.9506 - accuracy: 0.6332

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 0.9466 - accuracy: 0.6359

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 0.9509 - accuracy: 0.6378

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 0.9499 - accuracy: 0.6396

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 0.9500 - accuracy: 0.6393

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 0.9504 - accuracy: 0.6390

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 0.9518 - accuracy: 0.6387

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 0.9495 - accuracy: 0.6378

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 0.9503 - accuracy: 0.6381

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 0.9491 - accuracy: 0.6385

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 0.9491 - accuracy: 0.6370

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 0.9449 - accuracy: 0.6397

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 0.9402 - accuracy: 0.6412

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 0.9383 - accuracy: 0.6409

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 0.9381 - accuracy: 0.6412

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 0.9406 - accuracy: 0.6382

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 0.9430 - accuracy: 0.6358

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 0.9451 - accuracy: 0.6372

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 0.9468 - accuracy: 0.6354

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 0.9452 - accuracy: 0.6363

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 0.9469 - accuracy: 0.6346

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 0.9491 - accuracy: 0.6349

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 0.9480 - accuracy: 0.6367

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 0.9510 - accuracy: 0.6356

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 0.9494 - accuracy: 0.6359

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 0.9474 - accuracy: 0.6357

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 0.9450 - accuracy: 0.6392

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 0.9457 - accuracy: 0.6381

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 0.9477 - accuracy: 0.6362

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 0.9493 - accuracy: 0.6356

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 0.9487 - accuracy: 0.6359

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 0.9477 - accuracy: 0.6370

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 0.9509 - accuracy: 0.6347

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 0.9501 - accuracy: 0.6350

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 0.9478 - accuracy: 0.6357

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 0.9471 - accuracy: 0.6356

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 0.9497 - accuracy: 0.6354

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 0.9489 - accuracy: 0.6361

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 0.9459 - accuracy: 0.6359

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 0.9472 - accuracy: 0.6354

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 0.9462 - accuracy: 0.6349

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 0.9443 - accuracy: 0.6352

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 0.9420 - accuracy: 0.6358

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 0.9432 - accuracy: 0.6349

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 0.9399 - accuracy: 0.6363

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 0.9389 - accuracy: 0.6365

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 0.9367 - accuracy: 0.6374

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 0.9366 - accuracy: 0.6366

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 0.9396 - accuracy: 0.6354

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 0.9369 - accuracy: 0.6363

.. parsed-literal::

    92/92 [==============================] - 6s 64ms/step - loss: 0.9372 - accuracy: 0.6356 - val_loss: 0.9819 - val_accuracy: 0.6253


.. parsed-literal::

    Epoch 4/15


.. parsed-literal::

     1/92 [..............................] - ETA: 7s - loss: 0.8413 - accuracy: 0.7500

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 0.9868 - accuracy: 0.6250

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 0.9735 - accuracy: 0.6042

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 0.9745 - accuracy: 0.5859

.. parsed-literal::

     5/92 [>.............................] - ETA: 5s - loss: 0.9939 - accuracy: 0.5750

.. parsed-literal::

     6/92 [>.............................] - ETA: 5s - loss: 0.9713 - accuracy: 0.5938

.. parsed-literal::

     7/92 [=>............................] - ETA: 5s - loss: 0.9533 - accuracy: 0.6071

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 0.9529 - accuracy: 0.6094

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 0.9444 - accuracy: 0.6181

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 0.9599 - accuracy: 0.6125

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 0.9500 - accuracy: 0.6165

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 0.9693 - accuracy: 0.6120

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 0.9650 - accuracy: 0.6106

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 0.9589 - accuracy: 0.6116

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 0.9613 - accuracy: 0.6125

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 0.9671 - accuracy: 0.6113

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 0.9643 - accuracy: 0.6121

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 0.9484 - accuracy: 0.6181

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 0.9441 - accuracy: 0.6234

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 0.9482 - accuracy: 0.6250

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 0.9561 - accuracy: 0.6190

.. parsed-literal::

    22/92 [======>.......................] - ETA: 4s - loss: 0.9517 - accuracy: 0.6193

.. parsed-literal::

    23/92 [======>.......................] - ETA: 4s - loss: 0.9468 - accuracy: 0.6250

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 0.9563 - accuracy: 0.6211

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 0.9586 - accuracy: 0.6200

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 0.9533 - accuracy: 0.6202

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 0.9453 - accuracy: 0.6250

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 0.9416 - accuracy: 0.6261

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 0.9533 - accuracy: 0.6207

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 0.9476 - accuracy: 0.6260

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 0.9450 - accuracy: 0.6290

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 0.9522 - accuracy: 0.6299

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 0.9465 - accuracy: 0.6335

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 0.9410 - accuracy: 0.6388

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 0.9429 - accuracy: 0.6384

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 0.9471 - accuracy: 0.6389

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 0.9451 - accuracy: 0.6377

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 0.9319 - accuracy: 0.6411

.. parsed-literal::

    40/92 [============>.................] - ETA: 3s - loss: 0.9285 - accuracy: 0.6447

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 0.9218 - accuracy: 0.6488

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 0.9261 - accuracy: 0.6460

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 0.9321 - accuracy: 0.6418

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 0.9341 - accuracy: 0.6414

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 0.9355 - accuracy: 0.6404

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 0.9339 - accuracy: 0.6414

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 0.9331 - accuracy: 0.6410

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 0.9314 - accuracy: 0.6427

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 0.9313 - accuracy: 0.6436

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 0.9339 - accuracy: 0.6420

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 0.9309 - accuracy: 0.6422

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 0.9269 - accuracy: 0.6431

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 0.9260 - accuracy: 0.6440

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 0.9269 - accuracy: 0.6442

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 0.9258 - accuracy: 0.6444

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 0.9276 - accuracy: 0.6446

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 0.9288 - accuracy: 0.6437

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 0.9274 - accuracy: 0.6429

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 0.9248 - accuracy: 0.6431

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 0.9217 - accuracy: 0.6444

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 0.9250 - accuracy: 0.6415

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 0.9239 - accuracy: 0.6417

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 0.9220 - accuracy: 0.6419

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 0.9191 - accuracy: 0.6436

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 0.9190 - accuracy: 0.6458

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 0.9183 - accuracy: 0.6478

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 0.9169 - accuracy: 0.6484

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 0.9166 - accuracy: 0.6494

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 0.9176 - accuracy: 0.6491

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 0.9180 - accuracy: 0.6501

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 0.9144 - accuracy: 0.6519

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 0.9113 - accuracy: 0.6533

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 0.9074 - accuracy: 0.6538

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 0.9082 - accuracy: 0.6530

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 0.9059 - accuracy: 0.6538

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 0.9033 - accuracy: 0.6547

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 0.9011 - accuracy: 0.6559

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 0.9008 - accuracy: 0.6555

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 0.8999 - accuracy: 0.6556

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 0.9011 - accuracy: 0.6544

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 0.8991 - accuracy: 0.6556

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 0.8995 - accuracy: 0.6552

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 0.8979 - accuracy: 0.6563

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 0.8983 - accuracy: 0.6552

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 0.8951 - accuracy: 0.6563

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 0.8956 - accuracy: 0.6567

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 0.8955 - accuracy: 0.6563

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 0.8934 - accuracy: 0.6571

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 0.8924 - accuracy: 0.6574

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 0.8894 - accuracy: 0.6595

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 0.8905 - accuracy: 0.6584

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 0.8891 - accuracy: 0.6587

.. parsed-literal::

    92/92 [==============================] - 6s 64ms/step - loss: 0.8891 - accuracy: 0.6587 - val_loss: 1.0045 - val_accuracy: 0.6322


.. parsed-literal::

    Epoch 5/15


.. parsed-literal::

     1/92 [..............................] - ETA: 7s - loss: 0.5347 - accuracy: 0.7812

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 0.8282 - accuracy: 0.6875

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 0.8240 - accuracy: 0.6771

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 0.7797 - accuracy: 0.7031

.. parsed-literal::

     5/92 [>.............................] - ETA: 5s - loss: 0.8396 - accuracy: 0.6938

.. parsed-literal::

     6/92 [>.............................] - ETA: 5s - loss: 0.8157 - accuracy: 0.6875

.. parsed-literal::

     7/92 [=>............................] - ETA: 4s - loss: 0.8171 - accuracy: 0.6830

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 0.8465 - accuracy: 0.6719

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 0.8600 - accuracy: 0.6632

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 0.8476 - accuracy: 0.6750

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 0.8530 - accuracy: 0.6705

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 0.8602 - accuracy: 0.6615

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 0.8588 - accuracy: 0.6587

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 0.8581 - accuracy: 0.6607

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 0.8581 - accuracy: 0.6583

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 0.8535 - accuracy: 0.6641

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 0.8654 - accuracy: 0.6599

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 0.8635 - accuracy: 0.6597

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 0.8627 - accuracy: 0.6579

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 0.8587 - accuracy: 0.6609

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 0.8685 - accuracy: 0.6548

.. parsed-literal::

    22/92 [======>.......................] - ETA: 4s - loss: 0.8540 - accuracy: 0.6605

.. parsed-literal::

    23/92 [======>.......................] - ETA: 4s - loss: 0.8517 - accuracy: 0.6617

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 0.8508 - accuracy: 0.6654

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 0.8500 - accuracy: 0.6625

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 0.8434 - accuracy: 0.6683

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 0.8412 - accuracy: 0.6678

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 0.8394 - accuracy: 0.6685

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 0.8331 - accuracy: 0.6713

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 0.8308 - accuracy: 0.6719

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 0.8351 - accuracy: 0.6683

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 0.8378 - accuracy: 0.6660

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 0.8546 - accuracy: 0.6591

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 0.8543 - accuracy: 0.6590

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 0.8511 - accuracy: 0.6589

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 0.8442 - accuracy: 0.6623

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 0.8486 - accuracy: 0.6622

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 0.8449 - accuracy: 0.6645

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 0.8460 - accuracy: 0.6643

.. parsed-literal::

    40/92 [============>.................] - ETA: 3s - loss: 0.8448 - accuracy: 0.6664

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 0.8407 - accuracy: 0.6692

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 0.8451 - accuracy: 0.6674

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 0.8440 - accuracy: 0.6686

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 0.8454 - accuracy: 0.6676

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 0.8423 - accuracy: 0.6694

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 0.8432 - accuracy: 0.6698

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 0.8477 - accuracy: 0.6669

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 0.8454 - accuracy: 0.6686

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 0.8445 - accuracy: 0.6690

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 0.8387 - accuracy: 0.6731

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 0.8379 - accuracy: 0.6721

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 0.8428 - accuracy: 0.6688

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 0.8437 - accuracy: 0.6674

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 0.8423 - accuracy: 0.6689

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 0.8406 - accuracy: 0.6693

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 0.8403 - accuracy: 0.6707

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 0.8433 - accuracy: 0.6705

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 0.8440 - accuracy: 0.6702

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 0.8414 - accuracy: 0.6710

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 0.8417 - accuracy: 0.6708

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 0.8407 - accuracy: 0.6711

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 0.8390 - accuracy: 0.6703

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 0.8384 - accuracy: 0.6711

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 0.8421 - accuracy: 0.6699

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 0.8398 - accuracy: 0.6716

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 0.8425 - accuracy: 0.6704

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 0.8433 - accuracy: 0.6702

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 0.8431 - accuracy: 0.6709

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 0.8402 - accuracy: 0.6720

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 0.8377 - accuracy: 0.6731

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 0.8345 - accuracy: 0.6738

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 0.8345 - accuracy: 0.6723

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 0.8324 - accuracy: 0.6733

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 0.8316 - accuracy: 0.6731

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 0.8370 - accuracy: 0.6700

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 0.8352 - accuracy: 0.6694

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 0.8360 - accuracy: 0.6692

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 0.8342 - accuracy: 0.6694

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 0.8317 - accuracy: 0.6697

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 0.8337 - accuracy: 0.6695

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 0.8341 - accuracy: 0.6701

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 0.8341 - accuracy: 0.6699

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 0.8337 - accuracy: 0.6705

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 0.8340 - accuracy: 0.6704

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 0.8338 - accuracy: 0.6713

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 0.8355 - accuracy: 0.6718

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 0.8383 - accuracy: 0.6709

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 0.8385 - accuracy: 0.6718

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 0.8399 - accuracy: 0.6713

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 0.8397 - accuracy: 0.6708

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 0.8391 - accuracy: 0.6713

.. parsed-literal::

    92/92 [==============================] - 6s 63ms/step - loss: 0.8391 - accuracy: 0.6713 - val_loss: 0.8384 - val_accuracy: 0.6717


.. parsed-literal::

    Epoch 6/15


.. parsed-literal::

     1/92 [..............................] - ETA: 7s - loss: 0.6015 - accuracy: 0.7812

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 0.7103 - accuracy: 0.7500

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 0.6752 - accuracy: 0.7604

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 0.7022 - accuracy: 0.7578

.. parsed-literal::

     5/92 [>.............................] - ETA: 5s - loss: 0.7190 - accuracy: 0.7437

.. parsed-literal::

     6/92 [>.............................] - ETA: 5s - loss: 0.7377 - accuracy: 0.7240

.. parsed-literal::

     7/92 [=>............................] - ETA: 4s - loss: 0.7689 - accuracy: 0.7009

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 0.7552 - accuracy: 0.7188

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 0.7542 - accuracy: 0.7292

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 0.7553 - accuracy: 0.7188

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 0.7505 - accuracy: 0.7216

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 0.7368 - accuracy: 0.7292

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 0.7472 - accuracy: 0.7332

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 0.7420 - accuracy: 0.7344

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 0.7418 - accuracy: 0.7354

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 0.7379 - accuracy: 0.7363

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 0.7496 - accuracy: 0.7335

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 0.7593 - accuracy: 0.7292

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 0.7567 - accuracy: 0.7319

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 0.7587 - accuracy: 0.7281

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 0.7535 - accuracy: 0.7277

.. parsed-literal::

    22/92 [======>.......................] - ETA: 4s - loss: 0.7628 - accuracy: 0.7216

.. parsed-literal::

    23/92 [======>.......................] - ETA: 3s - loss: 0.7663 - accuracy: 0.7133

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 0.7635 - accuracy: 0.7188

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 0.7695 - accuracy: 0.7163

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 0.7762 - accuracy: 0.7127

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 0.7826 - accuracy: 0.7060

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 0.7755 - accuracy: 0.7098

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 0.7764 - accuracy: 0.7123

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 0.7845 - accuracy: 0.7073

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 0.7861 - accuracy: 0.7056

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 0.7851 - accuracy: 0.7061

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 0.7875 - accuracy: 0.7036

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 0.7856 - accuracy: 0.7040

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 0.7872 - accuracy: 0.7027

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 0.7830 - accuracy: 0.7066

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 0.7871 - accuracy: 0.7027

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 0.7895 - accuracy: 0.7015

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 0.7911 - accuracy: 0.6987

.. parsed-literal::

    40/92 [============>.................] - ETA: 3s - loss: 0.7879 - accuracy: 0.7016

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 0.7874 - accuracy: 0.7020

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 0.7844 - accuracy: 0.7024

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 0.7862 - accuracy: 0.7013

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 0.7804 - accuracy: 0.7024

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 0.7768 - accuracy: 0.7042

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 0.7726 - accuracy: 0.7058

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 0.7693 - accuracy: 0.7061

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 0.7739 - accuracy: 0.7044

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 0.7702 - accuracy: 0.7041

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 0.7671 - accuracy: 0.7050

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 0.7659 - accuracy: 0.7071

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 0.7685 - accuracy: 0.7055

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 0.7706 - accuracy: 0.7028

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 0.7693 - accuracy: 0.7025

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 0.7698 - accuracy: 0.7006

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 0.7716 - accuracy: 0.6987

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 0.7730 - accuracy: 0.6990

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 0.7735 - accuracy: 0.7004

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 0.7717 - accuracy: 0.7002

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 0.7700 - accuracy: 0.7010

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 0.7683 - accuracy: 0.7039

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 0.7662 - accuracy: 0.7046

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 0.7682 - accuracy: 0.7044

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 0.7728 - accuracy: 0.7036

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 0.7703 - accuracy: 0.7038

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 0.7701 - accuracy: 0.7036

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 0.7730 - accuracy: 0.7024

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 0.7730 - accuracy: 0.7017

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 0.7741 - accuracy: 0.7020

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 0.7743 - accuracy: 0.7018

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 0.7751 - accuracy: 0.7007

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 0.7783 - accuracy: 0.7005

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 0.7768 - accuracy: 0.7016

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 0.7776 - accuracy: 0.7006

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 0.7747 - accuracy: 0.7017

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 0.7767 - accuracy: 0.7015

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 0.7773 - accuracy: 0.7017

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 0.7751 - accuracy: 0.7031

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 0.7735 - accuracy: 0.7041

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 0.7745 - accuracy: 0.7035

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 0.7740 - accuracy: 0.7033

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 0.7738 - accuracy: 0.7035

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 0.7833 - accuracy: 0.7003

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 0.7832 - accuracy: 0.7001

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 0.7851 - accuracy: 0.6993

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 0.7876 - accuracy: 0.6984

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 0.7877 - accuracy: 0.6983

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 0.7879 - accuracy: 0.6985

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 0.7858 - accuracy: 0.6987

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 0.7869 - accuracy: 0.6986

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 0.7884 - accuracy: 0.6982

.. parsed-literal::

    92/92 [==============================] - 6s 64ms/step - loss: 0.7884 - accuracy: 0.6982 - val_loss: 0.9075 - val_accuracy: 0.6526


.. parsed-literal::

    Epoch 7/15


.. parsed-literal::

     1/92 [..............................] - ETA: 7s - loss: 0.7830 - accuracy: 0.6875

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 0.8130 - accuracy: 0.6562

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 0.7392 - accuracy: 0.6875

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 0.7903 - accuracy: 0.6484

.. parsed-literal::

     5/92 [>.............................] - ETA: 5s - loss: 0.8157 - accuracy: 0.6375

.. parsed-literal::

     6/92 [>.............................] - ETA: 5s - loss: 0.8039 - accuracy: 0.6615

.. parsed-literal::

     7/92 [=>............................] - ETA: 4s - loss: 0.7979 - accuracy: 0.6696

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 0.8426 - accuracy: 0.6523

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 0.8151 - accuracy: 0.6701

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 0.8119 - accuracy: 0.6844

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 0.7888 - accuracy: 0.6903

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 0.7753 - accuracy: 0.6979

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 0.7754 - accuracy: 0.6995

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 0.7600 - accuracy: 0.7031

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 0.7733 - accuracy: 0.6938

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 0.7810 - accuracy: 0.6875

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 0.7784 - accuracy: 0.6912

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 0.7903 - accuracy: 0.6823

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 0.7806 - accuracy: 0.6859

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 0.7759 - accuracy: 0.6875

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 0.7713 - accuracy: 0.6920

.. parsed-literal::

    22/92 [======>.......................] - ETA: 4s - loss: 0.7682 - accuracy: 0.6946

.. parsed-literal::

    23/92 [======>.......................] - ETA: 4s - loss: 0.7721 - accuracy: 0.6929

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 0.7675 - accuracy: 0.6940

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 0.7683 - accuracy: 0.6900

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 0.7659 - accuracy: 0.6875

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 0.7671 - accuracy: 0.6887

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 0.7677 - accuracy: 0.6897

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 0.7797 - accuracy: 0.6897

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 0.7766 - accuracy: 0.6927

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 0.7714 - accuracy: 0.6966

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 0.7676 - accuracy: 0.6992

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 0.7694 - accuracy: 0.6989

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 0.7595 - accuracy: 0.7050

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 0.7597 - accuracy: 0.7045

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 0.7663 - accuracy: 0.7023

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 0.7679 - accuracy: 0.7019

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 0.7672 - accuracy: 0.7023

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 0.7648 - accuracy: 0.7051

.. parsed-literal::

    40/92 [============>.................] - ETA: 3s - loss: 0.7620 - accuracy: 0.7078

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 0.7614 - accuracy: 0.7088

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 0.7614 - accuracy: 0.7098

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 0.7586 - accuracy: 0.7115

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 0.7566 - accuracy: 0.7116

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 0.7552 - accuracy: 0.7132

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 0.7541 - accuracy: 0.7133

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 0.7552 - accuracy: 0.7108

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 0.7485 - accuracy: 0.7148

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 0.7464 - accuracy: 0.7136

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 0.7436 - accuracy: 0.7131

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 0.7401 - accuracy: 0.7144

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 0.7399 - accuracy: 0.7156

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 0.7357 - accuracy: 0.7174

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 0.7350 - accuracy: 0.7175

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 0.7383 - accuracy: 0.7169

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 0.7417 - accuracy: 0.7170

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 0.7425 - accuracy: 0.7165

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 0.7471 - accuracy: 0.7149

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 0.7455 - accuracy: 0.7171

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 0.7435 - accuracy: 0.7171

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 0.7448 - accuracy: 0.7156

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 0.7429 - accuracy: 0.7161

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 0.7447 - accuracy: 0.7172

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 0.7468 - accuracy: 0.7167

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 0.7447 - accuracy: 0.7186

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 0.7413 - accuracy: 0.7196

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 0.7403 - accuracy: 0.7186

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 0.7400 - accuracy: 0.7177

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 0.7384 - accuracy: 0.7173

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 0.7434 - accuracy: 0.7151

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 0.7434 - accuracy: 0.7143

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 0.7439 - accuracy: 0.7143

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 0.7453 - accuracy: 0.7140

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 0.7426 - accuracy: 0.7153

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 0.7432 - accuracy: 0.7145

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 0.7414 - accuracy: 0.7158

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 0.7411 - accuracy: 0.7166

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 0.7416 - accuracy: 0.7159

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 0.7407 - accuracy: 0.7167

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 0.7386 - accuracy: 0.7187

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 0.7373 - accuracy: 0.7198

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 0.7342 - accuracy: 0.7213

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 0.7329 - accuracy: 0.7220

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 0.7304 - accuracy: 0.7223

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 0.7289 - accuracy: 0.7216

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 0.7285 - accuracy: 0.7215

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 0.7315 - accuracy: 0.7204

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 0.7373 - accuracy: 0.7173

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 0.7351 - accuracy: 0.7180

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 0.7364 - accuracy: 0.7176

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 0.7362 - accuracy: 0.7176

.. parsed-literal::

    92/92 [==============================] - 6s 64ms/step - loss: 0.7362 - accuracy: 0.7176 - val_loss: 0.7885 - val_accuracy: 0.7153


.. parsed-literal::

    Epoch 8/15


.. parsed-literal::

     1/92 [..............................] - ETA: 7s - loss: 0.7718 - accuracy: 0.5938

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 0.6838 - accuracy: 0.7188

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 0.6857 - accuracy: 0.7292

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 0.8084 - accuracy: 0.6719

.. parsed-literal::

     5/92 [>.............................] - ETA: 5s - loss: 0.7643 - accuracy: 0.6875

.. parsed-literal::

     6/92 [>.............................] - ETA: 5s - loss: 0.7444 - accuracy: 0.7188

.. parsed-literal::

     7/92 [=>............................] - ETA: 4s - loss: 0.7179 - accuracy: 0.7277

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 0.7093 - accuracy: 0.7266

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 0.7077 - accuracy: 0.7326

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 0.7119 - accuracy: 0.7281

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 0.7075 - accuracy: 0.7301

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 0.7122 - accuracy: 0.7318

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 0.7176 - accuracy: 0.7284

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 0.7003 - accuracy: 0.7388

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 0.7186 - accuracy: 0.7354

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 0.7232 - accuracy: 0.7324

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 0.7188 - accuracy: 0.7353

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 0.7073 - accuracy: 0.7378

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 0.7119 - accuracy: 0.7336

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 0.7041 - accuracy: 0.7344

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 0.7081 - accuracy: 0.7307

.. parsed-literal::

    22/92 [======>.......................] - ETA: 4s - loss: 0.7100 - accuracy: 0.7287

.. parsed-literal::

    23/92 [======>.......................] - ETA: 4s - loss: 0.7107 - accuracy: 0.7283

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 0.7087 - accuracy: 0.7318

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 0.7008 - accuracy: 0.7362

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 0.6952 - accuracy: 0.7344

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 0.6970 - accuracy: 0.7361

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 0.6964 - accuracy: 0.7344

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 0.6892 - accuracy: 0.7360

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 0.6948 - accuracy: 0.7333

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 0.7024 - accuracy: 0.7319

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 0.7038 - accuracy: 0.7295

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 0.7117 - accuracy: 0.7263

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 0.7091 - accuracy: 0.7298

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 0.7223 - accuracy: 0.7286

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 0.7273 - accuracy: 0.7266

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 0.7260 - accuracy: 0.7264

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 0.7245 - accuracy: 0.7270

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 0.7188 - accuracy: 0.7292

.. parsed-literal::

    40/92 [============>.................] - ETA: 3s - loss: 0.7193 - accuracy: 0.7305

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 0.7182 - accuracy: 0.7294

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 0.7138 - accuracy: 0.7314

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 0.7154 - accuracy: 0.7297

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 0.7176 - accuracy: 0.7273

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 0.7200 - accuracy: 0.7271

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 0.7178 - accuracy: 0.7283

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 0.7154 - accuracy: 0.7287

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 0.7198 - accuracy: 0.7292

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 0.7144 - accuracy: 0.7309

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 0.7090 - accuracy: 0.7331

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 0.7074 - accuracy: 0.7347

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 0.7062 - accuracy: 0.7356

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 0.7061 - accuracy: 0.7353

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 0.7062 - accuracy: 0.7344

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 0.7064 - accuracy: 0.7347

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 0.7106 - accuracy: 0.7327

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 0.7115 - accuracy: 0.7319

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 0.7098 - accuracy: 0.7317

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 0.7093 - accuracy: 0.7331

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 0.7103 - accuracy: 0.7323

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 0.7139 - accuracy: 0.7305

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 0.7130 - accuracy: 0.7308

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 0.7115 - accuracy: 0.7331

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 0.7144 - accuracy: 0.7329

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 0.7161 - accuracy: 0.7317

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 0.7135 - accuracy: 0.7330

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 0.7142 - accuracy: 0.7318

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 0.7146 - accuracy: 0.7298

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 0.7114 - accuracy: 0.7310

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 0.7112 - accuracy: 0.7304

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 0.7125 - accuracy: 0.7306

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 0.7145 - accuracy: 0.7292

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 0.7158 - accuracy: 0.7286

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 0.7133 - accuracy: 0.7302

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 0.7120 - accuracy: 0.7296

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 0.7106 - accuracy: 0.7307

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 0.7090 - accuracy: 0.7317

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 0.7091 - accuracy: 0.7312

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 0.7135 - accuracy: 0.7282

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 0.7134 - accuracy: 0.7285

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 0.7126 - accuracy: 0.7280

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 0.7118 - accuracy: 0.7290

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 0.7100 - accuracy: 0.7297

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 0.7119 - accuracy: 0.7295

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 0.7121 - accuracy: 0.7294

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 0.7164 - accuracy: 0.7286

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 0.7147 - accuracy: 0.7292

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 0.7183 - accuracy: 0.7280

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 0.7171 - accuracy: 0.7293

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 0.7150 - accuracy: 0.7302

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 0.7136 - accuracy: 0.7318

.. parsed-literal::

    92/92 [==============================] - 6s 63ms/step - loss: 0.7137 - accuracy: 0.7313 - val_loss: 0.7696 - val_accuracy: 0.7071


.. parsed-literal::

    Epoch 9/15


.. parsed-literal::

     1/92 [..............................] - ETA: 7s - loss: 0.4792 - accuracy: 0.7500

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 0.6647 - accuracy: 0.6719

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 0.7285 - accuracy: 0.6354

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 0.7083 - accuracy: 0.6484

.. parsed-literal::

     5/92 [>.............................] - ETA: 5s - loss: 0.7050 - accuracy: 0.6750

.. parsed-literal::

     6/92 [>.............................] - ETA: 4s - loss: 0.7311 - accuracy: 0.6562

.. parsed-literal::

     7/92 [=>............................] - ETA: 4s - loss: 0.7192 - accuracy: 0.6696

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 0.7084 - accuracy: 0.6836

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 0.7224 - accuracy: 0.6875

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 0.7144 - accuracy: 0.6906

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 0.6985 - accuracy: 0.7045

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 0.7074 - accuracy: 0.7083

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 0.7115 - accuracy: 0.7067

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 0.7303 - accuracy: 0.7009

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 0.7306 - accuracy: 0.7042

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 0.7299 - accuracy: 0.7012

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 0.7318 - accuracy: 0.7040

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 0.7238 - accuracy: 0.7083

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 0.7358 - accuracy: 0.7039

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 0.7288 - accuracy: 0.7063

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 0.7289 - accuracy: 0.7083

.. parsed-literal::

    22/92 [======>.......................] - ETA: 4s - loss: 0.7162 - accuracy: 0.7145

.. parsed-literal::

    23/92 [======>.......................] - ETA: 4s - loss: 0.7079 - accuracy: 0.7215

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 0.7088 - accuracy: 0.7188

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 0.7156 - accuracy: 0.7150

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 0.7143 - accuracy: 0.7175

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 0.7109 - accuracy: 0.7188

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 0.7029 - accuracy: 0.7232

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 0.6975 - accuracy: 0.7231

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 0.7135 - accuracy: 0.7167

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 0.7166 - accuracy: 0.7157

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 0.7190 - accuracy: 0.7158

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 0.7190 - accuracy: 0.7188

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 0.7136 - accuracy: 0.7233

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 0.7114 - accuracy: 0.7241

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 0.7085 - accuracy: 0.7274

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 0.7118 - accuracy: 0.7264

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 0.7115 - accuracy: 0.7270

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 0.7133 - accuracy: 0.7260

.. parsed-literal::

    40/92 [============>.................] - ETA: 3s - loss: 0.7082 - accuracy: 0.7281

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 0.7125 - accuracy: 0.7264

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 0.7135 - accuracy: 0.7240

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 0.7126 - accuracy: 0.7246

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 0.7125 - accuracy: 0.7259

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 0.7064 - accuracy: 0.7306

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 0.7035 - accuracy: 0.7330

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 0.7003 - accuracy: 0.7347

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 0.6982 - accuracy: 0.7357

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 0.7008 - accuracy: 0.7360

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 0.7062 - accuracy: 0.7344

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 0.7034 - accuracy: 0.7347

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 0.7019 - accuracy: 0.7338

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 0.7013 - accuracy: 0.7347

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 0.7028 - accuracy: 0.7344

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 0.7061 - accuracy: 0.7341

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 0.7122 - accuracy: 0.7316

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 0.7146 - accuracy: 0.7308

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 0.7114 - accuracy: 0.7317

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 0.7108 - accuracy: 0.7320

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 0.7085 - accuracy: 0.7333

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 0.7070 - accuracy: 0.7338

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 0.7046 - accuracy: 0.7336

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 0.7029 - accuracy: 0.7338

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 0.7045 - accuracy: 0.7331

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 0.7051 - accuracy: 0.7329

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 0.7030 - accuracy: 0.7346

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 0.7008 - accuracy: 0.7352

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 0.6997 - accuracy: 0.7359

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 0.7025 - accuracy: 0.7366

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 0.7032 - accuracy: 0.7363

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 0.7094 - accuracy: 0.7334

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 0.7118 - accuracy: 0.7320

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 0.7094 - accuracy: 0.7339

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 0.7112 - accuracy: 0.7333

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 0.7080 - accuracy: 0.7339

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 0.7060 - accuracy: 0.7353

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 0.7042 - accuracy: 0.7367

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 0.7040 - accuracy: 0.7361

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 0.7031 - accuracy: 0.7363

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 0.7029 - accuracy: 0.7368

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 0.7055 - accuracy: 0.7359

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 0.7045 - accuracy: 0.7360

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 0.7057 - accuracy: 0.7358

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 0.7101 - accuracy: 0.7338

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 0.7132 - accuracy: 0.7340

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 0.7106 - accuracy: 0.7349

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 0.7097 - accuracy: 0.7358

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 0.7118 - accuracy: 0.7349

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 0.7121 - accuracy: 0.7340

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 0.7112 - accuracy: 0.7342

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 0.7091 - accuracy: 0.7347

.. parsed-literal::

    92/92 [==============================] - 6s 64ms/step - loss: 0.7091 - accuracy: 0.7347 - val_loss: 0.7808 - val_accuracy: 0.7071


.. parsed-literal::

    Epoch 10/15


.. parsed-literal::

     1/92 [..............................] - ETA: 7s - loss: 0.8407 - accuracy: 0.6875

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 0.8091 - accuracy: 0.6562

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 0.7278 - accuracy: 0.7083

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 0.6750 - accuracy: 0.7500

.. parsed-literal::

     5/92 [>.............................] - ETA: 5s - loss: 0.6470 - accuracy: 0.7688

.. parsed-literal::

     6/92 [>.............................] - ETA: 5s - loss: 0.6259 - accuracy: 0.7708

.. parsed-literal::

     7/92 [=>............................] - ETA: 4s - loss: 0.6110 - accuracy: 0.7723

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 0.6122 - accuracy: 0.7656

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 0.5900 - accuracy: 0.7778

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 0.6032 - accuracy: 0.7812

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 0.6102 - accuracy: 0.7756

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 0.6092 - accuracy: 0.7708

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 0.6153 - accuracy: 0.7620

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 0.6080 - accuracy: 0.7656

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 0.6192 - accuracy: 0.7646

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 0.6076 - accuracy: 0.7695

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 0.6137 - accuracy: 0.7721

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 0.6003 - accuracy: 0.7812

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 0.6135 - accuracy: 0.7763

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 0.6102 - accuracy: 0.7797

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 0.6310 - accuracy: 0.7708

.. parsed-literal::

    22/92 [======>.......................] - ETA: 4s - loss: 0.6310 - accuracy: 0.7685

.. parsed-literal::

    23/92 [======>.......................] - ETA: 4s - loss: 0.6267 - accuracy: 0.7677

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 0.6296 - accuracy: 0.7656

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 0.6391 - accuracy: 0.7625

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 0.6352 - accuracy: 0.7644

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 0.6296 - accuracy: 0.7650

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 0.6431 - accuracy: 0.7578

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 0.6469 - accuracy: 0.7575

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 0.6405 - accuracy: 0.7604

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 0.6486 - accuracy: 0.7581

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 0.6497 - accuracy: 0.7578

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 0.6507 - accuracy: 0.7576

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 0.6514 - accuracy: 0.7564

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 0.6575 - accuracy: 0.7554

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 0.6556 - accuracy: 0.7561

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 0.6535 - accuracy: 0.7576

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 0.6507 - accuracy: 0.7582

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 0.6456 - accuracy: 0.7596

.. parsed-literal::

    40/92 [============>.................] - ETA: 3s - loss: 0.6429 - accuracy: 0.7633

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 0.6377 - accuracy: 0.7660

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 0.6387 - accuracy: 0.7671

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 0.6448 - accuracy: 0.7667

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 0.6430 - accuracy: 0.7670

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 0.6483 - accuracy: 0.7646

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 0.6473 - accuracy: 0.7643

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 0.6506 - accuracy: 0.7620

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 0.6531 - accuracy: 0.7598

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 0.6572 - accuracy: 0.7577

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 0.6612 - accuracy: 0.7550

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 0.6583 - accuracy: 0.7555

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 0.6571 - accuracy: 0.7554

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 0.6606 - accuracy: 0.7553

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 0.6624 - accuracy: 0.7558

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 0.6655 - accuracy: 0.7534

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 0.6676 - accuracy: 0.7533

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 0.6708 - accuracy: 0.7516

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 0.6728 - accuracy: 0.7505

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 0.6714 - accuracy: 0.7516

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 0.6724 - accuracy: 0.7516

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 0.6726 - accuracy: 0.7505

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 0.6810 - accuracy: 0.7470

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 0.6808 - accuracy: 0.7465

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 0.6821 - accuracy: 0.7466

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 0.6835 - accuracy: 0.7452

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 0.6887 - accuracy: 0.7429

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 0.6876 - accuracy: 0.7435

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 0.6869 - accuracy: 0.7440

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 0.6908 - accuracy: 0.7418

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 0.6896 - accuracy: 0.7415

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 0.6898 - accuracy: 0.7416

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 0.6901 - accuracy: 0.7435

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 0.6882 - accuracy: 0.7449

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 0.6892 - accuracy: 0.7441

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 0.6885 - accuracy: 0.7446

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 0.6871 - accuracy: 0.7451

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 0.6867 - accuracy: 0.7447

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 0.6847 - accuracy: 0.7452

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 0.6835 - accuracy: 0.7453

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 0.6852 - accuracy: 0.7437

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 0.6888 - accuracy: 0.7438

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 0.6866 - accuracy: 0.7450

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 0.6882 - accuracy: 0.7447

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 0.6890 - accuracy: 0.7444

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 0.6874 - accuracy: 0.7437

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 0.6880 - accuracy: 0.7442

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 0.6862 - accuracy: 0.7453

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 0.6876 - accuracy: 0.7443

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 0.6862 - accuracy: 0.7454

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 0.6847 - accuracy: 0.7452

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 0.6835 - accuracy: 0.7459

.. parsed-literal::

    92/92 [==============================] - 6s 63ms/step - loss: 0.6835 - accuracy: 0.7459 - val_loss: 0.9002 - val_accuracy: 0.6730


.. parsed-literal::

    Epoch 11/15


.. parsed-literal::

     1/92 [..............................] - ETA: 7s - loss: 0.6997 - accuracy: 0.7500

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 0.6637 - accuracy: 0.7969

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 0.5600 - accuracy: 0.8125

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 0.6066 - accuracy: 0.8125

.. parsed-literal::

     5/92 [>.............................] - ETA: 5s - loss: 0.6090 - accuracy: 0.8125

.. parsed-literal::

     6/92 [>.............................] - ETA: 5s - loss: 0.5763 - accuracy: 0.8281

.. parsed-literal::

     7/92 [=>............................] - ETA: 5s - loss: 0.6029 - accuracy: 0.8125

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 0.6080 - accuracy: 0.8047

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 0.5905 - accuracy: 0.8125

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 0.6063 - accuracy: 0.8031

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 0.6313 - accuracy: 0.7898

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 0.6357 - accuracy: 0.7760

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 0.6504 - accuracy: 0.7764

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 0.6348 - accuracy: 0.7812

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 0.6366 - accuracy: 0.7792

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 0.6385 - accuracy: 0.7773

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 0.6386 - accuracy: 0.7757

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 0.6438 - accuracy: 0.7760

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 0.6324 - accuracy: 0.7796

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 0.6364 - accuracy: 0.7734

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 0.6387 - accuracy: 0.7693

.. parsed-literal::

    22/92 [======>.......................] - ETA: 4s - loss: 0.6338 - accuracy: 0.7713

.. parsed-literal::

    23/92 [======>.......................] - ETA: 4s - loss: 0.6388 - accuracy: 0.7717

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 0.6430 - accuracy: 0.7708

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 0.6356 - accuracy: 0.7738

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 0.6364 - accuracy: 0.7728

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 0.6301 - accuracy: 0.7743

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 0.6232 - accuracy: 0.7779

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 0.6148 - accuracy: 0.7823

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 0.6184 - accuracy: 0.7802

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 0.6230 - accuracy: 0.7812

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 0.6242 - accuracy: 0.7812

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 0.6358 - accuracy: 0.7746

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 0.6331 - accuracy: 0.7767

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 0.6304 - accuracy: 0.7777

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 0.6332 - accuracy: 0.7769

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 0.6383 - accuracy: 0.7762

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 0.6389 - accuracy: 0.7730

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 0.6405 - accuracy: 0.7700

.. parsed-literal::

    40/92 [============>.................] - ETA: 3s - loss: 0.6408 - accuracy: 0.7703

.. parsed-literal::

    41/92 [============>.................] - ETA: 3s - loss: 0.6390 - accuracy: 0.7691

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 0.6378 - accuracy: 0.7701

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 0.6340 - accuracy: 0.7711

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 0.6344 - accuracy: 0.7720

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 0.6366 - accuracy: 0.7729

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 0.6338 - accuracy: 0.7751

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 0.6323 - accuracy: 0.7759

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 0.6350 - accuracy: 0.7728

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 0.6313 - accuracy: 0.7723

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 0.6308 - accuracy: 0.7725

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 0.6299 - accuracy: 0.7714

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 0.6294 - accuracy: 0.7728

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 0.6273 - accuracy: 0.7736

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 0.6318 - accuracy: 0.7708

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 0.6280 - accuracy: 0.7722

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 0.6313 - accuracy: 0.7718

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 0.6320 - accuracy: 0.7697

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 0.6299 - accuracy: 0.7710

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 0.6278 - accuracy: 0.7717

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 0.6281 - accuracy: 0.7724

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 0.6314 - accuracy: 0.7710

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 0.6315 - accuracy: 0.7712

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 0.6321 - accuracy: 0.7698

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 0.6350 - accuracy: 0.7695

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 0.6364 - accuracy: 0.7688

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 0.6380 - accuracy: 0.7694

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 0.6380 - accuracy: 0.7705

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 0.6354 - accuracy: 0.7707

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 0.6328 - accuracy: 0.7708

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 0.6329 - accuracy: 0.7701

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 0.6336 - accuracy: 0.7698

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 0.6338 - accuracy: 0.7700

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 0.6364 - accuracy: 0.7688

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 0.6333 - accuracy: 0.7690

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 0.6306 - accuracy: 0.7696

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 0.6292 - accuracy: 0.7701

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 0.6313 - accuracy: 0.7695

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 0.6321 - accuracy: 0.7692

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 0.6300 - accuracy: 0.7698

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 0.6285 - accuracy: 0.7707

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 0.6342 - accuracy: 0.7677

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 0.6359 - accuracy: 0.7670

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 0.6340 - accuracy: 0.7687

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 0.6325 - accuracy: 0.7692

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 0.6306 - accuracy: 0.7704

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 0.6291 - accuracy: 0.7709

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 0.6309 - accuracy: 0.7699

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 0.6307 - accuracy: 0.7697

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 0.6314 - accuracy: 0.7688

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 0.6320 - accuracy: 0.7676

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 0.6316 - accuracy: 0.7677

.. parsed-literal::

    92/92 [==============================] - 6s 64ms/step - loss: 0.6316 - accuracy: 0.7677 - val_loss: 0.7573 - val_accuracy: 0.7234


.. parsed-literal::

    Epoch 12/15


.. parsed-literal::

     1/92 [..............................] - ETA: 7s - loss: 0.8004 - accuracy: 0.6562

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 0.6768 - accuracy: 0.6719

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 0.6565 - accuracy: 0.6979

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 0.6688 - accuracy: 0.7109

.. parsed-literal::

     5/92 [>.............................] - ETA: 5s - loss: 0.6383 - accuracy: 0.7437

.. parsed-literal::

     6/92 [>.............................] - ETA: 5s - loss: 0.5995 - accuracy: 0.7604

.. parsed-literal::

     7/92 [=>............................] - ETA: 4s - loss: 0.6117 - accuracy: 0.7500

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 0.6030 - accuracy: 0.7539

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 0.5878 - accuracy: 0.7569

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 0.6006 - accuracy: 0.7563

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 0.6181 - accuracy: 0.7528

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 0.6226 - accuracy: 0.7448

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 0.6085 - accuracy: 0.7524

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 0.6014 - accuracy: 0.7589

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 0.5875 - accuracy: 0.7625

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 0.5869 - accuracy: 0.7695

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 0.5892 - accuracy: 0.7702

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 0.5839 - accuracy: 0.7717

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 0.5849 - accuracy: 0.7737

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 0.5859 - accuracy: 0.7756

.. parsed-literal::

    22/92 [======>.......................] - ETA: 3s - loss: 0.5909 - accuracy: 0.7759

.. parsed-literal::

    23/92 [======>.......................] - ETA: 3s - loss: 0.5911 - accuracy: 0.7761

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 0.5886 - accuracy: 0.7789

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 0.5934 - accuracy: 0.7765

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 0.5863 - accuracy: 0.7791

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 0.5797 - accuracy: 0.7815

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 0.5768 - accuracy: 0.7838

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 0.5767 - accuracy: 0.7826

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 0.5746 - accuracy: 0.7826

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 0.5717 - accuracy: 0.7856

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 0.5763 - accuracy: 0.7805

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 0.5833 - accuracy: 0.7796

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 0.5847 - accuracy: 0.7796

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 0.5855 - accuracy: 0.7797

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 0.5890 - accuracy: 0.7780

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 0.5891 - accuracy: 0.7781

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 0.6000 - accuracy: 0.7740

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 0.6020 - accuracy: 0.7726

.. parsed-literal::

    40/92 [============>.................] - ETA: 2s - loss: 0.5972 - accuracy: 0.7744

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 0.6068 - accuracy: 0.7738

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 0.6055 - accuracy: 0.7740

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 0.6037 - accuracy: 0.7749

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 0.6038 - accuracy: 0.7757

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 0.6059 - accuracy: 0.7744

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 0.6082 - accuracy: 0.7739

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 0.6079 - accuracy: 0.7734

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 0.6077 - accuracy: 0.7736

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 0.6086 - accuracy: 0.7744

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 0.6131 - accuracy: 0.7720

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 0.6204 - accuracy: 0.7709

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 0.6173 - accuracy: 0.7711

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 0.6150 - accuracy: 0.7719

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 0.6169 - accuracy: 0.7727

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 0.6156 - accuracy: 0.7734

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 0.6148 - accuracy: 0.7735

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 0.6142 - accuracy: 0.7737

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 0.6148 - accuracy: 0.7727

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 0.6169 - accuracy: 0.7723

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 0.6155 - accuracy: 0.7735

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 0.6160 - accuracy: 0.7731

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 0.6143 - accuracy: 0.7743

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 0.6152 - accuracy: 0.7754

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 0.6211 - accuracy: 0.7735

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 0.6203 - accuracy: 0.7746

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 0.6174 - accuracy: 0.7757

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 0.6154 - accuracy: 0.7762

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 0.6139 - accuracy: 0.7768

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 0.6139 - accuracy: 0.7768

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 0.6160 - accuracy: 0.7760

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 0.6155 - accuracy: 0.7769

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 0.6154 - accuracy: 0.7766

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 0.6150 - accuracy: 0.7771

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 0.6125 - accuracy: 0.7771

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 0.6164 - accuracy: 0.7747

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 0.6169 - accuracy: 0.7752

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 0.6176 - accuracy: 0.7748

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 0.6213 - accuracy: 0.7733

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 0.6188 - accuracy: 0.7746

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 0.6200 - accuracy: 0.7751

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 0.6205 - accuracy: 0.7740

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 0.6176 - accuracy: 0.7752

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 0.6150 - accuracy: 0.7768

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 0.6133 - accuracy: 0.7772

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 0.6123 - accuracy: 0.7769

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 0.6107 - accuracy: 0.7762

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 0.6103 - accuracy: 0.7767

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 0.6126 - accuracy: 0.7753

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 0.6118 - accuracy: 0.7761

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 0.6144 - accuracy: 0.7747

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 0.6128 - accuracy: 0.7755

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 0.6120 - accuracy: 0.7749

.. parsed-literal::

    92/92 [==============================] - 6s 63ms/step - loss: 0.6120 - accuracy: 0.7749 - val_loss: 0.6896 - val_accuracy: 0.7398


.. parsed-literal::

    Epoch 13/15


.. parsed-literal::

     1/92 [..............................] - ETA: 6s - loss: 0.4050 - accuracy: 0.7812

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 0.4805 - accuracy: 0.7812

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 0.4963 - accuracy: 0.7812

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 0.4684 - accuracy: 0.7969

.. parsed-literal::

     5/92 [>.............................] - ETA: 5s - loss: 0.4946 - accuracy: 0.8000

.. parsed-literal::

     6/92 [>.............................] - ETA: 4s - loss: 0.4732 - accuracy: 0.8073

.. parsed-literal::

     7/92 [=>............................] - ETA: 4s - loss: 0.4631 - accuracy: 0.8125

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 0.4665 - accuracy: 0.8203

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 0.4830 - accuracy: 0.8125

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 0.4927 - accuracy: 0.8062

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 0.4943 - accuracy: 0.8040

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 0.4980 - accuracy: 0.8073

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 0.5039 - accuracy: 0.8005

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 0.5134 - accuracy: 0.7924

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 0.5083 - accuracy: 0.7937

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 0.5067 - accuracy: 0.7910

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 0.5177 - accuracy: 0.7904

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 0.5172 - accuracy: 0.7882

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 0.5094 - accuracy: 0.7928

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 0.5067 - accuracy: 0.7984

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 0.5090 - accuracy: 0.8006

.. parsed-literal::

    22/92 [======>.......................] - ETA: 4s - loss: 0.5194 - accuracy: 0.7983

.. parsed-literal::

    23/92 [======>.......................] - ETA: 3s - loss: 0.5277 - accuracy: 0.7935

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 0.5301 - accuracy: 0.7930

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 0.5275 - accuracy: 0.7912

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 0.5255 - accuracy: 0.7897

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 0.5285 - accuracy: 0.7894

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 0.5388 - accuracy: 0.7846

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 0.5330 - accuracy: 0.7866

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 0.5328 - accuracy: 0.7886

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 0.5308 - accuracy: 0.7933

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 0.5324 - accuracy: 0.7920

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 0.5278 - accuracy: 0.7935

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 0.5300 - accuracy: 0.7914

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 0.5343 - accuracy: 0.7911

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 0.5335 - accuracy: 0.7908

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 0.5315 - accuracy: 0.7922

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 0.5387 - accuracy: 0.7879

.. parsed-literal::

    40/92 [============>.................] - ETA: 3s - loss: 0.5383 - accuracy: 0.7877

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 0.5372 - accuracy: 0.7876

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 0.5362 - accuracy: 0.7882

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 0.5348 - accuracy: 0.7895

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 0.5335 - accuracy: 0.7893

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 0.5338 - accuracy: 0.7891

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 0.5390 - accuracy: 0.7869

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 0.5438 - accuracy: 0.7854

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 0.5474 - accuracy: 0.7840

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 0.5477 - accuracy: 0.7833

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 0.5484 - accuracy: 0.7814

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 0.5504 - accuracy: 0.7789

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 0.5484 - accuracy: 0.7796

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 0.5506 - accuracy: 0.7784

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 0.5502 - accuracy: 0.7797

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 0.5529 - accuracy: 0.7780

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 0.5513 - accuracy: 0.7797

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 0.5535 - accuracy: 0.7786

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 0.5507 - accuracy: 0.7808

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 0.5537 - accuracy: 0.7787

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 0.5518 - accuracy: 0.7798

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 0.5506 - accuracy: 0.7809

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 0.5568 - accuracy: 0.7778

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 0.5559 - accuracy: 0.7789

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 0.5582 - accuracy: 0.7789

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 0.5572 - accuracy: 0.7790

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 0.5577 - accuracy: 0.7785

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 0.5552 - accuracy: 0.7790

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 0.5629 - accuracy: 0.7772

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 0.5612 - accuracy: 0.7786

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 0.5657 - accuracy: 0.7764

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 0.5634 - accuracy: 0.7774

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 0.5600 - accuracy: 0.7787

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 0.5610 - accuracy: 0.7779

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 0.5596 - accuracy: 0.7788

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 0.5602 - accuracy: 0.7793

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 0.5626 - accuracy: 0.7781

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 0.5642 - accuracy: 0.7773

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 0.5638 - accuracy: 0.7777

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 0.5642 - accuracy: 0.7778

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 0.5648 - accuracy: 0.7766

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 0.5657 - accuracy: 0.7771

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 0.5666 - accuracy: 0.7764

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 0.5695 - accuracy: 0.7745

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 0.5694 - accuracy: 0.7739

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 0.5682 - accuracy: 0.7740

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 0.5680 - accuracy: 0.7737

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 0.5726 - accuracy: 0.7716

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 0.5707 - accuracy: 0.7724

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 0.5726 - accuracy: 0.7725

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 0.5740 - accuracy: 0.7719

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 0.5721 - accuracy: 0.7727

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 0.5726 - accuracy: 0.7725

.. parsed-literal::

    92/92 [==============================] - 6s 63ms/step - loss: 0.5726 - accuracy: 0.7725 - val_loss: 0.7163 - val_accuracy: 0.7180


.. parsed-literal::

    Epoch 14/15


.. parsed-literal::

     1/92 [..............................] - ETA: 7s - loss: 0.5975 - accuracy: 0.9062

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 0.4781 - accuracy: 0.8906

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 0.4433 - accuracy: 0.8646

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 0.4570 - accuracy: 0.8594

.. parsed-literal::

     5/92 [>.............................] - ETA: 5s - loss: 0.4483 - accuracy: 0.8562

.. parsed-literal::

     6/92 [>.............................] - ETA: 5s - loss: 0.4651 - accuracy: 0.8438

.. parsed-literal::

     7/92 [=>............................] - ETA: 4s - loss: 0.4589 - accuracy: 0.8438

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 0.4525 - accuracy: 0.8516

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 0.4693 - accuracy: 0.8472

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 0.4783 - accuracy: 0.8406

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 0.4767 - accuracy: 0.8381

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 0.5061 - accuracy: 0.8203

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 0.5182 - accuracy: 0.8101

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 0.5111 - accuracy: 0.8080

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 0.5147 - accuracy: 0.8042

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 0.5136 - accuracy: 0.8047

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 0.5157 - accuracy: 0.7996

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 0.5223 - accuracy: 0.7969

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 0.5266 - accuracy: 0.7961

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 0.5238 - accuracy: 0.8000

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 0.5234 - accuracy: 0.8006

.. parsed-literal::

    22/92 [======>.......................] - ETA: 4s - loss: 0.5287 - accuracy: 0.7969

.. parsed-literal::

    23/92 [======>.......................] - ETA: 4s - loss: 0.5319 - accuracy: 0.7962

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 0.5435 - accuracy: 0.7917

.. parsed-literal::

    25/92 [=======>......................] - ETA: 3s - loss: 0.5417 - accuracy: 0.7937

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 0.5422 - accuracy: 0.7945

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 0.5354 - accuracy: 0.7963

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 0.5289 - accuracy: 0.7980

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 0.5267 - accuracy: 0.7985

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 0.5202 - accuracy: 0.8000

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 0.5184 - accuracy: 0.8014

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 0.5188 - accuracy: 0.8037

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 0.5214 - accuracy: 0.8011

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 0.5212 - accuracy: 0.8015

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 0.5197 - accuracy: 0.8027

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 0.5141 - accuracy: 0.8038

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 0.5158 - accuracy: 0.8041

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 0.5149 - accuracy: 0.8035

.. parsed-literal::

    40/92 [============>.................] - ETA: 3s - loss: 0.5187 - accuracy: 0.8011

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 0.5209 - accuracy: 0.8014

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 0.5211 - accuracy: 0.8001

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 0.5239 - accuracy: 0.8004

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 0.5223 - accuracy: 0.8007

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 0.5200 - accuracy: 0.8017

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 0.5191 - accuracy: 0.7999

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 0.5185 - accuracy: 0.8008

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 0.5174 - accuracy: 0.8010

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 0.5174 - accuracy: 0.8006

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 0.5170 - accuracy: 0.8009

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 0.5191 - accuracy: 0.8011

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 0.5198 - accuracy: 0.8001

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 0.5283 - accuracy: 0.7962

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 0.5264 - accuracy: 0.7965

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 0.5279 - accuracy: 0.7957

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 0.5335 - accuracy: 0.7943

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 0.5352 - accuracy: 0.7930

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 0.5343 - accuracy: 0.7927

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 0.5348 - accuracy: 0.7936

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 0.5425 - accuracy: 0.7929

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 0.5398 - accuracy: 0.7927

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 0.5407 - accuracy: 0.7920

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 0.5427 - accuracy: 0.7918

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 0.5428 - accuracy: 0.7926

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 0.5445 - accuracy: 0.7905

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 0.5455 - accuracy: 0.7894

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 0.5456 - accuracy: 0.7898

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 0.5447 - accuracy: 0.7901

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 0.5428 - accuracy: 0.7914

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 0.5448 - accuracy: 0.7894

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 0.5443 - accuracy: 0.7902

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 0.5436 - accuracy: 0.7901

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 0.5469 - accuracy: 0.7891

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 0.5501 - accuracy: 0.7877

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 0.5531 - accuracy: 0.7864

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 0.5520 - accuracy: 0.7880

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 0.5527 - accuracy: 0.7879

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 0.5535 - accuracy: 0.7886

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 0.5525 - accuracy: 0.7893

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 0.5530 - accuracy: 0.7884

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 0.5529 - accuracy: 0.7883

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 0.5514 - accuracy: 0.7890

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 0.5531 - accuracy: 0.7885

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 0.5520 - accuracy: 0.7881

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 0.5553 - accuracy: 0.7872

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 0.5580 - accuracy: 0.7864

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 0.5594 - accuracy: 0.7857

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 0.5613 - accuracy: 0.7849

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 0.5613 - accuracy: 0.7856

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 0.5606 - accuracy: 0.7855

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 0.5609 - accuracy: 0.7862

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 0.5600 - accuracy: 0.7864

.. parsed-literal::

    92/92 [==============================] - 6s 64ms/step - loss: 0.5600 - accuracy: 0.7864 - val_loss: 0.7119 - val_accuracy: 0.7302


.. parsed-literal::

    Epoch 15/15


.. parsed-literal::

     1/92 [..............................] - ETA: 6s - loss: 0.3944 - accuracy: 0.8750

.. parsed-literal::

     2/92 [..............................] - ETA: 5s - loss: 0.4978 - accuracy: 0.8125

.. parsed-literal::

     3/92 [..............................] - ETA: 5s - loss: 0.4878 - accuracy: 0.8229

.. parsed-literal::

     4/92 [>.............................] - ETA: 5s - loss: 0.5509 - accuracy: 0.7969

.. parsed-literal::

     5/92 [>.............................] - ETA: 5s - loss: 0.5954 - accuracy: 0.7875

.. parsed-literal::

     6/92 [>.............................] - ETA: 4s - loss: 0.5689 - accuracy: 0.7865

.. parsed-literal::

     7/92 [=>............................] - ETA: 4s - loss: 0.5410 - accuracy: 0.7991

.. parsed-literal::

     8/92 [=>............................] - ETA: 4s - loss: 0.5200 - accuracy: 0.8086

.. parsed-literal::

     9/92 [=>............................] - ETA: 4s - loss: 0.5297 - accuracy: 0.8021

.. parsed-literal::

    10/92 [==>...........................] - ETA: 4s - loss: 0.5428 - accuracy: 0.8000

.. parsed-literal::

    11/92 [==>...........................] - ETA: 4s - loss: 0.5856 - accuracy: 0.7869

.. parsed-literal::

    12/92 [==>...........................] - ETA: 4s - loss: 0.5691 - accuracy: 0.7969

.. parsed-literal::

    13/92 [===>..........................] - ETA: 4s - loss: 0.5601 - accuracy: 0.7957

.. parsed-literal::

    14/92 [===>..........................] - ETA: 4s - loss: 0.5623 - accuracy: 0.7969

.. parsed-literal::

    15/92 [===>..........................] - ETA: 4s - loss: 0.5447 - accuracy: 0.8000

.. parsed-literal::

    16/92 [====>.........................] - ETA: 4s - loss: 0.5389 - accuracy: 0.8027

.. parsed-literal::

    17/92 [====>.........................] - ETA: 4s - loss: 0.5455 - accuracy: 0.7960

.. parsed-literal::

    18/92 [====>.........................] - ETA: 4s - loss: 0.5510 - accuracy: 0.7917

.. parsed-literal::

    19/92 [=====>........................] - ETA: 4s - loss: 0.5543 - accuracy: 0.7878

.. parsed-literal::

    20/92 [=====>........................] - ETA: 4s - loss: 0.5629 - accuracy: 0.7859

.. parsed-literal::

    21/92 [=====>........................] - ETA: 4s - loss: 0.5559 - accuracy: 0.7872

.. parsed-literal::

    22/92 [======>.......................] - ETA: 4s - loss: 0.5459 - accuracy: 0.7926

.. parsed-literal::

    23/92 [======>.......................] - ETA: 3s - loss: 0.5404 - accuracy: 0.7935

.. parsed-literal::

    24/92 [======>.......................] - ETA: 3s - loss: 0.5349 - accuracy: 0.7956

.. parsed-literal::

    26/92 [=======>......................] - ETA: 3s - loss: 0.5428 - accuracy: 0.7925

.. parsed-literal::

    27/92 [=======>......................] - ETA: 3s - loss: 0.5342 - accuracy: 0.7979

.. parsed-literal::

    28/92 [========>.....................] - ETA: 3s - loss: 0.5344 - accuracy: 0.7973

.. parsed-literal::

    29/92 [========>.....................] - ETA: 3s - loss: 0.5290 - accuracy: 0.7978

.. parsed-literal::

    30/92 [========>.....................] - ETA: 3s - loss: 0.5224 - accuracy: 0.8004

.. parsed-literal::

    31/92 [=========>....................] - ETA: 3s - loss: 0.5260 - accuracy: 0.7978

.. parsed-literal::

    32/92 [=========>....................] - ETA: 3s - loss: 0.5270 - accuracy: 0.7953

.. parsed-literal::

    33/92 [=========>....................] - ETA: 3s - loss: 0.5270 - accuracy: 0.7968

.. parsed-literal::

    34/92 [==========>...................] - ETA: 3s - loss: 0.5271 - accuracy: 0.7944

.. parsed-literal::

    35/92 [==========>...................] - ETA: 3s - loss: 0.5231 - accuracy: 0.7968

.. parsed-literal::

    36/92 [==========>...................] - ETA: 3s - loss: 0.5205 - accuracy: 0.7990

.. parsed-literal::

    37/92 [===========>..................] - ETA: 3s - loss: 0.5204 - accuracy: 0.7993

.. parsed-literal::

    38/92 [===========>..................] - ETA: 3s - loss: 0.5250 - accuracy: 0.7972

.. parsed-literal::

    39/92 [===========>..................] - ETA: 3s - loss: 0.5248 - accuracy: 0.7976

.. parsed-literal::

    40/92 [============>.................] - ETA: 2s - loss: 0.5246 - accuracy: 0.7972

.. parsed-literal::

    41/92 [============>.................] - ETA: 2s - loss: 0.5209 - accuracy: 0.7975

.. parsed-literal::

    42/92 [============>.................] - ETA: 2s - loss: 0.5147 - accuracy: 0.8009

.. parsed-literal::

    43/92 [=============>................] - ETA: 2s - loss: 0.5117 - accuracy: 0.8019

.. parsed-literal::

    44/92 [=============>................] - ETA: 2s - loss: 0.5112 - accuracy: 0.8021

.. parsed-literal::

    45/92 [=============>................] - ETA: 2s - loss: 0.5144 - accuracy: 0.7996

.. parsed-literal::

    46/92 [==============>...............] - ETA: 2s - loss: 0.5111 - accuracy: 0.8012

.. parsed-literal::

    47/92 [==============>...............] - ETA: 2s - loss: 0.5115 - accuracy: 0.8021

.. parsed-literal::

    48/92 [==============>...............] - ETA: 2s - loss: 0.5125 - accuracy: 0.8010

.. parsed-literal::

    49/92 [==============>...............] - ETA: 2s - loss: 0.5179 - accuracy: 0.7974

.. parsed-literal::

    50/92 [===============>..............] - ETA: 2s - loss: 0.5167 - accuracy: 0.7977

.. parsed-literal::

    51/92 [===============>..............] - ETA: 2s - loss: 0.5183 - accuracy: 0.7962

.. parsed-literal::

    52/92 [===============>..............] - ETA: 2s - loss: 0.5164 - accuracy: 0.7983

.. parsed-literal::

    53/92 [================>.............] - ETA: 2s - loss: 0.5176 - accuracy: 0.7974

.. parsed-literal::

    54/92 [================>.............] - ETA: 2s - loss: 0.5213 - accuracy: 0.7942

.. parsed-literal::

    55/92 [================>.............] - ETA: 2s - loss: 0.5226 - accuracy: 0.7939

.. parsed-literal::

    56/92 [=================>............] - ETA: 2s - loss: 0.5251 - accuracy: 0.7932

.. parsed-literal::

    57/92 [=================>............] - ETA: 2s - loss: 0.5270 - accuracy: 0.7935

.. parsed-literal::

    58/92 [=================>............] - ETA: 1s - loss: 0.5305 - accuracy: 0.7917

.. parsed-literal::

    59/92 [==================>...........] - ETA: 1s - loss: 0.5329 - accuracy: 0.7920

.. parsed-literal::

    60/92 [==================>...........] - ETA: 1s - loss: 0.5347 - accuracy: 0.7913

.. parsed-literal::

    61/92 [==================>...........] - ETA: 1s - loss: 0.5334 - accuracy: 0.7917

.. parsed-literal::

    62/92 [===================>..........] - ETA: 1s - loss: 0.5363 - accuracy: 0.7915

.. parsed-literal::

    63/92 [===================>..........] - ETA: 1s - loss: 0.5376 - accuracy: 0.7903

.. parsed-literal::

    64/92 [===================>..........] - ETA: 1s - loss: 0.5347 - accuracy: 0.7912

.. parsed-literal::

    65/92 [====================>.........] - ETA: 1s - loss: 0.5372 - accuracy: 0.7910

.. parsed-literal::

    66/92 [====================>.........] - ETA: 1s - loss: 0.5384 - accuracy: 0.7909

.. parsed-literal::

    67/92 [====================>.........] - ETA: 1s - loss: 0.5386 - accuracy: 0.7898

.. parsed-literal::

    68/92 [=====================>........] - ETA: 1s - loss: 0.5378 - accuracy: 0.7906

.. parsed-literal::

    69/92 [=====================>........] - ETA: 1s - loss: 0.5373 - accuracy: 0.7914

.. parsed-literal::

    70/92 [=====================>........] - ETA: 1s - loss: 0.5381 - accuracy: 0.7912

.. parsed-literal::

    71/92 [======================>.......] - ETA: 1s - loss: 0.5366 - accuracy: 0.7915

.. parsed-literal::

    72/92 [======================>.......] - ETA: 1s - loss: 0.5383 - accuracy: 0.7918

.. parsed-literal::

    73/92 [======================>.......] - ETA: 1s - loss: 0.5375 - accuracy: 0.7921

.. parsed-literal::

    74/92 [=======================>......] - ETA: 1s - loss: 0.5358 - accuracy: 0.7928

.. parsed-literal::

    75/92 [=======================>......] - ETA: 0s - loss: 0.5347 - accuracy: 0.7935

.. parsed-literal::

    76/92 [=======================>......] - ETA: 0s - loss: 0.5352 - accuracy: 0.7933

.. parsed-literal::

    77/92 [========================>.....] - ETA: 0s - loss: 0.5357 - accuracy: 0.7928

.. parsed-literal::

    78/92 [========================>.....] - ETA: 0s - loss: 0.5353 - accuracy: 0.7934

.. parsed-literal::

    79/92 [========================>.....] - ETA: 0s - loss: 0.5348 - accuracy: 0.7940

.. parsed-literal::

    80/92 [=========================>....] - ETA: 0s - loss: 0.5354 - accuracy: 0.7935

.. parsed-literal::

    81/92 [=========================>....] - ETA: 0s - loss: 0.5333 - accuracy: 0.7945

.. parsed-literal::

    82/92 [=========================>....] - ETA: 0s - loss: 0.5386 - accuracy: 0.7920

.. parsed-literal::

    83/92 [==========================>...] - ETA: 0s - loss: 0.5378 - accuracy: 0.7923

.. parsed-literal::

    84/92 [==========================>...] - ETA: 0s - loss: 0.5362 - accuracy: 0.7933

.. parsed-literal::

    85/92 [==========================>...] - ETA: 0s - loss: 0.5346 - accuracy: 0.7931

.. parsed-literal::

    86/92 [===========================>..] - ETA: 0s - loss: 0.5351 - accuracy: 0.7930

.. parsed-literal::

    87/92 [===========================>..] - ETA: 0s - loss: 0.5371 - accuracy: 0.7921

.. parsed-literal::

    88/92 [===========================>..] - ETA: 0s - loss: 0.5347 - accuracy: 0.7934

.. parsed-literal::

    89/92 [============================>.] - ETA: 0s - loss: 0.5359 - accuracy: 0.7933

.. parsed-literal::

    90/92 [============================>.] - ETA: 0s - loss: 0.5359 - accuracy: 0.7935

.. parsed-literal::

    91/92 [============================>.] - ETA: 0s - loss: 0.5351 - accuracy: 0.7934

.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 0.5345 - accuracy: 0.7936

.. parsed-literal::

    92/92 [==============================] - 6s 64ms/step - loss: 0.5345 - accuracy: 0.7936 - val_loss: 0.7319 - val_accuracy: 0.7044



.. image:: 301-tensorflow-training-openvino-nncf-with-output_files/301-tensorflow-training-openvino-nncf-with-output_3_1452.png


.. parsed-literal::

    1/1 [==============================] - ETA: 0s

.. parsed-literal::

    1/1 [==============================] - 0s 75ms/step


.. parsed-literal::

    This image most likely belongs to sunflowers with a 81.50 percent confidence.


.. parsed-literal::

    2024-01-26 00:40:37.537773: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'random_flip_input' with dtype float and shape [?,180,180,3]
    	 [[{{node random_flip_input}}]]
    2024-01-26 00:40:37.623389: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-01-26 00:40:37.633313: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'random_flip_input' with dtype float and shape [?,180,180,3]
    	 [[{{node random_flip_input}}]]
    2024-01-26 00:40:37.644579: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-01-26 00:40:37.651808: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-01-26 00:40:37.658580: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-01-26 00:40:37.669430: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-01-26 00:40:37.709186: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'sequential_1_input' with dtype float and shape [?,180,180,3]
    	 [[{{node sequential_1_input}}]]


.. parsed-literal::

    2024-01-26 00:40:37.808931: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-01-26 00:40:37.829365: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'sequential_1_input' with dtype float and shape [?,180,180,3]
    	 [[{{node sequential_1_input}}]]
    2024-01-26 00:40:37.868335: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,22,22,64]
    	 [[{{node inputs}}]]
    2024-01-26 00:40:37.893339: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-01-26 00:40:37.967640: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]


.. parsed-literal::

    2024-01-26 00:40:38.111119: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-01-26 00:40:38.248703: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,22,22,64]
    	 [[{{node inputs}}]]
    2024-01-26 00:40:38.283618: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2024-01-26 00:40:38.311335: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]


.. parsed-literal::

    2024-01-26 00:40:38.358384: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
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
    This image most likely belongs to dandelion with a 99.60 percent confidence.



.. image:: 301-tensorflow-training-openvino-nncf-with-output_files/301-tensorflow-training-openvino-nncf-with-output_3_1464.png


Imports
~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

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

`back to top ⬆️ <#Table-of-contents:>`__

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

    2024-01-26 00:40:41.533104: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [734]
    	 [[{{node Placeholder/_0}}]]
    2024-01-26 00:40:41.533348: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [734]
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
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-598/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/live.py", line 32, in run
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    self.live.refresh()
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-598/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/live.py", line 223, in refresh
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    self._live_render.set_renderable(self.renderable)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-598/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/live.py", line 203, in renderable
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    renderable = self.get_renderable()
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-598/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/live.py", line 98, in get_renderable
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    self._get_renderable()
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-598/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1537, in get_renderable
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    renderable = Group(*self.get_renderables())
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-598/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1542, in get_renderables
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    table = self.make_tasks_table(self.tasks)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-598/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1566, in make_tasks_table
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    table.add_row(
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-598/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1571, in &lt;genexpr&gt;
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    else column(task)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-598/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 528, in __call__
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    renderable = self.render(task)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-598/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/nncf/common/logging/track_progress.py", line 58, in render
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    text = super().render(task)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-598/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 787, in render
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    task_time = task.time_remaining
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-598/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/rich/progress.py", line 1039, in time_remaining
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    estimate = ceil(remaining / speed)
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-598/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
    te-packages/tensorflow/python/util/traceback_utils.py", line 153, in error_handler
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    raise e.with_traceback(filtered_tb) from None
    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  File 
    "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-598/.workspace/scm/ov-notebook/.venv/lib/python3.8/si
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

`back to top ⬆️ <#Table-of-contents:>`__

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

`back to top ⬆️ <#Table-of-contents:>`__

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

    Accuracy of the original model: 0.703
    Accuracy of the quantized model: 0.711


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

`back to top ⬆️ <#Table-of-contents:>`__

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
    This image most likely belongs to dandelion with a 99.63 percent confidence.



.. image:: 301-tensorflow-training-openvino-nncf-with-output_files/301-tensorflow-training-openvino-nncf-with-output_27_1.png


Compare Inference Speed
-----------------------

`back to top ⬆️ <#Table-of-contents:>`__

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
    [ INFO ] CPU
    [ INFO ] Build ................................. 2023.3.0-13775-ceeafaf64f3-releases/2023/3
    [ INFO ] 
    [ INFO ] 
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(CPU) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files


.. parsed-literal::

    [ INFO ] Read model took 12.29 ms
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

    [ INFO ] Compile model took 61.58 ms
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
    [ INFO ] First inference took 8.29 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            57588 iterations
    [ INFO ] Duration:         15001.83 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        2.95 ms
    [ INFO ]    Average:       2.96 ms
    [ INFO ]    Min:           1.42 ms
    [ INFO ]    Max:           11.85 ms
    [ INFO ] Throughput:   3838.73 FPS


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
    [ INFO ] Read model took 14.03 ms
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

    [ INFO ] Compile model took 61.77 ms
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


.. parsed-literal::

    [ INFO ] First inference took 2.37 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            178980 iterations
    [ INFO ] Duration:         15001.64 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        0.94 ms
    [ INFO ]    Average:       0.97 ms
    [ INFO ]    Min:           0.59 ms
    [ INFO ]    Max:           6.96 ms
    [ INFO ] Throughput:   11930.70 FPS


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

    [ INFO ] Count:            57720 iterations
    [ INFO ] Duration:         15002.10 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        2.95 ms
    [ INFO ]    Average:       2.95 ms
    [ INFO ]    Min:           1.82 ms
    [ INFO ]    Max:           13.28 ms
    [ INFO ] Throughput:   3847.46 FPS


**Quantized IR model - CPU**

.. code:: ipython3

    benchmark_output = %sx benchmark_app -m $compressed_model_xml -t 15 -api async
    # Remove logging info from benchmark_app output and show only the results
    benchmark_result = benchmark_output[-8:]
    print("\n".join(benchmark_result))


.. parsed-literal::

    [ INFO ] Count:            178680 iterations
    [ INFO ] Duration:         15001.15 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        0.94 ms
    [ INFO ]    Average:       0.97 ms
    [ INFO ]    Min:           0.58 ms
    [ INFO ]    Max:           6.79 ms
    [ INFO ] Throughput:   11911.08 FPS


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

