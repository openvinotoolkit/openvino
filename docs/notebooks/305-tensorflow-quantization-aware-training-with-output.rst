Quantization Aware Training with NNCF, using TensorFlow Framework
=================================================================

The goal of this notebook to demonstrate how to use the Neural Network
Compression Framework `NNCF <https://github.com/openvinotoolkit/nncf>`__
8-bit quantization to optimize a TensorFlow model for inference with
OpenVINO™ Toolkit. The optimization process contains the following
steps:

-  Transforming the original ``FP32`` model to ``INT8``
-  Using fine-tuning to restore the accuracy.
-  Exporting optimized and original models to Frozen Graph and then to
   OpenVINO.
-  Measuring and comparing the performance of models.

For more advanced usage, refer to these
`examples <https://github.com/openvinotoolkit/nncf/tree/develop/examples>`__.

This tutorial uses the ResNet-18 model with Imagenette dataset.
Imagenette is a subset of 10 easily classified classes from the ImageNet
dataset. Using the smaller model and dataset will speed up training and
download time.

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Imports and Settings <#imports-and-settings>`__
-  `Dataset Preprocessing <#dataset-preprocessing>`__
-  `Define a Floating-Point Model <#define-a-floating-point-model>`__
-  `Pre-train a Floating-Point
   Model <#pre-train-a-floating-point-model>`__
-  `Create and Initialize
   Quantization <#create-and-initialize-quantization>`__
-  `Fine-tune the Compressed Model <#fine-tune-the-compressed-model>`__
-  `Export Models to OpenVINO Intermediate Representation
   (IR) <#export-models-to-openvino-intermediate-representation-ir>`__
-  `Benchmark Model Performance by Computing Inference
   Time <#benchmark-model-performance-by-computing-inference-time>`__

Imports and Settings
--------------------



Import NNCF and all auxiliary packages from your Python code. Set a name
for the model, input image size, used batch size, and the learning rate.
Also, define paths where Frozen Graph and OpenVINO IR versions of the
models will be stored.

   **NOTE**: All NNCF logging messages below ERROR level (INFO and
   WARNING) are disabled to simplify the tutorial. For production use,
   it is recommended to enable logging by removing
   ``set_log_level(logging.ERROR)``.

.. code:: ipython3

    import sys
    import importlib.util
    
    %pip install -q "openvino>=2023.1.0" "nncf>=2.5.0"
    if sys.platform == "win32":
        if importlib.util.find_spec("tensorflow_datasets"):
            %pip uninstall -q tensorflow-datasets
        %pip install -q --upgrade "tfds-nightly"
    else:
        %pip install -q "tensorflow-datasets>=4.8.0"


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    pytorch-lightning 1.6.5 requires protobuf<=3.20.1, but you have protobuf 3.20.3 which is incompatible.
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. code:: ipython3

    from pathlib import Path
    import logging
    
    import tensorflow as tf
    import tensorflow_datasets as tfds
    from tensorflow.keras import layers
    from tensorflow.keras import models
    
    from nncf import NNCFConfig
    from nncf.tensorflow.helpers.model_creation import create_compressed_model
    from nncf.tensorflow.initialization import register_default_init_args
    from nncf.common.logging.logger import set_log_level
    import openvino as ov
    
    set_log_level(logging.ERROR)
    
    MODEL_DIR = Path("model")
    OUTPUT_DIR = Path("output")
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    BASE_MODEL_NAME = "ResNet-18"
    
    fp32_h5_path = Path(MODEL_DIR / (BASE_MODEL_NAME + "_fp32")).with_suffix(".h5")
    fp32_ir_path = Path(OUTPUT_DIR / "saved_model").with_suffix(".xml")
    int8_pb_path = Path(OUTPUT_DIR / (BASE_MODEL_NAME + "_int8")).with_suffix(".pb")
    int8_ir_path = int8_pb_path.with_suffix(".xml")
    
    BATCH_SIZE = 128
    IMG_SIZE = (64, 64)  # Default Imagenet image size
    NUM_CLASSES = 10  # For Imagenette dataset
    
    LR = 1e-5
    
    MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)  # From Imagenet dataset
    STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)  # From Imagenet dataset
    
    fp32_pth_url = "https://storage.openvinotoolkit.org/repositories/nncf/openvino_notebook_ckpts/305_resnet18_imagenette_fp32_v1.h5"
    _ = tf.keras.utils.get_file(fp32_h5_path.resolve(), fp32_pth_url)
    print(f'Absolute path where the model weights are saved:\n {fp32_h5_path.resolve()}')


.. parsed-literal::

    2024-03-13 01:11:54.839379: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-03-13 01:11:54.874069: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


.. parsed-literal::

    2024-03-13 01:11:55.482764: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


.. parsed-literal::

    Downloading data from https://storage.openvinotoolkit.org/repositories/nncf/openvino_notebook_ckpts/305_resnet18_imagenette_fp32_v1.h5


.. parsed-literal::

    
     8192/134604992 [..............................] - ETA: 0s

.. parsed-literal::

    
   147456/134604992 [..............................] - ETA: 1:02

.. parsed-literal::

    
   655360/134604992 [..............................] - ETA: 27s 

.. parsed-literal::

    
  2719744/134604992 [..............................] - ETA: 9s 

.. parsed-literal::

    
  6914048/134604992 [>.............................] - ETA: 5s

.. parsed-literal::

    
 12705792/134604992 [=>............................] - ETA: 3s

.. parsed-literal::

    
 17588224/134604992 [==>...........................] - ETA: 2s

.. parsed-literal::

    
 20963328/134604992 [===>..........................] - ETA: 2s

.. parsed-literal::

    
 23764992/134604992 [====>.........................] - ETA: 2s

.. parsed-literal::

    
 26558464/134604992 [====>.........................] - ETA: 2s

.. parsed-literal::

    
 28844032/134604992 [=====>........................] - ETA: 2s

.. parsed-literal::

    
 31449088/134604992 [======>.......................] - ETA: 2s

.. parsed-literal::

    
 33857536/134604992 [======>.......................] - ETA: 2s

.. parsed-literal::

    
 36683776/134604992 [=======>......................] - ETA: 2s

.. parsed-literal::

    
 40517632/134604992 [========>.....................] - ETA: 2s

.. parsed-literal::

    
 41484288/134604992 [========>.....................] - ETA: 2s

.. parsed-literal::

    
 41934848/134604992 [========>.....................] - ETA: 2s

.. parsed-literal::

    
 43761664/134604992 [========>.....................] - ETA: 2s

.. parsed-literal::

    
 47177728/134604992 [=========>....................] - ETA: 2s

.. parsed-literal::

    
 52133888/134604992 [==========>...................] - ETA: 1s

.. parsed-literal::

    
 52420608/134604992 [==========>...................] - ETA: 1s

.. parsed-literal::

    
 52551680/134604992 [==========>...................] - ETA: 2s

.. parsed-literal::

    
 57663488/134604992 [===========>..................] - ETA: 1s

.. parsed-literal::

    
 61620224/134604992 [============>.................] - ETA: 1s

.. parsed-literal::

    
 64462848/134604992 [=============>................] - ETA: 1s

.. parsed-literal::

    
 68149248/134604992 [==============>...............] - ETA: 1s

.. parsed-literal::

    
 72540160/134604992 [===============>..............] - ETA: 1s

.. parsed-literal::

    
 73392128/134604992 [===============>..............] - ETA: 1s

.. parsed-literal::

    
 77783040/134604992 [================>.............] - ETA: 1s

.. parsed-literal::

    
 78635008/134604992 [================>.............] - ETA: 1s

.. parsed-literal::

    
 83025920/134604992 [=================>............] - ETA: 1s

.. parsed-literal::

    
 83877888/134604992 [=================>............] - ETA: 1s

.. parsed-literal::

    
 88268800/134604992 [==================>...........] - ETA: 1s

.. parsed-literal::

    
 89120768/134604992 [==================>...........] - ETA: 1s

.. parsed-literal::

    
 90234880/134604992 [===================>..........] - ETA: 1s

.. parsed-literal::

    
 92659712/134604992 [===================>..........] - ETA: 1s

.. parsed-literal::

    
 95322112/134604992 [====================>.........] - ETA: 1s

.. parsed-literal::

    
 99606528/134604992 [=====================>........] - ETA: 0s

.. parsed-literal::

    
104275968/134604992 [======================>.......] - ETA: 0s

.. parsed-literal::

    
104841216/134604992 [======================>.......] - ETA: 0s

.. parsed-literal::

    
109240320/134604992 [=======================>......] - ETA: 0s

.. parsed-literal::

    
110092288/134604992 [=======================>......] - ETA: 0s

.. parsed-literal::

    
112508928/134604992 [========================>.....] - ETA: 0s

.. parsed-literal::

    
115335168/134604992 [========================>.....] - ETA: 0s

.. parsed-literal::

    
120201216/134604992 [=========================>....] - ETA: 0s

.. parsed-literal::

    
120578048/134604992 [=========================>....] - ETA: 0s

.. parsed-literal::

    
125231104/134604992 [==========================>...] - ETA: 0s

.. parsed-literal::

    
125820928/134604992 [===========================>..] - ETA: 0s

.. parsed-literal::

    
130924544/134604992 [============================>.] - ETA: 0s

.. parsed-literal::

    
131538944/134604992 [============================>.] - ETA: 0s

.. parsed-literal::

    
134447104/134604992 [============================>.] - ETA: 0s

.. parsed-literal::

    
134604992/134604992 [==============================] - 3s 0us/step


.. parsed-literal::

    Absolute path where the model weights are saved:
     /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/305-tensorflow-quantization-aware-training/model/ResNet-18_fp32.h5


Dataset Preprocessing
---------------------



Download and prepare Imagenette 160px dataset. - Number of classes: 10 -
Download size: 94.18 MiB

::

   | Split        | Examples |
   |--------------|----------|
   | 'train'      | 12,894   |
   | 'validation' | 500      |

.. code:: ipython3

    datasets, datasets_info = tfds.load('imagenette/160px', shuffle_files=True, as_supervised=True, with_info=True,
                                        read_config=tfds.ReadConfig(shuffle_seed=0))
    train_dataset, validation_dataset = datasets['train'], datasets['validation']
    fig = tfds.show_examples(train_dataset, datasets_info)


.. parsed-literal::

    2024-03-13 01:12:03.781864: E tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:266] failed call to cuInit: CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: forward compatibility was attempted on non supported HW
    2024-03-13 01:12:03.781896: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:168] retrieving CUDA diagnostic information for host: iotg-dev-workstation-07
    2024-03-13 01:12:03.781901: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:175] hostname: iotg-dev-workstation-07
    2024-03-13 01:12:03.782051: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:199] libcuda reported version is: 470.223.2
    2024-03-13 01:12:03.782066: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:203] kernel reported version is: 470.182.3
    2024-03-13 01:12:03.782070: E tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:312] kernel version 470.182.3 does not match DSO version 470.223.2 -- cannot find working devices in this configuration
    2024-03-13 01:12:03.899468: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int64 and shape [1]
    	 [[{{node Placeholder/_4}}]]
    2024-03-13 01:12:03.899790: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_1' with dtype string and shape [1]
    	 [[{{node Placeholder/_1}}]]
    2024-03-13 01:12:03.971431: W tensorflow/core/kernels/data/cache_dataset_ops.cc:856] The calling iterator did not fully read the dataset being cached. In order to avoid unexpected truncation of the dataset, the partially cached contents of the dataset  will be discarded. This can happen if you have an input pipeline similar to `dataset.cache().take(k).repeat()`. You should use `dataset.take(k).cache().repeat()` instead.



.. image:: 305-tensorflow-quantization-aware-training-with-output_files/305-tensorflow-quantization-aware-training-with-output_6_1.png


.. code:: ipython3

    def preprocessing(image, label):
        image = tf.image.resize(image, IMG_SIZE)
        image = image - MEAN_RGB
        image = image / STDDEV_RGB
        label = tf.one_hot(label, NUM_CLASSES)
        return image, label
    
    
    train_dataset = (train_dataset.map(preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                                  .batch(BATCH_SIZE)
                                  .prefetch(tf.data.experimental.AUTOTUNE))
    
    validation_dataset = (validation_dataset.map(preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                                            .batch(BATCH_SIZE)
                                            .prefetch(tf.data.experimental.AUTOTUNE))

Define a Floating-Point Model
-----------------------------



.. code:: ipython3

    def residual_conv_block(filters, stage, block, strides=(1, 1), cut='pre'):
        def layer(input_tensor):
            x = layers.BatchNormalization(epsilon=2e-5)(input_tensor)
            x = layers.Activation('relu')(x)
    
            # Defining shortcut connection.
            if cut == 'pre':
                shortcut = input_tensor
            elif cut == 'post':
                shortcut = layers.Conv2D(filters, (1, 1), strides=strides, kernel_initializer='he_uniform', 
                                         use_bias=False)(x)
    
            # Continue with convolution layers.
            x = layers.ZeroPadding2D(padding=(1, 1))(x)
            x = layers.Conv2D(filters, (3, 3), strides=strides, kernel_initializer='he_uniform', use_bias=False)(x)
    
            x = layers.BatchNormalization(epsilon=2e-5)(x)
            x = layers.Activation('relu')(x)
            x = layers.ZeroPadding2D(padding=(1, 1))(x)
            x = layers.Conv2D(filters, (3, 3), kernel_initializer='he_uniform', use_bias=False)(x)
    
            # Add residual connection.
            x = layers.Add()([x, shortcut])
            return x
    
        return layer
    
    
    def ResNet18(input_shape=None):
        """Instantiates the ResNet18 architecture."""
        img_input = layers.Input(shape=input_shape, name='data')
    
        # ResNet18 bottom
        x = layers.BatchNormalization(epsilon=2e-5, scale=False)(img_input)
        x = layers.ZeroPadding2D(padding=(3, 3))(x)
        x = layers.Conv2D(64, (7, 7), strides=(2, 2), kernel_initializer='he_uniform', use_bias=False)(x)
        x = layers.BatchNormalization(epsilon=2e-5)(x)
        x = layers.Activation('relu')(x)
        x = layers.ZeroPadding2D(padding=(1, 1))(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)
    
        # ResNet18 body
        repetitions = (2, 2, 2, 2)
        for stage, rep in enumerate(repetitions):
            for block in range(rep):
                filters = 64 * (2 ** stage)
                if block == 0 and stage == 0:
                    x = residual_conv_block(filters, stage, block, strides=(1, 1), cut='post')(x)
                elif block == 0:
                    x = residual_conv_block(filters, stage, block, strides=(2, 2), cut='post')(x)
                else:
                    x = residual_conv_block(filters, stage, block, strides=(1, 1), cut='pre')(x)
        x = layers.BatchNormalization(epsilon=2e-5)(x)
        x = layers.Activation('relu')(x)
    
        # ResNet18 top
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(NUM_CLASSES)(x)
        x = layers.Activation('softmax')(x)
    
        # Create the model.
        model = models.Model(img_input, x)
    
        return model

.. code:: ipython3

    IMG_SHAPE = IMG_SIZE + (3,)
    fp32_model = ResNet18(input_shape=IMG_SHAPE)

Pre-train a Floating-Point Model
--------------------------------



Using NNCF for model compression assumes that the user has a pre-trained
model and a training pipeline.

   **NOTE** For the sake of simplicity of the tutorial, it is
   recommended to skip ``FP32`` model training and load the weights that
   are provided.

.. code:: ipython3

    # Load the floating-point weights.
    fp32_model.load_weights(fp32_h5_path)
    
    # Compile the floating-point model.
    fp32_model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name='acc@1')]
    )
    
    # Validate the floating-point model.
    test_loss, acc_fp32 = fp32_model.evaluate(
        validation_dataset,
        callbacks=tf.keras.callbacks.ProgbarLogger(stateful_metrics=['acc@1'])
    )
    print(f"\nAccuracy of FP32 model: {acc_fp32:.3f}")


.. parsed-literal::

    2024-03-13 01:12:04.910372: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int64 and shape [1]
    	 [[{{node Placeholder/_4}}]]
    2024-03-13 01:12:04.910741: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int64 and shape [1]
    	 [[{{node Placeholder/_4}}]]


.. parsed-literal::

    
      0/Unknown - 1s 0s/sample - loss: 1.0472 - acc@1: 0.7891

.. parsed-literal::

    
      0/Unknown - 1s 0s/sample - loss: 0.9818 - acc@1: 0.8203

.. parsed-literal::

    
      0/Unknown - 1s 0s/sample - loss: 0.9774 - acc@1: 0.8203

.. parsed-literal::

    
      0/Unknown - 1s 0s/sample - loss: 0.9807 - acc@1: 0.8220

.. parsed-literal::

    
4/4 [==============================] - 1s 254ms/sample - loss: 0.9807 - acc@1: 0.8220


.. parsed-literal::

    
    Accuracy of FP32 model: 0.822


Create and Initialize Quantization
----------------------------------



NNCF enables compression-aware training by integrating into regular
training pipelines. The framework is designed so that modifications to
your original training code are minor. Quantization is the simplest
scenario and requires only 3 modifications.

1. Configure NNCF parameters to specify compression

.. code:: ipython3

    nncf_config_dict = {
        "input_info": {"sample_size": [1, 3] + list(IMG_SIZE)},
        "log_dir": str(OUTPUT_DIR),  # The log directory for NNCF-specific logging outputs.
        "compression": {
            "algorithm": "quantization",  # Specify the algorithm here.
        },
    }
    nncf_config = NNCFConfig.from_dict(nncf_config_dict)

2. Provide a data loader to initialize the values of quantization ranges
   and determine which activation should be signed or unsigned from the
   collected statistics, using a given number of samples.

.. code:: ipython3

    nncf_config = register_default_init_args(nncf_config=nncf_config,
                                             data_loader=train_dataset,
                                             batch_size=BATCH_SIZE)

3. Create a wrapped model ready for compression fine-tuning from a
   pre-trained ``FP32`` model and a configuration object.

.. code:: ipython3

    compression_ctrl, int8_model = create_compressed_model(fp32_model, nncf_config)


.. parsed-literal::

    2024-03-13 01:12:07.816084: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int64 and shape [1]
    	 [[{{node Placeholder/_4}}]]
    2024-03-13 01:12:07.816469: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [1]
    	 [[{{node Placeholder/_0}}]]


.. parsed-literal::

    2024-03-13 01:12:08.749313: W tensorflow/core/kernels/data/cache_dataset_ops.cc:856] The calling iterator did not fully read the dataset being cached. In order to avoid unexpected truncation of the dataset, the partially cached contents of the dataset  will be discarded. This can happen if you have an input pipeline similar to `dataset.cache().take(k).repeat()`. You should use `dataset.take(k).cache().repeat()` instead.


.. parsed-literal::

    2024-03-13 01:12:09.358441: W tensorflow/core/kernels/data/cache_dataset_ops.cc:856] The calling iterator did not fully read the dataset being cached. In order to avoid unexpected truncation of the dataset, the partially cached contents of the dataset  will be discarded. This can happen if you have an input pipeline similar to `dataset.cache().take(k).repeat()`. You should use `dataset.take(k).cache().repeat()` instead.


.. parsed-literal::

    2024-03-13 01:12:17.326475: W tensorflow/core/kernels/data/cache_dataset_ops.cc:856] The calling iterator did not fully read the dataset being cached. In order to avoid unexpected truncation of the dataset, the partially cached contents of the dataset  will be discarded. This can happen if you have an input pipeline similar to `dataset.cache().take(k).repeat()`. You should use `dataset.take(k).cache().repeat()` instead.


Evaluate the new model on the validation set after initialization of
quantization. The accuracy should be not far from the accuracy of the
floating-point ``FP32`` model for a simple case like the one being
demonstrated here.

.. code:: ipython3

    # Compile the INT8 model.
    int8_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name='acc@1')]
    )
    
    # Validate the INT8 model.
    test_loss, test_acc = int8_model.evaluate(
        validation_dataset,
        callbacks=tf.keras.callbacks.ProgbarLogger(stateful_metrics=['acc@1'])
    )


.. parsed-literal::

    
      0/Unknown - 1s 0s/sample - loss: 1.0468 - acc@1: 0.7656

.. parsed-literal::

    
      0/Unknown - 1s 0s/sample - loss: 0.9804 - acc@1: 0.8008

.. parsed-literal::

    
      0/Unknown - 1s 0s/sample - loss: 0.9769 - acc@1: 0.8099

.. parsed-literal::

    
      0/Unknown - 1s 0s/sample - loss: 0.9766 - acc@1: 0.8120

.. parsed-literal::

    
4/4 [==============================] - 1s 301ms/sample - loss: 0.9766 - acc@1: 0.8120


Fine-tune the Compressed Model
------------------------------



At this step, a regular fine-tuning process is applied to further
improve quantized model accuracy. Normally, several epochs of tuning are
required with a small learning rate, the same that is usually used at
the end of the training of the original model. No other changes in the
training pipeline are required. Here is a simple example.

.. code:: ipython3

    print(f"\nAccuracy of INT8 model after initialization: {test_acc:.3f}")
    
    # Train the INT8 model.
    int8_model.fit(train_dataset, epochs=2)
    
    # Validate the INT8 model.
    test_loss, acc_int8 = int8_model.evaluate(
        validation_dataset, callbacks=tf.keras.callbacks.ProgbarLogger(stateful_metrics=['acc@1']))
    print(f"\nAccuracy of INT8 model after fine-tuning: {acc_int8:.3f}")
    print(
        f"\nAccuracy drop of tuned INT8 model over pre-trained FP32 model: {acc_fp32 - acc_int8:.3f}")


.. parsed-literal::

    
    Accuracy of INT8 model after initialization: 0.812


.. parsed-literal::

    Epoch 1/2


.. parsed-literal::

    
  1/101 [..............................] - ETA: 11:57 - loss: 0.6168 - acc@1: 0.9844

.. parsed-literal::

    
  2/101 [..............................] - ETA: 43s - loss: 0.6303 - acc@1: 0.9766  

.. parsed-literal::

    
  3/101 [..............................] - ETA: 42s - loss: 0.6613 - acc@1: 0.9609

.. parsed-literal::

    
  4/101 [>.............................] - ETA: 41s - loss: 0.6650 - acc@1: 0.9551

.. parsed-literal::

    
  5/101 [>.............................] - ETA: 40s - loss: 0.6783 - acc@1: 0.9469

.. parsed-literal::

    
  6/101 [>.............................] - ETA: 40s - loss: 0.6805 - acc@1: 0.9466

.. parsed-literal::

    
  7/101 [=>............................] - ETA: 39s - loss: 0.6796 - acc@1: 0.9442

.. parsed-literal::

    
  8/101 [=>............................] - ETA: 39s - loss: 0.6790 - acc@1: 0.9463

.. parsed-literal::

    
  9/101 [=>............................] - ETA: 38s - loss: 0.6828 - acc@1: 0.9462

.. parsed-literal::

    
 10/101 [=>............................] - ETA: 38s - loss: 0.6908 - acc@1: 0.9422

.. parsed-literal::

    
 11/101 [==>...........................] - ETA: 37s - loss: 0.6899 - acc@1: 0.9425

.. parsed-literal::

    
 12/101 [==>...........................] - ETA: 37s - loss: 0.6930 - acc@1: 0.9421

.. parsed-literal::

    
 13/101 [==>...........................] - ETA: 36s - loss: 0.6923 - acc@1: 0.9417

.. parsed-literal::

    
 14/101 [===>..........................] - ETA: 36s - loss: 0.6960 - acc@1: 0.9386

.. parsed-literal::

    
 15/101 [===>..........................] - ETA: 36s - loss: 0.6956 - acc@1: 0.9385

.. parsed-literal::

    
 16/101 [===>..........................] - ETA: 35s - loss: 0.6946 - acc@1: 0.9395

.. parsed-literal::

    
 17/101 [====>.........................] - ETA: 35s - loss: 0.6948 - acc@1: 0.9393

.. parsed-literal::

    
 18/101 [====>.........................] - ETA: 34s - loss: 0.6941 - acc@1: 0.9405

.. parsed-literal::

    
 19/101 [====>.........................] - ETA: 34s - loss: 0.6955 - acc@1: 0.9400

.. parsed-literal::

    
 20/101 [====>.........................] - ETA: 33s - loss: 0.6931 - acc@1: 0.9402

.. parsed-literal::

    
 21/101 [=====>........................] - ETA: 33s - loss: 0.6944 - acc@1: 0.9394

.. parsed-literal::

    
 22/101 [=====>........................] - ETA: 33s - loss: 0.6953 - acc@1: 0.9382

.. parsed-literal::

    
 23/101 [=====>........................] - ETA: 32s - loss: 0.6966 - acc@1: 0.9375

.. parsed-literal::

    
 24/101 [======>.......................] - ETA: 32s - loss: 0.6971 - acc@1: 0.9368

.. parsed-literal::

    
 25/101 [======>.......................] - ETA: 31s - loss: 0.6973 - acc@1: 0.9366

.. parsed-literal::

    
 26/101 [======>.......................] - ETA: 31s - loss: 0.6975 - acc@1: 0.9369

.. parsed-literal::

    
 27/101 [=======>......................] - ETA: 31s - loss: 0.6963 - acc@1: 0.9372

.. parsed-literal::

    
 28/101 [=======>......................] - ETA: 30s - loss: 0.6960 - acc@1: 0.9378

.. parsed-literal::

    
 29/101 [=======>......................] - ETA: 30s - loss: 0.6967 - acc@1: 0.9375

.. parsed-literal::

    
 30/101 [=======>......................] - ETA: 29s - loss: 0.6982 - acc@1: 0.9365

.. parsed-literal::

    
 31/101 [========>.....................] - ETA: 29s - loss: 0.6974 - acc@1: 0.9367

.. parsed-literal::

    
 32/101 [========>.....................] - ETA: 28s - loss: 0.6966 - acc@1: 0.9373

.. parsed-literal::

    
 33/101 [========>.....................] - ETA: 28s - loss: 0.6965 - acc@1: 0.9375

.. parsed-literal::

    
 34/101 [=========>....................] - ETA: 28s - loss: 0.6978 - acc@1: 0.9370

.. parsed-literal::

    
 35/101 [=========>....................] - ETA: 27s - loss: 0.6981 - acc@1: 0.9375

.. parsed-literal::

    
 36/101 [=========>....................] - ETA: 27s - loss: 0.6992 - acc@1: 0.9382

.. parsed-literal::

    
 37/101 [=========>....................] - ETA: 26s - loss: 0.7001 - acc@1: 0.9375

.. parsed-literal::

    
 38/101 [==========>...................] - ETA: 26s - loss: 0.7023 - acc@1: 0.9369

.. parsed-literal::

    
 39/101 [==========>...................] - ETA: 25s - loss: 0.7019 - acc@1: 0.9371

.. parsed-literal::

    
 40/101 [==========>...................] - ETA: 25s - loss: 0.7016 - acc@1: 0.9373

.. parsed-literal::

    
 41/101 [===========>..................] - ETA: 25s - loss: 0.7021 - acc@1: 0.9371

.. parsed-literal::

    
 42/101 [===========>..................] - ETA: 24s - loss: 0.7018 - acc@1: 0.9371

.. parsed-literal::

    
 43/101 [===========>..................] - ETA: 24s - loss: 0.7014 - acc@1: 0.9375

.. parsed-literal::

    
 44/101 [============>.................] - ETA: 23s - loss: 0.7016 - acc@1: 0.9373

.. parsed-literal::

    
 45/101 [============>.................] - ETA: 23s - loss: 0.7025 - acc@1: 0.9373

.. parsed-literal::

    
 46/101 [============>.................] - ETA: 22s - loss: 0.7028 - acc@1: 0.9372

.. parsed-literal::

    
 47/101 [============>.................] - ETA: 22s - loss: 0.7044 - acc@1: 0.9362

.. parsed-literal::

    
 48/101 [=============>................] - ETA: 22s - loss: 0.7045 - acc@1: 0.9357

.. parsed-literal::

    
 49/101 [=============>................] - ETA: 21s - loss: 0.7052 - acc@1: 0.9361

.. parsed-literal::

    
 50/101 [=============>................] - ETA: 21s - loss: 0.7052 - acc@1: 0.9359

.. parsed-literal::

    
 51/101 [==============>...............] - ETA: 20s - loss: 0.7061 - acc@1: 0.9357

.. parsed-literal::

    
 52/101 [==============>...............] - ETA: 20s - loss: 0.7057 - acc@1: 0.9358

.. parsed-literal::

    
 53/101 [==============>...............] - ETA: 20s - loss: 0.7061 - acc@1: 0.9350

.. parsed-literal::

    
 54/101 [===============>..............] - ETA: 19s - loss: 0.7055 - acc@1: 0.9355

.. parsed-literal::

    
 55/101 [===============>..............] - ETA: 19s - loss: 0.7052 - acc@1: 0.9357

.. parsed-literal::

    
 56/101 [===============>..............] - ETA: 18s - loss: 0.7050 - acc@1: 0.9357

.. parsed-literal::

    
 57/101 [===============>..............] - ETA: 18s - loss: 0.7053 - acc@1: 0.9352

.. parsed-literal::

    
 58/101 [================>.............] - ETA: 17s - loss: 0.7057 - acc@1: 0.9351

.. parsed-literal::

    
 59/101 [================>.............] - ETA: 17s - loss: 0.7062 - acc@1: 0.9345

.. parsed-literal::

    
 60/101 [================>.............] - ETA: 17s - loss: 0.7064 - acc@1: 0.9345

.. parsed-literal::

    
 61/101 [=================>............] - ETA: 16s - loss: 0.7064 - acc@1: 0.9343

.. parsed-literal::

    
 62/101 [=================>............] - ETA: 16s - loss: 0.7056 - acc@1: 0.9347

.. parsed-literal::

    
 63/101 [=================>............] - ETA: 15s - loss: 0.7060 - acc@1: 0.9345

.. parsed-literal::

    
 64/101 [==================>...........] - ETA: 15s - loss: 0.7063 - acc@1: 0.9342

.. parsed-literal::

    
 65/101 [==================>...........] - ETA: 15s - loss: 0.7073 - acc@1: 0.9337

.. parsed-literal::

    
 66/101 [==================>...........] - ETA: 14s - loss: 0.7077 - acc@1: 0.9332

.. parsed-literal::

    
 67/101 [==================>...........] - ETA: 14s - loss: 0.7083 - acc@1: 0.9327

.. parsed-literal::

    
 68/101 [===================>..........] - ETA: 13s - loss: 0.7081 - acc@1: 0.9330

.. parsed-literal::

    
 69/101 [===================>..........] - ETA: 13s - loss: 0.7087 - acc@1: 0.9330

.. parsed-literal::

    
 70/101 [===================>..........] - ETA: 12s - loss: 0.7091 - acc@1: 0.9326

.. parsed-literal::

    
 71/101 [====================>.........] - ETA: 12s - loss: 0.7081 - acc@1: 0.9330

.. parsed-literal::

    
 72/101 [====================>.........] - ETA: 12s - loss: 0.7083 - acc@1: 0.9329

.. parsed-literal::

    
 73/101 [====================>.........] - ETA: 11s - loss: 0.7075 - acc@1: 0.9334

.. parsed-literal::

    
 74/101 [====================>.........] - ETA: 11s - loss: 0.7079 - acc@1: 0.9334

.. parsed-literal::

    
 75/101 [=====================>........] - ETA: 10s - loss: 0.7085 - acc@1: 0.9329

.. parsed-literal::

    
 76/101 [=====================>........] - ETA: 10s - loss: 0.7082 - acc@1: 0.9332

.. parsed-literal::

    
 77/101 [=====================>........] - ETA: 9s - loss: 0.7078 - acc@1: 0.9333 

.. parsed-literal::

    
 78/101 [======================>.......] - ETA: 9s - loss: 0.7080 - acc@1: 0.9334

.. parsed-literal::

    
 79/101 [======================>.......] - ETA: 9s - loss: 0.7079 - acc@1: 0.9332

.. parsed-literal::

    
 80/101 [======================>.......] - ETA: 8s - loss: 0.7081 - acc@1: 0.9330

.. parsed-literal::

    
 81/101 [=======================>......] - ETA: 8s - loss: 0.7078 - acc@1: 0.9333

.. parsed-literal::

    
 82/101 [=======================>......] - ETA: 7s - loss: 0.7081 - acc@1: 0.9332

.. parsed-literal::

    
 83/101 [=======================>......] - ETA: 7s - loss: 0.7080 - acc@1: 0.9332

.. parsed-literal::

    
 84/101 [=======================>......] - ETA: 7s - loss: 0.7075 - acc@1: 0.9332

.. parsed-literal::

    
 85/101 [========================>.....] - ETA: 6s - loss: 0.7080 - acc@1: 0.9332

.. parsed-literal::

    
 86/101 [========================>.....] - ETA: 6s - loss: 0.7073 - acc@1: 0.9337

.. parsed-literal::

    
 87/101 [========================>.....] - ETA: 5s - loss: 0.7079 - acc@1: 0.9330

.. parsed-literal::

    
 88/101 [=========================>....] - ETA: 5s - loss: 0.7084 - acc@1: 0.9330

.. parsed-literal::

    
 89/101 [=========================>....] - ETA: 4s - loss: 0.7087 - acc@1: 0.9331

.. parsed-literal::

    
 90/101 [=========================>....] - ETA: 4s - loss: 0.7091 - acc@1: 0.9330

.. parsed-literal::

    
 91/101 [==========================>...] - ETA: 4s - loss: 0.7096 - acc@1: 0.9327

.. parsed-literal::

    
 92/101 [==========================>...] - ETA: 3s - loss: 0.7095 - acc@1: 0.9325

.. parsed-literal::

    
 93/101 [==========================>...] - ETA: 3s - loss: 0.7099 - acc@1: 0.9320

.. parsed-literal::

    
 94/101 [==========================>...] - ETA: 2s - loss: 0.7105 - acc@1: 0.9317

.. parsed-literal::

    
 95/101 [===========================>..] - ETA: 2s - loss: 0.7107 - acc@1: 0.9312

.. parsed-literal::

    
 96/101 [===========================>..] - ETA: 2s - loss: 0.7107 - acc@1: 0.9313

.. parsed-literal::

    
 97/101 [===========================>..] - ETA: 1s - loss: 0.7109 - acc@1: 0.9312

.. parsed-literal::

    
 98/101 [============================>.] - ETA: 1s - loss: 0.7111 - acc@1: 0.9311

.. parsed-literal::

    
 99/101 [============================>.] - ETA: 0s - loss: 0.7123 - acc@1: 0.9305

.. parsed-literal::

    
100/101 [============================>.] - ETA: 0s - loss: 0.7123 - acc@1: 0.9305

.. parsed-literal::

    
101/101 [==============================] - ETA: 0s - loss: 0.7134 - acc@1: 0.9299

.. parsed-literal::

    
101/101 [==============================] - 49s 415ms/step - loss: 0.7134 - acc@1: 0.9299


.. parsed-literal::

    Epoch 2/2


.. parsed-literal::

    
  1/101 [..............................] - ETA: 43s - loss: 0.5798 - acc@1: 1.0000

.. parsed-literal::

    
  2/101 [..............................] - ETA: 40s - loss: 0.5917 - acc@1: 1.0000

.. parsed-literal::

    
  3/101 [..............................] - ETA: 40s - loss: 0.6191 - acc@1: 0.9896

.. parsed-literal::

    
  4/101 [>.............................] - ETA: 39s - loss: 0.6225 - acc@1: 0.9844

.. parsed-literal::

    
  5/101 [>.............................] - ETA: 39s - loss: 0.6332 - acc@1: 0.9781

.. parsed-literal::

    
  6/101 [>.............................] - ETA: 39s - loss: 0.6378 - acc@1: 0.9753

.. parsed-literal::

    
  7/101 [=>............................] - ETA: 39s - loss: 0.6392 - acc@1: 0.9732

.. parsed-literal::

    
  8/101 [=>............................] - ETA: 38s - loss: 0.6395 - acc@1: 0.9736

.. parsed-literal::

    
  9/101 [=>............................] - ETA: 37s - loss: 0.6435 - acc@1: 0.9740

.. parsed-literal::

    
 10/101 [=>............................] - ETA: 37s - loss: 0.6508 - acc@1: 0.9688

.. parsed-literal::

    
 11/101 [==>...........................] - ETA: 37s - loss: 0.6517 - acc@1: 0.9695

.. parsed-literal::

    
 12/101 [==>...........................] - ETA: 36s - loss: 0.6548 - acc@1: 0.9681

.. parsed-literal::

    
 13/101 [==>...........................] - ETA: 36s - loss: 0.6551 - acc@1: 0.9681

.. parsed-literal::

    
 14/101 [===>..........................] - ETA: 36s - loss: 0.6592 - acc@1: 0.9660

.. parsed-literal::

    
 15/101 [===>..........................] - ETA: 35s - loss: 0.6590 - acc@1: 0.9656

.. parsed-literal::

    
 16/101 [===>..........................] - ETA: 35s - loss: 0.6580 - acc@1: 0.9673

.. parsed-literal::

    
 17/101 [====>.........................] - ETA: 34s - loss: 0.6583 - acc@1: 0.9665

.. parsed-literal::

    
 18/101 [====>.........................] - ETA: 34s - loss: 0.6584 - acc@1: 0.9666

.. parsed-literal::

    
 19/101 [====>.........................] - ETA: 33s - loss: 0.6601 - acc@1: 0.9659

.. parsed-literal::

    
 20/101 [====>.........................] - ETA: 33s - loss: 0.6586 - acc@1: 0.9656

.. parsed-literal::

    
 21/101 [=====>........................] - ETA: 33s - loss: 0.6599 - acc@1: 0.9639

.. parsed-literal::

    
 22/101 [=====>........................] - ETA: 32s - loss: 0.6610 - acc@1: 0.9634

.. parsed-literal::

    
 23/101 [=====>........................] - ETA: 32s - loss: 0.6623 - acc@1: 0.9620

.. parsed-literal::

    
 24/101 [======>.......................] - ETA: 31s - loss: 0.6630 - acc@1: 0.9609

.. parsed-literal::

    
 25/101 [======>.......................] - ETA: 31s - loss: 0.6632 - acc@1: 0.9606

.. parsed-literal::

    
 26/101 [======>.......................] - ETA: 31s - loss: 0.6638 - acc@1: 0.9603

.. parsed-literal::

    
 27/101 [=======>......................] - ETA: 30s - loss: 0.6631 - acc@1: 0.9604

.. parsed-literal::

    
 28/101 [=======>......................] - ETA: 30s - loss: 0.6629 - acc@1: 0.9609

.. parsed-literal::

    
 29/101 [=======>......................] - ETA: 29s - loss: 0.6636 - acc@1: 0.9604

.. parsed-literal::

    
 30/101 [=======>......................] - ETA: 29s - loss: 0.6652 - acc@1: 0.9594

.. parsed-literal::

    
 31/101 [========>.....................] - ETA: 29s - loss: 0.6645 - acc@1: 0.9592

.. parsed-literal::

    
 32/101 [========>.....................] - ETA: 28s - loss: 0.6641 - acc@1: 0.9592

.. parsed-literal::

    
 33/101 [========>.....................] - ETA: 28s - loss: 0.6641 - acc@1: 0.9593

.. parsed-literal::

    
 34/101 [=========>....................] - ETA: 27s - loss: 0.6655 - acc@1: 0.9586

.. parsed-literal::

    
 35/101 [=========>....................] - ETA: 27s - loss: 0.6657 - acc@1: 0.9587

.. parsed-literal::

    
 36/101 [=========>....................] - ETA: 27s - loss: 0.6665 - acc@1: 0.9588

.. parsed-literal::

    
 37/101 [=========>....................] - ETA: 26s - loss: 0.6674 - acc@1: 0.9578

.. parsed-literal::

    
 38/101 [==========>...................] - ETA: 26s - loss: 0.6695 - acc@1: 0.9570

.. parsed-literal::

    
 39/101 [==========>...................] - ETA: 25s - loss: 0.6692 - acc@1: 0.9569

.. parsed-literal::

    
 40/101 [==========>...................] - ETA: 25s - loss: 0.6689 - acc@1: 0.9574

.. parsed-literal::

    
 41/101 [===========>..................] - ETA: 24s - loss: 0.6692 - acc@1: 0.9571

.. parsed-literal::

    
 42/101 [===========>..................] - ETA: 24s - loss: 0.6692 - acc@1: 0.9568

.. parsed-literal::

    
 43/101 [===========>..................] - ETA: 24s - loss: 0.6689 - acc@1: 0.9571

.. parsed-literal::

    
 44/101 [============>.................] - ETA: 23s - loss: 0.6692 - acc@1: 0.9569

.. parsed-literal::

    
 45/101 [============>.................] - ETA: 23s - loss: 0.6700 - acc@1: 0.9564

.. parsed-literal::

    
 46/101 [============>.................] - ETA: 22s - loss: 0.6702 - acc@1: 0.9562

.. parsed-literal::

    
 47/101 [============>.................] - ETA: 22s - loss: 0.6715 - acc@1: 0.9551

.. parsed-literal::

    
 48/101 [=============>................] - ETA: 22s - loss: 0.6715 - acc@1: 0.9552

.. parsed-literal::

    
 49/101 [=============>................] - ETA: 21s - loss: 0.6722 - acc@1: 0.9554

.. parsed-literal::

    
 50/101 [=============>................] - ETA: 21s - loss: 0.6723 - acc@1: 0.9552

.. parsed-literal::

    
 51/101 [==============>...............] - ETA: 20s - loss: 0.6732 - acc@1: 0.9547

.. parsed-literal::

    
 52/101 [==============>...............] - ETA: 20s - loss: 0.6729 - acc@1: 0.9548

.. parsed-literal::

    
 53/101 [==============>...............] - ETA: 20s - loss: 0.6734 - acc@1: 0.9542

.. parsed-literal::

    
 54/101 [===============>..............] - ETA: 19s - loss: 0.6730 - acc@1: 0.9546

.. parsed-literal::

    
 55/101 [===============>..............] - ETA: 19s - loss: 0.6728 - acc@1: 0.9544

.. parsed-literal::

    
 56/101 [===============>..............] - ETA: 18s - loss: 0.6727 - acc@1: 0.9544

.. parsed-literal::

    
 57/101 [===============>..............] - ETA: 18s - loss: 0.6732 - acc@1: 0.9538

.. parsed-literal::

    
 58/101 [================>.............] - ETA: 17s - loss: 0.6735 - acc@1: 0.9537

.. parsed-literal::

    
 59/101 [================>.............] - ETA: 17s - loss: 0.6739 - acc@1: 0.9531

.. parsed-literal::

    
 60/101 [================>.............] - ETA: 17s - loss: 0.6741 - acc@1: 0.9530

.. parsed-literal::

    
 61/101 [=================>............] - ETA: 16s - loss: 0.6741 - acc@1: 0.9530

.. parsed-literal::

    
 62/101 [=================>............] - ETA: 16s - loss: 0.6735 - acc@1: 0.9533

.. parsed-literal::

    
 63/101 [=================>............] - ETA: 15s - loss: 0.6738 - acc@1: 0.9531

.. parsed-literal::

    
 64/101 [==================>...........] - ETA: 15s - loss: 0.6741 - acc@1: 0.9529

.. parsed-literal::

    
 65/101 [==================>...........] - ETA: 14s - loss: 0.6750 - acc@1: 0.9523

.. parsed-literal::

    
 66/101 [==================>...........] - ETA: 14s - loss: 0.6754 - acc@1: 0.9522

.. parsed-literal::

    
 67/101 [==================>...........] - ETA: 14s - loss: 0.6758 - acc@1: 0.9518

.. parsed-literal::

    
 68/101 [===================>..........] - ETA: 13s - loss: 0.6758 - acc@1: 0.9520

.. parsed-literal::

    
 69/101 [===================>..........] - ETA: 13s - loss: 0.6763 - acc@1: 0.9520

.. parsed-literal::

    
 70/101 [===================>..........] - ETA: 12s - loss: 0.6768 - acc@1: 0.9516

.. parsed-literal::

    
 71/101 [====================>.........] - ETA: 12s - loss: 0.6760 - acc@1: 0.9518

.. parsed-literal::

    
 72/101 [====================>.........] - ETA: 12s - loss: 0.6761 - acc@1: 0.9516

.. parsed-literal::

    
 73/101 [====================>.........] - ETA: 11s - loss: 0.6755 - acc@1: 0.9518

.. parsed-literal::

    
 74/101 [====================>.........] - ETA: 11s - loss: 0.6759 - acc@1: 0.9516

.. parsed-literal::

    
 75/101 [=====================>........] - ETA: 10s - loss: 0.6765 - acc@1: 0.9515

.. parsed-literal::

    
 76/101 [=====================>........] - ETA: 10s - loss: 0.6762 - acc@1: 0.9517

.. parsed-literal::

    
 77/101 [=====================>........] - ETA: 10s - loss: 0.6759 - acc@1: 0.9520

.. parsed-literal::

    
 78/101 [======================>.......] - ETA: 9s - loss: 0.6761 - acc@1: 0.9521 

.. parsed-literal::

    
 79/101 [======================>.......] - ETA: 9s - loss: 0.6760 - acc@1: 0.9518

.. parsed-literal::

    
 80/101 [======================>.......] - ETA: 8s - loss: 0.6762 - acc@1: 0.9514

.. parsed-literal::

    
 81/101 [=======================>......] - ETA: 8s - loss: 0.6759 - acc@1: 0.9516

.. parsed-literal::

    
 82/101 [=======================>......] - ETA: 7s - loss: 0.6762 - acc@1: 0.9516

.. parsed-literal::

    
 83/101 [=======================>......] - ETA: 7s - loss: 0.6761 - acc@1: 0.9515

.. parsed-literal::

    
 84/101 [=======================>......] - ETA: 7s - loss: 0.6757 - acc@1: 0.9517

.. parsed-literal::

    
 85/101 [========================>.....] - ETA: 6s - loss: 0.6762 - acc@1: 0.9517

.. parsed-literal::

    
 86/101 [========================>.....] - ETA: 6s - loss: 0.6756 - acc@1: 0.9521

.. parsed-literal::

    
 87/101 [========================>.....] - ETA: 5s - loss: 0.6762 - acc@1: 0.9516

.. parsed-literal::

    
 88/101 [=========================>....] - ETA: 5s - loss: 0.6766 - acc@1: 0.9513

.. parsed-literal::

    
 89/101 [=========================>....] - ETA: 5s - loss: 0.6768 - acc@1: 0.9515

.. parsed-literal::

    
 90/101 [=========================>....] - ETA: 4s - loss: 0.6771 - acc@1: 0.9515

.. parsed-literal::

    
 91/101 [==========================>...] - ETA: 4s - loss: 0.6775 - acc@1: 0.9512

.. parsed-literal::

    
 92/101 [==========================>...] - ETA: 3s - loss: 0.6775 - acc@1: 0.9511

.. parsed-literal::

    
 93/101 [==========================>...] - ETA: 3s - loss: 0.6778 - acc@1: 0.9509

.. parsed-literal::

    
 94/101 [==========================>...] - ETA: 2s - loss: 0.6783 - acc@1: 0.9507

.. parsed-literal::

    
 95/101 [===========================>..] - ETA: 2s - loss: 0.6785 - acc@1: 0.9502

.. parsed-literal::

    
 96/101 [===========================>..] - ETA: 2s - loss: 0.6785 - acc@1: 0.9504

.. parsed-literal::

    
 97/101 [===========================>..] - ETA: 1s - loss: 0.6787 - acc@1: 0.9501

.. parsed-literal::

    
 98/101 [============================>.] - ETA: 1s - loss: 0.6790 - acc@1: 0.9499

.. parsed-literal::

    
 99/101 [============================>.] - ETA: 0s - loss: 0.6800 - acc@1: 0.9493

.. parsed-literal::

    
100/101 [============================>.] - ETA: 0s - loss: 0.6800 - acc@1: 0.9493

.. parsed-literal::

    
101/101 [==============================] - ETA: 0s - loss: 0.6807 - acc@1: 0.9489

.. parsed-literal::

    
101/101 [==============================] - 42s 417ms/step - loss: 0.6807 - acc@1: 0.9489


.. parsed-literal::

    
      0/Unknown - 0s 0s/sample - loss: 1.0568 - acc@1: 0.7812

.. parsed-literal::

    
      0/Unknown - 0s 0s/sample - loss: 0.9848 - acc@1: 0.8086

.. parsed-literal::

    
      0/Unknown - 0s 0s/sample - loss: 0.9768 - acc@1: 0.8177

.. parsed-literal::

    
      0/Unknown - 1s 0s/sample - loss: 0.9760 - acc@1: 0.8160

.. parsed-literal::

    
4/4 [==============================] - 1s 146ms/sample - loss: 0.9760 - acc@1: 0.8160


.. parsed-literal::

    
    Accuracy of INT8 model after fine-tuning: 0.816
    
    Accuracy drop of tuned INT8 model over pre-trained FP32 model: 0.006


Export Models to OpenVINO Intermediate Representation (IR)
----------------------------------------------------------



Use model conversion Python API to convert the models to OpenVINO IR.

For more information about model conversion, see this
`page <https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html>`__.

Executing this command may take a while.

.. code:: ipython3

    model_ir_fp32 = ov.convert_model(fp32_model)


.. parsed-literal::

    WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.base has been moved to tensorflow.python.trackable.base. The old module will be deleted in version 2.11.


.. parsed-literal::

    WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.base has been moved to tensorflow.python.trackable.base. The old module will be deleted in version 2.11.


.. code:: ipython3

    model_ir_int8 = ov.convert_model(int8_model)

.. code:: ipython3

    ov.save_model(model_ir_fp32, fp32_ir_path, compress_to_fp16=False)
    ov.save_model(model_ir_int8, int8_ir_path, compress_to_fp16=False)


Benchmark Model Performance by Computing Inference Time
-------------------------------------------------------



Finally, measure the inference performance of the ``FP32`` and ``INT8``
models, using `Benchmark
Tool <https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html>`__
- an inference performance measurement tool in OpenVINO. By default,
Benchmark Tool runs inference for 60 seconds in asynchronous mode on
CPU. It returns inference speed as latency (milliseconds per image) and
throughput (frames per second) values.

   **NOTE**: This notebook runs ``benchmark_app`` for 15 seconds to give
   a quick indication of performance. For more accurate performance, it
   is recommended to run ``benchmark_app`` in a terminal/command prompt
   after closing other applications. Run
   ``benchmark_app -m model.xml -d CPU`` to benchmark async inference on
   CPU for one minute. Change CPU to GPU to benchmark on GPU. Run
   ``benchmark_app --help`` to see an overview of all command-line
   options.

Please select a benchmarking device using the dropdown list:

.. code:: ipython3

    import ipywidgets as widgets
    
    # Initialize OpenVINO runtime
    core = ov.Core()
    device = widgets.Dropdown(
        options=core.available_devices,
        value='CPU',
        description='Device:',
        disabled=False,
    )
    
    device




.. parsed-literal::

    Dropdown(description='Device:', options=('CPU',), value='CPU')



.. code:: ipython3

    def parse_benchmark_output(benchmark_output):
        parsed_output = [line for line in benchmark_output if 'FPS' in line]
        print(*parsed_output, sep='\n')
    
    
    print('Benchmark FP32 model (IR)')
    benchmark_output = ! benchmark_app -m $fp32_ir_path -d $device.value -api async -t 15 -shape [1,64,64,3]
    parse_benchmark_output(benchmark_output)
    
    print('\nBenchmark INT8 model (IR)')
    benchmark_output = ! benchmark_app -m $int8_ir_path -d $device.value -api async -t 15 -shape [1,64,64,3]
    parse_benchmark_output(benchmark_output)


.. parsed-literal::

    Benchmark FP32 model (IR)


.. parsed-literal::

    [ INFO ] Throughput:   2840.25 FPS
    
    Benchmark INT8 model (IR)


.. parsed-literal::

    [ INFO ] Throughput:   11202.29 FPS


Show Device Information for reference.

.. code:: ipython3

    core = ov.Core()
    core.get_property(device.value, "FULL_DEVICE_NAME")




.. parsed-literal::

    'Intel(R) Core(TM) i9-10920X CPU @ 3.50GHz'


