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

**Table of contents:**

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

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.0 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    Note: you may need to restart the kernel to use updated packages.
    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.0 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    onnxconverter-common 1.14.0 requires protobuf==3.20.2, but you have protobuf 3.20.3 which is incompatible.
    pytorch-lightning 1.6.5 requires protobuf<=3.20.1, but you have protobuf 3.20.3 which is incompatible.
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

    2023-11-15 00:29:06.329749: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2023-11-15 00:29:06.363853: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-11-15 00:29:06.956739: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino
    Downloading data from https://storage.openvinotoolkit.org/repositories/nncf/openvino_notebook_ckpts/305_resnet18_imagenette_fp32_v1.h5
    134604992/134604992 [==============================] - 38s 0us/step
    Absolute path where the model weights are saved:
     /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-545/.workspace/scm/ov-notebook/notebooks/305-tensorflow-quantization-aware-training/model/ResNet-18_fp32.h5


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

    2023-11-15 00:29:49.433840: E tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:266] failed call to cuInit: CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: forward compatibility was attempted on non supported HW
    2023-11-15 00:29:49.433872: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:168] retrieving CUDA diagnostic information for host: iotg-dev-workstation-07
    2023-11-15 00:29:49.433876: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:175] hostname: iotg-dev-workstation-07
    2023-11-15 00:29:49.434026: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:199] libcuda reported version is: 470.223.2
    2023-11-15 00:29:49.434042: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:203] kernel reported version is: 470.182.3
    2023-11-15 00:29:49.434046: E tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:312] kernel version 470.182.3 does not match DSO version 470.223.2 -- cannot find working devices in this configuration
    2023-11-15 00:29:49.527173: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int64 and shape [1]
    	 [[{{node Placeholder/_4}}]]
    2023-11-15 00:29:49.527491: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [1]
    	 [[{{node Placeholder/_0}}]]
    2023-11-15 00:29:49.604302: W tensorflow/core/kernels/data/cache_dataset_ops.cc:856] The calling iterator did not fully read the dataset being cached. In order to avoid unexpected truncation of the dataset, the partially cached contents of the dataset  will be discarded. This can happen if you have an input pipeline similar to `dataset.cache().take(k).repeat()`. You should use `dataset.take(k).cache().repeat()` instead.



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

    2023-11-15 00:29:50.670388: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_1' with dtype string and shape [1]
    	 [[{{node Placeholder/_1}}]]
    2023-11-15 00:29:50.670801: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_2' with dtype string and shape [1]
    	 [[{{node Placeholder/_2}}]]


.. parsed-literal::

    4/4 [==============================] - 1s 235ms/sample - loss: 0.9807 - acc@1: 0.8220
    
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

    2023-11-15 00:29:53.164614: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_1' with dtype string and shape [1]
    	 [[{{node Placeholder/_1}}]]
    2023-11-15 00:29:53.164992: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int64 and shape [1]
    	 [[{{node Placeholder/_4}}]]
    2023-11-15 00:29:54.320146: W tensorflow/core/kernels/data/cache_dataset_ops.cc:856] The calling iterator did not fully read the dataset being cached. In order to avoid unexpected truncation of the dataset, the partially cached contents of the dataset  will be discarded. This can happen if you have an input pipeline similar to `dataset.cache().take(k).repeat()`. You should use `dataset.take(k).cache().repeat()` instead.
    2023-11-15 00:29:54.969869: W tensorflow/core/kernels/data/cache_dataset_ops.cc:856] The calling iterator did not fully read the dataset being cached. In order to avoid unexpected truncation of the dataset, the partially cached contents of the dataset  will be discarded. This can happen if you have an input pipeline similar to `dataset.cache().take(k).repeat()`. You should use `dataset.take(k).cache().repeat()` instead.
    2023-11-15 00:30:03.554536: W tensorflow/core/kernels/data/cache_dataset_ops.cc:856] The calling iterator did not fully read the dataset being cached. In order to avoid unexpected truncation of the dataset, the partially cached contents of the dataset  will be discarded. This can happen if you have an input pipeline similar to `dataset.cache().take(k).repeat()`. You should use `dataset.take(k).cache().repeat()` instead.


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

    4/4 [==============================] - 1s 308ms/sample - loss: 0.9766 - acc@1: 0.8120


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
    Epoch 1/2
    101/101 [==============================] - 48s 408ms/step - loss: 0.7134 - acc@1: 0.9299
    Epoch 2/2
    101/101 [==============================] - 42s 416ms/step - loss: 0.6807 - acc@1: 0.9489
    4/4 [==============================] - 1s 146ms/sample - loss: 0.9760 - acc@1: 0.8160
    
    Accuracy of INT8 model after fine-tuning: 0.816
    
    Accuracy drop of tuned INT8 model over pre-trained FP32 model: 0.006


Export Models to OpenVINO Intermediate Representation (IR)
----------------------------------------------------------



Use model conversion Python API to convert the models to OpenVINO IR.

For more information about model conversion, see this
`page <https://docs.openvino.ai/2023.0/openvino_docs_model_processing_introduction.html>`__.

Executing this command may take a while.

.. code:: ipython3

    model_ir_fp32 = ov.convert_model(fp32_model)


.. parsed-literal::

    WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.base has been moved to tensorflow.python.trackable.base. The old module will be deleted in version 2.11.


.. parsed-literal::

    WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.base has been moved to tensorflow.python.trackable.base. The old module will be deleted in version 2.11.


.. code:: ipython3

    model_ir_int8 = ov.convert_model(int8_model)

Benchmark Model Performance by Computing Inference Time
-------------------------------------------------------



Finally, measure the inference performance of the ``FP32`` and ``INT8``
models, using `Benchmark
Tool <https://docs.openvino.ai/2023.0/openvino_inference_engine_tools_benchmark_tool_README.html>`__
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

.. code:: ipython3

    ov.save_model(model_ir_fp32, fp32_ir_path, compress_to_fp16=False)
    ov.save_model(model_ir_int8, int8_ir_path, compress_to_fp16=False)
    
    
    def parse_benchmark_output(benchmark_output):
        parsed_output = [line for line in benchmark_output if 'FPS' in line]
        print(*parsed_output, sep='\n')
    
    
    print('Benchmark FP32 model (IR)')
    benchmark_output = ! benchmark_app -m $fp32_ir_path -d CPU -api async -t 15 -shape [1,64,64,3]
    parse_benchmark_output(benchmark_output)
    
    print('\nBenchmark INT8 model (IR)')
    benchmark_output = ! benchmark_app -m $int8_ir_path -d CPU -api async -t 15 -shape [1,64,64,3]
    parse_benchmark_output(benchmark_output)


.. parsed-literal::

    Benchmark FP32 model (IR)
    [ INFO ] Throughput:   2859.55 FPS
    
    Benchmark INT8 model (IR)
    [ INFO ] Throughput:   11646.57 FPS


Show CPU Information for reference.

.. code:: ipython3

    core = ov.Core()
    core.get_property('CPU', "FULL_DEVICE_NAME")




.. parsed-literal::

    'Intel(R) Core(TM) i9-10920X CPU @ 3.50GHz'


