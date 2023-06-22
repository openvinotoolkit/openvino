Convert a TensorFlow Model to OpenVINO™
=======================================

This short tutorial shows how to convert a TensorFlow
`MobileNetV3 <https://docs.openvino.ai/latest/omz_models_model_mobilenet_v3_small_1_0_224_tf.html>`__
image classification model to OpenVINO `Intermediate
Representation <https://docs.openvino.ai/latest/openvino_docs_MO_DG_IR_and_opsets.html>`__
(OpenVINO IR) format, using `Model
Optimizer <https://docs.openvino.ai/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html>`__.
After creating the OpenVINO IR, load the model in `OpenVINO
Runtime <https://docs.openvino.ai/latest/openvino_docs_IE_DG_Deep_Learning_Inference_Engine_DevGuide.html>`__
and do inference with a sample image.

Imports
-------

.. code:: ipython3

    import time
    from pathlib import Path
    
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    from IPython.display import Markdown
    from openvino.runtime import Core


.. parsed-literal::

    2023-05-29 22:25:46.984005: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2023-05-29 22:25:47.018849: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-05-29 22:25:47.534965: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


Settings
--------

.. code:: ipython3

    # The paths of the source and converted models.
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)
    
    model_path = Path("model/v3-small_224_1.0_float")
    
    ir_path = Path("model/v3-small_224_1.0_float.xml")

Download model
--------------

Load model using `tf.keras.applications
api <https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV3Small>`__
and save it to the disk.

.. code:: ipython3

    model = tf.keras.applications.MobileNetV3Small()
    model.save(model_path)


.. parsed-literal::

    WARNING:tensorflow:`input_shape` is undefined or non-square, or `rows` is not 224. Weights for input shape (224, 224) will be loaded as the default.


.. parsed-literal::

    2023-05-29 22:25:48.396946: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1956] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
    Skipping registering GPU devices...


.. parsed-literal::

    WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.


.. parsed-literal::

    2023-05-29 22:25:52.606419: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,1,1,1024]
    	 [[{{node inputs}}]]
    2023-05-29 22:25:55.768588: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,1,1,1024]
    	 [[{{node inputs}}]]
    WARNING:absl:Found untraced functions such as _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op while saving (showing 5 of 54). These functions will not be directly callable after loading.


.. parsed-literal::

    INFO:tensorflow:Assets written to: model/v3-small_224_1.0_float/assets


.. parsed-literal::

    INFO:tensorflow:Assets written to: model/v3-small_224_1.0_float/assets


Convert a Model to OpenVINO IR Format
-------------------------------------

Convert a TensorFlow Model to OpenVINO IR Format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use Model Optimizer to convert a TensorFlow model to OpenVINO IR with
``FP16`` precision. The models are saved to the current directory. Add
mean values to the model and scale the output with the standard
deviation with ``--scale_values``. With these options, it is not
necessary to normalize input data before propagating it through the
network. The original model expects input images in ``RGB`` format. The
converted model also expects images in ``RGB`` format. If you want the
converted model to work with ``BGR`` images, use the
``--reverse-input-channels`` option. For more information about Model
Optimizer, including a description of the command-line options, see the
`Model Optimizer Developer
Guide <https://docs.openvino.ai/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html>`__.
For information about the model, including input shape, expected color
order and mean values, refer to the `model
documentation <https://docs.openvino.ai/latest/omz_models_model_mobilenet_v3_small_1_0_224_tf.html>`__.

First construct the command for Model Optimizer, and then execute this
command in the notebook by prepending the command with an ``!``. There
may be some errors or warnings in the output. When model optimization is
successful, the last lines of the output will include
``[ SUCCESS ] Generated IR version 11 model.``

.. code:: ipython3

    # Construct the command for Model Optimizer.
    mo_command = f"""mo
                     --saved_model_dir "{model_path}"
                     --input_shape "[1,224,224,3]"
                     --model_name "{model_path.name}"
                     --compress_to_fp16
                     --output_dir "{model_path.parent}"
                     """
    mo_command = " ".join(mo_command.split())
    print("Model Optimizer command to convert TensorFlow to OpenVINO:")
    display(Markdown(f"`{mo_command}`"))


.. parsed-literal::

    Model Optimizer command to convert TensorFlow to OpenVINO:



``mo --saved_model_dir "model/v3-small_224_1.0_float" --input_shape "[1,224,224,3]" --model_name "v3-small_224_1.0_float" --compress_to_fp16 --output_dir "model"``


.. code:: ipython3

    # Run Model Optimizer if the IR model file does not exist
    if not ir_path.exists():
        print("Exporting TensorFlow model to IR... This may take a few minutes.")
        ! $mo_command
    else:
        print(f"IR model {ir_path} already exists.")


.. parsed-literal::

    Exporting TensorFlow model to IR... This may take a few minutes.
    Check for a new version of Intel(R) Distribution of OpenVINO(TM) toolkit here https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/download.html?cid=other&source=prod&campid=ww_2023_bu_IOTG_OpenVINO-2022-3&content=upg_all&medium=organic or on https://github.com/openvinotoolkit/openvino
    [ INFO ] The model was converted to IR v11, the latest model format that corresponds to the source DL framework input/output format. While IR v11 is backwards compatible with OpenVINO Inference Engine API v1.0, please use API v2.0 (as of 2022.1) to take advantage of the latest improvements in IR v11.
    Find more information about API v2.0 and IR v11 at https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html
    [ SUCCESS ] Generated IR version 11 model.
    [ SUCCESS ] XML file: /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/notebooks/101-tensorflow-to-openvino/model/v3-small_224_1.0_float.xml
    [ SUCCESS ] BIN file: /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/notebooks/101-tensorflow-to-openvino/model/v3-small_224_1.0_float.bin


Test Inference on the Converted Model
-------------------------------------

Load the Model
~~~~~~~~~~~~~~

.. code:: ipython3

    ie = Core()
    model = ie.read_model(ir_path)
    compiled_model = ie.compile_model(model=model, device_name="CPU")

Get Model Information
~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    input_key = compiled_model.input(0)
    output_key = compiled_model.output(0)
    network_input_shape = input_key.shape 

Load an Image
~~~~~~~~~~~~~

Load an image, resize it, and convert it to the input shape of the
network.

.. code:: ipython3

    # The MobileNet network expects images in RGB format.
    image = cv2.cvtColor(cv2.imread(filename="../data/image/coco.jpg"), code=cv2.COLOR_BGR2RGB)
    
    # Resize the image to the network input shape.
    resized_image = cv2.resize(src=image, dsize=(224, 224))
    
    # Transpose the image to the network input shape.
    input_image = np.expand_dims(resized_image, 0)
    
    plt.imshow(image);



.. image:: 101-tensorflow-to-openvino-with-output_files/101-tensorflow-to-openvino-with-output_16_0.png


Do Inference
~~~~~~~~~~~~

.. code:: ipython3

    result = compiled_model(input_image)[output_key]
    
    result_index = np.argmax(result)

.. code:: ipython3

    # Convert the inference result to a class name.
    imagenet_classes = open("../data/datasets/imagenet/imagenet_2012.txt").read().splitlines()
    
    imagenet_classes[result_index]




.. parsed-literal::

    'n02099267 flat-coated retriever'



Timing
------

Measure the time it takes to do inference on thousand images. This gives
an indication of performance. For more accurate benchmarking, use the
`Benchmark
Tool <https://docs.openvino.ai/latest/openvino_inference_engine_tools_benchmark_tool_README.html>`__
in OpenVINO. Note that many optimizations are possible to improve the
performance.

.. code:: ipython3

    num_images = 1000
    
    start = time.perf_counter()
    
    for _ in range(num_images):
        compiled_model([input_image])
    
    end = time.perf_counter()
    time_ir = end - start
    
    print(
        f"IR model in OpenVINO Runtime/CPU: {time_ir/num_images:.4f} "
        f"seconds per image, FPS: {num_images/time_ir:.2f}"
    )


.. parsed-literal::

    IR model in OpenVINO Runtime/CPU: 0.0010 seconds per image, FPS: 1032.55

