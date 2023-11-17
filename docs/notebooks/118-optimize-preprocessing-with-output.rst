Optimize Preprocessing
======================

When input data does not fit the model input tensor perfectly,
additional operations/steps are needed to transform the data to the
format expected by the model. This tutorial demonstrates how it could be
performed with Preprocessing API. Preprocessing API is an easy-to-use
instrument, that enables integration of preprocessing steps into an
execution graph and performing it on a selected device, which can
improve device utilization. For more information about Preprocessing
API, see this
`overview <https://docs.openvino.ai/2023.0/openvino_docs_OV_UG_Preprocessing_Overview.html#>`__
and
`details <https://docs.openvino.ai/2023.0/openvino_docs_OV_UG_Preprocessing_Details.html>`__

This tutorial include following steps:

-  Downloading the model.
-  Setup preprocessing with Preprocessing API, loading the model and
   inference with original image.
-  Fitting image to the model input type and inference with prepared
   image.
-  Comparing results on one picture.
-  Comparing performance.

**Table of contents:**

-  `Settings <#settings>`__
-  `Imports <#imports>`__

   -  `Setup image and device <#setup-image-and-device>`__
   -  `Downloading the model <#downloading-the-model>`__
   -  `Create core <#create-core>`__
   -  `Check the original parameters of
      image <#check-the-original-parameters-of-image>`__

-  `Setup preprocessing steps with Preprocessing API and perform
   inference <#setup-preprocessing-steps-with-preprocessing-api-and-perform-inference>`__

   -  `Convert model to OpenVINO IR with model conversion
      API <#convert-model-to-openvino-ir-with-model-conversion-api>`__
   -  `Create PrePostProcessor
      Object <#create-prepostprocessor-object>`__
   -  `Declare User’s Data Format <#declare-users-data-format>`__
   -  `Declaring Model Layout <#declaring-model-layout>`__
   -  `Preprocessing Steps <#preprocessing-steps>`__
   -  `Integrating Steps into a
      Model <#integrating-steps-into-a-model>`__

-  `Load model and perform
   inference <#load-model-and-perform-inference>`__
-  `Fit image manually and perform
   inference <#fit-image-manually-and-perform-inference>`__

   -  `Load the model <#load-the-model>`__
   -  `Load image and fit it to model
      input <#load-image-and-fit-it-to-model-input>`__
   -  `Perform inference <#perform-inference>`__

-  `Compare results <#compare-results>`__

   -  `Compare results on one image <#compare-results-on-one-image>`__
   -  `Compare performance <#compare-performance>`__

Settings
--------



.. code:: ipython3

    # Install openvino package
    %pip install -q "openvino>=2023.1.0" tensorflow opencv-python matplotlib


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


Imports
-------



.. code:: ipython3

    import time
    from pathlib import Path
    
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import openvino as ov
    import tensorflow as tf
    
    # Fetch `notebook_utils` module
    import urllib.request
    urllib.request.urlretrieve(
        url='https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/utils/notebook_utils.py',
        filename='notebook_utils.py'
    )
    from notebook_utils import download_file


.. parsed-literal::

    2023-11-14 23:00:32.637266: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2023-11-14 23:00:32.671311: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-11-14 23:00:33.179278: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


Setup image and device
~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    # Download the image from the openvino_notebooks storage
    image_path = download_file(
        "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg",
        directory="data"
    )
    image_path = str(image_path)



.. parsed-literal::

    data/coco.jpg:   0%|          | 0.00/202k [00:00<?, ?B/s]


.. code:: ipython3

    import ipywidgets as widgets
    
    core = ov.Core()
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value='AUTO',
        description='Device:',
        disabled=False,
    )
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



Downloading the model
~~~~~~~~~~~~~~~~~~~~~



This tutorial uses the
`InceptionResNetV2 <https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_resnet_v2>`__.
The InceptionResNetV2 model is the second of the
`Inception <https://github.com/tensorflow/tpu/tree/master/models/experimental/inception>`__
family of models designed to perform image classification. Like other
Inception models, InceptionResNetV2 has been pre-trained on the
`ImageNet <https://image-net.org/>`__ data set. For more details about
this family of models, see the `research
paper <https://arxiv.org/abs/1602.07261>`__.

Load the model by using `tf.keras.applications
api <https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_resnet_v2>`__
and save it to the disk.

.. code:: ipython3

    model_name = "InceptionResNetV2"
    
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / model_name
    
    model = tf.keras.applications.InceptionV3()
    model.save(model_path)


.. parsed-literal::

    2023-11-14 23:00:37.345835: E tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:266] failed call to cuInit: CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: forward compatibility was attempted on non supported HW
    2023-11-14 23:00:37.345869: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:168] retrieving CUDA diagnostic information for host: iotg-dev-workstation-07
    2023-11-14 23:00:37.345874: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:175] hostname: iotg-dev-workstation-07
    2023-11-14 23:00:37.346012: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:199] libcuda reported version is: 470.223.2
    2023-11-14 23:00:37.346027: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:203] kernel reported version is: 470.182.3
    2023-11-14 23:00:37.346030: E tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:312] kernel version 470.182.3 does not match DSO version 470.223.2 -- cannot find working devices in this configuration


.. parsed-literal::

    WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.


.. parsed-literal::

    WARNING:absl:Found untraced functions such as _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op while saving (showing 5 of 94). These functions will not be directly callable after loading.


.. parsed-literal::

    INFO:tensorflow:Assets written to: model/InceptionResNetV2/assets


.. parsed-literal::

    INFO:tensorflow:Assets written to: model/InceptionResNetV2/assets


Create core
~~~~~~~~~~~



.. code:: ipython3

    core = ov.Core()

Check the original parameters of image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    image = cv2.imread(image_path)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB));
    print(f"The original shape of the image is {image.shape}")
    print(f"The original data type of the image is {image.dtype}")


.. parsed-literal::

    The original shape of the image is (577, 800, 3)
    The original data type of the image is uint8



.. image:: 118-optimize-preprocessing-with-output_files/118-optimize-preprocessing-with-output_14_1.png


Setup preprocessing steps with Preprocessing API and perform inference
----------------------------------------------------------------------



Intuitively, preprocessing API consists of the following parts:

-  Tensor - declares user data format, like shape, layout, precision,
   color format from actual user’s data.
-  Steps - describes sequence of preprocessing steps which need to be
   applied to user data.
-  Model - specifies model data format. Usually, precision and shape are
   already known for model, only additional information, like layout can
   be specified.

Graph modifications of a model shall be performed after the model is
read from a drive and before it is loaded on the actual device.

Pre-processing support following operations (please, see more details
`here <https://docs.openvino.ai/2023.0/classov_1_1preprocess_1_1PreProcessSteps.html#doxid-classov-1-1preprocess-1-1-pre-process-steps-1aeacaf406d72a238e31a359798ebdb3b7>`__)

-  Mean/Scale Normalization
-  Converting Precision
-  Converting layout (transposing)
-  Resizing Image
-  Color Conversion
-  Custom Operations

Convert model to OpenVINO IR with model conversion API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



The options for preprocessing are not required.

.. code:: ipython3

    ir_path = model_dir / "ir_model" / f"{model_name}.xml"
    
    ppp_model = None
    
    if ir_path.exists():
        ppp_model = core.read_model(model=ir_path)
        print(f"Model in OpenVINO format already exists: {ir_path}")
    else: 
        ppp_model = ov.convert_model(model_path,
                                     input=[1,299,299,3])
        ov.save_model(ppp_model, str(ir_path))

Create ``PrePostProcessor`` Object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



The
```PrePostProcessor()`` <https://docs.openvino.ai/2023.0/classov_1_1preprocess_1_1PrePostProcessor.html#doxid-classov-1-1preprocess-1-1-pre-post-processor>`__
class enables specifying the preprocessing and postprocessing steps for
a model.

.. code:: ipython3

    from openvino.preprocess import PrePostProcessor
    
    ppp = PrePostProcessor(ppp_model)

Declare User’s Data Format
~~~~~~~~~~~~~~~~~~~~~~~~~~



To address particular input of a model/preprocessor, use the
``PrePostProcessor.input(input_name)`` method. If the model has only one
input, then simple ``PrePostProcessor.input()`` will get a reference to
pre-processing builder for this input (a tensor, the steps, a model). In
general, when a model has multiple inputs/outputs, each one can be
addressed by a tensor name or by its index. By default, information
about user’s input tensor will be initialized to same data
(type/shape/etc) as model’s input parameter. User application can
override particular parameters according to application’s data. Refer to
the following
`page <https://docs.openvino.ai/2023.0/classov_1_1preprocess_1_1InputTensorInfo.html#doxid-classov-1-1preprocess-1-1-input-tensor-info-1a98fb73ff9178c8c71d809ddf8927faf5>`__
for more information about parameters for overriding.

Below is all the specified input information:

-  Precision is ``U8`` (unsigned 8-bit integer).
-  Size is non-fixed, setup of one determined shape size can be done
   with ``.set_shape([1, 577, 800, 3])``
-  Layout is ``“NHWC”``. It means, for example: height=577, width=800,
   channels=3.

The height and width are necessary for resizing, and channels are needed
for mean/scale normalization.

.. code:: ipython3

    # setup formant of data
    ppp.input().tensor().set_element_type(ov.Type.u8)\
                        .set_spatial_dynamic_shape()\
                        .set_layout(ov.Layout('NHWC'))




.. parsed-literal::

    <openvino._pyopenvino.preprocess.InputTensorInfo at 0x7fbffd787d70>



Declaring Model Layout
~~~~~~~~~~~~~~~~~~~~~~



Model input already has information about precision and shape.
Preprocessing API is not intended to modify this. The only thing that
may be specified is input data
`layout <https://docs.openvino.ai/2023.0/openvino_docs_OV_UG_Layout_Overview.html#doxid-openvino-docs-o-v-u-g-layout-overview>`__.

.. code:: ipython3

    input_layer_ir = next(iter(ppp_model.inputs))
    print(f"The input shape of the model is {input_layer_ir.shape}")
    
    ppp.input().model().set_layout(ov.Layout('NHWC'))


.. parsed-literal::

    The input shape of the model is [1,299,299,3]




.. parsed-literal::

    <openvino._pyopenvino.preprocess.InputModelInfo at 0x7fbffd7870b0>



Preprocessing Steps
~~~~~~~~~~~~~~~~~~~



Now, the sequence of preprocessing steps can be defined. For more
information about preprocessing steps, see
`here <https://docs.openvino.ai/2023.0/api/ie_python_api/_autosummary/openvino.preprocess.PreProcessSteps.html>`__.

Perform the following:

-  Convert ``U8`` to ``FP32`` precision.
-  Resize to height/width of a model. Be aware that if a model accepts
   dynamic size, for example, ``{?, 3, ?, ?}`` resize will not know how
   to resize the picture. Therefore, in this case, target height/ width
   should be specified. For more details, see also the
   ```PreProcessSteps.resize()`` <https://docs.openvino.ai/2023.0/classov_1_1preprocess_1_1PreProcessSteps.html#doxid-classov-1-1preprocess-1-1-pre-process-steps-1a40dab78be1222fee505ed6a13400efe6>`__.
-  Subtract mean from each channel.
-  Divide each pixel data to appropriate scale value.

There is no need to specify conversion layout. If layouts are different,
then such conversion will be added explicitly.

.. code:: ipython3

    from openvino.preprocess import ResizeAlgorithm
    
    ppp.input().preprocess().convert_element_type(ov.Type.f32) \
                            .resize(ResizeAlgorithm.RESIZE_LINEAR)\
                            .mean([127.5,127.5,127.5])\
                            .scale([127.5,127.5,127.5])




.. parsed-literal::

    <openvino._pyopenvino.preprocess.PreProcessSteps at 0x7fc0a02556b0>



Integrating Steps into a Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Once the preprocessing steps have been finished, the model can be
finally built. It is possible to display ``PrePostProcessor``
configuration for debugging purposes.

.. code:: ipython3

    print(f'Dump preprocessor: {ppp}')
    model_with_preprocess = ppp.build()


.. parsed-literal::

    Dump preprocessor: Input "input_1":
        User's input tensor: [1,?,?,3], [N,H,W,C], u8
        Model's expected tensor: [1,299,299,3], [N,H,W,C], f32
        Pre-processing steps (4):
          convert type (f32): ([1,?,?,3], [N,H,W,C], u8) -> ([1,?,?,3], [N,H,W,C], f32)
          resize to model width/height: ([1,?,?,3], [N,H,W,C], f32) -> ([1,299,299,3], [N,H,W,C], f32)
          mean (127.5,127.5,127.5): ([1,299,299,3], [N,H,W,C], f32) -> ([1,299,299,3], [N,H,W,C], f32)
          scale (127.5,127.5,127.5): ([1,299,299,3], [N,H,W,C], f32) -> ([1,299,299,3], [N,H,W,C], f32)
    


Load model and perform inference
--------------------------------



.. code:: ipython3

    def prepare_image_api_preprocess(image_path, model=None):
        image = cv2.imread(image_path)
        input_tensor = np.expand_dims(image, 0)
        return input_tensor
    
    
    compiled_model_with_preprocess_api = core.compile_model(model=ppp_model, device_name=device.value)
    
    ppp_output_layer = compiled_model_with_preprocess_api.output(0)
    
    ppp_input_tensor = prepare_image_api_preprocess(image_path)
    results = compiled_model_with_preprocess_api(ppp_input_tensor)[ppp_output_layer][0]

Fit image manually and perform inference
----------------------------------------



Load the model
~~~~~~~~~~~~~~



.. code:: ipython3

    model = core.read_model(model=ir_path)
    compiled_model = core.compile_model(model=model, device_name=device.value)

Load image and fit it to model input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    def manual_image_preprocessing(path_to_image, compiled_model):
        input_layer_ir = next(iter(compiled_model.inputs))
    
        # N, H, W, C = batch size, height, width, number of channels
        N, H, W, C = input_layer_ir.shape
        
        # load  image, image will be resized to model input size and converted to RGB
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(H, W), color_mode='rgb')
    
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
    
        # will scale input pixels between -1 and 1
        input_tensor = tf.keras.applications.inception_resnet_v2.preprocess_input(x)
    
        return input_tensor
    
    
    input_tensor = manual_image_preprocessing(image_path, compiled_model)
    print(f"The shape of the image is {input_tensor.shape}")
    print(f"The data type of the image is {input_tensor.dtype}")


.. parsed-literal::

    The shape of the image is (1, 299, 299, 3)
    The data type of the image is float32


Perform inference
~~~~~~~~~~~~~~~~~



.. code:: ipython3

    output_layer = compiled_model.output(0)
    
    result = compiled_model(input_tensor)[output_layer]

Compare results
---------------



Compare results on one image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    def check_results(input_tensor, compiled_model, imagenet_classes):
        output_layer = compiled_model.output(0)
    
        results = compiled_model(input_tensor)[output_layer][0]
    
        top_indices = np.argsort(results)[-5:][::-1]
        top_softmax = results[top_indices]
    
        for index, softmax_probability in zip(top_indices, top_softmax):
            print(f"{imagenet_classes[index]}, {softmax_probability:.5f}")
    
        return top_indices, top_softmax
    
    
    # Convert the inference result to a class name.
    imagenet_filename = download_file(
        "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/datasets/imagenet/imagenet_2012.txt",
        directory="data"
    )
    imagenet_classes = imagenet_filename.read_text().splitlines()
    imagenet_classes = ['background'] + imagenet_classes
    
    # get result for inference with preprocessing api
    print("Result of inference with Preprocessing API:")
    res = check_results(ppp_input_tensor, compiled_model_with_preprocess_api, imagenet_classes)
    
    print("\n")
    
    # get result for inference with the manual preparing of the image
    print("Result of inference with manual image setup:")
    res = check_results(input_tensor, compiled_model, imagenet_classes)



.. parsed-literal::

    data/imagenet_2012.txt:   0%|          | 0.00/30.9k [00:00<?, ?B/s]


.. parsed-literal::

    Result of inference with Preprocessing API:
    n02099601 golden retriever, 0.80560
    n02098413 Lhasa, Lhasa apso, 0.10039
    n02108915 French bulldog, 0.01915
    n02111129 Leonberg, 0.00825
    n02097047 miniature schnauzer, 0.00294
    
    
    Result of inference with manual image setup:
    n02098413 Lhasa, Lhasa apso, 0.76843
    n02099601 golden retriever, 0.19322
    n02111129 Leonberg, 0.00720
    n02097047 miniature schnauzer, 0.00287
    n02100877 Irish setter, red setter, 0.00115


Compare performance
~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    def check_performance(compiled_model, preprocessing_function=None):
        num_images = 1000
    
        start = time.perf_counter()
    
        for _ in range(num_images):
            input_tensor = preprocessing_function(image_path, compiled_model)
            compiled_model(input_tensor)
    
        end = time.perf_counter()
        time_ir = end - start
    
        return time_ir, num_images
    
    time_ir, num_images = check_performance(compiled_model, manual_image_preprocessing)
    print(
        f"IR model in OpenVINO Runtime/CPU with manual image preprocessing: {time_ir/num_images:.4f} "
        f"seconds per image, FPS: {num_images/time_ir:.2f}"
    )
    
    time_ir, num_images = check_performance(compiled_model_with_preprocess_api, prepare_image_api_preprocess)
    print(
        f"IR model in OpenVINO Runtime/CPU with preprocessing API: {time_ir/num_images:.4f} "
        f"seconds per image, FPS: {num_images/time_ir:.2f}"
    )


.. parsed-literal::

    IR model in OpenVINO Runtime/CPU with manual image preprocessing: 0.0152 seconds per image, FPS: 65.58
    IR model in OpenVINO Runtime/CPU with preprocessing API: 0.0187 seconds per image, FPS: 53.52

