Super Resolution with PaddleGAN and OpenVINO™
=============================================

This notebook demonstrates converting the RealSR (real-world
super-resolution) model from
`PaddlePaddle/PaddleGAN <https://github.com/PaddlePaddle/PaddleGAN>`__
to OpenVINO Intermediate Representation (OpenVINO IR) format, and shows
inference results on both the PaddleGAN and OpenVINO IR models.

For more information about the various PaddleGAN superresolution models,
refer to the `PaddleGAN
documentation <https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/en_US/tutorials/single_image_super_resolution.md>`__.
For more information about RealSR, see the `research
paper <https://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Ji_Real-World_Super-Resolution_via_Kernel_Estimation_and_Noise_Injection_CVPRW_2020_paper.pdf>`__
from CVPR 2020.

This notebook works best with small images (up to 800x600 resolution).

**Table of contents:**

- `Imports <#imports>`__
- `Settings <#settings>`__
- `Inference on PaddlePaddle Model <#inference-on-paddlepaddle-model>`__

  - `Investigate PaddleGAN Model <#investigate-paddlegan-model>`__
  - `Do Inference <#do-inference>`__

- `Convert PaddleGAN Model to ONNX and OpenVINO IR <#convert-paddlegan-model-to-onnx-and-openvino-ir>`__

  - `Convert PaddlePaddle Model to ONNX <#convert-paddlepaddle-model-to-onnx>`__
  - `Convert ONNX Model to OpenVINO IR with Model Conversion Python API <#convert-onnx-model-to-openvino-ir-with-model-conversion-python-api>`__

- `Do Inference on OpenVINO IR Model <#do-inference-on-openvino-ir-model>`__

  - `Select inference device <#select-inference-device>`__
  - `Show an Animated GIF <#show-an-animated-gif>`__
  - `Create a Comparison Video <#create-a-comparison-video>`__

Imports
###############################################################################################################################

.. code:: ipython3

    !pip install -q "openvino==2023.1.0.dev20230811"
    
    !pip install -q "paddlepaddle==2.5.0rc0" "paddle2onnx>=0.6"
    
    !pip install -q "imageio==2.9.0" "imageio-ffmpeg" "numba>=0.53.1" "easydict" "munch" "natsort"
    !pip install -q "git+https://github.com/PaddlePaddle/PaddleGAN.git" --no-deps
    !pip install -q scikit-image


.. parsed-literal::

    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    ppgan 2.1.0 requires imageio==2.9.0, but you have imageio 2.31.3 which is incompatible.
    ppgan 2.1.0 requires librosa==0.8.1, but you have librosa 0.10.1 which is incompatible.
    ppgan 2.1.0 requires opencv-python<=4.6.0.66, but you have opencv-python 4.8.0.76 which is incompatible.
    

.. code:: ipython3

    import sys
    import time
    import warnings
    from pathlib import Path
    
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import openvino as ov
    import paddle
    from IPython.display import HTML, FileLink, ProgressBar, clear_output, display
    from IPython.display import Image as DisplayImage
    from PIL import Image
    from paddle.static import InputSpec
    from ppgan.apps import RealSRPredictor
    
    sys.path.append("../utils")
    from notebook_utils import NotebookAlert

Settings
###############################################################################################################################

.. code:: ipython3

    # The filenames of the downloaded and converted models.
    MODEL_NAME = "paddlegan_sr"
    MODEL_DIR = Path("model")
    OUTPUT_DIR = Path("output")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    model_path = MODEL_DIR / MODEL_NAME
    ir_path = model_path.with_suffix(".xml")
    onnx_path = model_path.with_suffix(".onnx")

Inference on PaddlePaddle Model
###############################################################################################################################

Investigate PaddleGAN Model
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The `PaddleGAN
documentation <https://github.com/PaddlePaddle/PaddleGAN>`__ explains
how to run the model with ``sr.run()`` method. Find out what that
function does, and check other relevant functions that are called from
that function. Adding ``??`` to the methods shows the docstring and
source code.

.. code:: ipython3

    # Running this cell will download the model weights if they have not been downloaded before.
    # This may take a while.
    sr = RealSRPredictor()


.. parsed-literal::

    [09/08 23:31:15] ppgan INFO: Found /opt/home/k8sworker/.cache/ppgan/DF2K_JPEG.pdparams


.. code:: ipython3

    sr.run??

.. code:: ipython3

    sr.run_image??

.. code:: ipython3

    sr.norm??

.. code:: ipython3

    sr.denorm??

The run checks whether the input is an image or a video. For an image,
it loads the image as an ``RGB`` image, normalizes it, and converts it
to a Paddle tensor. It is propagated to the network by calling the
``self.model()`` method and then *“denormalized”*. The normalization
function simply divides all image values by 255. This converts an image
with integer values in the range of 0 to 255 to an image with floating
point values in the range of 0 to 1. The denormalization function
transforms the output from the (C,H,W) network shape to (H,W,C) image
shape. It then clips the image values between 0 and 255, and converts
the image to a standard ``RGB`` image with integer values in the range
of 0 to 255.

To get more information about how the model looks like, use the
``sr.model??`` command.

.. code:: ipython3

    # sr.model??

Do Inference
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

To show inference on the PaddlePaddle model, set ``PADDLEGAN_INFERENCE``
to ``True`` in the cell below. Keep in mind that performing inference
may take some time.

.. code:: ipython3

    # Set PADDLEGAN_INFERENCE to True to show inference on the PaddlePaddle model.
    # This may take a long time, especially for larger images.
    #
    PADDLEGAN_INFERENCE = False
    if PADDLEGAN_INFERENCE:
        # Load the input image and convert to a tensor with the input shape.
        IMAGE_PATH = Path("../data/image/coco_tulips.jpg")
        image = cv2.cvtColor(cv2.imread(str(IMAGE_PATH)), cv2.COLOR_BGR2RGB)
        input_image = image.transpose(2, 0, 1)[None, :, :, :] / 255
        input_tensor = paddle.to_tensor(input_image.astype(np.float32))
        if max(image.shape) > 400:
            NotebookAlert(
                f"This image has {image.shape} shape. Doing inference will be slow "
                "and the notebook may stop responding. Set PADDLEGAN_INFERENCE to False "
                "to skip doing inference on the PaddlePaddle model.",
                "warning",
            )

.. code:: ipython3

    if PADDLEGAN_INFERENCE:
        # Do inference and measure how long it takes.
        print(f"Start superresolution inference for {IMAGE_PATH.name} with shape {image.shape}...")
        start_time = time.perf_counter()
        sr.model.eval()
        with paddle.no_grad():
            result = sr.model(input_tensor)
        end_time = time.perf_counter()
        duration = end_time - start_time
        result_image = (
            (result.numpy().squeeze() * 255).clip(0, 255).astype("uint8").transpose((1, 2, 0))
        )
        print(f"Superresolution image shape: {result_image.shape}")
        print(f"Inference duration: {duration:.2f} seconds")
        plt.imshow(result_image);

Convert PaddleGAN Model to ONNX and OpenVINO IR
###############################################################################################################################

To convert the PaddlePaddle model to OpenVINO IR, first convert the
model to ONNX, and then convert the ONNX model to the OpenVINO IR
format.

Convert PaddlePaddle Model to ONNX
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. code:: ipython3

    # Ignore PaddlePaddle warnings:
    # The behavior of expression A + B has been unified with elementwise_add(X, Y, axis=-1).
    warnings.filterwarnings("ignore")
    sr.model.eval()
    # ONNX export requires an input shape in this format as a parameter.
    # Both OpenVINO and Paddle support `-1` placeholder for marking flexible dimensions
    input_shape = [-1, 3, -1, -1]
    x_spec = InputSpec(input_shape, "float32", "x")
    paddle.onnx.export(sr.model, str(model_path), input_spec=[x_spec], opset_version=13)


.. parsed-literal::

    2023-09-08 23:31:21 [INFO]	Static PaddlePaddle model saved in model/paddle_model_static_onnx_temp_dir.


.. parsed-literal::

    I0908 23:31:21.665750 670756 interpretercore.cc:267] New Executor is Running.


.. parsed-literal::

    [Paddle2ONNX] Start to parse PaddlePaddle model...
    [Paddle2ONNX] Model file path: model/paddle_model_static_onnx_temp_dir/model.pdmodel
    [Paddle2ONNX] Paramters file path: model/paddle_model_static_onnx_temp_dir/model.pdiparams
    [Paddle2ONNX] Start to parsing Paddle model...
    [Paddle2ONNX] Use opset_version = 13 for ONNX export.
    [Paddle2ONNX] PaddlePaddle model is exported as ONNX format now.
    2023-09-08 23:31:25 [INFO]	ONNX model saved in model/paddlegan_sr.onnx.


Convert ONNX Model to OpenVINO IR with `Model Conversion Python API <https://docs.openvino.ai/2023.0/openvino_docs_model_processing_introduction.html>`__
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
.. code:: ipython3

    print("Exporting ONNX model to OpenVINO IR... This may take a few minutes.")
    
    model = ov.convert_model(
        onnx_path,
        input=input_shape
    )
    
    # Serialize model in IR format
    ov.save_model(model, str(ir_path))


.. parsed-literal::

    Exporting ONNX model to OpenVINO IR... This may take a few minutes.


Do Inference on OpenVINO IR Model
###############################################################################################################################

.. code:: ipython3

    # Read the network and get input and output names.
    core = ov.Core()
    # Alternatively, the model obtained from `ov.convert_model()` may be used here
    model = core.read_model(model=ir_path)
    input_layer = model.input(0)

Select inference device
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Select device from dropdown list for running inference using OpenVINO:

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



.. code:: ipython3

    # Load and show the image.
    IMAGE_PATH = Path("../data/image/coco_tulips.jpg")
    image = cv2.cvtColor(cv2.imread(str(IMAGE_PATH)), cv2.COLOR_BGR2RGB)
    if max(image.shape) > 800:
        NotebookAlert(
            f"This image has shape {image.shape}. The notebook works best with images with "
            "a maximum side of 800x600. Larger images may work well, but inference may "
            "be slow",
            "warning",
        )
    plt.imshow(image)




.. parsed-literal::

    <matplotlib.image.AxesImage at 0x7fc37f61ac10>




.. image:: 207-vision-paddlegan-superresolution-with-output_files/207-vision-paddlegan-superresolution-with-output_25_1.png


.. code:: ipython3

    # Load the network to the CPU device (this may take a few seconds).
    compiled_model = core.compile_model(model=model, device_name=device.value)
    output_layer = compiled_model.output(0)

.. code:: ipython3

    # Convert the image to the network input shape and divide pixel values by 255.
    # See the "Investigate PaddleGAN model" section.
    input_image = image.transpose(2, 0, 1)[None, :, :, :] / 255
    start_time = time.perf_counter()
    # Do inference.
    ir_result = compiled_model([input_image])[output_layer]
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"Inference duration: {duration:.2f} seconds")


.. parsed-literal::

    Inference duration: 3.26 seconds


.. code:: ipython3

    # Get the result array in CHW format.
    result_array = ir_result.squeeze()
    # Convert the array to an image with the same method as PaddleGAN:
    # Multiply by 255, clip values between 0 and 255, convert to a HWC INT8 image.
    # See the "Investigate PaddleGAN model" section.
    image_super = (result_array * 255).clip(0, 255).astype("uint8").transpose((1, 2, 0))
    # Resize the image with bicubic upsampling for comparison.
    image_bicubic = cv2.resize(image, tuple(image_super.shape[:2][::-1]), interpolation=cv2.INTER_CUBIC)

.. code:: ipython3

    plt.imshow(image_super)




.. parsed-literal::

    <matplotlib.image.AxesImage at 0x7fc2dc11e250>




.. image:: 207-vision-paddlegan-superresolution-with-output_files/207-vision-paddlegan-superresolution-with-output_29_1.png


Show an Animated GIF
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

To visualize the difference between the bicubic image and the
superresolution image, create an animated GIF image that switches
between both versions.

.. code:: ipython3

    result_pil = Image.fromarray(image_super)
    bicubic_pil = Image.fromarray(image_bicubic)
    gif_image_path = OUTPUT_DIR / Path(IMAGE_PATH.stem + "_comparison.gif")
    final_image_path = OUTPUT_DIR / Path(IMAGE_PATH.stem + "_super.png")
    
    result_pil.save(
        fp=str(gif_image_path),
        format="GIF",
        append_images=[bicubic_pil],
        save_all=True,
        duration=1000,
        loop=0,
    )
    
    result_pil.save(fp=str(final_image_path), format="png")
    DisplayImage(open(gif_image_path, "rb").read(), width=1920 // 2)




.. image:: 207-vision-paddlegan-superresolution-with-output_files/207-vision-paddlegan-superresolution-with-output_31_0.png
   :width: 960px



Create a Comparison Video
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Create a video with a “slider”, showing the bicubic image to the right
and the superresolution image on the left.

For the video, the superresolution and bicubic image are resized to half
the original width and height, to improve processing speed. This gives
an indication of the superresolution effect. The video is saved as an
``.avi`` video file. You can click on the link to download the video, or
open it directly from the images directory, and play it locally.

.. code:: ipython3

    FOURCC = cv2.VideoWriter_fourcc(*"MJPG")
    IMAGE_PATH = Path(IMAGE_PATH)
    result_video_path = OUTPUT_DIR / Path(f"{IMAGE_PATH.stem}_comparison_paddlegan.avi")
    video_target_height, video_target_width = (
        image_super.shape[0] // 2,
        image_super.shape[1] // 2,
    )
    
    out_video = cv2.VideoWriter(
        str(result_video_path),
        FOURCC,
        90,
        (video_target_width, video_target_height),
    )
    
    resized_result_image = cv2.resize(image_super, (video_target_width, video_target_height))[
        :, :, (2, 1, 0)
    ]
    resized_bicubic_image = cv2.resize(image_bicubic, (video_target_width, video_target_height))[
        :, :, (2, 1, 0)
    ]
    
    progress_bar = ProgressBar(total=video_target_width)
    progress_bar.display()
    
    for i in range(2, video_target_width):
        # Create a frame where the left part (until i pixels width) contains the
        # superresolution image, and the right part (from i pixels width) contains
        # the bicubic image.
        comparison_frame = np.hstack(
            (
                resized_result_image[:, :i, :],
                resized_bicubic_image[:, i:, :],
            )
        )
    
        # Create a small black border line between the superresolution
        # and bicubic part of the image.
        comparison_frame[:, i - 1 : i + 1, :] = 0
        out_video.write(comparison_frame)
        progress_bar.progress = i
        progress_bar.update()
    out_video.release()
    clear_output()
    
    video_link = FileLink(result_video_path)
    video_link.html_link_str = "<a href='%s' download>%s</a>"
    display(HTML(f"The video has been saved to {video_link._repr_html_()}"))



.. raw:: html

    The video has been saved to output/coco_tulips_comparison_paddlegan.avi<br>

