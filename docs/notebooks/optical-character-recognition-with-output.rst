Optical Character Recognition (OCR) with OpenVINO™
==================================================

This tutorial demonstrates how to perform optical character recognition
(OCR) with OpenVINO models. It is a continuation of the
`hello-detection <hello-detection-with-output.html>`__ tutorial,
which shows only text detection.

The
`horizontal-text-detection-0001 <https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/horizontal-text-detection-0001/README.md>`__
and
`text-recognition-resnet <https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/text-recognition-resnet-fc/README.md>`__
models are used together for text detection and then text recognition.

In this tutorial, Open Model Zoo tools including Model Downloader, Model
Converter and Info Dumper are used to download and convert the models
from `Open Model
Zoo <https://github.com/openvinotoolkit/open_model_zoo>`__. For more
information, refer to the
`model-tools <model-tools-with-output.html>`__ tutorial.


**Table of contents:**


-  `Imports <#imports>`__
-  `Settings <#settings>`__
-  `Download Models <#download-models>`__
-  `Convert Models <#convert-models>`__
-  `Select inference device <#select-inference-device>`__
-  `Object Detection <#object-detection>`__

   -  `Load a Detection Model <#load-a-detection-model>`__
   -  `Load an Image <#load-an-image>`__
   -  `Do Inference <#do-inference>`__
   -  `Get Detection Results <#get-detection-results>`__

-  `Text Recognition <#text-recognition>`__

   -  `Load Text Recognition Model <#load-text-recognition-model>`__
   -  `Do Inference <#do-inference>`__

-  `Show Results <#show-results>`__

   -  `Show Detected Text Boxes and OCR Results for the
      Image <#show-detected-text-boxes-and-ocr-results-for-the-image>`__
   -  `Show the OCR Result per Bounding
      Box <#show-the-ocr-result-per-bounding-box>`__
   -  `Print Annotations in Plain Text
      Format <#print-annotations-in-plain-text-format>`__

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

.. code:: ipython3

    # Install openvino-dev package
    %pip install -q "openvino-dev>=2024.0.0"  "onnx<1.16.2" torch torchvision pillow opencv-python "matplotlib>=3.4" --extra-index-url https://download.pytorch.org/whl/cpu


.. parsed-literal::

    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    mobileclip 0.1.0 requires torchvision==0.14.1, but you have torchvision 0.17.2+cpu which is incompatible.
    torchaudio 2.4.1+cpu requires torch==2.4.1, but you have torch 2.2.2+cpu which is incompatible.
    Note: you may need to restart the kernel to use updated packages.


Imports
-------



.. code:: ipython3

    from pathlib import Path
    
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import openvino as ov
    from IPython.display import Markdown, display
    from PIL import Image
    
    # Fetch `notebook_utils` module
    import requests
    
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    
    open("notebook_utils.py", "w").write(r.text)
    from notebook_utils import load_image, device_widget

Settings
--------



.. code:: ipython3

    core = ov.Core()
    
    model_dir = Path("model")
    precision = "FP16"
    detection_model = "horizontal-text-detection-0001"
    recognition_model = "text-recognition-resnet-fc"
    
    model_dir.mkdir(exist_ok=True)

Download Models
---------------



The next cells will run Model Downloader to download the detection and
recognition models. If the models have been downloaded before, they will
not be downloaded again.

.. code:: ipython3

    download_command = (
        f"omz_downloader --name {detection_model},{recognition_model} --output_dir {model_dir} --cache_dir {model_dir} --precision {precision}  --num_attempts 5"
    )
    display(Markdown(f"Download command: `{download_command}`"))
    display(Markdown(f"Downloading {detection_model}, {recognition_model}..."))
    !$download_command
    display(Markdown(f"Finished downloading {detection_model}, {recognition_model}."))
    
    detection_model_path = (model_dir / "intel/horizontal-text-detection-0001" / precision / detection_model).with_suffix(".xml")
    recognition_model_path = (model_dir / "public/text-recognition-resnet-fc" / precision / recognition_model).with_suffix(".xml")



Download command:
``omz_downloader --name horizontal-text-detection-0001,text-recognition-resnet-fc --output_dir model --cache_dir model --precision FP16  --num_attempts 5``



Downloading horizontal-text-detection-0001, text-recognition-resnet-fc…


.. parsed-literal::

    ################|| Downloading horizontal-text-detection-0001 ||################
    
    ========== Downloading model/intel/horizontal-text-detection-0001/FP16/horizontal-text-detection-0001.xml
    
    
    ========== Downloading model/intel/horizontal-text-detection-0001/FP16/horizontal-text-detection-0001.bin
    
    
    ################|| Downloading text-recognition-resnet-fc ||################
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/models/__init__.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/models/builder.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/models/model.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/models/weight_init.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/models/registry.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/models/heads/__init__.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/models/heads/builder.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/models/heads/fc_head.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/models/heads/registry.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/models/bodies/__init__.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/models/bodies/builder.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/models/bodies/registry.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/models/bodies/body.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/models/bodies/component.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/models/bodies/sequences/__init__.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/models/bodies/sequences/builder.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/models/bodies/sequences/registry.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/__init__.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/builder.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/decoders/__init__.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/decoders/builder.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/decoders/registry.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/decoders/bricks/__init__.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/decoders/bricks/bricks.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/decoders/bricks/builder.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/decoders/bricks/registry.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/encoders/__init__.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/encoders/builder.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/encoders/backbones/__init__.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/encoders/backbones/builder.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/encoders/backbones/registry.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/encoders/backbones/resnet.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/encoders/enhance_modules/__init__.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/encoders/enhance_modules/builder.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/encoders/enhance_modules/registry.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/models/utils/__init__.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/models/utils/builder.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/models/utils/conv_module.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/models/utils/fc_module.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/models/utils/norm.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/models/utils/registry.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/utils/__init__.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/utils/common.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/utils/registry.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/utils/config.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/configs/resnet_fc.py
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/ckpt/resnet_fc.pth
    
    
    ========== Downloading model/public/text-recognition-resnet-fc/vedastr/addict-2.4.0-py3-none-any.whl
    
    
    ========== Replacing text in model/public/text-recognition-resnet-fc/vedastr/models/heads/__init__.py
    ========== Replacing text in model/public/text-recognition-resnet-fc/vedastr/models/bodies/__init__.py
    ========== Replacing text in model/public/text-recognition-resnet-fc/vedastr/models/bodies/sequences/__init__.py
    ========== Replacing text in model/public/text-recognition-resnet-fc/vedastr/models/bodies/component.py
    ========== Replacing text in model/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/decoders/__init__.py
    ========== Replacing text in model/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/decoders/bricks/__init__.py
    ========== Replacing text in model/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/encoders/backbones/__init__.py
    ========== Replacing text in model/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/encoders/enhance_modules/__init__.py
    ========== Replacing text in model/public/text-recognition-resnet-fc/vedastr/models/utils/__init__.py
    ========== Replacing text in model/public/text-recognition-resnet-fc/vedastr/utils/__init__.py
    ========== Replacing text in model/public/text-recognition-resnet-fc/vedastr/utils/config.py
    ========== Replacing text in model/public/text-recognition-resnet-fc/vedastr/utils/config.py
    ========== Replacing text in model/public/text-recognition-resnet-fc/vedastr/utils/config.py
    ========== Replacing text in model/public/text-recognition-resnet-fc/vedastr/utils/config.py
    ========== Replacing text in model/public/text-recognition-resnet-fc/vedastr/utils/config.py
    ========== Replacing text in model/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/encoders/backbones/resnet.py
    ========== Replacing text in model/public/text-recognition-resnet-fc/vedastr/models/bodies/feature_extractors/encoders/backbones/resnet.py
    ========== Unpacking model/public/text-recognition-resnet-fc/vedastr/addict-2.4.0-py3-none-any.whl
    



Finished downloading horizontal-text-detection-0001,
text-recognition-resnet-fc.


.. code:: ipython3

    ### The text-recognition-resnet-fc model consists of many files. All filenames are printed in
    ### the output of Model Downloader. Uncomment the next two lines to show this output.
    
    # for line in download_result:
    #    print(line)

Convert Models
--------------



The downloaded detection model is an Intel model, which is already in
OpenVINO Intermediate Representation (OpenVINO IR) format. The text
recognition model is a public model which needs to be converted to
OpenVINO IR. Since this model was downloaded from Open Model Zoo, use
Model Converter to convert the model to OpenVINO IR format.

The output of Model Converter will be displayed. When the conversion is
successful, the last lines of output will include
``[ SUCCESS ] Generated IR version 11 model.``

.. code:: ipython3

    convert_command = f"omz_converter --name {recognition_model} --precisions {precision} --download_dir {model_dir} --output_dir {model_dir}"
    display(Markdown(f"Convert command: `{convert_command}`"))
    display(Markdown(f"Converting {recognition_model}..."))
    ! $convert_command



Convert command:
``omz_converter --name text-recognition-resnet-fc --precisions FP16 --download_dir model --output_dir model``



Converting text-recognition-resnet-fc…


.. parsed-literal::

    ========== Converting text-recognition-resnet-fc to ONNX
    Conversion to ONNX command: /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/bin/python -- /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/omz_tools/internal_scripts/pytorch_to_onnx.py --model-path=/opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/omz_tools/models/public/text-recognition-resnet-fc --model-path=model/public/text-recognition-resnet-fc --model-name=get_model --import-module=model '--model-param=file_config=r"model/public/text-recognition-resnet-fc/vedastr/configs/resnet_fc.py"' '--model-param=weights=r"model/public/text-recognition-resnet-fc/vedastr/ckpt/resnet_fc.pth"' --input-shape=1,1,32,100 --input-names=input --output-names=output --output-file=model/public/text-recognition-resnet-fc/resnet_fc.onnx
    
    ONNX check passed successfully.
    
    ========== Converting text-recognition-resnet-fc to IR (FP16)
    Conversion command: /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/bin/python -- /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/bin/mo --framework=onnx --output_dir=model/public/text-recognition-resnet-fc/FP16 --model_name=text-recognition-resnet-fc --input=input '--mean_values=input[127.5]' '--scale_values=input[127.5]' --output=output --input_model=model/public/text-recognition-resnet-fc/resnet_fc.onnx '--layout=input(NCHW)' '--input_shape=[1, 1, 32, 100]' --compress_to_fp16=True
    
    [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
    In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API. 
    Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
    [ INFO ] Generated IR will be compressed to FP16. If you get lower accuracy, please consider disabling compression explicitly by adding argument --compress_to_fp16=False.
    Find more information about compression to FP16 at https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_FP16_Compression.html
    [ SUCCESS ] Generated IR version 11 model.
    [ SUCCESS ] XML file: /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/notebooks/optical-character-recognition/model/public/text-recognition-resnet-fc/FP16/text-recognition-resnet-fc.xml
    [ SUCCESS ] BIN file: /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/notebooks/optical-character-recognition/model/public/text-recognition-resnet-fc/FP16/text-recognition-resnet-fc.bin
    


Select inference device
-----------------------



select device from dropdown list for running inference using OpenVINO

.. code:: ipython3

    device = device_widget()
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



Object Detection
----------------



Load a detection model, load an image, do inference and get the
detection inference result.

Load a Detection Model
~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    detection_model = core.read_model(model=detection_model_path, weights=detection_model_path.with_suffix(".bin"))
    detection_compiled_model = core.compile_model(model=detection_model, device_name=device.value)
    
    detection_input_layer = detection_compiled_model.input(0)

Load an Image
~~~~~~~~~~~~~



.. code:: ipython3

    # The `image_file` variable can point to a URL or a local image.
    image_file = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/intel_rnb.jpg"
    
    image = load_image(image_file)
    
    # N,C,H,W = batch size, number of channels, height, width.
    N, C, H, W = detection_input_layer.shape
    
    # Resize the image to meet network expected input sizes.
    resized_image = cv2.resize(image, (W, H))
    
    # Reshape to the network input shape.
    input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)
    
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB));



.. image:: optical-character-recognition-with-output_files/optical-character-recognition-with-output_16_0.png


Do Inference
~~~~~~~~~~~~



Text boxes are detected in the images and returned as blobs of data in
the shape of ``[100, 5]``. Each description of detection has the
``[x_min, y_min, x_max, y_max, conf]`` format.

.. code:: ipython3

    output_key = detection_compiled_model.output("boxes")
    boxes = detection_compiled_model([input_image])[output_key]
    
    # Remove zero only boxes.
    boxes = boxes[~np.all(boxes == 0, axis=1)]

Get Detection Results
~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    def multiply_by_ratio(ratio_x, ratio_y, box):
        return [max(shape * ratio_y, 10) if idx % 2 else shape * ratio_x for idx, shape in enumerate(box[:-1])]
    
    
    def run_preprocesing_on_crop(crop, net_shape):
        temp_img = cv2.resize(crop, net_shape)
        temp_img = temp_img.reshape((1,) * 2 + temp_img.shape)
        return temp_img
    
    
    def convert_result_to_image(bgr_image, resized_image, boxes, threshold=0.3, conf_labels=True):
        # Define colors for boxes and descriptions.
        colors = {"red": (255, 0, 0), "green": (0, 255, 0), "white": (255, 255, 255)}
    
        # Fetch image shapes to calculate a ratio.
        (real_y, real_x), (resized_y, resized_x) = image.shape[:2], resized_image.shape[:2]
        ratio_x, ratio_y = real_x / resized_x, real_y / resized_y
    
        # Convert the base image from BGR to RGB format.
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    
        # Iterate through non-zero boxes.
        for box, annotation in boxes:
            # Pick a confidence factor from the last place in an array.
            conf = box[-1]
            if conf > threshold:
                # Convert float to int and multiply position of each box by x and y ratio.
                (x_min, y_min, x_max, y_max) = map(int, multiply_by_ratio(ratio_x, ratio_y, box))
    
                # Draw a box based on the position. Parameters in the `rectangle` function are: image, start_point, end_point, color, thickness.
                cv2.rectangle(rgb_image, (x_min, y_min), (x_max, y_max), colors["green"], 3)
    
                # Add a text to an image based on the position and confidence. Parameters in the `putText` function are: image, text, bottomleft_corner_textfield, font, font_scale, color, thickness, line_type
                if conf_labels:
                    # Create a background box based on annotation length.
                    (text_w, text_h), _ = cv2.getTextSize(f"{annotation}", cv2.FONT_HERSHEY_TRIPLEX, 0.8, 1)
                    image_copy = rgb_image.copy()
                    cv2.rectangle(
                        image_copy,
                        (x_min, y_min - text_h - 10),
                        (x_min + text_w, y_min - 10),
                        colors["white"],
                        -1,
                    )
                    # Add weighted image copy with white boxes under a text.
                    cv2.addWeighted(image_copy, 0.4, rgb_image, 0.6, 0, rgb_image)
                    cv2.putText(
                        rgb_image,
                        f"{annotation}",
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        colors["red"],
                        1,
                        cv2.LINE_AA,
                    )
    
        return rgb_image

Text Recognition
----------------



Load the text recognition model and do inference on the detected boxes
from the detection model.

Load Text Recognition Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    recognition_model = core.read_model(model=recognition_model_path, weights=recognition_model_path.with_suffix(".bin"))
    
    recognition_compiled_model = core.compile_model(model=recognition_model, device_name=device.value)
    
    recognition_output_layer = recognition_compiled_model.output(0)
    recognition_input_layer = recognition_compiled_model.input(0)
    
    # Get the height and width of the input layer.
    _, _, H, W = recognition_input_layer.shape

Do Inference
~~~~~~~~~~~~



.. code:: ipython3

    # Calculate scale for image resizing.
    (real_y, real_x), (resized_y, resized_x) = image.shape[:2], resized_image.shape[:2]
    ratio_x, ratio_y = real_x / resized_x, real_y / resized_y
    
    # Convert the image to grayscale for the text recognition model.
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Get a dictionary to encode output, based on the model documentation.
    letters = "~0123456789abcdefghijklmnopqrstuvwxyz"
    
    # Prepare an empty list for annotations.
    annotations = list()
    cropped_images = list()
    # fig, ax = plt.subplots(len(boxes), 1, figsize=(5,15), sharex=True, sharey=True)
    # Get annotations for each crop, based on boxes given by the detection model.
    for i, crop in enumerate(boxes):
        # Get coordinates on corners of a crop.
        (x_min, y_min, x_max, y_max) = map(int, multiply_by_ratio(ratio_x, ratio_y, crop))
        image_crop = run_preprocesing_on_crop(grayscale_image[y_min:y_max, x_min:x_max], (W, H))
    
        # Run inference with the recognition model.
        result = recognition_compiled_model([image_crop])[recognition_output_layer]
    
        # Squeeze the output to remove unnecessary dimension.
        recognition_results_test = np.squeeze(result)
    
        # Read an annotation based on probabilities from the output layer.
        annotation = list()
        for letter in recognition_results_test:
            parsed_letter = letters[letter.argmax()]
    
            # Returning 0 index from `argmax` signalizes an end of a string.
            if parsed_letter == letters[0]:
                break
            annotation.append(parsed_letter)
        annotations.append("".join(annotation))
        cropped_image = Image.fromarray(image[y_min:y_max, x_min:x_max])
        cropped_images.append(cropped_image)
    
    boxes_with_annotations = list(zip(boxes, annotations))

Show Results
------------



Show Detected Text Boxes and OCR Results for the Image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Visualize the result by drawing boxes around recognized text and showing
the OCR result from the text recognition model.

.. code:: ipython3

    plt.figure(figsize=(12, 12))
    plt.imshow(convert_result_to_image(image, resized_image, boxes_with_annotations, conf_labels=True));



.. image:: optical-character-recognition-with-output_files/optical-character-recognition-with-output_26_0.png


Show the OCR Result per Bounding Box
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Depending on the image, the OCR result may not be readable in the image
with boxes, as displayed in the cell above. Use the code below to
display the extracted boxes and the OCR result per box.

.. code:: ipython3

    for cropped_image, annotation in zip(cropped_images, annotations):
        display(cropped_image, Markdown("".join(annotation)))



.. image:: optical-character-recognition-with-output_files/optical-character-recognition-with-output_28_0.png



building



.. image:: optical-character-recognition-with-output_files/optical-character-recognition-with-output_28_2.png



noyce



.. image:: optical-character-recognition-with-output_files/optical-character-recognition-with-output_28_4.png



2200



.. image:: optical-character-recognition-with-output_files/optical-character-recognition-with-output_28_6.png



n



.. image:: optical-character-recognition-with-output_files/optical-character-recognition-with-output_28_8.png



center



.. image:: optical-character-recognition-with-output_files/optical-character-recognition-with-output_28_10.png



robert


Print Annotations in Plain Text Format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Print annotations for detected text based on their position in the input
image, starting from the upper left corner.

.. code:: ipython3

    [annotation for _, annotation in sorted(zip(boxes, annotations), key=lambda x: x[0][0] ** 2 + x[0][1] ** 2)]




.. parsed-literal::

    ['robert', 'n', 'noyce', 'building', '2200', 'center']


