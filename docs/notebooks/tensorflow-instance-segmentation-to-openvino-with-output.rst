Convert a TensorFlow Instance Segmentation Model to OpenVINO™
=============================================================

`TensorFlow <https://www.tensorflow.org/>`__, or TF for short, is an
open-source framework for machine learning.

The `TensorFlow Object Detection
API <https://github.com/tensorflow/models/tree/master/research/object_detection>`__
is an open-source computer vision framework built on top of TensorFlow.
It is used for building object detection and instance segmentation
models that can localize multiple objects in the same image. TensorFlow
Object Detection API supports various architectures and models, which
can be found and downloaded from the `TensorFlow
Hub <https://tfhub.dev/tensorflow/collections/object_detection/1>`__.

This tutorial shows how to convert a TensorFlow `Mask R-CNN with
Inception ResNet
V2 <https://tfhub.dev/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1>`__
instance segmentation model to OpenVINO `Intermediate
Representation <https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets.html>`__
(OpenVINO IR) format, using `Model Conversion
API <https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html>`__.
After creating the OpenVINO IR, load the model in `OpenVINO
Runtime <https://docs.openvino.ai/2024/openvino-workflow/running-inference.html>`__
and do inference with a sample image.


**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Imports <#imports>`__
-  `Settings <#settings>`__
-  `Download Model from TensorFlow
   Hub <#download-model-from-tensorflow-hub>`__
-  `Convert Model to OpenVINO IR <#convert-model-to-openvino-ir>`__
-  `Test Inference on the Converted
   Model <#test-inference-on-the-converted-model>`__
-  `Select inference device <#select-inference-device>`__

   -  `Load the Model <#load-the-model>`__
   -  `Get Model Information <#get-model-information>`__
   -  `Get an Image for Test
      Inference <#get-an-image-for-test-inference>`__
   -  `Perform Inference <#perform-inference>`__
   -  `Inference Result
      Visualization <#inference-result-visualization>`__

-  `Next Steps <#next-steps>`__

   -  `Async inference pipeline <#async-inference-pipeline>`__
   -  `Integration preprocessing to
      model <#integration-preprocessing-to-model>`__

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

Prerequisites
-------------



Install required packages:

.. code:: ipython3

    import platform
    
    %pip install -q "openvino>=2023.1.0" "numpy>=1.21.0" "opencv-python" "tqdm"
    
    if platform.system() != "Windows":
        %pip install -q "matplotlib>=3.4"
    else:
        %pip install -q "matplotlib>=3.4,<3.7"
    
    %pip install -q "tensorflow-macos>=2.5; sys_platform == 'darwin' and platform_machine == 'arm64' and python_version > '3.8'" # macOS M1 and M2
    %pip install -q "tensorflow>=2.5; sys_platform == 'darwin' and platform_machine != 'arm64' and python_version > '3.8'" # macOS x86
    %pip install -q "tensorflow>=2.5; sys_platform != 'darwin' and python_version > '3.8'"


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.


The notebook uses utility functions. The cell below will download the
``notebook_utils`` Python module from GitHub.

.. code:: ipython3

    # Fetch the notebook utils script from the openvino_notebooks repo
    import requests
    
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    
    open("notebook_utils.py", "w").write(r.text)




.. parsed-literal::

    24692



Imports
-------



.. code:: ipython3

    # Standard python modules
    from pathlib import Path
    
    # External modules and dependencies
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Notebook utils module
    from notebook_utils import download_file, device_widget
    
    # OpenVINO modules
    import openvino as ov

Settings
--------



Define model related variables and create corresponding directories:

.. code:: ipython3

    # Create directories for models files
    model_dir = Path("is-model")
    model_dir.mkdir(exist_ok=True)
    
    # Create directory for TensorFlow model
    tf_model_dir = model_dir / "tf"
    tf_model_dir.mkdir(exist_ok=True)
    
    # Create directory for OpenVINO IR model
    ir_model_dir = model_dir / "ir"
    ir_model_dir.mkdir(exist_ok=True)
    
    model_name = "mask_rcnn_inception_resnet_v2_1024x1024"
    
    openvino_ir_path = ir_model_dir / f"{model_name}.xml"
    
    tf_model_url = (
        "https://www.kaggle.com/models/tensorflow/mask-rcnn-inception-resnet-v2/frameworks/tensorFlow2/variations/1024x1024/versions/1?tf-hub-format=compressed"
    )
    
    tf_model_archive_filename = f"{model_name}.tar.gz"

Download Model from TensorFlow Hub
----------------------------------



Download archive with TensorFlow Instance Segmentation model
(`mask_rcnn_inception_resnet_v2_1024x1024 <https://tfhub.dev/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1>`__)
from TensorFlow Hub:

.. code:: ipython3

    download_file(url=tf_model_url, filename=tf_model_archive_filename, directory=tf_model_dir);



.. parsed-literal::

    is-model/tf/mask_rcnn_inception_resnet_v2_1024x1024.tar.gz:   0%|          | 0.00/232M [00:00<?, ?B/s]


Extract TensorFlow Instance Segmentation model from the downloaded
archive:

.. code:: ipython3

    import tarfile
    
    with tarfile.open(tf_model_dir / tf_model_archive_filename) as file:
        file.extractall(path=tf_model_dir)

Convert Model to OpenVINO IR
----------------------------



OpenVINO Model Optimizer Python API can be used to convert the
TensorFlow model to OpenVINO IR.

``mo.convert_model`` function accept path to TensorFlow model and
returns OpenVINO Model class instance which represents this model. Also
we need to provide model input shape (``input_shape``) that is described
at `model overview page on TensorFlow
Hub <https://tfhub.dev/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1>`__.
Optionally, we can apply compression to FP16 model weights using
``compress_to_fp16=True`` option and integrate preprocessing using this
approach.

The converted model is ready to load on a device using ``compile_model``
or saved on disk using the ``serialize`` function to reduce loading time
when the model is run in the future.

.. code:: ipython3

    ov_model = ov.convert_model(tf_model_dir)
    
    # Save converted OpenVINO IR model to the corresponding directory
    ov.save_model(ov_model, openvino_ir_path)

Test Inference on the Converted Model
-------------------------------------



Select inference device
-----------------------



select device from dropdown list for running inference using OpenVINO

.. code:: ipython3

    core = ov.Core()
    device = device_widget()
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



Load the Model
~~~~~~~~~~~~~~



.. code:: ipython3

    openvino_ir_model = core.read_model(openvino_ir_path)
    compiled_model = core.compile_model(model=openvino_ir_model, device_name=device.value)

Get Model Information
~~~~~~~~~~~~~~~~~~~~~



Mask R-CNN with Inception ResNet V2 instance segmentation model has one
input - a three-channel image of variable size. The input tensor shape
is ``[1, height, width, 3]`` with values in ``[0, 255]``.

Model output dictionary contains a lot of tensors, we will use only 5 of
them: - ``num_detections``: A ``tf.int`` tensor with only one value, the
number of detections ``[N]``. - ``detection_boxes``: A ``tf.float32``
tensor of shape ``[N, 4]`` containing bounding box coordinates in the
following order: ``[ymin, xmin, ymax, xmax]``. - ``detection_classes``:
A ``tf.int`` tensor of shape ``[N]`` containing detection class index
from the label file. - ``detection_scores``: A ``tf.float32`` tensor of
shape ``[N]`` containing detection scores. - ``detection_masks``: A
``[batch, max_detections, mask_height, mask_width]`` tensor. Note that a
pixel-wise sigmoid score converter is applied to the detection masks.

For more information about model inputs, outputs and their formats, see
the `model overview page on TensorFlow
Hub <https://tfhub.dev/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1>`__.

It is important to mention, that values of ``detection_boxes``,
``detection_classes``, ``detection_scores``, ``detection_masks``
correspond to each other and are ordered by the highest detection score:
the first detection mask corresponds to the first detection class and to
the first (and highest) detection score.

.. code:: ipython3

    model_inputs = compiled_model.inputs
    model_outputs = compiled_model.outputs
    
    print("Model inputs count:", len(model_inputs))
    print("Model inputs:")
    for _input in model_inputs:
        print("  ", _input)
    
    print("Model outputs count:", len(model_outputs))
    print("Model outputs:")
    for output in model_outputs:
        print("  ", output)


.. parsed-literal::

    Model inputs count: 1
    Model inputs:
       <ConstOutput: names[input_tensor] shape[1,?,?,3] type: u8>
    Model outputs count: 23
    Model outputs:
       <ConstOutput: names[] shape[49152,4] type: f32>
       <ConstOutput: names[box_classifier_features] shape[300,9,9,1536] type: f32>
       <ConstOutput: names[] shape[4] type: f32>
       <ConstOutput: names[mask_predictions] shape[100,90,33,33] type: f32>
       <ConstOutput: names[num_detections] shape[1] type: f32>
       <ConstOutput: names[num_proposals] shape[1] type: f32>
       <ConstOutput: names[proposal_boxes] shape[1,?,..8] type: f32>
       <ConstOutput: names[proposal_boxes_normalized, final_anchors] shape[1,?,..8] type: f32>
       <ConstOutput: names[raw_detection_boxes] shape[1,300,4] type: f32>
       <ConstOutput: names[raw_detection_scores] shape[1,300,91] type: f32>
       <ConstOutput: names[refined_box_encodings] shape[300,90,4] type: f32>
       <ConstOutput: names[rpn_box_encodings] shape[1,49152,4] type: f32>
       <ConstOutput: names[class_predictions_with_background] shape[300,91] type: f32>
       <ConstOutput: names[rpn_box_predictor_features] shape[1,64,64,512] type: f32>
       <ConstOutput: names[rpn_features_to_crop] shape[1,64,64,1088] type: f32>
       <ConstOutput: names[rpn_objectness_predictions_with_background] shape[1,49152,2] type: f32>
       <ConstOutput: names[detection_anchor_indices] shape[1,?] type: f32>
       <ConstOutput: names[detection_boxes] shape[1,?,..8] type: f32>
       <ConstOutput: names[detection_classes] shape[1,?] type: f32>
       <ConstOutput: names[detection_masks] shape[1,100,33,33] type: f32>
       <ConstOutput: names[detection_multiclass_scores] shape[1,?,..182] type: f32>
       <ConstOutput: names[detection_scores] shape[1,?] type: f32>
       <ConstOutput: names[proposal_boxes_normalized, final_anchors] shape[1,?,..8] type: f32>


Get an Image for Test Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Load and save an image:

.. code:: ipython3

    image_path = Path("./data/coco_bike.jpg")
    
    download_file(
        url="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco_bike.jpg",
        filename=image_path.name,
        directory=image_path.parent,
    );



.. parsed-literal::

    data/coco_bike.jpg:   0%|          | 0.00/182k [00:00<?, ?B/s]


Read the image, resize and convert it to the input shape of the network:

.. code:: ipython3

    # Read the image
    image = cv2.imread(filename=str(image_path))
    
    # The network expects images in RGB format
    image = cv2.cvtColor(image, code=cv2.COLOR_BGR2RGB)
    
    # Resize the image to the network input shape
    resized_image = cv2.resize(src=image, dsize=(255, 255))
    
    # Add batch dimension to image
    network_input_image = np.expand_dims(resized_image, 0)
    
    # Show the image
    plt.imshow(image)




.. parsed-literal::

    <matplotlib.image.AxesImage at 0x7f1444037a60>




.. image:: tensorflow-instance-segmentation-to-openvino-with-output_files/tensorflow-instance-segmentation-to-openvino-with-output_25_1.png


Perform Inference
~~~~~~~~~~~~~~~~~



.. code:: ipython3

    inference_result = compiled_model(network_input_image)

After model inference on the test image, instance segmentation data can
be extracted from the result. For further model result visualization
``detection_boxes``, ``detection_masks``, ``detection_classes`` and
``detection_scores`` outputs will be used.

.. code:: ipython3

    detection_boxes = compiled_model.output("detection_boxes")
    image_detection_boxes = inference_result[detection_boxes]
    print("image_detection_boxes:", image_detection_boxes.shape)
    
    detection_masks = compiled_model.output("detection_masks")
    image_detection_masks = inference_result[detection_masks]
    print("image_detection_masks:", image_detection_masks.shape)
    
    detection_classes = compiled_model.output("detection_classes")
    image_detection_classes = inference_result[detection_classes]
    print("image_detection_classes:", image_detection_classes.shape)
    
    detection_scores = compiled_model.output("detection_scores")
    image_detection_scores = inference_result[detection_scores]
    print("image_detection_scores:", image_detection_scores.shape)
    
    num_detections = compiled_model.output("num_detections")
    image_num_detections = inference_result[num_detections]
    print("image_detections_num:", image_num_detections)
    
    # Alternatively, inference result data can be extracted by model output name with `.get()` method
    assert (inference_result[detection_boxes] == inference_result.get("detection_boxes")).all(), "extracted inference result data should be equal"


.. parsed-literal::

    image_detection_boxes: (1, 100, 4)
    image_detection_masks: (1, 100, 33, 33)
    image_detection_classes: (1, 100)
    image_detection_scores: (1, 100)
    image_detections_num: [100.]


Inference Result Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Define utility functions to visualize the inference results

.. code:: ipython3

    from typing import Optional
    
    
    def add_detection_box(box: np.ndarray, image: np.ndarray, mask: np.ndarray, label: Optional[str] = None) -> np.ndarray:
        """
        Helper function for adding single bounding box to the image
    
        Parameters
        ----------
        box : np.ndarray
            Bounding box coordinates in format [ymin, xmin, ymax, xmax]
        image : np.ndarray
            The image to which detection box is added
        mask: np.ndarray
            Segmentation mask in format (H, W)
        label : str, optional
            Detection box label string, if not provided will not be added to result image (default is None)
    
        Returns
        -------
        np.ndarray
            NumPy array including image, detection box, and segmentation mask
    
        """
        ymin, xmin, ymax, xmax = box
        point1, point2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))
        box_color = [np.random.randint(0, 255) for _ in range(3)]
        line_thickness = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
    
        result = cv2.rectangle(
            img=image,
            pt1=point1,
            pt2=point2,
            color=box_color,
            thickness=line_thickness,
            lineType=cv2.LINE_AA,
        )
    
        if label:
            font_thickness = max(line_thickness - 1, 1)
            font_face = 0
            font_scale = line_thickness / 3
            font_color = (255, 255, 255)
            text_size = cv2.getTextSize(
                text=label,
                fontFace=font_face,
                fontScale=font_scale,
                thickness=font_thickness,
            )[0]
            # Calculate rectangle coordinates
            rectangle_point1 = point1
            rectangle_point2 = (point1[0] + text_size[0], point1[1] - text_size[1] - 3)
            # Add filled rectangle
            result = cv2.rectangle(
                img=result,
                pt1=rectangle_point1,
                pt2=rectangle_point2,
                color=box_color,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
            # Calculate text position
            text_position = point1[0], point1[1] - 3
            # Add text with label to filled rectangle
            result = cv2.putText(
                img=result,
                text=label,
                org=text_position,
                fontFace=font_face,
                fontScale=font_scale,
                color=font_color,
                thickness=font_thickness,
                lineType=cv2.LINE_AA,
            )
        mask_img = mask[:, :, np.newaxis] * box_color
        result = cv2.addWeighted(result, 1, mask_img.astype(np.uint8), 0.6, 0)
        return result

.. code:: ipython3

    def get_mask_frame(box, frame, mask):
        """
        Transform a binary mask to fit within a specified bounding box in a frame using perspective transformation.
    
        Args:
            box (tuple): A bounding box represented as a tuple (y_min, x_min, y_max, x_max).
            frame (numpy.ndarray): The larger frame or image where the mask will be placed.
            mask (numpy.ndarray): A binary mask image to be transformed.
    
        Returns:
            numpy.ndarray: A transformed mask image that fits within the specified bounding box in the frame.
        """
        x_min = frame.shape[1] * box[1]
        y_min = frame.shape[0] * box[0]
        x_max = frame.shape[1] * box[3]
        y_max = frame.shape[0] * box[2]
        rect_src = np.array(
            [
                [0, 0],
                [mask.shape[1], 0],
                [mask.shape[1], mask.shape[0]],
                [0, mask.shape[0]],
            ],
            dtype=np.float32,
        )
        rect_dst = np.array(
            [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(rect_src[:, :], rect_dst[:, :])
        mask_frame = cv2.warpPerspective(mask, M, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_CUBIC)
        return mask_frame

.. code:: ipython3

    from typing import Dict
    
    from openvino.runtime.utils.data_helpers import OVDict
    
    
    def visualize_inference_result(
        inference_result: OVDict,
        image: np.ndarray,
        labels_map: Dict,
        detections_limit: Optional[int] = None,
    ):
        """
        Helper function for visualizing inference result on the image
    
        Parameters
        ----------
        inference_result : OVDict
            Result of the compiled model inference on the test image
        image : np.ndarray
            Original image to use for visualization
        labels_map : Dict
            Dictionary with mappings of detection classes numbers and its names
        detections_limit : int, optional
            Number of detections to show on the image, if not provided all detections will be shown (default is None)
        """
        detection_boxes = inference_result.get("detection_boxes")
        detection_classes = inference_result.get("detection_classes")
        detection_scores = inference_result.get("detection_scores")
        num_detections = inference_result.get("num_detections")
        detection_masks = inference_result.get("detection_masks")
    
        detections_limit = int(min(detections_limit, num_detections[0]) if detections_limit is not None else num_detections[0])
    
        # Normalize detection boxes coordinates to original image size
        original_image_height, original_image_width, _ = image.shape
        normalized_detection_boxes = detection_boxes[0, :detections_limit] * [
            original_image_height,
            original_image_width,
            original_image_height,
            original_image_width,
        ]
        result = np.copy(image)
        for i in range(detections_limit):
            detected_class_name = labels_map[int(detection_classes[0, i])]
            score = detection_scores[0, i]
            mask = detection_masks[0, i]
            mask_reframed = get_mask_frame(detection_boxes[0, i], image, mask)
            mask_reframed = (mask_reframed > 0.5).astype(np.uint8)
            label = f"{detected_class_name} {score:.2f}"
            result = add_detection_box(
                box=normalized_detection_boxes[i],
                image=result,
                mask=mask_reframed,
                label=label,
            )
    
        plt.imshow(result)

TensorFlow Instance Segmentation model
(`mask_rcnn_inception_resnet_v2_1024x1024 <https://tfhub.dev/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1?tf-hub-format=compressed>`__)
used in this notebook was trained on `COCO
2017 <https://cocodataset.org/>`__ dataset with 91 classes. For better
visualization experience we can use COCO dataset labels with human
readable class names instead of class numbers or indexes.

We can download COCO dataset classes labels from `Open Model
Zoo <https://github.com/openvinotoolkit/open_model_zoo/>`__:

.. code:: ipython3

    coco_labels_file_path = Path("./data/coco_91cl.txt")
    
    download_file(
        url="https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/master/data/dataset_classes/coco_91cl.txt",
        filename=coco_labels_file_path.name,
        directory=coco_labels_file_path.parent,
    );



.. parsed-literal::

    data/coco_91cl.txt:   0%|          | 0.00/421 [00:00<?, ?B/s]


Then we need to create dictionary ``coco_labels_map`` with mappings
between detection classes numbers and its names from the downloaded
file:

.. code:: ipython3

    with open(coco_labels_file_path, "r") as file:
        coco_labels = file.read().strip().split("\n")
        coco_labels_map = dict(enumerate(coco_labels, 1))
    
    print(coco_labels_map)


.. parsed-literal::

    {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplan', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 12: 'street sign', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 26: 'hat', 27: 'backpack', 28: 'umbrella', 29: 'shoe', 30: 'eye glasses', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 45: 'plate', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 66: 'mirror', 67: 'dining table', 68: 'window', 69: 'desk', 70: 'toilet', 71: 'door', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 83: 'blender', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush', 91: 'hair brush'}


Finally, we are ready to visualize model inference results on the
original test image:

.. code:: ipython3

    visualize_inference_result(
        inference_result=inference_result,
        image=image,
        labels_map=coco_labels_map,
        detections_limit=5,
    )



.. image:: tensorflow-instance-segmentation-to-openvino-with-output_files/tensorflow-instance-segmentation-to-openvino-with-output_39_0.png


Next Steps
----------



This section contains suggestions on how to additionally improve the
performance of your application using OpenVINO.

Async inference pipeline
~~~~~~~~~~~~~~~~~~~~~~~~

The key advantage of the Async
API is that when a device is busy with inference, the application can
perform other tasks in parallel (for example, populating inputs or
scheduling other requests) rather than wait for the current inference to
complete first. To understand how to perform async inference using
openvino, refer to the `Async API
tutorial <async-api-with-output.html>`__.

Integration preprocessing to model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Preprocessing API enables making preprocessing a part of the model
reducing application code and dependency on additional image processing
libraries. The main advantage of Preprocessing API is that preprocessing
steps will be integrated into the execution graph and will be performed
on a selected device (CPU/GPU etc.) rather than always being executed on
CPU as part of an application. This will improve selected device
utilization.

For more information, refer to the `Optimize Preprocessing
tutorial <optimize-preprocessing-with-output.html>`__ and
to the overview of `Preprocessing
API <https://docs.openvino.ai/2024/openvino-workflow/running-inference/optimize-inference/optimize-preprocessing/preprocessing-api-details.html>`__.
