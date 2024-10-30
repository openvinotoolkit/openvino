Convert a TensorFlow Object Detection Model to OpenVINO™
========================================================

`TensorFlow <https://www.tensorflow.org/>`__, or TF for short, is an
open-source framework for machine learning.

The `TensorFlow Object Detection
API <https://github.com/tensorflow/models/tree/master/research/object_detection>`__
is an open-source computer vision framework built on top of TensorFlow.
It is used for building object detection and image segmentation models
that can localize multiple objects in the same image. TensorFlow Object
Detection API supports various architectures and models, which can be
found and downloaded from the `TensorFlow
Hub <https://tfhub.dev/tensorflow/collections/object_detection/1>`__.

This tutorial shows how to convert a TensorFlow `Faster R-CNN with
Resnet-50
V1 <https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1>`__
object detection model to OpenVINO `Intermediate
Representation <https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets.html>`__
(OpenVINO IR) format, using Model Converter. After creating the OpenVINO
IR, load the model in `OpenVINO
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
    
    # OpenVINO import
    import openvino as ov
    
    # Notebook utils module
    from notebook_utils import download_file, device_widget

Settings
--------



Define model related variables and create corresponding directories:

.. code:: ipython3

    # Create directories for models files
    model_dir = Path("od-model")
    model_dir.mkdir(exist_ok=True)
    
    # Create directory for TensorFlow model
    tf_model_dir = model_dir / "tf"
    tf_model_dir.mkdir(exist_ok=True)
    
    # Create directory for OpenVINO IR model
    ir_model_dir = model_dir / "ir"
    ir_model_dir.mkdir(exist_ok=True)
    
    model_name = "faster_rcnn_resnet50_v1_640x640"
    
    openvino_ir_path = ir_model_dir / f"{model_name}.xml"
    
    tf_model_url = "https://www.kaggle.com/models/tensorflow/faster-rcnn-resnet-v1/frameworks/tensorFlow2/variations/faster-rcnn-resnet50-v1-640x640/versions/1?tf-hub-format=compressed"
    
    tf_model_archive_filename = f"{model_name}.tar.gz"

Download Model from TensorFlow Hub
----------------------------------



Download archive with TensorFlow Object Detection model
(`faster_rcnn_resnet50_v1_640x640 <https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1>`__)
from TensorFlow Hub:

.. code:: ipython3

    download_file(url=tf_model_url, filename=tf_model_archive_filename, directory=tf_model_dir)



.. parsed-literal::

    od-model/tf/faster_rcnn_resnet50_v1_640x640.tar.gz:   0%|          | 0.00/101M [00:00<?, ?B/s]




.. parsed-literal::

    PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/notebooks/tensorflow-object-detection-to-openvino/od-model/tf/faster_rcnn_resnet50_v1_640x640.tar.gz')



Extract TensorFlow Object Detection model from the downloaded archive:

.. code:: ipython3

    import tarfile
    
    with tarfile.open(tf_model_dir / tf_model_archive_filename) as file:
        file.extractall(path=tf_model_dir)

Convert Model to OpenVINO IR
----------------------------



OpenVINO Model Conversion API can be used to convert the TensorFlow
model to OpenVINO IR.

``ov.convert_model`` function accept path to TensorFlow model and
returns OpenVINO Model class instance which represents this model. Also
we need to provide model input shape (``input_shape``) that is described
at `model overview page on TensorFlow
Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1>`__.

The converted model is ready to load on a device using ``compile_model``
or saved on disk using the ``save_model`` function to reduce loading
time when the model is run in the future.

See the `Model Preparation
Guide <https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html>`__
for more information about model conversion and TensorFlow `models
support <https://docs.openvino.ai/2024/openvino-workflow/model-preparation/convert-model-tensorflow.html>`__.

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

    core = ov.Core()
    openvino_ir_model = core.read_model(openvino_ir_path)
    compiled_model = core.compile_model(model=openvino_ir_model, device_name=device.value)

Get Model Information
~~~~~~~~~~~~~~~~~~~~~



Faster R-CNN with Resnet-50 V1 object detection model has one input - a
three-channel image of variable size. The input tensor shape is
``[1, height, width, 3]`` with values in ``[0, 255]``.

Model output dictionary contains several tensors:

-  ``num_detections`` - the number of detections in ``[N]`` format.
-  ``detection_boxes`` - bounding box coordinates for all ``N``
   detections in ``[ymin, xmin, ymax, xmax]`` format.
-  ``detection_classes`` - ``N`` detection class indexes size from the
   label file.
-  ``detection_scores`` - ``N`` detection scores (confidence) for each
   detected class.
-  ``raw_detection_boxes`` - decoded detection boxes without Non-Max
   suppression.
-  ``raw_detection_scores`` - class score logits for raw detection
   boxes.
-  ``detection_anchor_indices`` - the anchor indices of the detections
   after NMS.
-  ``detection_multiclass_scores`` - class score distribution (including
   background) for detection boxes in the image including background
   class.

In this tutorial we will mostly use ``detection_boxes``,
``detection_classes``, ``detection_scores`` tensors. It is important to
mention, that values of these tensors correspond to each other and are
ordered by the highest detection score: the first detection box
corresponds to the first detection class and to the first (and highest)
detection score.

See the `model overview page on TensorFlow
Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1>`__
for more information about model inputs, outputs and their formats.

.. code:: ipython3

    model_inputs = compiled_model.inputs
    model_input = compiled_model.input(0)
    model_outputs = compiled_model.outputs
    
    print("Model inputs count:", len(model_inputs))
    print("Model input:", model_input)
    
    print("Model outputs count:", len(model_outputs))
    print("Model outputs:")
    for output in model_outputs:
        print("  ", output)


.. parsed-literal::

    Model inputs count: 1
    Model input: <ConstOutput: names[input_tensor] shape[1,?,?,3] type: u8>
    Model outputs count: 8
    Model outputs:
       <ConstOutput: names[detection_anchor_indices] shape[1,?] type: f32>
       <ConstOutput: names[detection_boxes] shape[1,?,..8] type: f32>
       <ConstOutput: names[detection_classes] shape[1,?] type: f32>
       <ConstOutput: names[detection_multiclass_scores] shape[1,?,..182] type: f32>
       <ConstOutput: names[detection_scores] shape[1,?] type: f32>
       <ConstOutput: names[num_detections] shape[1] type: f32>
       <ConstOutput: names[raw_detection_boxes] shape[1,300,4] type: f32>
       <ConstOutput: names[raw_detection_scores] shape[1,300,91] type: f32>


Get an Image for Test Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Load and save an image:

.. code:: ipython3

    image_path = Path("./data/coco_bike.jpg")
    
    download_file(
        url="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco_bike.jpg",
        filename=image_path.name,
        directory=image_path.parent,
    )


.. parsed-literal::

    'data/coco_bike.jpg' already exists.




.. parsed-literal::

    PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/notebooks/tensorflow-object-detection-to-openvino/data/coco_bike.jpg')



Read the image, resize and convert it to the input shape of the network:

.. code:: ipython3

    # Read the image
    image = cv2.imread(filename=str(image_path))
    
    # The network expects images in RGB format
    image = cv2.cvtColor(image, code=cv2.COLOR_BGR2RGB)
    
    # Resize the image to the network input shape
    resized_image = cv2.resize(src=image, dsize=(255, 255))
    
    # Transpose the image to the network input shape
    network_input_image = np.expand_dims(resized_image, 0)
    
    # Show the image
    plt.imshow(image)




.. parsed-literal::

    <matplotlib.image.AxesImage at 0x7f101c03e220>




.. image:: tensorflow-object-detection-to-openvino-with-output_files/tensorflow-object-detection-to-openvino-with-output_25_1.png


Perform Inference
~~~~~~~~~~~~~~~~~



.. code:: ipython3

    inference_result = compiled_model(network_input_image)

After model inference on the test image, object detection data can be
extracted from the result. For further model result visualization
``detection_boxes``, ``detection_classes`` and ``detection_scores``
outputs will be used.

.. code:: ipython3

    (
        _,
        detection_boxes,
        detection_classes,
        _,
        detection_scores,
        num_detections,
        _,
        _,
    ) = model_outputs
    
    image_detection_boxes = inference_result[detection_boxes]
    print("image_detection_boxes:", image_detection_boxes)
    
    image_detection_classes = inference_result[detection_classes]
    print("image_detection_classes:", image_detection_classes)
    
    image_detection_scores = inference_result[detection_scores]
    print("image_detection_scores:", image_detection_scores)
    
    image_num_detections = inference_result[num_detections]
    print("image_detections_num:", image_num_detections)
    
    # Alternatively, inference result data can be extracted by model output name with `.get()` method
    assert (inference_result[detection_boxes] == inference_result.get("detection_boxes")).all(), "extracted inference result data should be equal"


.. parsed-literal::

    image_detection_boxes: [[[0.16447833 0.5460326  0.89537144 0.8550827 ]
      [0.6717681  0.01238852 0.9843284  0.53113335]
      [0.49202633 0.01172762 0.98052186 0.8866133 ]
      ...
      [0.46021447 0.5924625  0.48734403 0.6187243 ]
      [0.4360505  0.5933398  0.4692526  0.6341007 ]
      [0.68998176 0.4135669  0.9760198  0.8143897 ]]]
    image_detection_classes: [[18.  2.  2.  3.  2.  8.  2.  2.  3.  2.  4.  4.  2.  4. 16.  1.  1.  2.
      27.  8. 62.  2.  2.  4.  4.  2. 18. 41.  4.  4.  2. 18.  2.  2.  4.  2.
      27.  2. 27.  2.  1.  2. 16.  1. 16.  2.  2.  2.  2. 16.  2.  2.  4.  2.
       1. 33.  4. 15.  3.  2.  2.  1.  2.  1.  4.  2. 11.  3.  4. 35.  4.  1.
      40.  2. 62.  2.  4.  4. 36.  1. 36. 36. 77. 31.  2.  1. 51.  1. 34.  3.
      90.  3.  2.  2.  1.  2.  2.  1.  1.  1.  2. 18.  4.  3.  2.  2. 31.  1.
       2.  1.  2. 41. 33. 41. 31.  3.  3.  1. 36. 15. 27.  4. 27.  2.  4. 15.
       3. 37.  1. 27.  4. 35. 36. 88.  4.  2.  3. 15.  2.  4.  2.  1.  3. 27.
       4.  3.  4. 16. 23. 44.  1.  1.  4.  1.  4.  3. 15.  4. 62. 36. 77.  3.
      28.  1. 27. 35.  2. 36. 28. 27. 75.  8.  3. 36.  4. 44.  2.  4. 35.  1.
       3.  1.  1. 35. 87.  1.  1.  1. 15.  1. 84.  1.  3.  1.  1. 35.  1.  2.
       1.  1. 15. 62.  1. 15. 44.  1. 41.  1. 62.  4. 35.  4. 43.  3. 16. 15.
       2.  4. 34. 14.  3. 62. 33. 41.  4.  2. 35. 18.  3. 15.  1. 27.  4. 21.
      19. 87.  1.  1. 27.  1.  3.  2.  3. 15. 38.  1. 27.  1. 15. 84.  4.  4.
       3. 38.  1. 15. 20.  3. 62. 41. 20. 58.  2. 88.  4. 62.  1. 15. 14. 31.
      19.  4. 31.  1.  2.  8. 18. 15.  4.  2.  2.  2. 31. 84. 15.  3. 18.  2.
      27. 28. 15. 31. 28.  1.  1.  8. 20.  3.  1. 41.]]
    image_detection_scores: [[0.98100936 0.94071937 0.932054   0.87772274 0.84029174 0.5898775
      0.5533583  0.5398071  0.49383202 0.47797197 0.46248457 0.4405343
      0.40156218 0.34709066 0.3174982  0.27442312 0.24709812 0.23665425
      0.23217288 0.22382483 0.21970391 0.2021361  0.19405638 0.14689012
      0.14507614 0.14343795 0.12780006 0.12564348 0.11809891 0.10874528
      0.10462027 0.09282681 0.09071824 0.08906853 0.08674242 0.0808276
      0.08010086 0.079368   0.06617683 0.0628278  0.06066268 0.0602232
      0.0580567  0.053602   0.05180356 0.04988255 0.048532   0.04689693
      0.04476341 0.04134317 0.0408088  0.03969054 0.03504278 0.03275277
      0.03109965 0.02965053 0.02862901 0.02858275 0.0257968  0.02342912
      0.02333545 0.02142582 0.02137399 0.02088613 0.02024864 0.01939381
      0.0193674  0.01934038 0.01863845 0.0184786  0.01844665 0.0183451
      0.01803045 0.01781685 0.01730029 0.01667061 0.01585764 0.01565674
      0.01565629 0.01524817 0.01516375 0.01505281 0.01435965 0.01434395
      0.01415888 0.01369895 0.01359102 0.0129866  0.01253129 0.0120007
      0.01156755 0.01149271 0.01135033 0.01133145 0.01113621 0.01108707
      0.01100362 0.01090855 0.01044954 0.01028427 0.01001238 0.00976972
      0.00976233 0.00964447 0.00960519 0.00954092 0.0094881  0.00940329
      0.00935068 0.00933121 0.00906878 0.00887597 0.0088425  0.00881775
      0.00860451 0.00854638 0.0084926  0.00848049 0.00845459 0.00824691
      0.00814731 0.00789408 0.00785361 0.00773962 0.00770773 0.00766053
      0.00765653 0.00765338 0.00744546 0.00704072 0.00697901 0.00689811
      0.00689055 0.00659724 0.00649199 0.0063755  0.00635564 0.00623979
      0.00622121 0.00599785 0.0058857  0.00585696 0.00579975 0.0057361
      0.00572549 0.0056205  0.00558006 0.00556709 0.00549531 0.00547659
      0.00547634 0.00546918 0.00541863 0.00540305 0.00535539 0.00534113
      0.00524252 0.00522422 0.00505857 0.0050541  0.00490434 0.00482884
      0.00479049 0.00470287 0.00461144 0.0046054  0.00460464 0.00457361
      0.00455593 0.00455155 0.00454144 0.0044696  0.00437295 0.00425156
      0.00421544 0.00415256 0.0041001  0.00407984 0.0040696  0.00404598
      0.00403254 0.00399533 0.00396139 0.00393393 0.00391581 0.00389289
      0.00383419 0.00383254 0.00381891 0.00376752 0.0037526  0.00373114
      0.0037009  0.00367086 0.0036602  0.00359289 0.00351931 0.00350436
      0.00348357 0.00345003 0.00343477 0.00343364 0.00336449 0.00332134
      0.00331493 0.00329596 0.0032774  0.00312507 0.00311955 0.00307898
      0.00307835 0.00307419 0.00306389 0.0030464  0.00302192 0.003013
      0.00299757 0.00297221 0.00292418 0.00289839 0.00289729 0.00289356
      0.00287951 0.00281861 0.00280929 0.00275672 0.0027263  0.00269611
      0.00267223 0.00263109 0.00260242 0.00256464 0.0025561  0.00251843
      0.00250994 0.00250275 0.00248212 0.002474   0.0024659  0.00242074
      0.00239178 0.00237558 0.0023748  0.00235467 0.00234726 0.00234068
      0.00232315 0.00232086 0.00231538 0.00230753 0.00229496 0.00229319
      0.00226935 0.00223911 0.00221997 0.00220866 0.00219945 0.00219268
      0.00218071 0.00216285 0.00215859 0.00215483 0.0021313  0.00211466
      0.00210661 0.00204844 0.00204042 0.00204004 0.00202383 0.00202068
      0.00199253 0.00198849 0.00198765 0.00198162 0.00197627 0.00195188
      0.00193299 0.00191865 0.00190285 0.00188111 0.00185229 0.00182701
      0.00178874 0.00177356 0.00176628 0.00176079 0.0017537  0.00174401
      0.00171574 0.00169506 0.00168347 0.00168053 0.00167159 0.00167045
      0.00163559 0.00163302 0.00163038 0.00162886 0.00162866 0.00162236]]
    image_detections_num: [300.]


Inference Result Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Define utility functions to visualize the inference results

.. code:: ipython3

    from typing import Optional
    
    
    def add_detection_box(box: np.ndarray, image: np.ndarray, label: Optional[str] = None) -> np.ndarray:
        """
        Helper function for adding single bounding box to the image
    
        Parameters
        ----------
        box : np.ndarray
            Bounding box coordinates in format [ymin, xmin, ymax, xmax]
        image : np.ndarray
            The image to which detection box is added
        label : str, optional
            Detection box label string, if not provided will not be added to result image (default is None)
    
        Returns
        -------
        np.ndarray
            NumPy array including both image and detection box
    
        """
        ymin, xmin, ymax, xmax = box
        point1, point2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))
        box_color = [np.random.randint(0, 255) for _ in range(3)]
        line_thickness = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
    
        cv2.rectangle(
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
            cv2.rectangle(
                img=image,
                pt1=rectangle_point1,
                pt2=rectangle_point2,
                color=box_color,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
            # Calculate text position
            text_position = point1[0], point1[1] - 3
            # Add text with label to filled rectangle
            cv2.putText(
                img=image,
                text=label,
                org=text_position,
                fontFace=font_face,
                fontScale=font_scale,
                color=font_color,
                thickness=font_thickness,
                lineType=cv2.LINE_AA,
            )
        return image

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
        detection_boxes: np.ndarray = inference_result.get("detection_boxes")
        detection_classes: np.ndarray = inference_result.get("detection_classes")
        detection_scores: np.ndarray = inference_result.get("detection_scores")
        num_detections: np.ndarray = inference_result.get("num_detections")
    
        detections_limit = int(min(detections_limit, num_detections[0]) if detections_limit is not None else num_detections[0])
    
        # Normalize detection boxes coordinates to original image size
        original_image_height, original_image_width, _ = image.shape
        normalized_detection_boxex = detection_boxes[::] * [
            original_image_height,
            original_image_width,
            original_image_height,
            original_image_width,
        ]
    
        image_with_detection_boxex = np.copy(image)
    
        for i in range(detections_limit):
            detected_class_name = labels_map[int(detection_classes[0, i])]
            score = detection_scores[0, i]
            label = f"{detected_class_name} {score:.2f}"
            add_detection_box(
                box=normalized_detection_boxex[0, i],
                image=image_with_detection_boxex,
                label=label,
            )
    
        plt.imshow(image_with_detection_boxex)

TensorFlow Object Detection model
(`faster_rcnn_resnet50_v1_640x640 <https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1>`__)
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
    )



.. parsed-literal::

    data/coco_91cl.txt:   0%|          | 0.00/421 [00:00<?, ?B/s]




.. parsed-literal::

    PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/notebooks/tensorflow-object-detection-to-openvino/data/coco_91cl.txt')



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



.. image:: tensorflow-object-detection-to-openvino-with-output_files/tensorflow-object-detection-to-openvino-with-output_38_0.png


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
