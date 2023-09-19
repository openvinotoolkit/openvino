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
Representation <https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_IR_and_opsets.html>`__
(OpenVINO IR) format, using Model Converter. After creating the OpenVINO
IR, load the model in `OpenVINO
Runtime <https://docs.openvino.ai/nightly/openvino_docs_OV_UG_OV_Runtime_User_Guide.html>`__
and do inference with a sample image.

**Table of contents:**

- `Prerequisites <#prerequisites>`__
- `Imports <#imports>`__
- `Settings <#settings>`__
- `Download Model from TensorFlow Hub <#download-model-from-tensorflow-hub>`__
- `Convert Model to OpenVINO IR <#convert-model-to-openvino-ir>`__
- `Test Inference on the Converted Model <#test-inference-on-the-converted-model>`__
- `Select inference device <#select-inference-device>`__

  - `Load the Model <#load-the-model>`__
  - `Get Model Information <#get-model-information>`__
  - `Get an Image for Test Inference <#get-an-image-for-test-inference>`__
  - `Perform Inference <#perform-inference>`__
  - `Inference Result Visualization <#inference-result-visualization>`__

- `Next Steps <#next-steps>`__

  - `Async inference pipeline <#async-inference-pipeline>`__
  - `Integration preprocessing to model <#integration-preprocessing-to-model>`__

Prerequisites
###############################################################################################################################

Install required packages:

.. code:: ipython3

    !pip install -q "openvino==2023.1.0.dev20230811" "numpy>=1.21.0" "opencv-python" "matplotlib>=3.4,<3.5.3"

The notebook uses utility functions. The cell below will download the
``notebook_utils`` Python module from GitHub.

.. code:: ipython3

    # Fetch the notebook utils script from the openvino_notebooks repo
    import urllib.request
    
    urllib.request.urlretrieve(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/utils/notebook_utils.py",
        filename="notebook_utils.py",
    );

Imports
###############################################################################################################################

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
    from notebook_utils import download_file

Settings
###############################################################################################################################

Define model related variables and create corresponding directories:

.. code:: ipython3

    # Create directories for models files
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)
    
    # Create directory for TensorFlow model
    tf_model_dir = model_dir / "tf"
    tf_model_dir.mkdir(exist_ok=True)
    
    # Create directory for OpenVINO IR model
    ir_model_dir = model_dir / "ir"
    ir_model_dir.mkdir(exist_ok=True)
    
    model_name = "faster_rcnn_resnet50_v1_640x640"
    
    openvino_ir_path = ir_model_dir / f"{model_name}.xml"
    
    tf_model_url = "https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1?tf-hub-format=compressed"
    
    tf_model_archive_filename = f"{model_name}.tar.gz"

Download Model from TensorFlow Hub
###############################################################################################################################

Download archive with TensorFlow Object Detection model
(`faster_rcnn_resnet50_v1_640x640 <https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1>`__)
from TensorFlow Hub:

.. code:: ipython3

    download_file(
        url=tf_model_url,
        filename=tf_model_archive_filename,
        directory=tf_model_dir
    )



.. parsed-literal::

    model/tf/faster_rcnn_resnet50_v1_640x640.tar.gz:   0%|          | 0.00/101M [00:00<?, ?B/s]




.. parsed-literal::

    PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/notebooks/120-tensorflow-object-detection-to-openvino/model/tf/faster_rcnn_resnet50_v1_640x640.tar.gz')



Extract TensorFlow Object Detection model from the downloaded archive:

.. code:: ipython3

    import tarfile
    
    with tarfile.open(tf_model_dir / tf_model_archive_filename) as file:
        file.extractall(path=tf_model_dir)

Convert Model to OpenVINO IR
###############################################################################################################################

OpenVINO Model Converter Python API can be used to convert the
TensorFlow model to OpenVINO IR.

``ov.convert_model`` function accept path to TensorFlow model and
returns OpenVINO Model class instance which represents this model. Also
we need to provide model input shape (``input_shape``) that is described
at `model overview page on TensorFlow
Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1>`__.

The converted model is ready to load on a device using ``compile_model``
or saved on disk using the ``save_model`` function to reduce loading
time when the model is run in the future.

See the `Model Converter Developer
Guide <https://docs.openvino.ai/2023.1/openvino_docs_model_processing_introduction.html>`__
for more information about Model Converter and TensorFlow `models
support <https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html>`__.

.. code:: ipython3

    ov_model = ov.convert_model(
        tf_model_dir,
        input=[[1, 255, 255, 3]]
    )
    
    # Save converted OpenVINO IR model to the corresponding directory
    ov.save_model(ov_model, openvino_ir_path)

Test Inference on the Converted Model
###############################################################################################################################

Select inference device
###############################################################################################################################

Select device from dropdown list for running inference using OpenVINO:

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



Load the Model
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. code:: ipython3

    core = ov.Core()
    openvino_ir_model = core.read_model(openvino_ir_path)
    compiled_model = core.compile_model(model=openvino_ir_model, device_name=device.value)

Get Model Information
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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
    Model input: <ConstOutput: names[input_tensor] shape[1,255,255,3] type: u8>
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
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Load and save an image:

.. code:: ipython3

    image_path = Path("./data/coco_bike.jpg")
    
    download_file(
        url="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco_bike.jpg",
        filename=image_path.name,
        directory=image_path.parent,
    )



.. parsed-literal::

    data/coco_bike.jpg:   0%|          | 0.00/182k [00:00<?, ?B/s]




.. parsed-literal::

    PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/notebooks/120-tensorflow-object-detection-to-openvino/data/coco_bike.jpg')



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

    <matplotlib.image.AxesImage at 0x7f3b40532ee0>




.. image:: 120-tensorflow-object-detection-to-openvino-with-output_files/120-tensorflow-object-detection-to-openvino-with-output_25_1.png


Perform Inference
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. code:: ipython3

    inference_result = compiled_model(network_input_image)

After model inference on the test image, object detection data can be
extracted from the result. For further model result visualization
``detection_boxes``, ``detection_classes`` and ``detection_scores``
outputs will be used.

.. code:: ipython3

    _, detection_boxes, detection_classes, _, detection_scores, num_detections, _, _ = model_outputs
    
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

    image_detection_boxes: [[[0.1645457  0.54601336 0.8953864  0.85500604]
      [0.67189544 0.01240013 0.9843237  0.5308594 ]
      [0.49188587 0.0117609  0.98050654 0.8866383 ]
      ...
      [0.43604603 0.59332204 0.4692565  0.6341099 ]
      [0.46022677 0.59246916 0.48732638 0.61871874]
      [0.47092935 0.4351712  0.5583364  0.5072162 ]]]
    image_detection_classes: [[18.  2.  2.  3.  2.  8.  2.  2.  3.  2.  4.  4.  2.  4. 16.  1.  1.  2.
      27.  8. 62.  2.  2.  4.  4.  2. 18. 41.  4.  4.  2. 18.  2.  2.  4.  2.
      27.  2. 27.  2.  1.  2. 16.  1. 16.  2.  2.  2.  2. 16.  2.  2.  4.  2.
       1. 33.  4. 15.  3.  2.  2.  1.  2.  1.  4.  2.  3. 11.  4. 35.  4.  1.
      40.  2. 62.  2.  4.  4. 36.  1. 36. 36. 31. 77.  2.  1. 51.  1. 34.  3.
       2.  3. 90.  2.  1.  2.  1.  2.  1.  1.  2.  4. 18.  2.  3.  2. 31.  1.
       1.  2.  2. 33. 41. 41. 31.  3.  1. 36.  3. 15. 27. 27.  4.  4.  2. 37.
       3. 15.  1. 35. 27.  4. 36.  4. 88.  3.  2. 15.  2.  4.  2.  1.  3.  4.
      27.  4.  3. 16. 44.  1.  1. 23.  4.  1.  4.  3.  4. 15. 62. 36. 77.  3.
       1. 28. 27. 35.  2. 36. 75. 28. 27.  8.  3. 36.  4. 44.  2. 35.  4.  1.
       3.  1.  1. 35. 87.  1.  1.  1. 15. 84.  1.  1.  1.  3.  1. 35.  1.  1.
       1. 62. 15.  1. 15. 44.  1. 41.  1. 62.  4.  4.  3. 43. 16. 35. 15.  2.
       4. 34. 14.  3. 62. 33.  4. 41.  2. 35. 18.  3. 15.  1. 27.  4. 87.  2.
      19. 21.  1.  1. 27.  1.  3.  3.  2. 15. 38.  1.  1. 15. 27.  4.  4.  3.
      84. 38.  1. 15.  3. 20. 62. 58. 41. 20.  2.  4. 88. 62. 15. 31.  1. 31.
      14. 19.  4.  1.  2.  8. 18. 15.  4.  2.  2.  2. 31. 84. 15.  3. 28.  2.
      27. 18. 15.  1. 31. 28.  1. 41.  8.  1.  3. 20.]]
    image_detection_scores: [[0.9810079  0.9406672  0.9318088  0.87736803 0.8406418  0.590001
      0.5544931  0.5395725  0.49390146 0.48142615 0.46272704 0.44070086
      0.40116653 0.3470845  0.31795666 0.27489564 0.2474634  0.23632632
      0.23248206 0.22401379 0.21871325 0.20231566 0.19377239 0.14768396
      0.14555264 0.14337891 0.12709722 0.12582931 0.11867397 0.11002139
      0.10564936 0.09225632 0.08963246 0.08887175 0.08704519 0.08072548
      0.08002183 0.07911441 0.0666113  0.06338128 0.06100732 0.06005874
      0.05798699 0.05364133 0.05204991 0.05011017 0.04850946 0.04709009
      0.04469202 0.04128509 0.04075823 0.03989557 0.03523415 0.03272378
      0.03108068 0.02970159 0.02872299 0.02845932 0.02585638 0.02348834
      0.02330401 0.02148149 0.02133745 0.02086147 0.0203565  0.01959799
      0.01931953 0.01926655 0.01872199 0.01856231 0.018533   0.01838779
      0.0181897  0.01780706 0.01727113 0.0166365  0.01586579 0.01579068
      0.01573388 0.01528254 0.01502856 0.01451417 0.01439991 0.01428939
      0.01419332 0.01380482 0.01360496 0.01299109 0.01249149 0.01198874
      0.0114887  0.01145835 0.01144462 0.01139608 0.01113943 0.01108595
      0.01089338 0.01082359 0.01051233 0.01027331 0.01006837 0.00979451
      0.00973239 0.00960592 0.00957181 0.00953101 0.00949827 0.00942653
      0.00942553 0.00931231 0.00907305 0.00887801 0.00884456 0.00881256
      0.00864554 0.00854315 0.00849876 0.00849663 0.00846909 0.00820139
      0.00816586 0.00791354 0.0079015  0.00769929 0.00768903 0.00766408
      0.00766067 0.00764458 0.00745573 0.00721994 0.00706666 0.00700596
      0.0067884  0.00648051 0.00646964 0.00638165 0.00635813 0.00625102
      0.00622972 0.00599667 0.00591933 0.00585055 0.00578007 0.00576509
      0.00572359 0.00560451 0.00558354 0.00556508 0.00553865 0.00548295
      0.00547358 0.00543471 0.00543379 0.0054083  0.0053792  0.00535764
      0.00523385 0.00518936 0.00505314 0.00505005 0.00492085 0.00482561
      0.00471782 0.00470318 0.00464702 0.00461123 0.00458301 0.00457273
      0.00455804 0.00454316 0.00454089 0.00441311 0.00437611 0.0042632
      0.00420744 0.00415997 0.00409999 0.00409556 0.00407972 0.00405195
      0.00404086 0.00399852 0.00399512 0.00393439 0.00390283 0.00387304
      0.0038489  0.00382758 0.00380029 0.00379529 0.00376791 0.00374193
      0.0037119  0.00369629 0.00366445 0.00358808 0.00351782 0.0035044
      0.00344527 0.00343268 0.00342918 0.0033823  0.00332239 0.00330844
      0.00329753 0.00327268 0.00315135 0.0031098  0.00308979 0.00308363
      0.00305497 0.00304868 0.00304043 0.00303659 0.00302582 0.00301236
      0.0029885  0.00291268 0.00290264 0.00289243 0.00287722 0.00286564
      0.0028257  0.00282503 0.00275258 0.00274533 0.0027204  0.00268618
      0.00261918 0.00260795 0.00256593 0.00254094 0.00252855 0.00250768
      0.00249793 0.00249551 0.00248255 0.00247912 0.00246619 0.00241695
      0.00240165 0.00236032 0.00235902 0.00234437 0.00234337 0.00233791
      0.00233535 0.00230773 0.00230558 0.00229112 0.00228888 0.0022631
      0.00225214 0.00224187 0.00222553 0.00219966 0.00219677 0.00217865
      0.00217776 0.00215922 0.0021541  0.00214997 0.00212955 0.00211928
      0.0021005  0.00205066 0.00204869 0.00203888 0.00203537 0.00203026
      0.00201357 0.00199936 0.00199387 0.00197951 0.00197288 0.00195503
      0.00194848 0.00192129 0.00189951 0.00187286 0.0018519  0.00182989
      0.00179158 0.00177909 0.00176328 0.00176319 0.00175034 0.00173788
      0.00172983 0.00172819 0.00168273 0.0016768  0.00167542 0.00167398
      0.0016395  0.00163637 0.00163319 0.00162887 0.00162824 0.00162028]]
    image_detections_num: [300.]


Inference Result Visualization
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Define utility functions to visualize the inference results

.. code:: ipython3

    import random
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
        box_color = [random.randint(0, 255) for _ in range(3)]
        line_thickness = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
    
        cv2.rectangle(img=image, pt1=point1, pt2=point2, color=box_color, thickness=line_thickness, lineType=cv2.LINE_AA)
    
        if label:
            font_thickness = max(line_thickness - 1, 1)
            font_face = 0
            font_scale = line_thickness / 3
            font_color = (255, 255, 255)
            text_size = cv2.getTextSize(text=label, fontFace=font_face, fontScale=font_scale, thickness=font_thickness)[0]
            # Calculate rectangle coordinates
            rectangle_point1 = point1
            rectangle_point2 = (point1[0] + text_size[0], point1[1] - text_size[1] - 3)
            # Add filled rectangle
            cv2.rectangle(img=image, pt1=rectangle_point1, pt2=rectangle_point2, color=box_color, thickness=-1, lineType=cv2.LINE_AA)
            # Calculate text position
            text_position = point1[0], point1[1] - 3
            # Add text with label to filled rectangle
            cv2.putText(img=image, text=label, org=text_position, fontFace=font_face, fontScale=font_scale, color=font_color, thickness=font_thickness, lineType=cv2.LINE_AA)
        return image

.. code:: ipython3

    from typing import Dict
    
    from openvino.runtime.utils.data_helpers import OVDict
    
    
    def visualize_inference_result(inference_result: OVDict, image: np.ndarray, labels_map: Dict, detections_limit: Optional[int] = None):
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
    
        detections_limit = int(
            min(detections_limit, num_detections[0])
            if detections_limit is not None
            else num_detections[0]
        )
    
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

    PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/notebooks/120-tensorflow-object-detection-to-openvino/data/coco_91cl.txt')



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



.. image:: 120-tensorflow-object-detection-to-openvino-with-output_files/120-tensorflow-object-detection-to-openvino-with-output_38_0.png


Next Steps
###############################################################################################################################

This section contains suggestions on how to additionally improve the
performance of your application using OpenVINO.

Async inference pipeline
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The key advantage of the Async API is that when a device is busy with
inference, the application can perform other tasks in parallel (for
example, populating inputs or scheduling other requests) rather than
wait for the current inference to complete first. To understand how to
perform async inference using openvino, refer to the `Async API
tutorial <115-async-api-with-output.html>`__.

Integration preprocessing to model
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Preprocessing API enables making preprocessing a part of the model
reducing application code and dependency on additional image processing
libraries. The main advantage of Preprocessing API is that preprocessing
steps will be integrated into the execution graph and will be performed
on a selected device (CPU/GPU etc.) rather than always being executed on
CPU as part of an application. This will improve selected device
utilization.

For more information, refer to the `Optimize Preprocessing
tutorial <118-optimize-preprocessing-with-output.html>`__
and to the overview of `Preprocessing
API <https://docs.openvino.ai/2023.0/openvino_docs_OV_UG_Preprocessing_Overview.html>`__.
