Convert a TensorFlow Object Detection Model to OpenVINO™
========================================================

.. _top:

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
(OpenVINO IR) format, using `Model
Optimizer <https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html>`__.
After creating the OpenVINO IR, load the model in `OpenVINO
Runtime <https://docs.openvino.ai/2023.0/openvino_docs_OV_UG_OV_Runtime_User_Guide.html>`__
and do inference with a sample image. 

**Table of contents**:

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

Prerequisites `⇑ <#top>`__
###############################################################################################################################


Install required packages:

.. code:: ipython3

    !pip install -q "openvino-dev>=2023.0.0" "numpy>=1.21.0" "opencv-python" "matplotlib>=3.4,<3.5.3"

The notebook uses utility functions. The cell below will download the
``notebook_utils`` Python module from GitHub.

.. code:: ipython3

    # Fetch the notebook utils script from the openvino_notebooks repo
    import urllib.request
    
    urllib.request.urlretrieve(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/utils/notebook_utils.py",
        filename="notebook_utils.py",
    );

Imports `⇑ <#top>`__
###############################################################################################################################


.. code:: ipython3

    # Standard python modules
    from pathlib import Path
    
    # External modules and dependencies
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Notebook utils module
    from notebook_utils import download_file
    
    # OpenVINO modules
    from openvino.runtime import Core, serialize
    from openvino.tools import mo

Settings `⇑ <#top>`__
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

Download Model from TensorFlow Hub `⇑ <#top>`__
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

    PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-475/.workspace/scm/ov-notebook/notebooks/120-tensorflow-object-detection-to-openvino/model/tf/faster_rcnn_resnet50_v1_640x640.tar.gz')



Extract TensorFlow Object Detection model from the downloaded archive:

.. code:: ipython3

    import tarfile
    
    with tarfile.open(tf_model_dir / tf_model_archive_filename) as file:
        file.extractall(path=tf_model_dir)

Convert Model to OpenVINO IR `⇑ <#top>`__
###############################################################################################################################


OpenVINO Model Optimizer Python API can be used to convert the
TensorFlow model to OpenVINO IR.

``mo.convert_model`` function accept path to TensorFlow model and
returns OpenVINO Model class instance which represents this model. Also
we need to provide model input shape (``input_shape``) that is described
at `model overview page on TensorFlow
Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1>`__.
Optionally, we can apply compression to FP16 model weights using
``compress_to_fp16=True`` option and integrate preprocessing using this
approach.

The converted model is ready to load on a device using ``compile_model``
or saved on disk using the ``serialize`` function to reduce loading time
when the model is run in the future.

See the `Model Optimizer Developer
Guide <https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html>`__
for more information about Model Optimizer and TensorFlow `models
support <https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html>`__.

.. code:: ipython3

    ov_model = mo.convert_model(
        saved_model_dir=tf_model_dir,
        input_shape=[[1, 255, 255, 3]]
    )
    
    # Save converted OpenVINO IR model to the corresponding directory
    serialize(ov_model, openvino_ir_path)

Test Inference on the Converted Model `⇑ <#top>`__
###############################################################################################################################


Select inference device `⇑ <#top>`__
###############################################################################################################################


Select device from dropdown list for running inference using OpenVINO:

.. code:: ipython3

    import ipywidgets as widgets
    
    core = Core()
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value='AUTO',
        description='Device:',
        disabled=False,
    )
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



Load the Model `⇑ <#top>`__
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


.. code:: ipython3

    core = Core()
    openvino_ir_model = core.read_model(openvino_ir_path)
    compiled_model = core.compile_model(model=openvino_ir_model, device_name=device.value)

Get Model Information `⇑ <#top>`__
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


Get an Image for Test Inference `⇑ <#top>`__
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

    PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-475/.workspace/scm/ov-notebook/notebooks/120-tensorflow-object-detection-to-openvino/data/coco_bike.jpg')



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

    <matplotlib.image.AxesImage at 0x7f9b48184ca0>




.. image:: 120-tensorflow-object-detection-to-openvino-with-output_files/120-tensorflow-object-detection-to-openvino-with-output_25_1.png


Perform Inference `⇑ <#top>`__
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

    image_detection_boxes: [[[0.16453631 0.54612625 0.89533776 0.85469896]
      [0.6721994  0.01249559 0.98444635 0.53168815]
      [0.4910983  0.01171527 0.98045075 0.88644964]
      ...
      [0.5012431  0.5489591  0.6030575  0.61094964]
      [0.45808432 0.3619884  0.8841141  0.83722156]
      [0.4652153  0.02054662 0.48204365 0.0438836 ]]]
    image_detection_classes: [[18.  2.  2.  3.  2.  8.  2.  2.  3.  2.  4.  4.  2.  4. 16.  1.  1. 27.
       2.  8. 62.  2.  2.  4.  4.  2. 41. 18.  4.  2.  4. 18.  2.  2.  4. 27.
       2.  2. 27.  2.  1.  1. 16.  2.  2.  2. 16.  2.  2.  4.  2.  1. 33.  4.
      15.  2.  3.  2.  2.  1.  2.  1.  4.  2.  3. 11.  4. 35. 40.  4.  1. 62.
       2.  2.  4. 36.  4. 36.  1. 31. 77.  2. 36.  1. 51.  1. 34.  3. 90.  2.
       3.  2.  1.  2.  2.  1.  1.  2.  1.  4. 18.  2.  2.  3. 31.  1. 41.  1.
       2.  2. 33. 41.  3. 31.  1.  3. 36. 27. 27. 15.  4.  4. 15.  3.  2. 37.
       1. 35. 27.  4. 36. 88.  4.  2.  3. 15.  2.  4.  2.  1.  3.  3. 27.  4.
       4. 44. 16.  1.  1. 23.  4.  3.  1.  4.  4. 62. 15. 36. 77.  3. 28.  1.
      35. 27.  2. 27. 75. 36.  8. 28.  3.  4. 36. 35. 44.  4.  3.  1.  2.  1.
       1. 35. 87.  1. 84.  1.  1.  1. 15.  1.  3.  1. 35.  1.  1.  1.  1. 62.
      15.  1. 44. 15.  1. 41. 62.  1.  4. 43. 15.  4.  3.  4. 16. 35.  2. 33.
       3. 14. 62. 34. 41.  2. 35.  4. 18.  3. 15.  1. 27. 87.  1.  4. 19. 21.
      27.  1.  3.  2.  1. 27. 15.  4.  3.  1. 38.  1.  2. 15. 38.  4. 15.  1.
       3.  3. 62. 84. 20. 58.  2.  4. 41. 20. 88. 15.  1. 19. 31. 62. 31.  4.
      14.  1.  8. 18. 15.  2.  4.  2.  2.  2. 31. 84.  2. 15. 28.  3. 27. 18.
      15.  1. 31. 41.  1. 28.  3.  1.  8. 15.  1. 16.]]
    image_detection_scores: [[0.9808771  0.9418091  0.9318733  0.8789291  0.8423196  0.5888979
      0.5630133  0.53731316 0.4974923  0.48222807 0.4673298  0.4398691
      0.39919445 0.33909947 0.3190495  0.27470118 0.24837914 0.23406433
      0.23351488 0.22481255 0.22016802 0.20236589 0.19338816 0.14771679
      0.14576106 0.14285511 0.12738948 0.12668392 0.12027147 0.10873836
      0.10812037 0.09577218 0.09060974 0.08950701 0.08673717 0.08170561
      0.08120535 0.0789713  0.06743153 0.06118729 0.06112184 0.05309067
      0.05216556 0.05023476 0.04783678 0.04460874 0.04213375 0.04042179
      0.04019568 0.03522961 0.03165065 0.0310733  0.03000823 0.02873152
      0.02782036 0.02706797 0.0266978  0.02341437 0.02291683 0.02147149
      0.02130841 0.02099001 0.02032206 0.01978395 0.01961209 0.01902091
      0.01893682 0.01863261 0.01858075 0.01846547 0.01823624 0.0176264
      0.01760109 0.01703349 0.01584588 0.01582033 0.01547665 0.01527787
      0.01522782 0.01430391 0.01428877 0.01422195 0.0141238  0.01411421
      0.0135575  0.01288707 0.01269312 0.01218521 0.01160688 0.01143213
      0.01142005 0.01137567 0.0111644  0.01107758 0.0109348  0.01073039
      0.0106188  0.01016685 0.01010454 0.00983268 0.00977985 0.00967134
      0.00965687 0.00964259 0.00962718 0.00956944 0.00950549 0.00937742
      0.00927729 0.00916896 0.00897371 0.00891221 0.00866699 0.00863667
      0.00855941 0.00836656 0.00835135 0.00816708 0.00795946 0.00793826
      0.00789131 0.00781442 0.00773429 0.00767627 0.00765273 0.00752015
      0.00749519 0.00744095 0.00715925 0.00700314 0.00692652 0.00655058
      0.00643994 0.00641626 0.00629459 0.00628646 0.00627907 0.00612065
      0.00593393 0.00582955 0.00582755 0.00570769 0.00569362 0.00564996
      0.00563695 0.00558055 0.00557034 0.00551842 0.00549368 0.00544169
      0.00544044 0.00542281 0.00540061 0.00525593 0.00524985 0.00515946
      0.00515553 0.00511156 0.00489827 0.00484957 0.00472266 0.00465891
      0.00464309 0.00463513 0.00459531 0.00456809 0.0045585  0.00455432
      0.00443505 0.00443078 0.00440637 0.00422725 0.00416438 0.0041492
      0.00413432 0.00413151 0.00409415 0.00409274 0.00407757 0.00405691
      0.00396555 0.00393284 0.00391471 0.00388586 0.00385833 0.00385633
      0.00385035 0.00379386 0.00378297 0.00378109 0.00377772 0.00370916
      0.00364531 0.00363934 0.00358231 0.00354156 0.0035037  0.00348796
      0.00344136 0.00340937 0.00334414 0.00330951 0.00329006 0.00321436
      0.00320603 0.00312488 0.00309948 0.00307925 0.00307775 0.00306451
      0.00303381 0.00302188 0.00299367 0.00299316 0.00298596 0.00296609
      0.00293693 0.00288884 0.0028709  0.00283928 0.00283312 0.00281894
      0.00276538 0.00276278 0.00270719 0.00268026 0.00258883 0.00258464
      0.00254383 0.00253249 0.00250638 0.00250605 0.00250558 0.0025017
      0.00249729 0.00248757 0.00246982 0.00243592 0.0024358  0.00235382
      0.0023404  0.00233721 0.00233374 0.00233181 0.0023271  0.00230558
      0.00230428 0.00229607 0.00227586 0.00226048 0.00223509 0.00222384
      0.00220214 0.00219295 0.00219229 0.00218538 0.00218472 0.00217254
      0.00216129 0.00214788 0.00213485 0.00213233 0.00208789 0.00206768
      0.00206485 0.00206409 0.00204371 0.00203812 0.00201267 0.00200125
      0.00199629 0.00199346 0.00198402 0.00192943 0.00191091 0.0019036
      0.0018943  0.00188735 0.00188038 0.00186264 0.00179476 0.00177307
      0.00176998 0.00176099 0.0017542  0.00174639 0.00171193 0.0017064
      0.00169167 0.00168484 0.00167157 0.00166569 0.00166213 0.00166009
      0.00164244 0.00164076 0.00163557 0.00162898 0.00160348 0.00159898]]
    image_detections_num: [300.]


Inference Result Visualization `⇑ <#top>`__
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

    PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-475/.workspace/scm/ov-notebook/notebooks/120-tensorflow-object-detection-to-openvino/data/coco_91cl.txt')



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


Next Steps `⇑ <#top>`__
###############################################################################################################################


This section contains suggestions on how to additionally improve the
performance of your application using OpenVINO.

Async inference pipeline `⇑ <#top>`__
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The key advantage of the Async API is that when a device is busy with inference, 
the application can perform other tasks in parallel (for example, populating inputs or 
scheduling other requests) rather than wait for the current inference to 
complete first. To understand how to perform async inference using 
openvino, refer to the `Async API tutorial <115-async-api-with-output.html>`__.

Integration preprocessing to model `⇑ <#top>`__
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
and to the overview of :doc:`Preprocessing API <openvino_docs_OV_UG_Preprocessing_Overview>`.
