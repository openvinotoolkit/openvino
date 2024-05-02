Person Tracking with OpenVINO™
==============================

This notebook demonstrates live person tracking with OpenVINO: it reads
frames from an input video sequence, detects people in the frames,
uniquely identifies each one of them and tracks all of them until they
leave the frame. We will use the `Deep
SORT <https://arxiv.org/abs/1703.07402>`__ algorithm to perform object
tracking, an extension to SORT (Simple Online and Realtime Tracking).

Detection vs Tracking
---------------------

-  In object detection, we detect an object in a frame, put a bounding
   box or a mask around it, and classify the object. Note that, the job
   of the detector ends here. It processes each frame independently and
   identifies numerous objects in that particular frame.
-  An object tracker on the other hand needs to track a particular
   object across the entire video. If the detector detects three cars in
   the frame, the object tracker has to identify the three separate
   detections and needs to track it across the subsequent frames (with
   the help of a unique ID).

Deep SORT
---------

`Deep SORT <https://arxiv.org/abs/1703.07402>`__ can be defined as the
tracking algorithm which tracks objects not only based on the velocity
and motion of the object but also the appearance of the object. It is
made of three key components which are as follows: |deepsort|

1. **Detection**

   This is the first step in the tracking module. In this step, a deep
   learning model will be used to detect the objects in the frame that
   are to be tracked. These detections are then passed on to the next
   step.

2. **Prediction**

   In this step, we use Kalman filter [1] framework to predict a target
   bounding box of each tracking object in the next frame. There are two
   states of prediction output: ``confirmed`` and ``unconfirmed``. A new
   track comes with a state of ``unconfirmed`` by default, and it can be
   turned into ``confirmed`` when a certain number of consecutive
   detections are matched with this new track. Meanwhile, if a matched
   track is missed over a specific time, it will be deleted as well.

3. **Data association and update**

   Now, we have to match the target bounding box with the detected
   bounding box, and update track identities. A conventional way to
   solve the association between the predicted Kalman states and newly
   arrived measurements is to build an assignment problem with the
   Hungarian algorithm [2]. In this problem formulation, we integrate
   motion and appearance information through a combination of two
   appropriate metrics. The cost used for the first matching step is set
   as a combination of the Mahalanobis and the cosine distances. The
   `Mahalanobis
   distance <https://en.wikipedia.org/wiki/Mahalanobis_distance>`__ is
   used to incorporate motion information and the cosine distance is
   used to calculate similarity between two objects. Cosine distance is
   a metric that helps the tracker recover identities in case of
   long-term occlusion and motion estimation also fails. For this
   purposes, a reidentification model will be implemented to produce a
   vector in high-dimensional space that represents the appearance of
   the object. Using these simple things can make the tracker even more
   powerful and accurate.

   In the second matching stage, we will run intersection over
   union(IOU) association as proposed in the original SORT algorithm [3]
   on the set of unconfirmed and unmatched tracks from the previous
   step. If the IOU of detection and target is less than a certain
   threshold value called ``IOUmin`` then that assignment is rejected.
   This helps to account for sudden appearance changes, for example, due
   to partial occlusion with static scene geometry, and to increase
   robustness against erroneous.

   When detection result is associated with a target, the detected
   bounding box is used to update the target state.

--------------

[1] R. Kalman, “A New Approach to Linear Filtering and Prediction
Problems”, Journal of Basic Engineering, vol. 82, no. Series D,
pp. 35-45, 1960.

[2] H. W. Kuhn, “The Hungarian method for the assignment problem”, Naval
Research Logistics Quarterly, vol. 2, pp. 83-97, 1955.

[3] A. Bewley, G. Zongyuan, F. Ramos, and B. Upcroft, “Simple online and
realtime tracking,” in ICIP, 2016, pp. 3464–3468.

.. |deepsort| image:: https://user-images.githubusercontent.com/91237924/221744683-0042eff8-2c41-43b8-b3ad-b5929bafb60b.png

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Imports <#imports>`__
-  `Download the Model <#download-the-model>`__
-  `Load model <#load-model>`__

   -  `Select inference device <#select-inference-device>`__

-  `Data Processing <#data-processing>`__
-  `Test person reidentification
   model <#test-person-reidentification-model>`__

   -  `Visualize data <#visualize-data>`__
   -  `Compare two persons <#compare-two-persons>`__

-  `Main Processing Function <#main-processing-function>`__
-  `Run <#run>`__

   -  `Initialize tracker <#initialize-tracker>`__
   -  `Run Live Person Tracking <#run-live-person-tracking>`__

.. code:: ipython3

    import platform
    
    %pip install -q "openvino-dev>=2024.0.0"
    %pip install -q opencv-python requests scipy tqdm
    
    if platform.system() != "Windows":
        %pip install -q "matplotlib>=3.4"
    else:
        %pip install -q "matplotlib>=3.4,<3.7"


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


Imports
-------



.. code:: ipython3

    import collections
    from pathlib import Path
    import time
    
    import numpy as np
    import cv2
    from IPython import display
    import matplotlib.pyplot as plt
    import openvino as ov

.. code:: ipython3

    # Import local modules
    
    if not Path("./notebook_utils.py").exists():
        # Fetch `notebook_utils` module
        import requests
    
        r = requests.get(
            url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
        )
    
        open("notebook_utils.py", "w").write(r.text)
    
    import notebook_utils as utils
    from deepsort_utils.tracker import Tracker
    from deepsort_utils.nn_matching import NearestNeighborDistanceMetric
    from deepsort_utils.detection import (
        Detection,
        compute_color_for_labels,
        xywh_to_xyxy,
        xywh_to_tlwh,
        tlwh_to_xyxy,
    )

Download the Model
------------------



We will use pre-trained models from OpenVINO’s `Open Model
Zoo <https://docs.openvino.ai/2024/documentation/legacy-features/model-zoo.html>`__
to start the test.

Use ``omz_downloader``, which is a command-line tool from the
``openvino-dev`` package. It automatically creates a directory structure
and downloads the selected model. This step is skipped if the model is
already downloaded. The selected model comes from the public directory,
which means it must be converted into OpenVINO Intermediate
Representation (OpenVINO IR).

   **NOTE**: Using a model outside the list can require different pre-
   and post-processing.

In this case, `person detection
model <https://docs.openvino.ai/2024/omz_models_model_person_detection_0202.html>`__
is deployed to detect the person in each frame of the video, and
`reidentification
model <https://docs.openvino.ai/2024/omz_models_model_person_reidentification_retail_0287.html>`__
is used to output embedding vector to match a pair of images of a person
by the cosine distance.

If you want to download another model (``person-detection-xxx`` from
`Object Detection Models
list <https://docs.openvino.ai/2024/omz_models_group_intel.html#object-detection-models>`__,
``person-reidentification-retail-xxx`` from `Reidentification Models
list <https://docs.openvino.ai/2024/omz_models_group_intel.html#reidentification-models>`__),
replace the name of the model in the code below.

.. code:: ipython3

    # A directory where the model will be downloaded.
    base_model_dir = "model"
    precision = "FP16"
    # The name of the model from Open Model Zoo
    detection_model_name = "person-detection-0202"
    
    download_command = (
        f"omz_downloader " f"--name {detection_model_name} " f"--precisions {precision} " f"--output_dir {base_model_dir} " f"--cache_dir {base_model_dir}"
    )
    ! $download_command
    
    detection_model_path = f"model/intel/{detection_model_name}/{precision}/{detection_model_name}.xml"
    
    
    reidentification_model_name = "person-reidentification-retail-0287"
    
    download_command = (
        f"omz_downloader " f"--name {reidentification_model_name} " f"--precisions {precision} " f"--output_dir {base_model_dir} " f"--cache_dir {base_model_dir}"
    )
    ! $download_command
    
    reidentification_model_path = f"model/intel/{reidentification_model_name}/{precision}/{reidentification_model_name}.xml"


.. parsed-literal::

    ################|| Downloading person-detection-0202 ||################
    
    ========== Downloading model/intel/person-detection-0202/FP16/person-detection-0202.xml


.. parsed-literal::

    ... 12%, 32 KB, 900 KB/s, 0 seconds passed

.. parsed-literal::

    ... 25%, 64 KB, 937 KB/s, 0 seconds passed
... 38%, 96 KB, 1364 KB/s, 0 seconds passed
... 51%, 128 KB, 1254 KB/s, 0 seconds passed
... 64%, 160 KB, 1552 KB/s, 0 seconds passed
... 77%, 192 KB, 1828 KB/s, 0 seconds passed
... 89%, 224 KB, 2097 KB/s, 0 seconds passed
... 100%, 248 KB, 2307 KB/s, 0 seconds passed

    
    ========== Downloading model/intel/person-detection-0202/FP16/person-detection-0202.bin


.. parsed-literal::

    ... 0%, 32 KB, 867 KB/s, 0 seconds passed
... 1%, 64 KB, 952 KB/s, 0 seconds passed

.. parsed-literal::

    ... 2%, 96 KB, 1141 KB/s, 0 seconds passed
... 3%, 128 KB, 1273 KB/s, 0 seconds passed
... 4%, 160 KB, 1575 KB/s, 0 seconds passed
... 5%, 192 KB, 1774 KB/s, 0 seconds passed
... 6%, 224 KB, 1964 KB/s, 0 seconds passed
... 7%, 256 KB, 2147 KB/s, 0 seconds passed

.. parsed-literal::

    ... 8%, 288 KB, 2143 KB/s, 0 seconds passed
... 9%, 320 KB, 2371 KB/s, 0 seconds passed
... 9%, 352 KB, 2597 KB/s, 0 seconds passed
... 10%, 384 KB, 2813 KB/s, 0 seconds passed
... 11%, 416 KB, 2976 KB/s, 0 seconds passed
... 12%, 448 KB, 3065 KB/s, 0 seconds passed
... 13%, 480 KB, 3215 KB/s, 0 seconds passed
... 14%, 512 KB, 3349 KB/s, 0 seconds passed
... 15%, 544 KB, 3487 KB/s, 0 seconds passed
... 16%, 576 KB, 3642 KB/s, 0 seconds passed
... 17%, 608 KB, 3611 KB/s, 0 seconds passed
... 18%, 640 KB, 3786 KB/s, 0 seconds passed
... 18%, 672 KB, 3958 KB/s, 0 seconds passed
... 19%, 704 KB, 4128 KB/s, 0 seconds passed
... 20%, 736 KB, 4280 KB/s, 0 seconds passed
... 21%, 768 KB, 4411 KB/s, 0 seconds passed
... 22%, 800 KB, 4515 KB/s, 0 seconds passed
... 23%, 832 KB, 4615 KB/s, 0 seconds passed
... 24%, 864 KB, 4755 KB/s, 0 seconds passed
... 25%, 896 KB, 4879 KB/s, 0 seconds passed

.. parsed-literal::

    ... 26%, 928 KB, 5017 KB/s, 0 seconds passed
... 27%, 960 KB, 5135 KB/s, 0 seconds passed
... 27%, 992 KB, 5250 KB/s, 0 seconds passed
... 28%, 1024 KB, 5363 KB/s, 0 seconds passed
... 29%, 1056 KB, 5472 KB/s, 0 seconds passed
... 30%, 1088 KB, 5610 KB/s, 0 seconds passed
... 31%, 1120 KB, 5717 KB/s, 0 seconds passed
... 32%, 1152 KB, 5869 KB/s, 0 seconds passed
... 33%, 1184 KB, 5860 KB/s, 0 seconds passed
... 34%, 1216 KB, 6002 KB/s, 0 seconds passed
... 35%, 1248 KB, 6147 KB/s, 0 seconds passed
... 36%, 1280 KB, 6292 KB/s, 0 seconds passed
... 36%, 1312 KB, 6412 KB/s, 0 seconds passed
... 37%, 1344 KB, 6552 KB/s, 0 seconds passed
... 38%, 1376 KB, 6647 KB/s, 0 seconds passed
... 39%, 1408 KB, 6735 KB/s, 0 seconds passed
... 40%, 1440 KB, 6822 KB/s, 0 seconds passed
... 41%, 1472 KB, 6939 KB/s, 0 seconds passed
... 42%, 1504 KB, 7032 KB/s, 0 seconds passed
... 43%, 1536 KB, 7148 KB/s, 0 seconds passed
... 44%, 1568 KB, 7266 KB/s, 0 seconds passed
... 45%, 1600 KB, 7365 KB/s, 0 seconds passed
... 45%, 1632 KB, 7482 KB/s, 0 seconds passed
... 46%, 1664 KB, 7616 KB/s, 0 seconds passed
... 47%, 1696 KB, 7744 KB/s, 0 seconds passed
... 48%, 1728 KB, 7855 KB/s, 0 seconds passed
... 49%, 1760 KB, 7962 KB/s, 0 seconds passed
... 50%, 1792 KB, 8069 KB/s, 0 seconds passed
... 51%, 1824 KB, 8173 KB/s, 0 seconds passed
... 52%, 1856 KB, 8291 KB/s, 0 seconds passed
... 53%, 1888 KB, 8388 KB/s, 0 seconds passed
... 54%, 1920 KB, 8517 KB/s, 0 seconds passed
... 54%, 1952 KB, 8630 KB/s, 0 seconds passed
... 55%, 1984 KB, 8734 KB/s, 0 seconds passed
... 56%, 2016 KB, 8836 KB/s, 0 seconds passed
... 57%, 2048 KB, 8939 KB/s, 0 seconds passed
... 58%, 2080 KB, 9041 KB/s, 0 seconds passed
... 59%, 2112 KB, 9148 KB/s, 0 seconds passed
... 60%, 2144 KB, 9269 KB/s, 0 seconds passed
... 61%, 2176 KB, 9379 KB/s, 0 seconds passed
... 62%, 2208 KB, 9479 KB/s, 0 seconds passed
... 63%, 2240 KB, 9600 KB/s, 0 seconds passed
... 64%, 2272 KB, 9710 KB/s, 0 seconds passed
... 64%, 2304 KB, 9810 KB/s, 0 seconds passed
... 65%, 2336 KB, 9929 KB/s, 0 seconds passed
... 66%, 2368 KB, 10032 KB/s, 0 seconds passed

.. parsed-literal::

    ... 67%, 2400 KB, 9992 KB/s, 0 seconds passed
... 68%, 2432 KB, 10108 KB/s, 0 seconds passed
... 69%, 2464 KB, 10224 KB/s, 0 seconds passed
... 70%, 2496 KB, 10342 KB/s, 0 seconds passed
... 71%, 2528 KB, 10296 KB/s, 0 seconds passed
... 72%, 2560 KB, 10408 KB/s, 0 seconds passed
... 73%, 2592 KB, 10520 KB/s, 0 seconds passed
... 73%, 2624 KB, 10631 KB/s, 0 seconds passed
... 74%, 2656 KB, 10743 KB/s, 0 seconds passed
... 75%, 2688 KB, 10856 KB/s, 0 seconds passed
... 76%, 2720 KB, 10967 KB/s, 0 seconds passed
... 77%, 2752 KB, 11079 KB/s, 0 seconds passed
... 78%, 2784 KB, 11191 KB/s, 0 seconds passed
... 79%, 2816 KB, 11302 KB/s, 0 seconds passed
... 80%, 2848 KB, 11413 KB/s, 0 seconds passed
... 81%, 2880 KB, 11524 KB/s, 0 seconds passed
... 82%, 2912 KB, 11635 KB/s, 0 seconds passed
... 82%, 2944 KB, 11747 KB/s, 0 seconds passed
... 83%, 2976 KB, 11856 KB/s, 0 seconds passed
... 84%, 3008 KB, 11966 KB/s, 0 seconds passed
... 85%, 3040 KB, 12078 KB/s, 0 seconds passed
... 86%, 3072 KB, 12187 KB/s, 0 seconds passed
... 87%, 3104 KB, 12297 KB/s, 0 seconds passed
... 88%, 3136 KB, 12407 KB/s, 0 seconds passed
... 89%, 3168 KB, 12516 KB/s, 0 seconds passed
... 90%, 3200 KB, 12623 KB/s, 0 seconds passed
... 91%, 3232 KB, 12733 KB/s, 0 seconds passed
... 91%, 3264 KB, 12840 KB/s, 0 seconds passed
... 92%, 3296 KB, 12948 KB/s, 0 seconds passed
... 93%, 3328 KB, 13059 KB/s, 0 seconds passed
... 94%, 3360 KB, 13169 KB/s, 0 seconds passed
... 95%, 3392 KB, 13280 KB/s, 0 seconds passed
... 96%, 3424 KB, 13392 KB/s, 0 seconds passed
... 97%, 3456 KB, 13494 KB/s, 0 seconds passed
... 98%, 3488 KB, 13602 KB/s, 0 seconds passed
... 99%, 3520 KB, 13702 KB/s, 0 seconds passed
... 100%, 3549 KB, 13798 KB/s, 0 seconds passed

    


.. parsed-literal::

    ################|| Downloading person-reidentification-retail-0287 ||################
    
    ========== Downloading model/intel/person-reidentification-retail-0287/person-reidentification-retail-0267.onnx


.. parsed-literal::

    ... 0%, 32 KB, 948 KB/s, 0 seconds passed

.. parsed-literal::

    ... 1%, 64 KB, 949 KB/s, 0 seconds passed
... 2%, 96 KB, 1371 KB/s, 0 seconds passed
... 3%, 128 KB, 1763 KB/s, 0 seconds passed
... 4%, 160 KB, 1573 KB/s, 0 seconds passed
... 5%, 192 KB, 1852 KB/s, 0 seconds passed
... 6%, 224 KB, 2124 KB/s, 0 seconds passed
... 7%, 256 KB, 2387 KB/s, 0 seconds passed
... 8%, 288 KB, 2635 KB/s, 0 seconds passed

.. parsed-literal::

    ... 9%, 320 KB, 2365 KB/s, 0 seconds passed
... 10%, 352 KB, 2577 KB/s, 0 seconds passed
... 11%, 384 KB, 2802 KB/s, 0 seconds passed
... 11%, 416 KB, 3027 KB/s, 0 seconds passed
... 12%, 448 KB, 3252 KB/s, 0 seconds passed
... 13%, 480 KB, 3469 KB/s, 0 seconds passed
... 14%, 512 KB, 3687 KB/s, 0 seconds passed
... 15%, 544 KB, 3897 KB/s, 0 seconds passed
... 16%, 576 KB, 4115 KB/s, 0 seconds passed
... 17%, 608 KB, 4268 KB/s, 0 seconds passed
... 18%, 640 KB, 4465 KB/s, 0 seconds passed

.. parsed-literal::

    ... 19%, 672 KB, 3961 KB/s, 0 seconds passed
... 20%, 704 KB, 4102 KB/s, 0 seconds passed
... 21%, 736 KB, 4279 KB/s, 0 seconds passed
... 22%, 768 KB, 4456 KB/s, 0 seconds passed
... 22%, 800 KB, 4632 KB/s, 0 seconds passed
... 23%, 832 KB, 4809 KB/s, 0 seconds passed
... 24%, 864 KB, 4985 KB/s, 0 seconds passed
... 25%, 896 KB, 5158 KB/s, 0 seconds passed
... 26%, 928 KB, 5333 KB/s, 0 seconds passed
... 27%, 960 KB, 5506 KB/s, 0 seconds passed
... 28%, 992 KB, 5678 KB/s, 0 seconds passed
... 29%, 1024 KB, 5849 KB/s, 0 seconds passed
... 30%, 1056 KB, 6020 KB/s, 0 seconds passed
... 31%, 1088 KB, 6190 KB/s, 0 seconds passed
... 32%, 1120 KB, 6362 KB/s, 0 seconds passed
... 33%, 1152 KB, 6531 KB/s, 0 seconds passed
... 33%, 1184 KB, 6700 KB/s, 0 seconds passed
... 34%, 1216 KB, 6868 KB/s, 0 seconds passed
... 35%, 1248 KB, 7036 KB/s, 0 seconds passed
... 36%, 1280 KB, 7203 KB/s, 0 seconds passed
... 37%, 1312 KB, 6540 KB/s, 0 seconds passed
... 38%, 1344 KB, 6550 KB/s, 0 seconds passed
... 39%, 1376 KB, 6691 KB/s, 0 seconds passed
... 40%, 1408 KB, 6834 KB/s, 0 seconds passed
... 41%, 1440 KB, 6978 KB/s, 0 seconds passed
... 42%, 1472 KB, 7121 KB/s, 0 seconds passed
... 43%, 1504 KB, 7263 KB/s, 0 seconds passed
... 44%, 1536 KB, 7406 KB/s, 0 seconds passed
... 44%, 1568 KB, 7547 KB/s, 0 seconds passed
... 45%, 1600 KB, 7689 KB/s, 0 seconds passed
... 46%, 1632 KB, 7830 KB/s, 0 seconds passed
... 47%, 1664 KB, 7971 KB/s, 0 seconds passed
... 48%, 1696 KB, 8111 KB/s, 0 seconds passed
... 49%, 1728 KB, 8252 KB/s, 0 seconds passed
... 50%, 1760 KB, 8391 KB/s, 0 seconds passed
... 51%, 1792 KB, 8531 KB/s, 0 seconds passed
... 52%, 1824 KB, 8669 KB/s, 0 seconds passed
... 53%, 1856 KB, 8808 KB/s, 0 seconds passed
... 54%, 1888 KB, 8944 KB/s, 0 seconds passed
... 55%, 1920 KB, 9081 KB/s, 0 seconds passed
... 55%, 1952 KB, 9218 KB/s, 0 seconds passed
... 56%, 1984 KB, 9354 KB/s, 0 seconds passed
... 57%, 2016 KB, 9491 KB/s, 0 seconds passed
... 58%, 2048 KB, 9616 KB/s, 0 seconds passed
... 59%, 2080 KB, 9751 KB/s, 0 seconds passed
... 60%, 2112 KB, 9885 KB/s, 0 seconds passed
... 61%, 2144 KB, 10019 KB/s, 0 seconds passed
... 62%, 2176 KB, 10153 KB/s, 0 seconds passed
... 63%, 2208 KB, 10286 KB/s, 0 seconds passed
... 64%, 2240 KB, 10419 KB/s, 0 seconds passed
... 65%, 2272 KB, 10551 KB/s, 0 seconds passed
... 66%, 2304 KB, 10683 KB/s, 0 seconds passed
... 66%, 2336 KB, 10815 KB/s, 0 seconds passed
... 67%, 2368 KB, 10946 KB/s, 0 seconds passed
... 68%, 2400 KB, 11077 KB/s, 0 seconds passed
... 69%, 2432 KB, 11208 KB/s, 0 seconds passed
... 70%, 2464 KB, 11339 KB/s, 0 seconds passed

.. parsed-literal::

    ... 71%, 2496 KB, 11470 KB/s, 0 seconds passed
... 72%, 2528 KB, 11600 KB/s, 0 seconds passed
... 73%, 2560 KB, 11730 KB/s, 0 seconds passed
... 74%, 2592 KB, 11861 KB/s, 0 seconds passed
... 75%, 2624 KB, 11992 KB/s, 0 seconds passed
... 76%, 2656 KB, 11112 KB/s, 0 seconds passed
... 77%, 2688 KB, 11225 KB/s, 0 seconds passed
... 77%, 2720 KB, 11341 KB/s, 0 seconds passed
... 78%, 2752 KB, 11459 KB/s, 0 seconds passed
... 79%, 2784 KB, 11577 KB/s, 0 seconds passed
... 80%, 2816 KB, 11693 KB/s, 0 seconds passed
... 81%, 2848 KB, 11810 KB/s, 0 seconds passed
... 82%, 2880 KB, 11927 KB/s, 0 seconds passed
... 83%, 2912 KB, 12042 KB/s, 0 seconds passed
... 84%, 2944 KB, 12158 KB/s, 0 seconds passed
... 85%, 2976 KB, 12272 KB/s, 0 seconds passed
... 86%, 3008 KB, 12388 KB/s, 0 seconds passed
... 87%, 3040 KB, 12501 KB/s, 0 seconds passed
... 88%, 3072 KB, 12616 KB/s, 0 seconds passed
... 88%, 3104 KB, 12729 KB/s, 0 seconds passed
... 89%, 3136 KB, 12844 KB/s, 0 seconds passed
... 90%, 3168 KB, 12958 KB/s, 0 seconds passed
... 91%, 3200 KB, 13070 KB/s, 0 seconds passed
... 92%, 3232 KB, 13184 KB/s, 0 seconds passed
... 93%, 3264 KB, 13296 KB/s, 0 seconds passed
... 94%, 3296 KB, 13409 KB/s, 0 seconds passed
... 95%, 3328 KB, 13522 KB/s, 0 seconds passed
... 96%, 3360 KB, 13633 KB/s, 0 seconds passed
... 97%, 3392 KB, 13745 KB/s, 0 seconds passed
... 98%, 3424 KB, 13856 KB/s, 0 seconds passed
... 99%, 3456 KB, 13968 KB/s, 0 seconds passed
... 100%, 3487 KB, 14073 KB/s, 0 seconds passed

    
    ========== Downloading model/intel/person-reidentification-retail-0287/FP16/person-reidentification-retail-0287.xml


.. parsed-literal::

    ... 5%, 32 KB, 972 KB/s, 0 seconds passed

.. parsed-literal::

    ... 10%, 64 KB, 958 KB/s, 0 seconds passed
... 15%, 96 KB, 1427 KB/s, 0 seconds passed

.. parsed-literal::

    ... 21%, 128 KB, 1286 KB/s, 0 seconds passed
... 26%, 160 KB, 1600 KB/s, 0 seconds passed
... 31%, 192 KB, 1907 KB/s, 0 seconds passed
... 37%, 224 KB, 2208 KB/s, 0 seconds passed
... 42%, 256 KB, 2500 KB/s, 0 seconds passed
... 47%, 288 KB, 2144 KB/s, 0 seconds passed
... 53%, 320 KB, 2363 KB/s, 0 seconds passed
... 58%, 352 KB, 2581 KB/s, 0 seconds passed
... 63%, 384 KB, 2795 KB/s, 0 seconds passed
... 69%, 416 KB, 3009 KB/s, 0 seconds passed
... 74%, 448 KB, 3231 KB/s, 0 seconds passed
... 79%, 480 KB, 3438 KB/s, 0 seconds passed
... 85%, 512 KB, 3641 KB/s, 0 seconds passed

.. parsed-literal::

    ... 90%, 544 KB, 3760 KB/s, 0 seconds passed
... 95%, 576 KB, 3464 KB/s, 0 seconds passed
... 100%, 600 KB, 3598 KB/s, 0 seconds passed

    
    ========== Downloading model/intel/person-reidentification-retail-0287/FP16/person-reidentification-retail-0287.bin


.. parsed-literal::

    ... 2%, 32 KB, 1015 KB/s, 0 seconds passed

.. parsed-literal::

    ... 5%, 64 KB, 990 KB/s, 0 seconds passed
... 8%, 96 KB, 1460 KB/s, 0 seconds passed
... 11%, 128 KB, 1917 KB/s, 0 seconds passed
... 13%, 160 KB, 1648 KB/s, 0 seconds passed
... 16%, 192 KB, 1957 KB/s, 0 seconds passed
... 19%, 224 KB, 2260 KB/s, 0 seconds passed
... 22%, 256 KB, 2556 KB/s, 0 seconds passed
... 24%, 288 KB, 2847 KB/s, 0 seconds passed

.. parsed-literal::

    ... 27%, 320 KB, 2510 KB/s, 0 seconds passed
... 30%, 352 KB, 2676 KB/s, 0 seconds passed
... 33%, 384 KB, 2874 KB/s, 0 seconds passed
... 36%, 416 KB, 3069 KB/s, 0 seconds passed
... 38%, 448 KB, 3255 KB/s, 0 seconds passed
... 41%, 480 KB, 3461 KB/s, 0 seconds passed
... 44%, 512 KB, 3679 KB/s, 0 seconds passed
... 47%, 544 KB, 3892 KB/s, 0 seconds passed
... 49%, 576 KB, 4108 KB/s, 0 seconds passed
... 52%, 608 KB, 4326 KB/s, 0 seconds passed
... 55%, 640 KB, 4541 KB/s, 0 seconds passed

.. parsed-literal::

    ... 58%, 672 KB, 4170 KB/s, 0 seconds passed
... 61%, 704 KB, 4356 KB/s, 0 seconds passed
... 63%, 736 KB, 4542 KB/s, 0 seconds passed
... 66%, 768 KB, 4728 KB/s, 0 seconds passed
... 69%, 800 KB, 4906 KB/s, 0 seconds passed
... 72%, 832 KB, 5089 KB/s, 0 seconds passed
... 74%, 864 KB, 5274 KB/s, 0 seconds passed
... 77%, 896 KB, 5456 KB/s, 0 seconds passed
... 80%, 928 KB, 5638 KB/s, 0 seconds passed
... 83%, 960 KB, 5818 KB/s, 0 seconds passed
... 86%, 992 KB, 5997 KB/s, 0 seconds passed
... 88%, 1024 KB, 6176 KB/s, 0 seconds passed
... 91%, 1056 KB, 6354 KB/s, 0 seconds passed
... 94%, 1088 KB, 6531 KB/s, 0 seconds passed
... 97%, 1120 KB, 6708 KB/s, 0 seconds passed
... 99%, 1152 KB, 6886 KB/s, 0 seconds passed
... 100%, 1153 KB, 6882 KB/s, 0 seconds passed

    


Load model
----------



Define a common class for model loading and predicting.

There are four main steps for OpenVINO model initialization, and they
are required to run for only once before inference loop. 1. Initialize
OpenVINO Runtime. 2. Read the network from ``*.bin`` and ``*.xml`` files
(weights and architecture). 3. Compile the model for device. 4. Get
input and output names of nodes.

In this case, we can put them all in a class constructor function.

To let OpenVINO automatically select the best device for inference just
use ``AUTO``. In most cases, the best device to use is ``GPU`` (better
performance, but slightly longer startup time).

.. code:: ipython3

    core = ov.Core()
    
    
    class Model:
        """
        This class represents a OpenVINO model object.
    
        """
    
        def __init__(self, model_path, batchsize=1, device="AUTO"):
            """
            Initialize the model object
    
            Parameters
            ----------
            model_path: path of inference model
            batchsize: batch size of input data
            device: device used to run inference
            """
            self.model = core.read_model(model=model_path)
            self.input_layer = self.model.input(0)
            self.input_shape = self.input_layer.shape
            self.height = self.input_shape[2]
            self.width = self.input_shape[3]
    
            for layer in self.model.inputs:
                input_shape = layer.partial_shape
                input_shape[0] = batchsize
                self.model.reshape({layer: input_shape})
            self.compiled_model = core.compile_model(model=self.model, device_name=device)
            self.output_layer = self.compiled_model.output(0)
    
        def predict(self, input):
            """
            Run inference
    
            Parameters
            ----------
            input: array of input data
            """
            result = self.compiled_model(input)[self.output_layer]
            return result

Select inference device
~~~~~~~~~~~~~~~~~~~~~~~



select device from dropdown list for running inference using OpenVINO

.. code:: ipython3

    import ipywidgets as widgets
    
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value="AUTO",
        description="Device:",
        disabled=False,
    )
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



.. code:: ipython3

    detector = Model(detection_model_path, device=device.value)
    # since the number of detection object is uncertain, the input batch size of reid model should be dynamic
    extractor = Model(reidentification_model_path, -1, device.value)

Data Processing
---------------



Data Processing includes data preprocess and postprocess functions. -
Data preprocess function is used to change the layout and shape of input
data, according to requirement of the network input format. - Data
postprocess function is used to extract the useful information from
network’s original output and visualize it.

.. code:: ipython3

    def preprocess(frame, height, width):
        """
        Preprocess a single image
    
        Parameters
        ----------
        frame: input frame
        height: height of model input data
        width: width of model input data
        """
        resized_image = cv2.resize(frame, (width, height))
        resized_image = resized_image.transpose((2, 0, 1))
        input_image = np.expand_dims(resized_image, axis=0).astype(np.float32)
        return input_image
    
    
    def batch_preprocess(img_crops, height, width):
        """
        Preprocess batched images
    
        Parameters
        ----------
        img_crops: batched input images
        height: height of model input data
        width: width of model input data
        """
        img_batch = np.concatenate([preprocess(img, height, width) for img in img_crops], axis=0)
        return img_batch
    
    
    def process_results(h, w, results, thresh=0.5):
        """
        postprocess detection results
    
        Parameters
        ----------
        h, w: original height and width of input image
        results: raw detection network output
        thresh: threshold for low confidence filtering
        """
        # The 'results' variable is a [1, 1, N, 7] tensor.
        detections = results.reshape(-1, 7)
        boxes = []
        labels = []
        scores = []
        for i, detection in enumerate(detections):
            _, label, score, xmin, ymin, xmax, ymax = detection
            # Filter detected objects.
            if score > thresh:
                # Create a box with pixels coordinates from the box with normalized coordinates [0,1].
                boxes.append(
                    [
                        (xmin + xmax) / 2 * w,
                        (ymin + ymax) / 2 * h,
                        (xmax - xmin) * w,
                        (ymax - ymin) * h,
                    ]
                )
                labels.append(int(label))
                scores.append(float(score))
    
        if len(boxes) == 0:
            boxes = np.array([]).reshape(0, 4)
            scores = np.array([])
            labels = np.array([])
        return np.array(boxes), np.array(scores), np.array(labels)
    
    
    def draw_boxes(img, bbox, identities=None):
        """
        Draw bounding box in original image
    
        Parameters
        ----------
        img: original image
        bbox: coordinate of bounding box
        identities: identities IDs
        """
        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            # box text and bar
            id = int(identities[i]) if identities is not None else 0
            color = compute_color_for_labels(id)
            label = "{}{:d}".format("", id)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
            cv2.putText(
                img,
                label,
                (x1, y1 + t_size[1] + 4),
                cv2.FONT_HERSHEY_PLAIN,
                1.6,
                [255, 255, 255],
                2,
            )
        return img
    
    
    def cosin_metric(x1, x2):
        """
        Calculate the consin distance of two vector
    
        Parameters
        ----------
        x1, x2: input vectors
        """
        return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

Test person reidentification model
----------------------------------



The reidentification network outputs a blob with the ``(1, 256)`` shape
named ``reid_embedding``, which can be compared with other descriptors
using the cosine distance.

Visualize data
~~~~~~~~~~~~~~



.. code:: ipython3

    base_file_link = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/person_"
    image_indices = ["1_1.png", "1_2.png", "2_1.png"]
    image_paths = [utils.download_file(base_file_link + image_index, directory="data") for image_index in image_indices]
    image1, image2, image3 = [cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB) for image_path in image_paths]
    
    # Define titles with images.
    data = {"Person 1": image1, "Person 2": image2, "Person 3": image3}
    
    # Create a subplot to visualize images.
    fig, axs = plt.subplots(1, len(data.items()), figsize=(5, 5))
    
    # Fill the subplot.
    for ax, (name, image) in zip(axs, data.items()):
        ax.axis("off")
        ax.set_title(name)
        ax.imshow(image)
    
    # Display an image.
    plt.show(fig)



.. parsed-literal::

    data/person_1_1.png:   0%|          | 0.00/68.3k [00:00<?, ?B/s]



.. parsed-literal::

    data/person_1_2.png:   0%|          | 0.00/68.9k [00:00<?, ?B/s]



.. parsed-literal::

    data/person_2_1.png:   0%|          | 0.00/70.3k [00:00<?, ?B/s]



.. image:: person-tracking-with-output_files/person-tracking-with-output_17_3.png


Compare two persons
~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    # Metric parameters
    MAX_COSINE_DISTANCE = 0.6  # threshold of matching object
    input_data = [image2, image3]
    img_batch = batch_preprocess(input_data, extractor.height, extractor.width)
    features = extractor.predict(img_batch)
    sim = cosin_metric(features[0], features[1])
    if sim >= 1 - MAX_COSINE_DISTANCE:
        print(f"Same person (confidence: {sim})")
    else:
        print(f"Different person (confidence: {sim})")


.. parsed-literal::

    Different person (confidence: 0.02726624347269535)


Main Processing Function
------------------------



Run person tracking on the specified source. Either a webcam feed or a
video file.

.. code:: ipython3

    # Main processing function to run person tracking.
    def run_person_tracking(source=0, flip=False, use_popup=False, skip_first_frames=0):
        """
        Main function to run the person tracking:
        1. Create a video player to play with target fps (utils.VideoPlayer).
        2. Prepare a set of frames for person tracking.
        3. Run AI inference for person tracking.
        4. Visualize the results.
    
        Parameters:
        ----------
            source: The webcam number to feed the video stream with primary webcam set to "0", or the video path.
            flip: To be used by VideoPlayer function for flipping capture image.
            use_popup: False for showing encoded frames over this notebook, True for creating a popup window.
            skip_first_frames: Number of frames to skip at the beginning of the video.
        """
        player = None
        try:
            # Create a video player to play with target fps.
            player = utils.VideoPlayer(
                source=source,
                size=(700, 450),
                flip=flip,
                fps=24,
                skip_first_frames=skip_first_frames,
            )
            # Start capturing.
            player.start()
            if use_popup:
                title = "Press ESC to Exit"
                cv2.namedWindow(winname=title, flags=cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
    
            processing_times = collections.deque()
            while True:
                # Grab the frame.
                frame = player.next()
                if frame is None:
                    print("Source ended")
                    break
                # If the frame is larger than full HD, reduce size to improve the performance.
    
                # Resize the image and change dims to fit neural network input.
                h, w = frame.shape[:2]
                input_image = preprocess(frame, detector.height, detector.width)
    
                # Measure processing time.
                start_time = time.time()
                # Get the results.
                output = detector.predict(input_image)
                stop_time = time.time()
                processing_times.append(stop_time - start_time)
                if len(processing_times) > 200:
                    processing_times.popleft()
    
                _, f_width = frame.shape[:2]
                # Mean processing time [ms].
                processing_time = np.mean(processing_times) * 1100
                fps = 1000 / processing_time
    
                # Get poses from detection results.
                bbox_xywh, score, label = process_results(h, w, results=output)
    
                img_crops = []
                for box in bbox_xywh:
                    x1, y1, x2, y2 = xywh_to_xyxy(box, h, w)
                    img = frame[y1:y2, x1:x2]
                    img_crops.append(img)
    
                # Get reidentification feature of each person.
                if img_crops:
                    # preprocess
                    img_batch = batch_preprocess(img_crops, extractor.height, extractor.width)
                    features = extractor.predict(img_batch)
                else:
                    features = np.array([])
    
                # Wrap the detection and reidentification results together
                bbox_tlwh = xywh_to_tlwh(bbox_xywh)
                detections = [Detection(bbox_tlwh[i], features[i]) for i in range(features.shape[0])]
    
                # predict the position of tracking target
                tracker.predict()
    
                # update tracker
                tracker.update(detections)
    
                # update bbox identities
                outputs = []
                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    box = track.to_tlwh()
                    x1, y1, x2, y2 = tlwh_to_xyxy(box, h, w)
                    track_id = track.track_id
                    outputs.append(np.array([x1, y1, x2, y2, track_id], dtype=np.int32))
                if len(outputs) > 0:
                    outputs = np.stack(outputs, axis=0)
    
                # draw box for visualization
                if len(outputs) > 0:
                    bbox_tlwh = []
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    frame = draw_boxes(frame, bbox_xyxy, identities)
    
                cv2.putText(
                    img=frame,
                    text=f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
                    org=(20, 40),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=f_width / 1000,
                    color=(0, 0, 255),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )
    
                if use_popup:
                    cv2.imshow(winname=title, mat=frame)
                    key = cv2.waitKey(1)
                    # escape = 27
                    if key == 27:
                        break
                else:
                    # Encode numpy array to jpg.
                    _, encoded_img = cv2.imencode(ext=".jpg", img=frame, params=[cv2.IMWRITE_JPEG_QUALITY, 100])
                    # Create an IPython image.
                    i = display.Image(data=encoded_img)
                    # Display the image in this notebook.
                    display.clear_output(wait=True)
                    display.display(i)
    
        # ctrl-c
        except KeyboardInterrupt:
            print("Interrupted")
        # any different error
        except RuntimeError as e:
            print(e)
        finally:
            if player is not None:
                # Stop capturing.
                player.stop()
            if use_popup:
                cv2.destroyAllWindows()

Run
---



Initialize tracker
~~~~~~~~~~~~~~~~~~



Before running a new tracking task, we have to reinitialize a Tracker
object

.. code:: ipython3

    NN_BUDGET = 100
    MAX_COSINE_DISTANCE = 0.6  # threshold of matching object
    metric = NearestNeighborDistanceMetric("cosine", MAX_COSINE_DISTANCE, NN_BUDGET)
    tracker = Tracker(metric, max_iou_distance=0.7, max_age=70, n_init=3)

Run Live Person Tracking
~~~~~~~~~~~~~~~~~~~~~~~~



Use a webcam as the video input. By default, the primary webcam is set
with ``source=0``. If you have multiple webcams, each one will be
assigned a consecutive number starting at 0. Set ``flip=True`` when
using a front-facing camera. Some web browsers, especially Mozilla
Firefox, may cause flickering. If you experience flickering, set
``use_popup=True``.

If you do not have a webcam, you can still run this demo with a video
file. Any `format supported by
OpenCV <https://docs.opencv.org/4.5.1/dd/d43/tutorial_py_video_display.html>`__
will work.

.. code:: ipython3

    USE_WEBCAM = False
    
    cam_id = 0
    video_file = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/people.mp4"
    source = cam_id if USE_WEBCAM else video_file
    
    run_person_tracking(source=source, flip=USE_WEBCAM, use_popup=False)



.. image:: person-tracking-with-output_files/person-tracking-with-output_25_0.png


.. parsed-literal::

    Source ended

