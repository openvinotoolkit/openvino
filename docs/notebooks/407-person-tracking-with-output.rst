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

    %pip install -q "openvino-dev>=2023.1.0"
    %pip install -q opencv-python matplotlib requests scipy


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.0 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.0 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


Imports
-------



.. code:: ipython3

    import collections
    from pathlib import Path
    import sys
    import time
    
    import numpy as np
    import cv2
    from IPython import display
    import matplotlib.pyplot as plt
    import openvino as ov

.. code:: ipython3

    # Import local modules
    
    utils_file_path = Path('../utils/notebook_utils.py')
    notebook_directory_path = Path('.')
    
    if not utils_file_path.exists():
        !git clone --depth 1 https://github.com/igor-davidyuk/openvino_notebooks.git -b moving_data_to_cloud openvino_notebooks
        utils_file_path = Path('./openvino_notebooks/notebooks/utils/notebook_utils.py')
        notebook_directory_path = Path('./openvino_notebooks/notebooks/407-person-tracking-webcam/')
    
    sys.path.append(str(utils_file_path.parent))
    sys.path.append(str(notebook_directory_path))
    
    import notebook_utils as utils
    from deepsort_utils.tracker import Tracker
    from deepsort_utils.nn_matching import NearestNeighborDistanceMetric
    from deepsort_utils.detection import Detection, compute_color_for_labels, xywh_to_xyxy, xywh_to_tlwh, tlwh_to_xyxy

Download the Model
------------------



We will use pre-trained models from OpenVINO’s `Open Model
Zoo <https://docs.openvino.ai/nightly/model_zoo.html>`__ to start the
test.

Use ``omz_downloader``, which is a command-line tool from the
``openvino-dev`` package. It automatically creates a directory structure
and downloads the selected model. This step is skipped if the model is
already downloaded. The selected model comes from the public directory,
which means it must be converted into OpenVINO Intermediate
Representation (OpenVINO IR).

   **NOTE**: Using a model outside the list can require different pre-
   and post-processing.

In this case, `person detection
model <https://docs.openvino.ai/2023.0/omz_models_model_person_detection_0202.html>`__
is deployed to detect the person in each frame of the video, and
`reidentification
model <https://docs.openvino.ai/2023.0/omz_models_model_person_reidentification_retail_0287.html>`__
is used to output embedding vector to match a pair of images of a person
by the cosine distance.

If you want to download another model (``person-detection-xxx`` from
`Object Detection Models
list <https://docs.openvino.ai/2023.0/omz_models_group_intel.html#object-detection-models>`__,
``person-reidentification-retail-xxx`` from `Reidentification Models
list <https://docs.openvino.ai/2023.0/omz_models_group_intel.html#reidentification-models>`__),
replace the name of the model in the code below.

.. code:: ipython3

    # A directory where the model will be downloaded.
    base_model_dir = "model"
    precision = "FP16"
    # The name of the model from Open Model Zoo
    detection_model_name = "person-detection-0202"
    
    download_command = f"omz_downloader " \
                       f"--name {detection_model_name} " \
                       f"--precisions {precision} " \
                       f"--output_dir {base_model_dir} " \
                       f"--cache_dir {base_model_dir}"
    ! $download_command
    
    detection_model_path = f"model/intel/{detection_model_name}/{precision}/{detection_model_name}.xml"
    
    
    reidentification_model_name = "person-reidentification-retail-0287"
    
    download_command = f"omz_downloader " \
                       f"--name {reidentification_model_name} " \
                       f"--precisions {precision} " \
                       f"--output_dir {base_model_dir} " \
                       f"--cache_dir {base_model_dir}"
    ! $download_command
    
    reidentification_model_path = f"model/intel/{reidentification_model_name}/{precision}/{reidentification_model_name}.xml"


.. parsed-literal::

    ################|| Downloading person-detection-0202 ||################
    
    ========== Downloading model/intel/person-detection-0202/FP16/person-detection-0202.xml


.. parsed-literal::

    ... 12%, 32 KB, 1348 KB/s, 0 seconds passed
... 25%, 64 KB, 1353 KB/s, 0 seconds passed

.. parsed-literal::

    ... 38%, 96 KB, 1600 KB/s, 0 seconds passed
... 51%, 128 KB, 1500 KB/s, 0 seconds passed

.. parsed-literal::

    ... 64%, 160 KB, 1319 KB/s, 0 seconds passed
... 77%, 192 KB, 1575 KB/s, 0 seconds passed
... 89%, 224 KB, 1684 KB/s, 0 seconds passed
... 100%, 248 KB, 1865 KB/s, 0 seconds passed

    
    ========== Downloading model/intel/person-detection-0202/FP16/person-detection-0202.bin


.. parsed-literal::

    ... 0%, 32 KB, 1532 KB/s, 0 seconds passed
... 1%, 64 KB, 1424 KB/s, 0 seconds passed
... 2%, 96 KB, 2094 KB/s, 0 seconds passed

.. parsed-literal::

    ... 3%, 128 KB, 1853 KB/s, 0 seconds passed
... 4%, 160 KB, 2245 KB/s, 0 seconds passed
... 5%, 192 KB, 2341 KB/s, 0 seconds passed
... 6%, 224 KB, 2712 KB/s, 0 seconds passed

.. parsed-literal::

    ... 7%, 256 KB, 2395 KB/s, 0 seconds passed
... 8%, 288 KB, 2402 KB/s, 0 seconds passed
... 9%, 320 KB, 2655 KB/s, 0 seconds passed
... 9%, 352 KB, 2912 KB/s, 0 seconds passed
... 10%, 384 KB, 2678 KB/s, 0 seconds passed
... 11%, 416 KB, 2880 KB/s, 0 seconds passed

.. parsed-literal::

    ... 12%, 448 KB, 2855 KB/s, 0 seconds passed
... 13%, 480 KB, 3045 KB/s, 0 seconds passed
... 14%, 512 KB, 2844 KB/s, 0 seconds passed
... 15%, 544 KB, 2820 KB/s, 0 seconds passed
... 16%, 576 KB, 2979 KB/s, 0 seconds passed
... 17%, 608 KB, 3137 KB/s, 0 seconds passed

.. parsed-literal::

    ... 18%, 640 KB, 2954 KB/s, 0 seconds passed
... 18%, 672 KB, 3080 KB/s, 0 seconds passed
... 19%, 704 KB, 3051 KB/s, 0 seconds passed
... 20%, 736 KB, 3184 KB/s, 0 seconds passed
... 21%, 768 KB, 3032 KB/s, 0 seconds passed
... 22%, 800 KB, 3138 KB/s, 0 seconds passed

.. parsed-literal::

    ... 23%, 832 KB, 3111 KB/s, 0 seconds passed
... 24%, 864 KB, 3224 KB/s, 0 seconds passed
... 25%, 896 KB, 3084 KB/s, 0 seconds passed
... 26%, 928 KB, 3041 KB/s, 0 seconds passed
... 27%, 960 KB, 3142 KB/s, 0 seconds passed
... 27%, 992 KB, 3243 KB/s, 0 seconds passed

.. parsed-literal::

    ... 28%, 1024 KB, 3129 KB/s, 0 seconds passed
... 29%, 1056 KB, 3090 KB/s, 0 seconds passed
... 30%, 1088 KB, 3179 KB/s, 0 seconds passed
... 31%, 1120 KB, 3270 KB/s, 0 seconds passed

.. parsed-literal::

    ... 32%, 1152 KB, 3167 KB/s, 0 seconds passed
... 33%, 1184 KB, 3245 KB/s, 0 seconds passed
... 34%, 1216 KB, 3210 KB/s, 0 seconds passed
... 35%, 1248 KB, 3291 KB/s, 0 seconds passed
... 36%, 1280 KB, 3194 KB/s, 0 seconds passed

.. parsed-literal::

    ... 36%, 1312 KB, 3161 KB/s, 0 seconds passed
... 37%, 1344 KB, 3233 KB/s, 0 seconds passed
... 38%, 1376 KB, 3307 KB/s, 0 seconds passed
... 39%, 1408 KB, 3219 KB/s, 0 seconds passed
... 40%, 1440 KB, 3187 KB/s, 0 seconds passed
... 41%, 1472 KB, 3253 KB/s, 0 seconds passed
... 42%, 1504 KB, 3321 KB/s, 0 seconds passed

.. parsed-literal::

    ... 43%, 1536 KB, 3243 KB/s, 0 seconds passed
... 44%, 1568 KB, 3300 KB/s, 0 seconds passed
... 45%, 1600 KB, 3271 KB/s, 0 seconds passed
... 45%, 1632 KB, 3333 KB/s, 0 seconds passed
... 46%, 1664 KB, 3262 KB/s, 0 seconds passed
... 47%, 1696 KB, 3315 KB/s, 0 seconds passed

.. parsed-literal::

    ... 48%, 1728 KB, 3287 KB/s, 0 seconds passed
... 49%, 1760 KB, 3344 KB/s, 0 seconds passed
... 50%, 1792 KB, 3275 KB/s, 0 seconds passed
... 51%, 1824 KB, 3242 KB/s, 0 seconds passed
... 52%, 1856 KB, 3296 KB/s, 0 seconds passed
... 53%, 1888 KB, 3350 KB/s, 0 seconds passed

.. parsed-literal::

    ... 54%, 1920 KB, 3285 KB/s, 0 seconds passed
... 54%, 1952 KB, 3255 KB/s, 0 seconds passed
... 55%, 1984 KB, 3305 KB/s, 0 seconds passed
... 56%, 2016 KB, 3357 KB/s, 0 seconds passed

.. parsed-literal::

    ... 57%, 2048 KB, 3293 KB/s, 0 seconds passed
... 58%, 2080 KB, 3269 KB/s, 0 seconds passed
... 59%, 2112 KB, 3317 KB/s, 0 seconds passed
... 60%, 2144 KB, 3364 KB/s, 0 seconds passed
... 61%, 2176 KB, 3305 KB/s, 0 seconds passed

.. parsed-literal::

    ... 62%, 2208 KB, 3281 KB/s, 0 seconds passed
... 63%, 2240 KB, 3325 KB/s, 0 seconds passed
... 64%, 2272 KB, 3371 KB/s, 0 seconds passed
... 64%, 2304 KB, 3314 KB/s, 0 seconds passed
... 65%, 2336 KB, 3291 KB/s, 0 seconds passed
... 66%, 2368 KB, 3332 KB/s, 0 seconds passed

.. parsed-literal::

    ... 67%, 2400 KB, 3289 KB/s, 0 seconds passed
... 68%, 2432 KB, 3319 KB/s, 0 seconds passed
... 69%, 2464 KB, 3291 KB/s, 0 seconds passed
... 70%, 2496 KB, 3331 KB/s, 0 seconds passed
... 71%, 2528 KB, 3297 KB/s, 0 seconds passed
... 72%, 2560 KB, 3328 KB/s, 0 seconds passed

.. parsed-literal::

    ... 73%, 2592 KB, 3304 KB/s, 0 seconds passed
... 73%, 2624 KB, 3339 KB/s, 0 seconds passed
... 74%, 2656 KB, 3378 KB/s, 0 seconds passed
... 75%, 2688 KB, 3340 KB/s, 0 seconds passed

.. parsed-literal::

    ... 76%, 2720 KB, 3313 KB/s, 0 seconds passed
... 77%, 2752 KB, 3346 KB/s, 0 seconds passed
... 78%, 2784 KB, 3382 KB/s, 0 seconds passed
... 79%, 2816 KB, 3343 KB/s, 0 seconds passed
... 80%, 2848 KB, 3317 KB/s, 0 seconds passed
... 81%, 2880 KB, 3351 KB/s, 0 seconds passed

.. parsed-literal::

    ... 82%, 2912 KB, 3318 KB/s, 0 seconds passed
... 82%, 2944 KB, 3345 KB/s, 0 seconds passed
... 83%, 2976 KB, 3323 KB/s, 0 seconds passed
... 84%, 3008 KB, 3357 KB/s, 0 seconds passed
... 85%, 3040 KB, 3322 KB/s, 0 seconds passed
... 86%, 3072 KB, 3347 KB/s, 0 seconds passed

.. parsed-literal::

    ... 87%, 3104 KB, 3332 KB/s, 0 seconds passed
... 88%, 3136 KB, 3361 KB/s, 0 seconds passed
... 89%, 3168 KB, 3329 KB/s, 0 seconds passed
... 90%, 3200 KB, 3355 KB/s, 0 seconds passed
... 91%, 3232 KB, 3336 KB/s, 0 seconds passed
... 91%, 3264 KB, 3363 KB/s, 0 seconds passed

.. parsed-literal::

    ... 92%, 3296 KB, 3331 KB/s, 0 seconds passed
... 93%, 3328 KB, 3356 KB/s, 0 seconds passed
... 94%, 3360 KB, 3342 KB/s, 1 seconds passed
... 95%, 3392 KB, 3367 KB/s, 1 seconds passed

.. parsed-literal::

    ... 96%, 3424 KB, 3334 KB/s, 1 seconds passed
... 97%, 3456 KB, 3361 KB/s, 1 seconds passed
... 98%, 3488 KB, 3346 KB/s, 1 seconds passed
... 99%, 3520 KB, 3371 KB/s, 1 seconds passed
... 100%, 3549 KB, 3397 KB/s, 1 seconds passed

    


.. parsed-literal::

    ################|| Downloading person-reidentification-retail-0287 ||################
    
    ========== Downloading model/intel/person-reidentification-retail-0287/person-reidentification-retail-0267.onnx


.. parsed-literal::

    ... 0%, 32 KB, 1215 KB/s, 0 seconds passed

.. parsed-literal::

    ... 1%, 64 KB, 1020 KB/s, 0 seconds passed
... 2%, 96 KB, 1323 KB/s, 0 seconds passed
... 3%, 128 KB, 1276 KB/s, 0 seconds passed

.. parsed-literal::

    ... 4%, 160 KB, 1289 KB/s, 0 seconds passed
... 5%, 192 KB, 1412 KB/s, 0 seconds passed
... 6%, 224 KB, 1517 KB/s, 0 seconds passed
... 7%, 256 KB, 1729 KB/s, 0 seconds passed
... 8%, 288 KB, 1941 KB/s, 0 seconds passed

.. parsed-literal::

    ... 9%, 320 KB, 1856 KB/s, 0 seconds passed
... 10%, 352 KB, 1911 KB/s, 0 seconds passed
... 11%, 384 KB, 2080 KB/s, 0 seconds passed
... 11%, 416 KB, 2249 KB/s, 0 seconds passed

.. parsed-literal::

    ... 12%, 448 KB, 2139 KB/s, 0 seconds passed
... 13%, 480 KB, 2172 KB/s, 0 seconds passed
... 14%, 512 KB, 2312 KB/s, 0 seconds passed
... 15%, 544 KB, 2322 KB/s, 0 seconds passed
... 16%, 576 KB, 2342 KB/s, 0 seconds passed
... 17%, 608 KB, 2362 KB/s, 0 seconds passed
... 18%, 640 KB, 2478 KB/s, 0 seconds passed

.. parsed-literal::

    ... 19%, 672 KB, 2474 KB/s, 0 seconds passed
... 20%, 704 KB, 2482 KB/s, 0 seconds passed
... 21%, 736 KB, 2501 KB/s, 0 seconds passed
... 22%, 768 KB, 2604 KB/s, 0 seconds passed
... 22%, 800 KB, 2597 KB/s, 0 seconds passed

.. parsed-literal::

    ... 23%, 832 KB, 2601 KB/s, 0 seconds passed
... 24%, 864 KB, 2607 KB/s, 0 seconds passed
... 25%, 896 KB, 2697 KB/s, 0 seconds passed
... 26%, 928 KB, 2672 KB/s, 0 seconds passed
... 27%, 960 KB, 2693 KB/s, 0 seconds passed

.. parsed-literal::

    ... 28%, 992 KB, 2692 KB/s, 0 seconds passed
... 29%, 1024 KB, 2774 KB/s, 0 seconds passed
... 30%, 1056 KB, 2750 KB/s, 0 seconds passed
... 31%, 1088 KB, 2762 KB/s, 0 seconds passed
... 32%, 1120 KB, 2764 KB/s, 0 seconds passed
... 33%, 1152 KB, 2840 KB/s, 0 seconds passed

.. parsed-literal::

    ... 33%, 1184 KB, 2816 KB/s, 0 seconds passed
... 34%, 1216 KB, 2829 KB/s, 0 seconds passed
... 35%, 1248 KB, 2822 KB/s, 0 seconds passed
... 36%, 1280 KB, 2891 KB/s, 0 seconds passed
... 37%, 1312 KB, 2867 KB/s, 0 seconds passed

.. parsed-literal::

    ... 38%, 1344 KB, 2879 KB/s, 0 seconds passed
... 39%, 1376 KB, 2872 KB/s, 0 seconds passed
... 40%, 1408 KB, 2935 KB/s, 0 seconds passed
... 41%, 1440 KB, 2914 KB/s, 0 seconds passed
... 42%, 1472 KB, 2924 KB/s, 0 seconds passed

.. parsed-literal::

    ... 43%, 1504 KB, 2916 KB/s, 0 seconds passed
... 44%, 1536 KB, 2975 KB/s, 0 seconds passed
... 44%, 1568 KB, 2949 KB/s, 0 seconds passed
... 45%, 1600 KB, 2956 KB/s, 0 seconds passed
... 46%, 1632 KB, 2952 KB/s, 0 seconds passed
... 47%, 1664 KB, 3006 KB/s, 0 seconds passed

.. parsed-literal::

    ... 48%, 1696 KB, 2985 KB/s, 0 seconds passed
... 49%, 1728 KB, 2991 KB/s, 0 seconds passed
... 50%, 1760 KB, 2986 KB/s, 0 seconds passed
... 51%, 1792 KB, 3037 KB/s, 0 seconds passed
... 52%, 1824 KB, 3016 KB/s, 0 seconds passed
... 53%, 1856 KB, 3019 KB/s, 0 seconds passed

.. parsed-literal::

    ... 54%, 1888 KB, 3014 KB/s, 0 seconds passed
... 55%, 1920 KB, 3063 KB/s, 0 seconds passed
... 55%, 1952 KB, 3042 KB/s, 0 seconds passed
... 56%, 1984 KB, 3044 KB/s, 0 seconds passed
... 57%, 2016 KB, 3041 KB/s, 0 seconds passed
... 58%, 2048 KB, 3085 KB/s, 0 seconds passed

.. parsed-literal::

    ... 59%, 2080 KB, 3067 KB/s, 0 seconds passed
... 60%, 2112 KB, 3068 KB/s, 0 seconds passed
... 61%, 2144 KB, 3062 KB/s, 0 seconds passed
... 62%, 2176 KB, 3104 KB/s, 0 seconds passed
... 63%, 2208 KB, 3087 KB/s, 0 seconds passed

.. parsed-literal::

    ... 64%, 2240 KB, 3045 KB/s, 0 seconds passed
... 65%, 2272 KB, 3084 KB/s, 0 seconds passed
... 66%, 2304 KB, 3124 KB/s, 0 seconds passed
... 66%, 2336 KB, 3105 KB/s, 0 seconds passed
... 67%, 2368 KB, 3066 KB/s, 0 seconds passed

.. parsed-literal::

    ... 68%, 2400 KB, 3104 KB/s, 0 seconds passed
... 69%, 2432 KB, 3139 KB/s, 0 seconds passed
... 70%, 2464 KB, 3123 KB/s, 0 seconds passed
... 71%, 2496 KB, 3084 KB/s, 0 seconds passed
... 72%, 2528 KB, 3119 KB/s, 0 seconds passed
... 73%, 2560 KB, 3153 KB/s, 0 seconds passed

.. parsed-literal::

    ... 74%, 2592 KB, 3138 KB/s, 0 seconds passed
... 75%, 2624 KB, 3099 KB/s, 0 seconds passed
... 76%, 2656 KB, 3135 KB/s, 0 seconds passed
... 77%, 2688 KB, 3169 KB/s, 0 seconds passed
... 77%, 2720 KB, 3155 KB/s, 0 seconds passed

.. parsed-literal::

    ... 78%, 2752 KB, 3115 KB/s, 0 seconds passed
... 79%, 2784 KB, 3149 KB/s, 0 seconds passed
... 80%, 2816 KB, 3182 KB/s, 0 seconds passed
... 81%, 2848 KB, 3170 KB/s, 0 seconds passed
... 82%, 2880 KB, 3129 KB/s, 0 seconds passed
... 83%, 2912 KB, 3161 KB/s, 0 seconds passed
... 84%, 2944 KB, 3192 KB/s, 0 seconds passed

.. parsed-literal::

    ... 85%, 2976 KB, 3179 KB/s, 0 seconds passed
... 86%, 3008 KB, 3141 KB/s, 0 seconds passed
... 87%, 3040 KB, 3173 KB/s, 0 seconds passed
... 88%, 3072 KB, 3205 KB/s, 0 seconds passed

.. parsed-literal::

    ... 88%, 3104 KB, 3170 KB/s, 0 seconds passed
... 89%, 3136 KB, 3154 KB/s, 0 seconds passed
... 90%, 3168 KB, 3185 KB/s, 0 seconds passed
... 91%, 3200 KB, 3215 KB/s, 0 seconds passed
... 92%, 3232 KB, 3202 KB/s, 1 seconds passed

.. parsed-literal::

    ... 93%, 3264 KB, 3167 KB/s, 1 seconds passed
... 94%, 3296 KB, 3195 KB/s, 1 seconds passed
... 95%, 3328 KB, 3224 KB/s, 1 seconds passed
... 96%, 3360 KB, 3217 KB/s, 1 seconds passed
... 97%, 3392 KB, 3216 KB/s, 1 seconds passed
... 98%, 3424 KB, 3203 KB/s, 1 seconds passed
... 99%, 3456 KB, 3231 KB/s, 1 seconds passed

.. parsed-literal::

    ... 100%, 3487 KB, 3226 KB/s, 1 seconds passed

    
    ========== Downloading model/intel/person-reidentification-retail-0287/FP16/person-reidentification-retail-0287.xml


.. parsed-literal::

    ... 5%, 32 KB, 1208 KB/s, 0 seconds passed

.. parsed-literal::

    ... 10%, 64 KB, 1236 KB/s, 0 seconds passed
... 15%, 96 KB, 1653 KB/s, 0 seconds passed
... 21%, 128 KB, 1696 KB/s, 0 seconds passed
... 26%, 160 KB, 1983 KB/s, 0 seconds passed

.. parsed-literal::

    ... 31%, 192 KB, 2123 KB/s, 0 seconds passed
... 37%, 224 KB, 2308 KB/s, 0 seconds passed
... 42%, 256 KB, 2274 KB/s, 0 seconds passed
... 47%, 288 KB, 2475 KB/s, 0 seconds passed
... 53%, 320 KB, 2543 KB/s, 0 seconds passed
... 58%, 352 KB, 2654 KB/s, 0 seconds passed

.. parsed-literal::

    ... 63%, 384 KB, 2601 KB/s, 0 seconds passed
... 69%, 416 KB, 2691 KB/s, 0 seconds passed
... 74%, 448 KB, 2727 KB/s, 0 seconds passed
... 79%, 480 KB, 2816 KB/s, 0 seconds passed

.. parsed-literal::

    ... 85%, 512 KB, 2755 KB/s, 0 seconds passed
... 90%, 544 KB, 2838 KB/s, 0 seconds passed
... 95%, 576 KB, 2888 KB/s, 0 seconds passed
... 100%, 600 KB, 2937 KB/s, 0 seconds passed

    
    ========== Downloading model/intel/person-reidentification-retail-0287/FP16/person-reidentification-retail-0287.bin


.. parsed-literal::

    ... 2%, 32 KB, 1326 KB/s, 0 seconds passed

.. parsed-literal::

    ... 5%, 64 KB, 1384 KB/s, 0 seconds passed
... 8%, 96 KB, 1945 KB/s, 0 seconds passed
... 11%, 128 KB, 1848 KB/s, 0 seconds passed
... 13%, 160 KB, 2230 KB/s, 0 seconds passed
... 16%, 192 KB, 2307 KB/s, 0 seconds passed
... 19%, 224 KB, 2608 KB/s, 0 seconds passed

.. parsed-literal::

    ... 22%, 256 KB, 2412 KB/s, 0 seconds passed
... 24%, 288 KB, 2675 KB/s, 0 seconds passed
... 27%, 320 KB, 2675 KB/s, 0 seconds passed
... 30%, 352 KB, 2875 KB/s, 0 seconds passed

.. parsed-literal::

    ... 33%, 384 KB, 2698 KB/s, 0 seconds passed
... 36%, 416 KB, 2883 KB/s, 0 seconds passed
... 38%, 448 KB, 2857 KB/s, 0 seconds passed
... 41%, 480 KB, 3018 KB/s, 0 seconds passed
... 44%, 512 KB, 2810 KB/s, 0 seconds passed
... 47%, 544 KB, 2977 KB/s, 0 seconds passed

.. parsed-literal::

    ... 49%, 576 KB, 2963 KB/s, 0 seconds passed
... 52%, 608 KB, 3068 KB/s, 0 seconds passed
... 55%, 640 KB, 2924 KB/s, 0 seconds passed
... 58%, 672 KB, 3062 KB/s, 0 seconds passed
... 61%, 704 KB, 3037 KB/s, 0 seconds passed
... 63%, 736 KB, 3130 KB/s, 0 seconds passed

.. parsed-literal::

    ... 66%, 768 KB, 3000 KB/s, 0 seconds passed
... 69%, 800 KB, 3120 KB/s, 0 seconds passed
... 72%, 832 KB, 3097 KB/s, 0 seconds passed
... 74%, 864 KB, 3180 KB/s, 0 seconds passed
... 77%, 896 KB, 3062 KB/s, 0 seconds passed

.. parsed-literal::

    ... 80%, 928 KB, 3166 KB/s, 0 seconds passed
... 83%, 960 KB, 3146 KB/s, 0 seconds passed
... 86%, 992 KB, 3222 KB/s, 0 seconds passed
... 88%, 1024 KB, 3112 KB/s, 0 seconds passed
... 91%, 1056 KB, 3203 KB/s, 0 seconds passed
... 94%, 1088 KB, 3182 KB/s, 0 seconds passed

.. parsed-literal::

    ... 97%, 1120 KB, 3253 KB/s, 0 seconds passed
... 99%, 1152 KB, 3150 KB/s, 0 seconds passed
... 100%, 1153 KB, 3150 KB/s, 0 seconds passed

    


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
        value='AUTO',
        description='Device:',
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
        img_batch = np.concatenate([
            preprocess(img, height, width)
            for img in img_crops
        ], axis=0)
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
                    [(xmin + xmax) / 2 * w, (ymin + ymax) / 2 * h, (xmax - xmin) * w, (ymax - ymin) * h]
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
            label = '{}{:d}'.format("", id)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(
                img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
            cv2.putText(
                img,
                label,
                (x1, y1 + t_size[1] + 4),
                cv2.FONT_HERSHEY_PLAIN,
                1.6,
                [255, 255, 255],
                2
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

    base_file_link = 'https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/person_'
    image_indices = ['1_1.png', '1_2.png', '2_1.png']
    image_paths = [utils.download_file(base_file_link + image_index, directory='data') for image_index in image_indices]
    image1, image2, image3 = [cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB) for image_path in image_paths]
    
    # Define titles with images.
    data = {"Person 1": image1, "Person 2": image2, "Person 3": image3}
    
    # Create a subplot to visualize images.
    fig, axs = plt.subplots(1, len(data.items()), figsize=(5, 5))
    
    # Fill the subplot.
    for ax, (name, image) in zip(axs, data.items()):
        ax.axis('off')
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



.. image:: 407-person-tracking-with-output_files/407-person-tracking-with-output_17_3.png


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
        print(f'Same person (confidence: {sim})')
    else:
        print(f'Different person (confidence: {sim})')


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
                source=source, size=(700, 450), flip=flip, fps=24, skip_first_frames=skip_first_frames
            )
            # Start capturing.
            player.start()
            if use_popup:
                title = "Press ESC to Exit"
                cv2.namedWindow(
                    winname=title, flags=cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE
                )
    
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
                detections = [
                    Detection(bbox_tlwh[i], features[i])
                    for i in range(features.shape[0])
                ]
    
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
                    _, encoded_img = cv2.imencode(
                        ext=".jpg", img=frame, params=[cv2.IMWRITE_JPEG_QUALITY, 100]
                    )
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
    metric = NearestNeighborDistanceMetric(
        "cosine", MAX_COSINE_DISTANCE, NN_BUDGET
    )
    tracker = Tracker(
        metric,
        max_iou_distance=0.7,
        max_age=70,
        n_init=3
    )

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
    video_file = 'https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/people.mp4'
    source = cam_id if USE_WEBCAM else video_file
    
    run_person_tracking(source=source, flip=USE_WEBCAM, use_popup=False)



.. image:: 407-person-tracking-with-output_files/407-person-tracking-with-output_25_0.png


.. parsed-literal::

    Source ended

