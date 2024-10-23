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


**Table of contents:**


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

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

.. code:: ipython3

    %pip install -q "openvino-dev>=2024.0.0"
    %pip install -q opencv-python requests scipy tqdm "matplotlib>=3.4"


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.
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
model <https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/person-detection-0202/README.md>`__
is deployed to detect the person in each frame of the video, and
`reidentification
model <https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/person-reidentification-retail-0287/README.md>`__
is used to output embedding vector to match a pair of images of a person
by the cosine distance.

If you want to download another model (``person-detection-xxx`` from
`Object Detection Models
list <https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/index.md#object-detection-models>`__,
``person-reidentification-retail-xxx`` from `Reidentification Models
list <https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/index.md#reidentification-models>`__),
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


    ========== Downloading model/intel/person-detection-0202/FP16/person-detection-0202.bin


    ################|| Downloading person-reidentification-retail-0287 ||################

    ========== Downloading model/intel/person-reidentification-retail-0287/person-reidentification-retail-0267.onnx


    ========== Downloading model/intel/person-reidentification-retail-0287/FP16/person-reidentification-retail-0287.xml


    ========== Downloading model/intel/person-reidentification-retail-0287/FP16/person-reidentification-retail-0287.bin




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

    device = utils.device_widget()

    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



.. code:: ipython3

    detector = Model(detection_model_path, device=device.value)
    # since the number of detection object is uncertain, the input batch size of reid model should be dynamic
    extractor = Model(reidentification_model_path, -1, device.value)

Data Processing
---------------



Data Processing includes data preprocess and postprocess functions.

- Data preprocess function is used to change the layout and shape of input
  data, according to requirement of the network input format.
- Datapostprocess function is used to extract the useful information from
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

