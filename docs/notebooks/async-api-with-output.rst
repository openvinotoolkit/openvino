Asynchronous Inference with OpenVINO™
=====================================

This notebook demonstrates how to use the `Async
API <https://docs.openvino.ai/2024/openvino-workflow/running-inference/optimize-inference/general-optimizations.html>`__
for asynchronous execution with OpenVINO.

OpenVINO Runtime supports inference in either synchronous or
asynchronous mode. The key advantage of the Async API is that when a
device is busy with inference, the application can perform other tasks
in parallel (for example, populating inputs or scheduling other
requests) rather than wait for the current inference to complete first.

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Imports <#imports>`__
-  `Prepare model and data
   processing <#prepare-model-and-data-processing>`__

   -  `Download test model <#download-test-model>`__
   -  `Load the model <#load-the-model>`__
   -  `Create functions for data
      processing <#create-functions-for-data-processing>`__
   -  `Get the test video <#get-the-test-video>`__

-  `How to improve the throughput of video
   processing <#how-to-improve-the-throughput-of-video-processing>`__

   -  `Sync Mode (default) <#sync-mode-default>`__
   -  `Test performance in Sync Mode <#test-performance-in-sync-mode>`__
   -  `Async Mode <#async-mode>`__
   -  `Test the performance in Async
      Mode <#test-the-performance-in-async-mode>`__
   -  `Compare the performance <#compare-the-performance>`__

-  `AsyncInferQueue <#asyncinferqueue>`__

   -  `Setting Callback <#setting-callback>`__
   -  `Test the performance with
      AsyncInferQueue <#test-the-performance-with-asyncinferqueue>`__

Imports
-------



.. code:: ipython3

    import platform
    
    %pip install -q "openvino>=2023.1.0"
    %pip install -q opencv-python
    if platform.system() != "windows":
        %pip install -q "matplotlib>=3.4"
    else:
        %pip install -q "matplotlib>=3.4,<3.7"


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. code:: ipython3

    import cv2
    import time
    import numpy as np
    import openvino as ov
    from IPython import display
    import matplotlib.pyplot as plt
    
    # Fetch the notebook utils script from the openvino_notebooks repo
    import requests
    
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    open("notebook_utils.py", "w").write(r.text)
    
    import notebook_utils as utils

Prepare model and data processing
---------------------------------



Download test model
~~~~~~~~~~~~~~~~~~~



We use a pre-trained model from OpenVINO’s `Open Model
Zoo <https://docs.openvino.ai/2024/documentation/legacy-features/model-zoo.html>`__
to start the test. In this case, the model will be executed to detect
the person in each frame of the video.

.. code:: ipython3

    # directory where model will be downloaded
    base_model_dir = "model"
    
    # model name as named in Open Model Zoo
    model_name = "person-detection-0202"
    precision = "FP16"
    model_path = f"model/intel/{model_name}/{precision}/{model_name}.xml"
    download_command = f"omz_downloader " f"--name {model_name} " f"--precision {precision} " f"--output_dir {base_model_dir} " f"--cache_dir {base_model_dir}"
    ! $download_command


.. parsed-literal::

    ################|| Downloading person-detection-0202 ||################
    
    ========== Downloading model/intel/person-detection-0202/FP16/person-detection-0202.xml


.. parsed-literal::

    ... 12%, 32 KB, 998 KB/s, 0 seconds passed

.. parsed-literal::

    ... 25%, 64 KB, 1010 KB/s, 0 seconds passed
... 38%, 96 KB, 1446 KB/s, 0 seconds passed
... 51%, 128 KB, 1336 KB/s, 0 seconds passed
... 64%, 160 KB, 1638 KB/s, 0 seconds passed
... 77%, 192 KB, 1927 KB/s, 0 seconds passed
... 89%, 224 KB, 2206 KB/s, 0 seconds passed
... 100%, 248 KB, 2423 KB/s, 0 seconds passed

    
    ========== Downloading model/intel/person-detection-0202/FP16/person-detection-0202.bin


.. parsed-literal::

    ... 0%, 32 KB, 646 KB/s, 0 seconds passed

.. parsed-literal::

    ... 1%, 64 KB, 991 KB/s, 0 seconds passed
... 2%, 96 KB, 1232 KB/s, 0 seconds passed
... 3%, 128 KB, 1330 KB/s, 0 seconds passed
... 4%, 160 KB, 1640 KB/s, 0 seconds passed
... 5%, 192 KB, 1860 KB/s, 0 seconds passed

.. parsed-literal::

    ... 6%, 224 KB, 2064 KB/s, 0 seconds passed
... 7%, 256 KB, 2271 KB/s, 0 seconds passed
... 8%, 288 KB, 2239 KB/s, 0 seconds passed
... 9%, 320 KB, 2476 KB/s, 0 seconds passed
... 9%, 352 KB, 2698 KB/s, 0 seconds passed
... 10%, 384 KB, 2906 KB/s, 0 seconds passed
... 11%, 416 KB, 3122 KB/s, 0 seconds passed
... 12%, 448 KB, 3335 KB/s, 0 seconds passed
... 13%, 480 KB, 3537 KB/s, 0 seconds passed
... 14%, 512 KB, 3723 KB/s, 0 seconds passed
... 15%, 544 KB, 3847 KB/s, 0 seconds passed
... 16%, 576 KB, 4016 KB/s, 0 seconds passed

.. parsed-literal::

    ... 17%, 608 KB, 3756 KB/s, 0 seconds passed
... 18%, 640 KB, 3938 KB/s, 0 seconds passed
... 18%, 672 KB, 4121 KB/s, 0 seconds passed
... 19%, 704 KB, 4300 KB/s, 0 seconds passed
... 20%, 736 KB, 4484 KB/s, 0 seconds passed
... 21%, 768 KB, 4666 KB/s, 0 seconds passed
... 22%, 800 KB, 4751 KB/s, 0 seconds passed
... 23%, 832 KB, 4819 KB/s, 0 seconds passed
... 24%, 864 KB, 4944 KB/s, 0 seconds passed
... 25%, 896 KB, 5048 KB/s, 0 seconds passed
... 26%, 928 KB, 5115 KB/s, 0 seconds passed
... 27%, 960 KB, 5202 KB/s, 0 seconds passed
... 27%, 992 KB, 5285 KB/s, 0 seconds passed
... 28%, 1024 KB, 5369 KB/s, 0 seconds passed
... 29%, 1056 KB, 5452 KB/s, 0 seconds passed
... 30%, 1088 KB, 5568 KB/s, 0 seconds passed
... 31%, 1120 KB, 5674 KB/s, 0 seconds passed
... 32%, 1152 KB, 5773 KB/s, 0 seconds passed
... 33%, 1184 KB, 5877 KB/s, 0 seconds passed
... 34%, 1216 KB, 5951 KB/s, 0 seconds passed
... 35%, 1248 KB, 6041 KB/s, 0 seconds passed

.. parsed-literal::

    ... 36%, 1280 KB, 6132 KB/s, 0 seconds passed
... 36%, 1312 KB, 6225 KB/s, 0 seconds passed
... 37%, 1344 KB, 6354 KB/s, 0 seconds passed
... 38%, 1376 KB, 6446 KB/s, 0 seconds passed
... 39%, 1408 KB, 6522 KB/s, 0 seconds passed
... 40%, 1440 KB, 6650 KB/s, 0 seconds passed
... 41%, 1472 KB, 6739 KB/s, 0 seconds passed
... 42%, 1504 KB, 6827 KB/s, 0 seconds passed
... 43%, 1536 KB, 6935 KB/s, 0 seconds passed
... 44%, 1568 KB, 7017 KB/s, 0 seconds passed
... 45%, 1600 KB, 7124 KB/s, 0 seconds passed
... 45%, 1632 KB, 7201 KB/s, 0 seconds passed
... 46%, 1664 KB, 7306 KB/s, 0 seconds passed
... 47%, 1696 KB, 7411 KB/s, 0 seconds passed
... 48%, 1728 KB, 7526 KB/s, 0 seconds passed
... 49%, 1760 KB, 7611 KB/s, 0 seconds passed
... 50%, 1792 KB, 7697 KB/s, 0 seconds passed
... 51%, 1824 KB, 7816 KB/s, 0 seconds passed
... 52%, 1856 KB, 7912 KB/s, 0 seconds passed
... 53%, 1888 KB, 8016 KB/s, 0 seconds passed
... 54%, 1920 KB, 8115 KB/s, 0 seconds passed
... 54%, 1952 KB, 8215 KB/s, 0 seconds passed
... 55%, 1984 KB, 8313 KB/s, 0 seconds passed
... 56%, 2016 KB, 8409 KB/s, 0 seconds passed
... 57%, 2048 KB, 8507 KB/s, 0 seconds passed
... 58%, 2080 KB, 8608 KB/s, 0 seconds passed
... 59%, 2112 KB, 8706 KB/s, 0 seconds passed
... 60%, 2144 KB, 8801 KB/s, 0 seconds passed
... 61%, 2176 KB, 8893 KB/s, 0 seconds passed
... 62%, 2208 KB, 8987 KB/s, 0 seconds passed
... 63%, 2240 KB, 9085 KB/s, 0 seconds passed
... 64%, 2272 KB, 9202 KB/s, 0 seconds passed
... 64%, 2304 KB, 9302 KB/s, 0 seconds passed
... 65%, 2336 KB, 9396 KB/s, 0 seconds passed
... 66%, 2368 KB, 9489 KB/s, 0 seconds passed
... 67%, 2400 KB, 9579 KB/s, 0 seconds passed
... 68%, 2432 KB, 9694 KB/s, 0 seconds passed
... 69%, 2464 KB, 9791 KB/s, 0 seconds passed
... 70%, 2496 KB, 9879 KB/s, 0 seconds passed
... 71%, 2528 KB, 9993 KB/s, 0 seconds passed
... 72%, 2560 KB, 10088 KB/s, 0 seconds passed
... 73%, 2592 KB, 10186 KB/s, 0 seconds passed
... 73%, 2624 KB, 10299 KB/s, 0 seconds passed
... 74%, 2656 KB, 10376 KB/s, 0 seconds passed
... 75%, 2688 KB, 10472 KB/s, 0 seconds passed
... 76%, 2720 KB, 10566 KB/s, 0 seconds passed
... 77%, 2752 KB, 10673 KB/s, 0 seconds passed

.. parsed-literal::

    ... 78%, 2784 KB, 10764 KB/s, 0 seconds passed
... 79%, 2816 KB, 10874 KB/s, 0 seconds passed
... 80%, 2848 KB, 10964 KB/s, 0 seconds passed
... 81%, 2880 KB, 11062 KB/s, 0 seconds passed
... 82%, 2912 KB, 11172 KB/s, 0 seconds passed
... 82%, 2944 KB, 11241 KB/s, 0 seconds passed
... 83%, 2976 KB, 11346 KB/s, 0 seconds passed
... 84%, 3008 KB, 11451 KB/s, 0 seconds passed
... 85%, 3040 KB, 11528 KB/s, 0 seconds passed
... 86%, 3072 KB, 11634 KB/s, 0 seconds passed
... 87%, 3104 KB, 11720 KB/s, 0 seconds passed
... 88%, 3136 KB, 11817 KB/s, 0 seconds passed
... 89%, 3168 KB, 11921 KB/s, 0 seconds passed
... 90%, 3200 KB, 12009 KB/s, 0 seconds passed
... 91%, 3232 KB, 12114 KB/s, 0 seconds passed
... 91%, 3264 KB, 12197 KB/s, 0 seconds passed
... 92%, 3296 KB, 12274 KB/s, 0 seconds passed
... 93%, 3328 KB, 12377 KB/s, 0 seconds passed
... 94%, 3360 KB, 12462 KB/s, 0 seconds passed
... 95%, 3392 KB, 12563 KB/s, 0 seconds passed
... 96%, 3424 KB, 12652 KB/s, 0 seconds passed
... 97%, 3456 KB, 12756 KB/s, 0 seconds passed
... 98%, 3488 KB, 12845 KB/s, 0 seconds passed
... 99%, 3520 KB, 12947 KB/s, 0 seconds passed
... 100%, 3549 KB, 13037 KB/s, 0 seconds passed

    


Select inference device
~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    import ipywidgets as widgets
    
    core = ov.Core()
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value="CPU",
        description="Device:",
        disabled=False,
    )
    
    device




.. parsed-literal::

    Dropdown(description='Device:', options=('CPU', 'AUTO'), value='CPU')



Load the model
~~~~~~~~~~~~~~



.. code:: ipython3

    # initialize OpenVINO runtime
    core = ov.Core()
    
    # read the network and corresponding weights from file
    model = core.read_model(model=model_path)
    
    # compile the model for the CPU (you can choose manually CPU, GPU etc.)
    # or let the engine choose the best available device (AUTO)
    compiled_model = core.compile_model(model=model, device_name=device.value)
    
    # get input node
    input_layer_ir = model.input(0)
    N, C, H, W = input_layer_ir.shape
    shape = (H, W)

Create functions for data processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    def preprocess(image):
        """
        Define the preprocess function for input data
    
        :param: image: the orignal input frame
        :returns:
                resized_image: the image processed
        """
        resized_image = cv2.resize(image, shape)
        resized_image = cv2.cvtColor(np.array(resized_image), cv2.COLOR_BGR2RGB)
        resized_image = resized_image.transpose((2, 0, 1))
        resized_image = np.expand_dims(resized_image, axis=0).astype(np.float32)
        return resized_image
    
    
    def postprocess(result, image, fps):
        """
        Define the postprocess function for output data
    
        :param: result: the inference results
                image: the orignal input frame
                fps: average throughput calculated for each frame
        :returns:
                image: the image with bounding box and fps message
        """
        detections = result.reshape(-1, 7)
        for i, detection in enumerate(detections):
            _, image_id, confidence, xmin, ymin, xmax, ymax = detection
            if confidence > 0.5:
                xmin = int(max((xmin * image.shape[1]), 10))
                ymin = int(max((ymin * image.shape[0]), 10))
                xmax = int(min((xmax * image.shape[1]), image.shape[1] - 10))
                ymax = int(min((ymax * image.shape[0]), image.shape[0] - 10))
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(
                    image,
                    str(round(fps, 2)) + " fps",
                    (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    3,
                )
        return image

Get the test video
~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    video_path = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/CEO%20Pat%20Gelsinger%20on%20Leading%20Intel.mp4"

How to improve the throughput of video processing
-------------------------------------------------



Below, we compare the performance of the synchronous and async-based
approaches:

Sync Mode (default)
~~~~~~~~~~~~~~~~~~~



Let us see how video processing works with the default approach. Using
the synchronous approach, the frame is captured with OpenCV and then
immediately processed:

.. figure:: https://user-images.githubusercontent.com/91237924/168452573-d354ea5b-7966-44e5-813d-f9053be4338a.png
   :alt: drawing

   drawing

::

   while(true) {
   // capture frame
   // populate CURRENT InferRequest
   // Infer CURRENT InferRequest
   //this call is synchronous
   // display CURRENT result
   }

\``\`

.. code:: ipython3

    def sync_api(source, flip, fps, use_popup, skip_first_frames):
        """
        Define the main function for video processing in sync mode
    
        :param: source: the video path or the ID of your webcam
        :returns:
                sync_fps: the inference throughput in sync mode
        """
        frame_number = 0
        infer_request = compiled_model.create_infer_request()
        player = None
        try:
            # Create a video player
            player = utils.VideoPlayer(source, flip=flip, fps=fps, skip_first_frames=skip_first_frames)
            # Start capturing
            start_time = time.time()
            player.start()
            if use_popup:
                title = "Press ESC to Exit"
                cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
            while True:
                frame = player.next()
                if frame is None:
                    print("Source ended")
                    break
                resized_frame = preprocess(frame)
                infer_request.set_tensor(input_layer_ir, ov.Tensor(resized_frame))
                # Start the inference request in synchronous mode
                infer_request.infer()
                res = infer_request.get_output_tensor(0).data
                stop_time = time.time()
                total_time = stop_time - start_time
                frame_number = frame_number + 1
                sync_fps = frame_number / total_time
                frame = postprocess(res, frame, sync_fps)
                # Display the results
                if use_popup:
                    cv2.imshow(title, frame)
                    key = cv2.waitKey(1)
                    # escape = 27
                    if key == 27:
                        break
                else:
                    # Encode numpy array to jpg
                    _, encoded_img = cv2.imencode(".jpg", frame, params=[cv2.IMWRITE_JPEG_QUALITY, 90])
                    # Create IPython image
                    i = display.Image(data=encoded_img)
                    # Display the image in this notebook
                    display.clear_output(wait=True)
                    display.display(i)
        # ctrl-c
        except KeyboardInterrupt:
            print("Interrupted")
        # Any different error
        except RuntimeError as e:
            print(e)
        finally:
            if use_popup:
                cv2.destroyAllWindows()
            if player is not None:
                # stop capturing
                player.stop()
            return sync_fps

Test performance in Sync Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    sync_fps = sync_api(source=video_path, flip=False, fps=30, use_popup=False, skip_first_frames=800)
    print(f"average throuput in sync mode: {sync_fps:.2f} fps")



.. image:: async-api-with-output_files/async-api-with-output_17_0.png


.. parsed-literal::

    Source ended
    average throuput in sync mode: 44.01 fps


Async Mode
~~~~~~~~~~



Let us see how the OpenVINO Async API can improve the overall frame rate
of an application. The key advantage of the Async approach is as
follows: while a device is busy with the inference, the application can
do other things in parallel (for example, populating inputs or
scheduling other requests) rather than wait for the current inference to
complete first.

.. figure:: https://user-images.githubusercontent.com/91237924/168452572-c2ff1c59-d470-4b85-b1f6-b6e1dac9540e.png
   :alt: drawing

   drawing

In the example below, inference is applied to the results of the video
decoding. So it is possible to keep multiple infer requests, and while
the current request is processed, the input frame for the next is being
captured. This essentially hides the latency of capturing, so that the
overall frame rate is rather determined only by the slowest part of the
pipeline (decoding vs inference) and not by the sum of the stages.

::

   while(true) {
   // capture frame
   // populate NEXT InferRequest
   // start NEXT InferRequest
   // this call is async and returns immediately
   // wait for the CURRENT InferRequest
   // display CURRENT result
   // swap CURRENT and NEXT InferRequests
   }

.. code:: ipython3

    def async_api(source, flip, fps, use_popup, skip_first_frames):
        """
        Define the main function for video processing in async mode
    
        :param: source: the video path or the ID of your webcam
        :returns:
                async_fps: the inference throughput in async mode
        """
        frame_number = 0
        # Create 2 infer requests
        curr_request = compiled_model.create_infer_request()
        next_request = compiled_model.create_infer_request()
        player = None
        async_fps = 0
        try:
            # Create a video player
            player = utils.VideoPlayer(source, flip=flip, fps=fps, skip_first_frames=skip_first_frames)
            # Start capturing
            start_time = time.time()
            player.start()
            if use_popup:
                title = "Press ESC to Exit"
                cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
            # Capture CURRENT frame
            frame = player.next()
            resized_frame = preprocess(frame)
            curr_request.set_tensor(input_layer_ir, ov.Tensor(resized_frame))
            # Start the CURRENT inference request
            curr_request.start_async()
            while True:
                # Capture NEXT frame
                next_frame = player.next()
                if next_frame is None:
                    print("Source ended")
                    break
                resized_frame = preprocess(next_frame)
                next_request.set_tensor(input_layer_ir, ov.Tensor(resized_frame))
                # Start the NEXT inference request
                next_request.start_async()
                # Waiting for CURRENT inference result
                curr_request.wait()
                res = curr_request.get_output_tensor(0).data
                stop_time = time.time()
                total_time = stop_time - start_time
                frame_number = frame_number + 1
                async_fps = frame_number / total_time
                frame = postprocess(res, frame, async_fps)
                # Display the results
                if use_popup:
                    cv2.imshow(title, frame)
                    key = cv2.waitKey(1)
                    # escape = 27
                    if key == 27:
                        break
                else:
                    # Encode numpy array to jpg
                    _, encoded_img = cv2.imencode(".jpg", frame, params=[cv2.IMWRITE_JPEG_QUALITY, 90])
                    # Create IPython image
                    i = display.Image(data=encoded_img)
                    # Display the image in this notebook
                    display.clear_output(wait=True)
                    display.display(i)
                # Swap CURRENT and NEXT frames
                frame = next_frame
                # Swap CURRENT and NEXT infer requests
                curr_request, next_request = next_request, curr_request
        # ctrl-c
        except KeyboardInterrupt:
            print("Interrupted")
        # Any different error
        except RuntimeError as e:
            print(e)
        finally:
            if use_popup:
                cv2.destroyAllWindows()
            if player is not None:
                # stop capturing
                player.stop()
            return async_fps

Test the performance in Async Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    async_fps = async_api(source=video_path, flip=False, fps=30, use_popup=False, skip_first_frames=800)
    print(f"average throuput in async mode: {async_fps:.2f} fps")



.. image:: async-api-with-output_files/async-api-with-output_21_0.png


.. parsed-literal::

    Source ended
    average throuput in async mode: 73.74 fps


Compare the performance
~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    width = 0.4
    fontsize = 14
    
    plt.rc("font", size=fontsize)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    rects1 = ax.bar([0], sync_fps, width, color="#557f2d")
    rects2 = ax.bar([width], async_fps, width)
    ax.set_ylabel("frames per second")
    ax.set_xticks([0, width])
    ax.set_xticklabels(["Sync mode", "Async mode"])
    ax.set_xlabel("Higher is better")
    
    fig.suptitle("Sync mode VS Async mode")
    fig.tight_layout()
    
    plt.show()



.. image:: async-api-with-output_files/async-api-with-output_23_0.png


``AsyncInferQueue``
-------------------



Asynchronous mode pipelines can be supported with the
`AsyncInferQueue <https://docs.openvino.ai/2024/openvino-workflow/running-inference/integrate-openvino-with-your-application/python-api-exclusives.html#asyncinferqueue>`__
wrapper class. This class automatically spawns the pool of
``InferRequest`` objects (also called “jobs”) and provides
synchronization mechanisms to control the flow of the pipeline. It is a
simpler way to manage the infer request queue in Asynchronous mode.

Setting Callback
~~~~~~~~~~~~~~~~



When ``callback`` is set, any job that ends inference calls upon the
Python function. The ``callback`` function must have two arguments: one
is the request that calls the ``callback``, which provides the
``InferRequest`` API; the other is called “user data”, which provides
the possibility of passing runtime values.

.. code:: ipython3

    def callback(infer_request, info) -> None:
        """
        Define the callback function for postprocessing
    
        :param: infer_request: the infer_request object
                info: a tuple includes original frame and starts time
        :returns:
                None
        """
        global frame_number
        global total_time
        global inferqueue_fps
        stop_time = time.time()
        frame, start_time = info
        total_time = stop_time - start_time
        frame_number = frame_number + 1
        inferqueue_fps = frame_number / total_time
    
        res = infer_request.get_output_tensor(0).data[0]
        frame = postprocess(res, frame, inferqueue_fps)
        # Encode numpy array to jpg
        _, encoded_img = cv2.imencode(".jpg", frame, params=[cv2.IMWRITE_JPEG_QUALITY, 90])
        # Create IPython image
        i = display.Image(data=encoded_img)
        # Display the image in this notebook
        display.clear_output(wait=True)
        display.display(i)

.. code:: ipython3

    def inferqueue(source, flip, fps, skip_first_frames) -> None:
        """
        Define the main function for video processing with async infer queue
    
        :param: source: the video path or the ID of your webcam
        :retuns:
            None
        """
        # Create infer requests queue
        infer_queue = ov.AsyncInferQueue(compiled_model, 2)
        infer_queue.set_callback(callback)
        player = None
        try:
            # Create a video player
            player = utils.VideoPlayer(source, flip=flip, fps=fps, skip_first_frames=skip_first_frames)
            # Start capturing
            start_time = time.time()
            player.start()
            while True:
                # Capture frame
                frame = player.next()
                if frame is None:
                    print("Source ended")
                    break
                resized_frame = preprocess(frame)
                # Start the inference request with async infer queue
                infer_queue.start_async({input_layer_ir.any_name: resized_frame}, (frame, start_time))
        except KeyboardInterrupt:
            print("Interrupted")
        # Any different error
        except RuntimeError as e:
            print(e)
        finally:
            infer_queue.wait_all()
            player.stop()

Test the performance with ``AsyncInferQueue``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    frame_number = 0
    total_time = 0
    inferqueue(source=video_path, flip=False, fps=30, skip_first_frames=800)
    print(f"average throughput in async mode with async infer queue: {inferqueue_fps:.2f} fps")



.. image:: async-api-with-output_files/async-api-with-output_29_0.png


.. parsed-literal::

    average throughput in async mode with async infer queue: 112.89 fps

