Asynchronous Inference with OpenVINO™
=====================================

This notebook demonstrates how to use the `Async
API <https://docs.openvino.ai/nightly/openvino_docs_deployment_optimization_guide_common.html>`__
for asynchronous execution with OpenVINO.

OpenVINO Runtime supports inference in either synchronous or
asynchronous mode. The key advantage of the Async API is that when a
device is busy with inference, the application can perform other tasks
in parallel (for example, populating inputs or scheduling other
requests) rather than wait for the current inference to complete first.

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Imports <#Imports>`__
-  `Prepare model and data
   processing <#Prepare-model-and-data-processing>`__

   -  `Download test model <#Download-test-model>`__
   -  `Load the model <#Load-the-model>`__
   -  `Create functions for data
      processing <#Create-functions-for-data-processing>`__
   -  `Get the test video <#Get-the-test-video>`__

-  `How to improve the throughput of video
   processing <#How-to-improve-the-throughput-of-video-processing>`__

   -  `Sync Mode (default) <#Sync-Mode-(default)>`__
   -  `Test performance in Sync Mode <#Test-performance-in-Sync-Mode>`__
   -  `Async Mode <#Async-Mode>`__
   -  `Test the performance in Async
      Mode <#Test-the-performance-in-Async-Mode>`__
   -  `Compare the performance <#Compare-the-performance>`__

-  ```AsyncInferQueue`` <#AsyncInferQueue>`__

   -  `Setting Callback <#Setting-Callback>`__
   -  `Test the performance with
      ``AsyncInferQueue`` <#Test-the-performance-with-AsyncInferQueue>`__

Imports
-------

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    %pip install -q "openvino>=2023.1.0"
    %pip install -q opencv-python matplotlib


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
    import urllib.request
    urllib.request.urlretrieve(
        url='https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/utils/notebook_utils.py',
        filename='notebook_utils.py'
    )
    
    import notebook_utils as utils

Prepare model and data processing
---------------------------------

`back to top ⬆️ <#Table-of-contents:>`__

Download test model
~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

We use a pre-trained model from OpenVINO’s `Open Model
Zoo <https://docs.openvino.ai/nightly/model_zoo.html>`__ to start the
test. In this case, the model will be executed to detect the person in
each frame of the video.

.. code:: ipython3

    # directory where model will be downloaded
    base_model_dir = "model"
    
    # model name as named in Open Model Zoo
    model_name = "person-detection-0202"
    precision = "FP16"
    model_path = (
        f"model/intel/{model_name}/{precision}/{model_name}.xml"
    )
    download_command = f"omz_downloader " \
                       f"--name {model_name} " \
                       f"--precision {precision} " \
                       f"--output_dir {base_model_dir} " \
                       f"--cache_dir {base_model_dir}"
    ! $download_command


.. parsed-literal::

    ################|| Downloading person-detection-0202 ||################
    
    ========== Downloading model/intel/person-detection-0202/FP16/person-detection-0202.xml


.. parsed-literal::

    ... 12%, 32 KB, 1452 KB/s, 0 seconds passed

.. parsed-literal::

    ... 25%, 64 KB, 1385 KB/s, 0 seconds passed... 38%, 96 KB, 1632 KB/s, 0 seconds passed

.. parsed-literal::

    ... 51%, 128 KB, 1424 KB/s, 0 seconds passed... 64%, 160 KB, 1393 KB/s, 0 seconds passed... 77%, 192 KB, 1664 KB/s, 0 seconds passed... 89%, 224 KB, 1795 KB/s, 0 seconds passed... 100%, 248 KB, 1986 KB/s, 0 seconds passed


.. parsed-literal::

    
    ========== Downloading model/intel/person-detection-0202/FP16/person-detection-0202.bin


.. parsed-literal::

    ... 0%, 32 KB, 1284 KB/s, 0 seconds passed... 1%, 64 KB, 1284 KB/s, 0 seconds passed... 2%, 96 KB, 1894 KB/s, 0 seconds passed

.. parsed-literal::

    ... 3%, 128 KB, 1699 KB/s, 0 seconds passed... 4%, 160 KB, 2092 KB/s, 0 seconds passed... 5%, 192 KB, 2274 KB/s, 0 seconds passed... 6%, 224 KB, 2623 KB/s, 0 seconds passed... 7%, 256 KB, 2328 KB/s, 0 seconds passed... 8%, 288 KB, 2599 KB/s, 0 seconds passed

.. parsed-literal::

    ... 9%, 320 KB, 2693 KB/s, 0 seconds passed... 9%, 352 KB, 2939 KB/s, 0 seconds passed... 10%, 384 KB, 2655 KB/s, 0 seconds passed... 11%, 416 KB, 2858 KB/s, 0 seconds passed... 12%, 448 KB, 2918 KB/s, 0 seconds passed... 13%, 480 KB, 3100 KB/s, 0 seconds passed

.. parsed-literal::

    ... 14%, 512 KB, 2858 KB/s, 0 seconds passed... 15%, 544 KB, 2904 KB/s, 0 seconds passed... 16%, 576 KB, 3065 KB/s, 0 seconds passed... 17%, 608 KB, 3219 KB/s, 0 seconds passed... 18%, 640 KB, 2998 KB/s, 0 seconds passed... 18%, 672 KB, 3133 KB/s, 0 seconds passed

.. parsed-literal::

    ... 19%, 704 KB, 3164 KB/s, 0 seconds passed... 20%, 736 KB, 3293 KB/s, 0 seconds passed... 21%, 768 KB, 3098 KB/s, 0 seconds passed... 22%, 800 KB, 3120 KB/s, 0 seconds passed... 23%, 832 KB, 3228 KB/s, 0 seconds passed

.. parsed-literal::

    ... 24%, 864 KB, 3062 KB/s, 0 seconds passed... 25%, 896 KB, 3170 KB/s, 0 seconds passed... 26%, 928 KB, 3274 KB/s, 0 seconds passed... 27%, 960 KB, 3291 KB/s, 0 seconds passed... 27%, 992 KB, 3388 KB/s, 0 seconds passed... 28%, 1024 KB, 3231 KB/s, 0 seconds passed... 29%, 1056 KB, 3322 KB/s, 0 seconds passed

.. parsed-literal::

    ... 30%, 1088 KB, 3331 KB/s, 0 seconds passed... 31%, 1120 KB, 3189 KB/s, 0 seconds passed... 32%, 1152 KB, 3275 KB/s, 0 seconds passed... 33%, 1184 KB, 3360 KB/s, 0 seconds passed... 34%, 1216 KB, 3370 KB/s, 0 seconds passed... 35%, 1248 KB, 3448 KB/s, 0 seconds passed

.. parsed-literal::

    ... 36%, 1280 KB, 3316 KB/s, 0 seconds passed... 36%, 1312 KB, 3390 KB/s, 0 seconds passed... 37%, 1344 KB, 3401 KB/s, 0 seconds passed... 38%, 1376 KB, 3472 KB/s, 0 seconds passed... 39%, 1408 KB, 3349 KB/s, 0 seconds passed... 40%, 1440 KB, 3418 KB/s, 0 seconds passed

.. parsed-literal::

    ... 41%, 1472 KB, 3426 KB/s, 0 seconds passed... 42%, 1504 KB, 3488 KB/s, 0 seconds passed... 43%, 1536 KB, 3372 KB/s, 0 seconds passed... 44%, 1568 KB, 3437 KB/s, 0 seconds passed... 45%, 1600 KB, 3446 KB/s, 0 seconds passed... 45%, 1632 KB, 3505 KB/s, 0 seconds passed

.. parsed-literal::

    ... 46%, 1664 KB, 3400 KB/s, 0 seconds passed... 47%, 1696 KB, 3457 KB/s, 0 seconds passed... 48%, 1728 KB, 3468 KB/s, 0 seconds passed... 49%, 1760 KB, 3520 KB/s, 0 seconds passed... 50%, 1792 KB, 3417 KB/s, 0 seconds passed... 51%, 1824 KB, 3472 KB/s, 0 seconds passed

.. parsed-literal::

    ... 52%, 1856 KB, 3478 KB/s, 0 seconds passed... 53%, 1888 KB, 3383 KB/s, 0 seconds passed... 54%, 1920 KB, 3437 KB/s, 0 seconds passed... 54%, 1952 KB, 3487 KB/s, 0 seconds passed... 55%, 1984 KB, 3495 KB/s, 0 seconds passed... 56%, 2016 KB, 3544 KB/s, 0 seconds passed

.. parsed-literal::

    ... 57%, 2048 KB, 3451 KB/s, 0 seconds passed... 58%, 2080 KB, 3500 KB/s, 0 seconds passed... 59%, 2112 KB, 3506 KB/s, 0 seconds passed... 60%, 2144 KB, 3554 KB/s, 0 seconds passed... 61%, 2176 KB, 3466 KB/s, 0 seconds passed

.. parsed-literal::

    ... 62%, 2208 KB, 3510 KB/s, 0 seconds passed... 63%, 2240 KB, 3513 KB/s, 0 seconds passed... 64%, 2272 KB, 3561 KB/s, 0 seconds passed... 64%, 2304 KB, 3478 KB/s, 0 seconds passed... 65%, 2336 KB, 3518 KB/s, 0 seconds passed... 66%, 2368 KB, 3521 KB/s, 0 seconds passed

.. parsed-literal::

    ... 67%, 2400 KB, 3444 KB/s, 0 seconds passed... 68%, 2432 KB, 3486 KB/s, 0 seconds passed... 69%, 2464 KB, 3495 KB/s, 0 seconds passed... 70%, 2496 KB, 3528 KB/s, 0 seconds passed

.. parsed-literal::

    ... 71%, 2528 KB, 3454 KB/s, 0 seconds passed... 72%, 2560 KB, 3496 KB/s, 0 seconds passed... 73%, 2592 KB, 3505 KB/s, 0 seconds passed... 73%, 2624 KB, 3515 KB/s, 0 seconds passed... 74%, 2656 KB, 3463 KB/s, 0 seconds passed... 75%, 2688 KB, 3502 KB/s, 0 seconds passed... 76%, 2720 KB, 3514 KB/s, 0 seconds passed... 77%, 2752 KB, 3523 KB/s, 0 seconds passed

.. parsed-literal::

    ... 78%, 2784 KB, 3474 KB/s, 0 seconds passed... 79%, 2816 KB, 3510 KB/s, 0 seconds passed... 80%, 2848 KB, 3522 KB/s, 0 seconds passed... 81%, 2880 KB, 3530 KB/s, 0 seconds passed

.. parsed-literal::

    ... 82%, 2912 KB, 3482 KB/s, 0 seconds passed... 82%, 2944 KB, 3518 KB/s, 0 seconds passed... 83%, 2976 KB, 3529 KB/s, 0 seconds passed... 84%, 3008 KB, 3535 KB/s, 0 seconds passed... 85%, 3040 KB, 3489 KB/s, 0 seconds passed... 86%, 3072 KB, 3523 KB/s, 0 seconds passed... 87%, 3104 KB, 3532 KB/s, 0 seconds passed

.. parsed-literal::

    ... 88%, 3136 KB, 3540 KB/s, 0 seconds passed... 89%, 3168 KB, 3498 KB/s, 0 seconds passed... 90%, 3200 KB, 3530 KB/s, 0 seconds passed... 91%, 3232 KB, 3539 KB/s, 0 seconds passed... 91%, 3264 KB, 3544 KB/s, 0 seconds passed

.. parsed-literal::

    ... 92%, 3296 KB, 3504 KB/s, 0 seconds passed... 93%, 3328 KB, 3536 KB/s, 0 seconds passed... 94%, 3360 KB, 3544 KB/s, 0 seconds passed... 95%, 3392 KB, 3550 KB/s, 0 seconds passed... 96%, 3424 KB, 3509 KB/s, 0 seconds passed... 97%, 3456 KB, 3540 KB/s, 0 seconds passed... 98%, 3488 KB, 3548 KB/s, 0 seconds passed

.. parsed-literal::

    ... 99%, 3520 KB, 3560 KB/s, 0 seconds passed... 100%, 3549 KB, 3521 KB/s, 1 seconds passed
    


Load the model
~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    # initialize OpenVINO runtime
    core = ov.Core()
    
    # read the network and corresponding weights from file
    model = core.read_model(model=model_path)
    
    # compile the model for the CPU (you can choose manually CPU, GPU etc.)
    # or let the engine choose the best available device (AUTO)
    compiled_model = core.compile_model(model=model, device_name="CPU")
    
    # get input node
    input_layer_ir = model.input(0)
    N, C, H, W = input_layer_ir.shape
    shape = (H, W)

Create functions for data processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

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
                cv2.putText(image, str(round(fps, 2)) + " fps", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 3) 
        return image

Get the test video
~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    video_path = 'https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/CEO%20Pat%20Gelsinger%20on%20Leading%20Intel.mp4'

How to improve the throughput of video processing
-------------------------------------------------

`back to top ⬆️ <#Table-of-contents:>`__

Below, we compare the performance of the synchronous and async-based
approaches:

Sync Mode (default)
~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

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

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    sync_fps = sync_api(source=video_path, flip=False, fps=30, use_popup=False, skip_first_frames=800)
    print(f"average throuput in sync mode: {sync_fps:.2f} fps")



.. image:: 115-async-api-with-output_files/115-async-api-with-output_15_0.png


.. parsed-literal::

    Source ended
    average throuput in sync mode: 43.51 fps


Async Mode
~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

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

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    async_fps = async_api(source=video_path, flip=False, fps=30, use_popup=False, skip_first_frames=800)
    print(f"average throuput in async mode: {async_fps:.2f} fps")



.. image:: 115-async-api-with-output_files/115-async-api-with-output_19_0.png


.. parsed-literal::

    Source ended
    average throuput in async mode: 74.01 fps


Compare the performance
~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    width = 0.4
    fontsize = 14
    
    plt.rc('font', size=fontsize)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    rects1 = ax.bar([0], sync_fps, width, color='#557f2d')
    rects2 = ax.bar([width], async_fps, width)
    ax.set_ylabel("frames per second")
    ax.set_xticks([0, width]) 
    ax.set_xticklabels(["Sync mode", "Async mode"])
    ax.set_xlabel("Higher is better")
    
    fig.suptitle('Sync mode VS Async mode')
    fig.tight_layout()
    
    plt.show()



.. image:: 115-async-api-with-output_files/115-async-api-with-output_21_0.png


``AsyncInferQueue``
-------------------

`back to top ⬆️ <#Table-of-contents:>`__

Asynchronous mode pipelines can be supported with the
```AsyncInferQueue`` <https://docs.openvino.ai/2023.3/openvino_docs_OV_UG_Python_API_exclusives.html#asyncinferqueue>`__
wrapper class. This class automatically spawns the pool of
``InferRequest`` objects (also called “jobs”) and provides
synchronization mechanisms to control the flow of the pipeline. It is a
simpler way to manage the infer request queue in Asynchronous mode.

Setting Callback
~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

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

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    frame_number = 0
    total_time = 0
    inferqueue(source=video_path, flip=False, fps=30, skip_first_frames=800)
    print(f"average throughput in async mode with async infer queue: {inferqueue_fps:.2f} fps")



.. image:: 115-async-api-with-output_files/115-async-api-with-output_27_0.png


.. parsed-literal::

    average throughput in async mode with async infer queue: 110.03 fps

