Image Classification Async Sample
=================================


.. meta::
   :description: Learn how to do inference of image classification models
                 using Asynchronous Inference Request API (Python, C++).


This sample demonstrates how to do inference of image classification models
using Asynchronous Inference Request API. Before using the sample, refer to the
following requirements:

- Models with only one input and output are supported.
- The sample accepts any file format supported by ``core.read_model``.
- To build the sample, use instructions available at :ref:`Build the Sample Applications <build-samples>`
  section in "Get Started with Samples" guide.


How It Works
####################

At startup, the sample application reads command-line parameters, prepares input data, and
loads a specified model and an image to the OpenVINO™ Runtime plugin.
The batch size of the model is set according to the number of read images. The
batch mode is an independent attribute on the asynchronous mode.
The asynchronous mode works efficiently with any batch size.

Then, the sample creates an inference request object and assigns completion callback
for it. In scope of the completion callback handling, the inference request is executed again.

After that, the application starts inference for the first infer request and waits
until 10th inference request execution has been completed.
The asynchronous mode might increase the throughput of the pictures.

When inference is done, the application outputs data to the standard output stream.
You can place labels in ``.labels`` file near the model to get pretty output.

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. scrollbox::

         .. doxygensnippet:: samples/python/classification_sample_async/classification_sample_async.py
            :language: python

   .. tab-item:: C++
      :sync: cpp

      .. scrollbox::

         .. doxygensnippet:: samples/cpp/classification_sample_async/main.cpp
            :language: cpp


You can see the explicit description of each sample step at
:doc:`Integration Steps <../../openvino-workflow/running-inference/integrate-openvino-with-your-application>`
section of "Integrate OpenVINO™ Runtime with Your Application" guide.


Running
####################

Run the application with the ``-h`` option to see the usage message:

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: console

         python classification_sample_async.py -h

      Usage message:

      .. code-block:: console

         usage: classification_sample_async.py [-h] -m MODEL -i INPUT [INPUT ...]
                                               [-d DEVICE]

         Options:
           -h, --help            Show this help message and exit.
           -m MODEL, --model MODEL
                                 Required. Path to an .xml or .onnx file with a trained
                                 model.
           -i INPUT [INPUT ...], --input INPUT [INPUT ...]
                                 Required. Path to an image file(s).
           -d DEVICE, --device DEVICE
                                 Optional. Specify the target device to infer on; CPU,
                                 GPU or HETERO: is acceptable. The sample
                                 will look for a suitable plugin for device specified.
                                 Default value is CPU.

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: console

         classification_sample_async -h

      Usage instructions:

      .. code-block:: console

         [ INFO ] OpenVINO Runtime version ......... <version>
         [ INFO ] Build ........... <build>

         classification_sample_async [OPTION]
         Options:

             -h                      Print usage instructions.
             -m "<path>"             Required. Path to an .xml file with a trained model.
             -i "<path>"             Required. Path to a folder with images or path to image files: a .ubyte file for LeNet and a .bmp file for other models.
             -d "<device>"           Optional. Specify the target device to infer on (the list of available devices is shown below). Default value is CPU. Use "-d HETERO:<comma_separated_devices_list>" format to specify the HETERO plugin. Sample will look for a suitable plugin for the device specified.

         Available target devices: <devices>


To run the sample, you need to specify a model and an image:

- You can get a model specific for your inference task from one of model
  repositories, such as TensorFlow Zoo, HuggingFace, or TensorFlow Hub.
- You can use images from the media files collection available at
  `the storage <https://storage.openvinotoolkit.org/data/test_data>`__.


.. note::

   - By default, OpenVINO™ Toolkit Samples and demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the sample or demo application or reconvert your model using model conversion API with ``reverse_input_channels`` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of :doc:`Embedding Preprocessing Computation <../../documentation/legacy-features/transition-legacy-conversion-api/legacy-conversion-api/[legacy]-setting-input-shapes>`.

   - Before running the sample with a trained model, make sure the model is converted to the intermediate representation (IR) format (\*.xml + \*.bin) using :doc:`model conversion API <../../documentation/legacy-features/transition-legacy-conversion-api/legacy-conversion-api>`.

   - The sample accepts models in ONNX format (.onnx) that do not require preprocessing.

   - The sample supports NCHW model layout only.

   - When you specify single options multiple times, only the last value will be used. For example, the ``-m`` flag:

     .. tab-set::

        .. tab-item:: Python
           :sync: python

           .. code-block:: console

              python classification_sample_async.py -m model.xml -m model2.xml

        .. tab-item:: C++
           :sync: cpp

           .. code-block:: console

              ./classification_sample_async -m model.xml -m model2.xml


Example
++++++++++++++++++++


1. Download a pre-trained model:
2. You can convert it by using:

   .. tab-set::

      .. tab-item:: Python
         :sync: python

         .. code-block:: python

            import openvino as ov

            ov_model = ov.convert_model('./models/alexnet')
            # or, when model is a Python model object
            ov_model = ov.convert_model(alexnet)

      .. tab-item:: CLI
         :sync: cli

         .. code-block:: console

            ovc ./models/alexnet

4. Perform inference of image files, using a model on a ``GPU``, for example:

   .. tab-set::

      .. tab-item:: Python
         :sync: python

         .. code-block:: console

            python classification_sample_async.py -m ./models/alexnet.xml -i ./test_data/images/banana.jpg ./test_data/images/car.bmp -d GPU

      .. tab-item:: C++
         :sync: cpp

         .. code-block:: console

            classification_sample_async -m ./models/googlenet-v1.xml -i ./images/dog.bmp -d GPU


Sample Output
####################

.. tab-set::

   .. tab-item:: Python
      :sync: python

      The sample application logs each step in a standard output stream and
      outputs top-10 inference results.

      .. code-block:: console

         [ INFO ] Creating OpenVINO Runtime Core
         [ INFO ] Reading the model: C:/test_data/models/alexnet.xml
         [ INFO ] Loading the model to the plugin
         [ INFO ] Starting inference in asynchronous mode
         [ INFO ] Image path: /test_data/images/banana.jpg
         [ INFO ] Top 10 results:
         [ INFO ] class_id probability
         [ INFO ] --------------------
         [ INFO ] 954      0.9707602
         [ INFO ] 666      0.0216788
         [ INFO ] 659      0.0032558
         [ INFO ] 435      0.0008082
         [ INFO ] 809      0.0004359
         [ INFO ] 502      0.0003860
         [ INFO ] 618      0.0002867
         [ INFO ] 910      0.0002866
         [ INFO ] 951      0.0002410
         [ INFO ] 961      0.0002193
         [ INFO ]
         [ INFO ] Image path: /test_data/images/car.bmp
         [ INFO ] Top 10 results:
         [ INFO ] class_id probability
         [ INFO ] --------------------
         [ INFO ] 656      0.5120340
         [ INFO ] 874      0.1142275
         [ INFO ] 654      0.0697167
         [ INFO ] 436      0.0615163
         [ INFO ] 581      0.0552262
         [ INFO ] 705      0.0304179
         [ INFO ] 675      0.0151660
         [ INFO ] 734      0.0151582
         [ INFO ] 627      0.0148493
         [ INFO ] 757      0.0120964
         [ INFO ]
         [ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool

   .. tab-item:: C++
      :sync: cpp

      The sample application logs each step in a standard output stream and
      outputs top-10 inference results.

      .. code-block:: console

         [ INFO ] OpenVINO Runtime version ......... <version>
         [ INFO ] Build ........... <build>
         [ INFO ]
         [ INFO ] Parsing input parameters
         [ INFO ] Files were added: 1
         [ INFO ]     /images/dog.bmp
         [ INFO ] Loading model files:
         [ INFO ] /models/googlenet-v1.xml
         [ INFO ] model name: GoogleNet
         [ INFO ]     inputs
         [ INFO ]         input name: data
         [ INFO ]         input type: f32
         [ INFO ]         input shape: {1, 3, 224, 224}
         [ INFO ]     outputs
         [ INFO ]         output name: prob
         [ INFO ]         output type: f32
         [ INFO ]         output shape: {1, 1000}
         [ INFO ] Read input images
         [ INFO ] Set batch size 1
         [ INFO ] model name: GoogleNet
         [ INFO ]     inputs
         [ INFO ]         input name: data
         [ INFO ]         input type: u8
         [ INFO ]         input shape: {1, 224, 224, 3}
         [ INFO ]     outputs
         [ INFO ]         output name: prob
         [ INFO ]         output type: f32
         [ INFO ]         output shape: {1, 1000}
         [ INFO ] Loading model to the device GPU
         [ INFO ] Create infer request
         [ INFO ] Start inference (asynchronous executions)
         [ INFO ] Completed 1 async request execution
         [ INFO ] Completed 2 async request execution
         [ INFO ] Completed 3 async request execution
         [ INFO ] Completed 4 async request execution
         [ INFO ] Completed 5 async request execution
         [ INFO ] Completed 6 async request execution
         [ INFO ] Completed 7 async request execution
         [ INFO ] Completed 8 async request execution
         [ INFO ] Completed 9 async request execution
         [ INFO ] Completed 10 async request execution
         [ INFO ] Completed async requests execution

         Top 10 results:

         Image /images/dog.bmp

         classid probability
         ------- -----------
         156     0.8935547
         218     0.0608215
         215     0.0217133
         219     0.0105667
         212     0.0018835
         217     0.0018730
         152     0.0018730
         157     0.0015745
         154     0.0012817
         220     0.0010099


Additional Resources
####################

- :doc:`Integrate the OpenVINO™ Runtime with Your Application <../../openvino-workflow/running-inference/integrate-openvino-with-your-application>`
- :doc:`Get Started with Samples <get-started-demos>`
- :doc:`Using OpenVINO™ Toolkit Samples <../openvino-samples>`
- :doc:`Convert a Model <../../documentation/legacy-features/transition-legacy-conversion-api/legacy-conversion-api>`
- `Image Classification Async Python Sample on Github <https://github.com/openvinotoolkit/openvino/blob/master/samples/python/classification_sample_async/README.md>`__
- `Image Classification Async C++ Sample on Github <https://github.com/openvinotoolkit/openvino/blob/master/samples/cpp/classification_sample_async/README.md>`__
