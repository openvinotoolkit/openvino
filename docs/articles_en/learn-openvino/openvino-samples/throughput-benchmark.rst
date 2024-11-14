Throughput Benchmark Sample
===========================


.. meta::
   :description: Learn how to estimate performance of a model using Asynchronous Inference Request API in throughput mode (Python, C++).


This sample demonstrates how to estimate performance of a model using Asynchronous
Inference Request API in throughput mode. Unlike `demos <https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/README.md>`__ this sample
does not have other configurable command-line arguments. Feel free to modify sample's
source code to try out different options.

The reported results may deviate from what :doc:`benchmark_app <benchmark-tool>`
reports. One example is model input precision for computer vision tasks. benchmark_app
sets ``uint8``, while the sample uses default model precision which is usually ``float32``.

Before using the sample, refer to the following requirements:

- The sample accepts any file format supported by ``core.read_model``.
- The sample has been validated with: `yolo-v3-tf <https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/yolo-v3-tf/README.md>`__,
  `face-detection-0200 <https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/face-detection-0200/README.md>`__ models.
- To build the sample, use instructions available at :ref:`Build the Sample Applications <build-samples>`
  section in "Get Started with Samples" guide.

How It Works
####################

The sample compiles a model for a given device, randomly generates input data,
performs asynchronous inference multiple times for a given number of seconds.
Then, it processes and reports performance results.

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. scrollbox::

         .. doxygensnippet:: samples/python/benchmark/throughput_benchmark/throughput_benchmark.py
            :language: python

   .. tab-item:: C++
      :sync: cpp

      .. scrollbox::

         .. doxygensnippet:: samples/cpp/benchmark/throughput_benchmark/main.cpp
            :language: cpp


You can see the explicit description of each sample step at
:doc:`Integration Steps <../../openvino-workflow/running-inference/integrate-openvino-with-your-application>`
section of "Integrate OpenVINO™ Runtime with Your Application" guide.

Running
####################

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: console

         python throughput_benchmark.py <path_to_model> <device_name>(default: CPU)


   .. tab-item:: C++
      :sync: cpp

      .. code-block:: console

         throughput_benchmark <path_to_model> <device_name>(default: CPU)


To run the sample, you need to specify a model. You can get a model specific for
your inference task from one of model repositories, such as TensorFlow Zoo, HuggingFace, or TensorFlow Hub.

Example
++++++++++++++++++++

1. Download a pre-trained model.
2. You can convert it by using:

   .. tab-set::

      .. tab-item:: Python
         :sync: python

         .. code-block:: python

            import openvino as ov

            ov_model = ov.convert_model('./models/googlenet-v1')
            # or, when model is a Python model object
            ov_model = ov.convert_model(googlenet-v1)

      .. tab-item:: CLI
         :sync: cli

         .. code-block:: console

            ovc ./models/googlenet-v1


3. Perform benchmarking, using the ``googlenet-v1`` model on a ``CPU``:

   .. tab-set::

      .. tab-item:: Python
         :sync: python

         .. code-block:: console

            python throughput_benchmark.py ./models/googlenet-v1.xml

      .. tab-item:: C++
         :sync: cpp

         .. code-block:: console

            throughput_benchmark ./models/googlenet-v1.xml


Sample Output
####################

.. tab-set::

   .. tab-item:: Python
      :sync: python

      The application outputs performance results.

      .. code-block:: console

         [ INFO ] OpenVINO:
         [ INFO ] Build ................................. <version>
         [ INFO ] Count:          2817 iterations
         [ INFO ] Duration:       10012.65 ms
         [ INFO ] Latency:
         [ INFO ]     Median:     13.80 ms
         [ INFO ]     Average:    14.10 ms
         [ INFO ]     Min:        8.35 ms
         [ INFO ]     Max:        28.38 ms
         [ INFO ] Throughput: 281.34 FPS

   .. tab-item:: C++
      :sync: cpp

      The application outputs performance results.

      .. code-block:: console

         [ INFO ] OpenVINO:
         [ INFO ] Build ................................. <version>
         [ INFO ] Count:      1577 iterations
         [ INFO ] Duration:   15024.2 ms
         [ INFO ] Latency:
         [ INFO ]        Median:     38.02 ms
         [ INFO ]        Average:    38.08 ms
         [ INFO ]        Min:        25.23 ms
         [ INFO ]        Max:        49.16 ms
         [ INFO ] Throughput: 104.96 FPS


Additional Resources
####################

- :doc:`Integrate the OpenVINO™ Runtime with Your Application <../../openvino-workflow/running-inference/integrate-openvino-with-your-application>`
- :doc:`Get Started with Samples <get-started-demos>`
- :doc:`Using OpenVINO Samples <../openvino-samples>`
- :doc:`Convert a Model <../../documentation/legacy-features/transition-legacy-conversion-api/legacy-conversion-api>`
- `Throughput Benchmark Python Sample on Github <https://github.com/openvinotoolkit/openvino/blob/master/samples/python/benchmark/throughput_benchmark/README.md>`__
- `Throughput Benchmark C++ Sample on Github <https://github.com/openvinotoolkit/openvino/blob/master/samples/cpp/benchmark/throughput_benchmark/README.md>`__
