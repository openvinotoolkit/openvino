Bert Benchmark Python Sample
============================


.. meta::
   :description: Learn how to estimate performance of a Bert model using Asynchronous Inference Request (Python) API.


This sample demonstrates how to estimate performance of a Bert model using Asynchronous
Inference Request API. This sample does not have
configurable command line arguments. Feel free to modify sample's source code to
try out different options.


How It Works
####################

The sample downloads a model and a tokenizer, exports the model to ONNX format, reads the
exported model and reshapes it to enforce dynamic input shapes. Then, it compiles the
resulting model, downloads a dataset and runs a benchmark on the dataset.

.. scrollbox::

   .. doxygensnippet:: samples/python/benchmark/bert_benchmark/bert_benchmark.py
      :language: python


You can see the explicit description of each sample step at
:doc:`Integration Steps <../../../openvino-workflow/running-inference>`
section of "Integrate OpenVINO™ Runtime with Your Application" guide.

Running
####################

1. Install the ``openvino`` Python package:

   .. code-block:: console

      python -m pip install openvino


2. Install packages from ``requirements.txt``:

   .. code-block:: console

      python -m pip install -r requirements.txt

3. Run the sample

   .. code-block:: console

      python bert_benchmark.py


Sample Output
####################

The sample outputs how long it takes to process a dataset.

Additional Resources
####################

- :doc:`Integrate the OpenVINO™ Runtime with Your Application <../../../openvino-workflow/running-inference>`
- :doc:`Get Started with Samples <get-started-demos>`
- :doc:`Using OpenVINO Samples <../openvino-samples>`
- :doc:`Convert a Model <../../../openvino-workflow/model-preparation/convert-model-to-ir>`
- `Bert Benchmark Python Sample on Github <https://github.com/openvinotoolkit/openvino/blob/master/samples/python/benchmark/bert_benchmark/README.md>`__
