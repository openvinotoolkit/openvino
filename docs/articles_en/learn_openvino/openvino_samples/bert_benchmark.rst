.. {#openvino_sample_bert_benchmark}

Bert Benchmark Python Sample
============================


.. meta::
   :description: Learn how to estimate performance of a Bert model using Asynchronous Inference Request (Python) API.


This sample demonstrates how to estimate performance of a Bert model using Asynchronous 
Inference Request API. Unlike :doc:`demos <omz_demos>` this sample does not have 
configurable command line arguments. Feel free to modify sample's source code to 
try out different options.


How It Works
####################

The sample downloads a model and a tokenizer, exports the model to ONNX format, reads the 
exported model and reshapes it to enforce dynamic input shapes. Then, it compiles the 
resulting model, downloads a dataset and runs a benchmark on the dataset.

.. tab-set::

   .. tab-item:: Sample Code

      .. scrollbox::

         .. doxygensnippet:: samples/python/benchmark/bert_benchmark/bert_benchmark.py
            :language: python

   .. tab-item:: Python API

      The following API is used in the application:

      +--------------------------------+-------------------------------------------------+----------------------------------------------+
      | Feature                        | API                                             | Description                                  |
      +================================+=================================================+==============================================+
      | OpenVINO Runtime Version       | [openvino.runtime.get_version]                  | Get Openvino API version.                    |
      +--------------------------------+-------------------------------------------------+----------------------------------------------+
      | Basic Infer Flow               | [openvino.runtime.Core],                        | Common API to do inference: compile a model. |
      |                                | [openvino.runtime.Core.compile_model]           |                                              |
      +--------------------------------+-------------------------------------------------+----------------------------------------------+
      | Asynchronous Infer             | [openvino.runtime.AsyncInferQueue],             | Do asynchronous inference.                   |
      |                                | [openvino.runtime.AsyncInferQueue.start_async], |                                              |
      |                                | [openvino.runtime.AsyncInferQueue.wait_all]     |                                              |
      +--------------------------------+-------------------------------------------------+----------------------------------------------+
      | Model Operations               | [openvino.runtime.CompiledModel.inputs]         | Get inputs of a model.                       |
      +--------------------------------+-------------------------------------------------+----------------------------------------------+


You can see the explicit description of each sample step at 
:doc:`Integration Steps <openvino_docs_OV_UG_Integrate_OV_with_your_application>` 
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

* :doc:`Integrate the OpenVINO™ Runtime with Your Application <openvino_docs_OV_UG_Integrate_OV_with_your_application>`
* :doc:`Using OpenVINO Samples <openvino_docs_OV_UG_Samples_Overview>`
* :doc:`Convert a Model <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`
