.. {#openvino_inference_engine_ie_bridges_python_sample_bert_benchmark_README}

Bert Benchmark Python Sample
============================


.. meta::
   :description: Learn how to estimate performance of a Bert model using Asynchronous Inference Request (Python) API.


This sample demonstrates how to estimate performance of a Bert model using Asynchronous Inference Request API. Unlike :doc:`demos <omz_demos>` this sample doesn't have configurable command line arguments. Feel free to modify sample's source code to try out different options.

The following Python API is used in the application:

.. tab-set::

   .. tab-item:: Python API 

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
   
   .. tab-item:: Sample Code

      .. doxygensnippet:: samples/python/benchmark/bert_benchmark/bert_benchmark.py
         :language: python

How It Works
####################

The sample downloads a model and a tokenizer, export the model to onnx, reads the exported model and reshapes it to enforce dynamic input shapes, compiles the resulting model, downloads a dataset and runs benchmarking on the dataset.

You can see the explicit description of
each sample step at :doc:`Integration Steps <openvino_docs_OV_UG_Integrate_OV_with_your_application>` section of "Integrate OpenVINO™ Runtime with Your Application" guide.

Running
####################

Install the ``openvino`` Python package:

.. code-block:: sh

   python -m pip install openvino


Install packages from ``requirements.txt``:

.. code-block:: sh

   python -m pip install -r requirements.txt


Run the sample

.. code-block:: sh

   python bert_benchmark.py


Sample Output
####################

The sample outputs how long it takes to process a dataset.

See Also
####################

* :doc:`Integrate the OpenVINO™ Runtime with Your Application <openvino_docs_OV_UG_Integrate_OV_with_your_application>`
* :doc:`Using OpenVINO Samples <openvino_docs_OV_UG_Samples_Overview>`
* :doc:`Model Downloader <omz_tools_downloader>`
* :doc:`Convert a Model <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`

