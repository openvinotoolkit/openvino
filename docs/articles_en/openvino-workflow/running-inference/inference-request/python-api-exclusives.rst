OpenVINO™ Python API Exclusives
=================================


.. meta::
   :description: OpenVINO™ Runtime Python API includes additional features to
                 improve user experience and provide simple yet powerful tool
                 for Python users.


OpenVINO™ Runtime Python API offers additional features and helpers to enhance user experience. The main goal of Python API is to provide user-friendly and simple yet powerful tool for Python users.

Easier Model Compilation
########################

``CompiledModel`` can be easily created with the helper method. It hides the creation of ``Core`` and applies ``AUTO`` inference mode by default.


.. doxygensnippet:: docs/articles_en/assets/snippets/ov_python_exclusives.py
   :language: python
   :fragment: [auto_compilation]


Model/CompiledModel Inputs and Outputs
######################################

Besides functions aligned to C++ API, some of them have their Python counterparts or extensions. For example, ``Model`` and ``CompiledModel`` inputs/outputs can be accessed via properties.


.. doxygensnippet:: docs/articles_en/assets/snippets/ov_python_exclusives.py
   :language: python
   :fragment: [properties_example]


Refer to `Python API documentation <../../../api/ie_python_api/api.html>`__,
where helper functions or properties are available for different classes.


Working with Tensor
####################

Python API allows passing data as tensors. The ``Tensor`` object holds a copy of the data from the given array. The ``dtype`` of *numpy* arrays is converted to OpenVINO™ types automatically.


.. doxygensnippet:: docs/articles_en/assets/snippets/ov_python_exclusives.py
   :language: python
   :fragment: [tensor_basics]


Shared Memory Mode
++++++++++++++++++

``Tensor`` objects can share the memory with *numpy* arrays. By specifying the ``shared_memory`` argument, the ``Tensor`` object does not copy data. Instead, it has access to the memory of the *numpy* array.


.. doxygensnippet:: docs/articles_en/assets/snippets/ov_python_exclusives.py
   :language: python
   :fragment: [tensor_shared_mode]


Running Inference
####################

Python API supports extra calling methods to synchronous and asynchronous modes for inference.

All infer methods allow users to pass data as popular *numpy* arrays, gathered in either Python dicts or lists.


.. doxygensnippet:: docs/articles_en/assets/snippets/ov_python_exclusives.py
   :language: python
   :fragment: [passing_numpy_array]


Results from inference can be obtained in various ways:


.. doxygensnippet:: docs/articles_en/assets/snippets/ov_python_exclusives.py
   :language: python
   :fragment: [getting_results]


Synchronous Mode - Extended
+++++++++++++++++++++++++++

Python API provides different synchronous calls to infer model, which block the application execution. Additionally, these calls return results of inference:


.. doxygensnippet:: docs/articles_en/assets/snippets/ov_python_exclusives.py
   :language: python
   :fragment: [sync_infer]


Inference Results - OVDict
++++++++++++++++++++++++++


Synchronous calls return a special data structure called ``OVDict``. It can be compared to a "frozen dictionary". There are various ways of accessing the object's elements:


.. doxygensnippet:: docs/articles_en/assets/snippets/ov_python_exclusives.py
   :language: python
   :fragment: [ov_dict]


.. note::

   It is possible to convert ``OVDict`` to a native dictionary using the ``to_dict()`` method.


.. warning::

   Using ``to_dict()`` results in losing access via strings and integers. Additionally,
   it performs a shallow copy, thus any modifications may affect the original
   object as well.


AsyncInferQueue
++++++++++++++++++++

Asynchronous mode pipelines can be supported with a wrapper class called ``AsyncInferQueue``. This class automatically spawns the pool of ``InferRequest`` objects (also called "jobs") and provides synchronization mechanisms to control the flow of the pipeline.

Each job is distinguishable by a unique ``id``, which is in the range from 0 up to the number of jobs specified in the ``AsyncInferQueue`` constructor.

The ``start_async`` function call is not required to be synchronized - it waits for any available job if the queue is busy/overloaded. Every ``AsyncInferQueue`` code block should end with the ``wait_all`` function which provides the "global" synchronization of all jobs in the pool and ensure that access to them is safe.


.. doxygensnippet:: docs/articles_en/assets/snippets/ov_python_exclusives.py
   :language: python
   :fragment: [asyncinferqueue]

.. warning::

   ``InferRequest`` objects that can be acquired by iterating over a ``AsyncInferQueue`` object or by ``[id]`` guaranteed to work with read-only methods like getting tensors.
   Any mutating methods (e.g. start_async, set_callback) of a single request will put the parent AsyncInferQueue object in an invalid state.

Acquiring Results from Requests
-------------------------------

After the call to ``wait_all``, jobs and their data can be safely accessed. Acquiring a specific job with ``[id]`` will return the ``InferRequest`` object, which will result in seamless retrieval of the output data.


.. doxygensnippet:: docs/articles_en/assets/snippets/ov_python_exclusives.py
   :language: python
   :fragment: [asyncinferqueue_access]


Setting Callbacks
--------------------

Another feature of ``AsyncInferQueue`` is the ability to set callbacks. When callback is set, any job that ends inference calls upon the Python function. The callback function must have two arguments: one is the request that calls the callback, which provides the ``InferRequest`` API; the other is called "userdata", which provides the possibility of passing runtime values. Those values can be of any Python type and later used within the callback function.

The callback of ``AsyncInferQueue`` is uniform for every job. When executed, GIL is acquired to ensure safety of data manipulation inside the function.


.. doxygensnippet:: docs/articles_en/assets/snippets/ov_python_exclusives.py
   :language: python
   :fragment: [asyncinferqueue_set_callback]


Working with u1, u4 and i4 Element Types
++++++++++++++++++++++++++++++++++++++++

Since OpenVINO™ supports low precision element types, there are a few ways to handle them in Python.
To create an input tensor with such element types, you may need to pack your data in the new *numpy* array, with which the byte size matches the original input size:


.. doxygensnippet:: docs/articles_en/assets/snippets/ov_python_exclusives.py
   :language: python
   :fragment: [packing_data]


To extract low precision values from a tensor into the *numpy* array, you can use the following helper:


.. doxygensnippet:: docs/articles_en/assets/snippets/ov_python_exclusives.py
   :language: python
   :fragment: [unpacking]


Release of GIL
++++++++++++++++++++

Some functions in Python API release the Global Lock Interpreter (GIL) while running work-intensive code. This can help you achieve more parallelism in your application, using Python threads. For more information about GIL, refer to the `Python API documentation <../../../api/ie_python_api/api.html>`__.


.. doxygensnippet:: docs/articles_en/assets/snippets/ov_python_exclusives.py
   :language: python
   :fragment: [releasing_gil]


.. note:: While GIL is released, functions can still modify and/or operate on Python objects in C++. Hence, there is no reference counting. You should pay attention to thread safety in case sharing of these objects with another thread occurs. It might affect code only if multiple threads are spawned in Python.


List of Functions that Release the GIL
--------------------------------------

* openvino.runtime.AsyncInferQueue.start_async
* openvino.runtime.AsyncInferQueue.is_ready
* openvino.runtime.AsyncInferQueue.wait_all
* openvino.runtime.AsyncInferQueue.get_idle_request_id
* openvino.runtime.CompiledModel.create_infer_request
* openvino.runtime.CompiledModel.infer_new_request
* openvino.runtime.CompiledModel.__call__
* openvino.runtime.CompiledModel.export
* openvino.runtime.CompiledModel.get_runtime_model
* openvino.runtime.Core.compile_model
* openvino.runtime.Core.read_model
* openvino.runtime.Core.import_model
* openvino.runtime.Core.query_model
* openvino.runtime.Core.get_available_devices
* openvino.runtime.InferRequest.infer
* openvino.runtime.InferRequest.start_async
* openvino.runtime.InferRequest.wait
* openvino.runtime.InferRequest.wait_for
* openvino.runtime.InferRequest.get_profiling_info
* openvino.runtime.InferRequest.query_state
* openvino.runtime.Model.reshape
* openvino.preprocess.PrePostProcessor.build

