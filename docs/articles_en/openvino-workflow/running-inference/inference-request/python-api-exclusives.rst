Python API Exclusives
===============================================================================================


.. meta::
   :description: OpenVINO™ Python API includes additional features to
                 improve user experience and provide simple yet powerful tool
                 for Python users.


The main goal of OpenVINO™ Python API is to provide user-friendly and simple yet powerful tool.
The API offers additional features and helpers that enhance user experience.

Easy Model Compilation
###############################################################################################

``CompiledModel`` can be easily created with the helper method.
It hides the creation of ``Core`` and applies the ``AUTO`` inference mode by default:

.. doxygensnippet:: docs/articles_en/assets/snippets/ov_python_exclusives.py
   :language: python
   :fragment: [auto_compilation]


Model/CompiledModel Input and Output
###############################################################################################

Some of the C++ functions have their counterparts or extensions in Python API.
For example, inputs/outputs for the ``Model`` and ``CompiledModel`` functions can be
accessed via properties:

.. doxygensnippet:: docs/articles_en/assets/snippets/ov_python_exclusives.py
   :language: python
   :fragment: [properties_example]


Refer to the `Python API documentation <../../../api/ie_python_api/api.html>`__,
describing helper functions or properties available for different classes.

Working with Tensor
###############################################################################################

Python API enables passing data as tensors. A ``Tensor`` object holds a copy of data
from a given *numpy* array, where the ``dtype`` is automatically converted to OpenVINO types:

.. doxygensnippet:: docs/articles_en/assets/snippets/ov_python_exclusives.py
   :language: python
   :fragment: [tensor_basics]


Shared Memory Mode
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

``Tensor`` objects can share the memory with *numpy* arrays. When you specify the
``shared_memory`` argument, the ``Tensor`` object will not copy data. Instead, it has
access to the memory of the *numpy* array.

.. doxygensnippet:: docs/articles_en/assets/snippets/ov_python_exclusives.py
   :language: python
   :fragment: [tensor_shared_mode]


Running Inference
###############################################################################################

Python API supports additional call methods to synchronous and asynchronous modes
for inference. All inference methods enable users to pass data as popular *numpy* arrays,
gathered in either Python dicts or lists:

.. doxygensnippet:: docs/articles_en/assets/snippets/ov_python_exclusives.py
   :language: python
   :fragment: [passing_numpy_array]


You can get the inference results in various ways:

.. doxygensnippet:: docs/articles_en/assets/snippets/ov_python_exclusives.py
   :language: python
   :fragment: [getting_results]


Synchronous Mode - Extended
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

You can run model inference in Python API, using different synchronous calls, which block the
execution of application. You can also use these calls to get the inference results:

.. doxygensnippet:: docs/articles_en/assets/snippets/ov_python_exclusives.py
   :language: python
   :fragment: [sync_infer]


.. _inference_results_ovdict:

Inference Results - OVDict
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Synchronous calls return an ``OVDict`` object of a special data structure that can be compared
to a “frozen dictionary”. You can use several different ways to access its elements:

.. doxygensnippet:: docs/articles_en/assets/snippets/ov_python_exclusives.py
   :language: python
   :fragment: [ov_dict]


.. note::

   It is possible to convert ``OVDict`` to a native dictionary, using
   the ``to_dict()`` method.

.. warning::

   Using ``to_dict()`` results in losing access via strings and integers. Additionally,
   it performs a shallow copy, so any modifications may also affect the original object.

AsyncInferQueue
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Asynchronous mode pipelines can be supported with the ``AsyncInferQueue`` wrapper class.
This class automatically generates the pool of ``InferRequest``
objects (also called “jobs”) and provides synchronization mechanisms to control
the flow of the pipeline.

Each job is distinguishable by a unique ``id``, ranged from 0 to the specified number
in the ``AsyncInferQueue`` constructor.

The ``start_async`` function call does not require synchronization - it waits for
any available job if the queue is busy/overloaded. Every ``AsyncInferQueue`` code
block should end with the ``wait_all`` function, which provides “global"
synchronization of all jobs in the pool and ensures safe access to them.

.. doxygensnippet:: docs/articles_en/assets/snippets/ov_python_exclusives.py
   :language: python
   :fragment: [asyncinferqueue]


.. warning::

   * ``InferRequest`` objects acquired by iterating over an ``AsyncInferQueue`` object
     or by ``[id]`` are guaranteed to work with read-only methods like ``get_tensor()``.
   * Any mutating methods of a single inference request, for example, ``start_async``,
     ``set_callback`` will put the parent ``AsyncInferQueue`` object in an invalid state.

Acquiring Results from Requests
-----------------------------------------------------------------------------------------------

After the call to ``wait_all``, jobs and their data can be safely accessed.
Acquiring a specific job with ``[id]`` will return the ``InferRequest`` object,
which will result in seamless retrieval of the output data:

.. doxygensnippet:: docs/articles_en/assets/snippets/ov_python_exclusives.py
   :language: python
   :fragment: [asyncinferqueue_access]


Setting Callbacks
-----------------------------------------------------------------------------------------------

Another feature of ``AsyncInferQueue`` is the ability to set callbacks. When
callback is set, any job that ends inference calls upon the Python function.
The callback function must have two arguments: one is the request that calls the
callback, which provides the ``InferRequest`` API; the other is called "userdata",
which provides the possibility of passing runtime values. Those values can be of any
Python type and later used within the callback function.

The callback of ``AsyncInferQueue`` is uniform for every job. When executed, GIL is
acquired to ensure safety of data manipulation inside the function:

.. doxygensnippet:: docs/articles_en/assets/snippets/ov_python_exclusives.py
   :language: python
   :fragment: [asyncinferqueue_set_callback]


u1, u4 and i4 Low Precision Element Types
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Using Python API, you can handle low precision element types.
For instance, to create an input tensor with such element types, you can
pack data in a new *numpy* array, where the byte size matches the original input size:

.. doxygensnippet:: docs/articles_en/assets/snippets/ov_python_exclusives.py
   :language: python
   :fragment: [packing_data]


To extract low precision values from a tensor to the *numpy* array, you can use
the following helper:

.. doxygensnippet:: docs/articles_en/assets/snippets/ov_python_exclusives.py
   :language: python
   :fragment: [unpacking]


Release of GIL
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Python threads release the the Global Lock Interpreter (GIL) while running
work-intensive code. You can use it to introduce more parallelism in your application:

.. doxygensnippet:: docs/articles_en/assets/snippets/ov_python_exclusives.py
   :language: python
   :fragment: [releasing_gil]


.. note::

   While GIL is released, functions can still modify and/or operate on Python objects
   in C++. Hence, there is no reference counting. You should pay attention to thread
   safety when sharing the objects with another thread. It might affect the
   code only when multiple threads are created in Python.

For more information about GIL, refer to the
`Python API documentation <../../../api/ie_python_api/api.html>`__.

List of Functions that Release the GIL
-----------------------------------------------------------------------------------------------

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
