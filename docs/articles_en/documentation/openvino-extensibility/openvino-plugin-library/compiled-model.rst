Compiled Model
==============


.. meta::
   :description: Use the ov::CompiledModel class as the base class for a compiled
                 model and to create an arbitrary number of ov::InferRequest objects.

``ov::CompiledModel`` class functionality:

* Compile an ``ov::Model`` instance to a backend specific graph representation
* Create an arbitrary number of ``ov::InferRequest`` objects
* Hold some common resources shared between different instances of ``ov::InferRequest``. For example:

  * ``ov::ICompiledModel::m_task_executor`` task executor to implement asynchronous execution
  * ``ov::ICompiledModel::m_callback_executor`` task executor to run an asynchronous inference request callback in a separate thread

CompiledModel Class
###################

OpenVINO Plugin API provides the interface ``ov::ICompiledModel`` which should be used as a base class
for a compiled model. Based on that, a declaration of an compiled model class can look as follows:

.. doxygensnippet:: src/plugins/template/src/compiled_model.hpp
   :language: cpp
   :fragment: [compiled_model:header]


Class Fields
++++++++++++

The example class has several fields:

* ``m_request_id`` - Tracks a number of created inference requests, which is used to distinguish
  different inference requests during profiling via the Intel® Instrumentation and Tracing Technology (ITT) library.
* ``m_cfg`` - Defines a configuration a compiled model was compiled with.
* ``m_model`` - Keeps a reference to transformed ``ov::Model`` which is used in OpenVINO reference
  backend computations. Note, in case of other backends with backend specific graph representation
  ``m_model`` has different type and represents backend specific graph or just a set of computational kernels to perform an inference.
* ``m_loaded_from_cache`` - Allows to understand that model was loaded from cache.

CompiledModel Constructor
+++++++++++++++++++++++++

This constructor accepts a generic representation of a model as an ov::Model and is compiled into a backend specific device graph:

.. doxygensnippet:: src/plugins/template/src/compiled_model.cpp
   :language: cpp
   :fragment: [compiled_model:ctor]

The implementation ``compile_model()`` is fully device-specific.

compile_model()
+++++++++++++++

The function accepts a const shared pointer to ``ov::Model`` object and applies OpenVINO passes
using ``transform_model()`` function, which defines plugin-specific conversion pipeline. To support
low precision inference, the pipeline can include Low Precision Transformations. These
transformations are usually hardware specific. You can find how to use and configure Low Precisions
Transformations in :doc:`Low Precision Transformations <advanced-guides/low-precision-transformations>` guide.

.. doxygensnippet:: src/plugins/template/src/compiled_model.cpp
   :language: cpp
   :fragment: [compiled_model:compile_model]


.. note::

   After all these steps, the backend specific graph is ready to create inference requests and perform inference.

export_model()
++++++++++++++

The implementation of the method should write all data to the ``model_stream``, which is required
to import a backend specific graph later in the ``Plugin::import_model`` method:

.. doxygensnippet:: src/plugins/template/src/compiled_model.cpp
   :language: cpp
   :fragment: [compiled_model:export_model]

create_sync_infer_request()
+++++++++++++++++++++++++++

The method creates an synchronous inference request and returns it.

.. doxygensnippet:: src/plugins/template/src/compiled_model.cpp
   :language: cpp
   :fragment: [compiled_model:create_sync_infer_request]

While the public OpenVINO API has a single interface for inference request, which can be executed
in synchronous and asynchronous modes, a plugin library implementation has two separate classes:

* :doc:`Synchronous inference request <synch-inference-request>`, which defines pipeline stages and runs them synchronously in the ``infer`` method.

* :doc:`Asynchronous inference request <asynch-inference-request>`, which is a wrapper for a synchronous
  inference request and can run a pipeline asynchronously. Depending on a device pipeline structure,
  it can have one or several stages:

  * For single-stage pipelines, there is no need to define this method and create a class derived
    from ``ov::IAsyncInferRequest``. For single stage pipelines, a default implementation of this
    method creates ``ov::IAsyncInferRequest`` wrapping a synchronous inference request and runs
    it asynchronously in the ``m_request_executor`` executor.
  * For pipelines with multiple stages, such as performing some preprocessing on host, uploading
    input data to a device, running inference on a device, or downloading and postprocessing output
    data, schedule stages on several task executors to achieve better device use and performance.
    You can do it by creating a sufficient number of inference requests running in parallel.
    In this case, device stages of different inference requests are overlapped with preprocessing
    and postprocessing stage giving better performance.

.. important::

   It is up to you to decide how many task executors you need to optimally execute a device pipeline.


create_infer_request()
++++++++++++++++++++++

The method creates an asynchronous inference request and returns it.

.. doxygensnippet:: src/plugins/template/src/compiled_model.cpp
   :language: cpp
   :fragment: [compiled_model:create_infer_request]

get_property()
++++++++++++++

Returns a current value for a property with the name ``name``. The method extracts configuration values a compiled model is compiled with.

.. doxygensnippet:: src/plugins/template/src/compiled_model.cpp
   :language: cpp
   :fragment: [compiled_model:get_property]

This function is the only way to get configuration values when a model is imported and compiled by other developers and tools.

set_property()
++++++++++++++

The methods allows to set compiled model specific properties.

.. doxygensnippet:: src/plugins/template/src/compiled_model.cpp
   :language: cpp
   :fragment: [compiled_model:set_property]

get_runtime_model()
+++++++++++++++++++

The methods returns the runtime model with backend specific information.

.. doxygensnippet:: src/plugins/template/src/compiled_model.cpp
   :language: cpp
   :fragment: [compiled_model:get_runtime_model]

The next step in plugin library implementation is the :doc:`Synchronous Inference Request <synch-inference-request>` class.

