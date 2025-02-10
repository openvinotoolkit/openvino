Model Caching Overview
======================


.. meta::
   :description: Enabling model caching to export compiled model
                 automatically and reusing it can significantly
                 reduce duration of model compilation on application startup.


As described in :doc:`Integrate OpenVINO™ with Your Application <../../integrate-openvino-with-your-application>`,
a common workflow consists of the following steps:

1. | **Create a Core object**:
   |   First step to manage available devices and read model objects
2. | **Read the Intermediate Representation**:
   |   Read an Intermediate Representation file into the `ov::Model <https://docs.openvino.ai/2025/api/c_cpp_api/classov_1_1_model.html>`__ object
3. | **Prepare inputs and outputs**:
   |   If needed, manipulate precision, memory layout, size or color format
4. | **Set configuration**:
   |   Add device-specific loading configurations to the device
5. | **Compile and Load Network to device**:
   |   Use the `ov::Core::compile_model() <https://docs.openvino.ai/2025/api/c_cpp_api/classov_1_1_core.html>`__ method with a specific device
6. | **Set input data**:
   |   Specify input tensor
7. | **Execute**:
   |   Carry out inference and process results

Step 5 can potentially perform several time-consuming device-specific optimizations and network compilations.
To reduce the resulting delays at application startup, you can use Model Caching. It exports the compiled model
automatically and reuses it to significantly reduce the model compilation time.

.. important::

   Not all devices support import/export of models. They will perform normally but will not
   enable the compilation stage speed-up.


Set configuration options
+++++++++++++++++++++++++++++++++++++++++++++++++++++

| Use the ``device_name`` option to specify the inference device.
| Specify ``cache_dir`` to enable model caching.

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_caching.py
         :language: py
         :fragment: [ov:caching:part0]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_caching.cpp
         :language: cpp
         :fragment: [ov:caching:part0]


If the specified device supports import/export of models,
a cached blob file: ``.cl_cache`` (GPU) or  ``.blob`` (CPU) is automatically
created inside the ``/path/to/cache/dir`` folder.
If the device does not support import/export of models, the cache is not
created and no error is thrown.

Note that the first ``compile_model`` operation takes slightly more time,
as the cache needs to be created - the compiled blob is saved into a file:

.. image:: ../../../../assets/images/caching_enabled.svg


Use optimized methods
+++++++++++++++++++++++++++++++++++++++++++++++++++

Applications do not always require an initial customization of inputs and
outputs, as they can call ``model = core.read_model(...)``, then ``core.compile_model(model, ..)``,
which can be further optimized. Thus, the model can be compiled conveniently in a single call,
skipping the read step:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_caching.py
         :language: py
         :fragment: [ov:caching:part1]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_caching.cpp
         :language: cpp
         :fragment: [ov:caching:part1]


The total load time is even shorter, when model caching is enabled and ``read_model`` is optimized as well.

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_caching.py
         :language: py
         :fragment: [ov:caching:part2]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_caching.cpp
         :language: cpp
         :fragment: [ov:caching:part2]


.. image:: ../../../../assets/images/caching_times.svg

Advanced Examples
++++++++++++++++++++

Enabling model caching has no effect when the specified device does not support
import/export of models. To check in advance if a particular device supports
model caching, use the following code in your application:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_caching.py
         :language: py
         :fragment: [ov:caching:part3]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_caching.cpp
         :language: cpp
         :fragment: [ov:caching:part3]

Enable cache encryption
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

If model caching is enabled in the CPU Plugin, set the "cache_encryption_callbacks"
config option to encrypt the model while caching it and decrypt it when
loading it from the cache. Currently, this property can be set only in ``compile_model``.

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_caching.py
         :language: py
         :fragment: [ov:caching:part4]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_caching.cpp
         :language: cpp
         :fragment: [ov:caching:part4]

Full encryption only works when the ``CacheMode`` property is set to ``OPTIMIZE_SIZE``.

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_caching.py
         :language: py
         :fragment: [ov:caching:part5]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_caching.cpp
         :language: cpp
         :fragment: [ov:caching:part5]

.. important::

   Currently, encryption is supported only by the CPU and GPU plugins. Enabling this
   feature for other HW plugins will not encrypt/decrypt model topology in the
   cache and will not affect performance.
