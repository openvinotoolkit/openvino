Model Caching Overview
======================


.. meta::
   :description: Enabling model caching to export compiled model
                 automatically and reusing it can significantly
                 reduce duration of model compilation on application startup.


As described in :doc:`Integrate OpenVINO™ with Your Application <../../integrate-openvino-with-your-application>`,
a common application flow consists of the following steps:

1. | **Create a Core object**:
   |   First step to manage available devices and read model objects
2. | **Read the Intermediate Representation**:
   |   Read an Intermediate Representation file into an object of the `ov::Model <https://docs.openvino.ai/2024/api/c_cpp_api/classov_1_1_model.html>`__
3. | **Prepare inputs and outputs**:
   |   If needed, manipulate precision, memory layout, size or color format
4. | **Set configuration**:
   |   Pass device-specific loading configurations to the device
5. | **Compile and Load Network to device**:
   |   Use the `ov::Core::compile_model() <https://docs.openvino.ai/2024/api/c_cpp_api/classov_1_1_core.html>`__ method with a specific device
6. | **Set input data**:
   |   Specify input tensor
7. | **Execute**:
   |   Carry out inference and process results

Step 5 can potentially perform several time-consuming device-specific optimizations and network compilations.
To reduce the resulting delays at application startup, you can use Model Caching. It exports the compiled model
automatically and reuses it to significantly reduce the model compilation time.

.. important::

   Not all devices support the network import/export feature. They will perform normally but will not
   enable the compilation stage speed-up.


Set "cache_dir" config option to enable model caching
+++++++++++++++++++++++++++++++++++++++++++++++++++++

To enable model caching, the application must specify a folder to store the cached blobs:

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


With this code, if the device specified by ``device_name`` supports import/export model capability,
a cached blob (the ``.cl_cache`` and ``.blob`` file for GPU and CPU respectively) is automatically
created inside the ``/path/to/cache/dir`` folder.
If the device does not support the import/export capability, cache is not created and no error is thrown.

Note that the first ``compile_model`` operation takes slightly longer, as the cache needs to be created -
the compiled blob is saved into a cache file:

.. image:: ../../../../assets/images/caching_enabled.svg


Make it even faster: use compile_model(modelPath)
+++++++++++++++++++++++++++++++++++++++++++++++++++

In some cases, applications do not need to customize inputs and outputs every time. Such application always
call ``model = core.read_model(...)``, then ``core.compile_model(model, ..)``, which can be further optimized.
For these cases, there is a more convenient API to compile the model in a single call, skipping the read step:

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


With model caching enabled, the total load time is even shorter, if ``read_model`` is optimized as well.

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

Not every device supports the network import/export capability. For those that don't, enabling caching has no effect.
To check in advance if a particular device supports model caching, your application can use the following code:

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

Set "cache_encryption_callbacks" config option to enable cache encryption
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

If model caching is enabled, the model topology can be encrypted when saving to the cache and decrypted when loading from the cache. This property can currently be set only in ``compile_model``.

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

.. important::

   Currently, this property is supported only by the CPU plugin. For other HW plugins, setting this property will not encrypt/decrypt the model topology in cache and will not affect performance. 
