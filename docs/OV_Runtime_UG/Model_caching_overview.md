# Model Caching Overview {#openvino_docs_OV_UG_Model_caching_overview}

@sphinxdirective

As described in the :doc:`Integrate OpenVINO™ with Your Application <openvino_docs_OV_UG_Integrate_OV_with_your_application>`, a common application flow consists of the following steps:

1. **Create a Core object**: First step to manage available devices and read model objects

2. **Read the Intermediate Representation**: Read an Intermediate Representation file into an object of the `ov::Model <classov_1_1Model.html#doxid-classov-1-1-model>`__

3. **Prepare inputs and outputs**: If needed, manipulate precision, memory layout, size or color format

4. **Set configuration**: Pass device-specific loading configurations to the device

5. **Compile and Load Network to device**: Use the `ov::Core::compile_model() <classov_1_1Core.html#doxid-classov-1-1-core-1a46555f0803e8c29524626be08e7f5c5a>`__ method with a specific device

6. **Set input data**: Specify input tensor

7. **Execute**: Carry out inference and process results

Step 5 can potentially perform several time-consuming device-specific optimizations and network compilations,
and such delays can lead to a bad user experience on application startup. To avoid this, some devices offer
import/export network capability, and it is possible to either use the :doc:`Compile tool <openvino_inference_engine_tools_compile_tool_README>`
or enable model caching to export compiled model automatically. Reusing cached model can significantly reduce compile model time.

Set "cache_dir" config option to enable model caching
+++++++++++++++++++++++++++++++++++++++++++++++++++++

To enable model caching, the application must specify a folder to store cached blobs, which is done like this:


.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_caching.cpp
         :language: cpp
         :fragment: [ov:caching:part0]

.. tab:: Python

      .. doxygensnippet:: docs/snippets/ov_caching.py
         :language: python
         :fragment: [ov:caching:part0]


With this code, if the device specified by ``device_name`` supports import/export model capability, a cached blob is automatically created inside the ``/path/to/cache/dir`` folder.
If the device does not support import/export capability, cache is not created and no error is thrown.

Depending on your device, total time for compiling model on application startup can be significantly reduced.
Also note that the very first ``compile_model`` (when cache is not yet created) takes slightly longer time to "export" the compiled blob into a cache file:


.. image:: _static/images/caching_enabled.svg


Even faster: use compile_model(modelPath)
+++++++++++++++++++++++++++++++++++++++++

In some cases, applications do not need to customize inputs and outputs every time. Such application always
call ``model = core.read_model(...)``, then ``core.compile_model(model, ..)`` and it can be further optimized.
For these cases, there is a more convenient API to compile the model in a single call, skipping the read step:


.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_caching.cpp
         :language: cpp
         :fragment: [ov:caching:part1]

.. tab:: Python

      .. doxygensnippet:: docs/snippets/ov_caching.py
         :language: python
         :fragment: [ov:caching:part1]


With model caching enabled, total load time is even smaller, if ``read_model`` is optimized as well.


.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_caching.cpp
         :language: cpp
         :fragment: [ov:caching:part2]

.. tab:: Python

      .. doxygensnippet:: docs/snippets/ov_caching.py
         :language: python
         :fragment: [ov:caching:part2]


.. image:: _static/images/caching_times.svg

Advanced Examples
++++++++++++++++++++

Not every device supports network import/export capability. For those that don't, enabling caching has no effect.
To check in advance if a particular device supports model caching, your application can use the following code:


.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_caching.cpp
         :language: cpp
         :fragment: [ov:caching:part3]

.. tab:: Python

      .. doxygensnippet:: docs/snippets/ov_caching.py
         :language: python
         :fragment: [ov:caching:part3]


.. note::

   For GPU, model caching is currently implemented as a preview feature. Before it is fully supported, kernel caching can be used in the same manner: by setting the CACHE_DIR configuration key to a folder where the cache should be stored (see the :doc:`GPU plugin documentation <openvino_docs_OV_UG_supported_plugins_GPU>`). To activate the preview feature of model caching, set the OV_GPU_CACHE_MODEL environment variable to 1.

@endsphinxdirective
