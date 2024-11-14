Query Device Properties - Configuration
=======================================

.. meta::
   :description: Learn the details on the process of querying different device
                 properties and configuration values at runtime.


This article provides an overview of how to query different device properties
and configuration values at runtime.

OpenVINO runtime has two types of properties:

- **Read only properties** which provide information about devices, such as device
  name and execution capabilities, and information about configuration values
  used to compile the model - ``ov::CompiledModel``.
- **Mutable properties**, primarily used to configure the ``ov::Core::compile_model``
  process and affect final inference on a specific set of devices. Such properties
  can be set globally per device via ``ov::Core::set_property`` or locally for a
  particular model in the ``ov::Core::compile_model`` and ``ov::Core::query_model``
  calls.


An OpenVINO property is represented as a named constexpr variable with a given string
name and a type. The following example represents a read-only property with the C++ name
of ``ov::available_devices``, the string name of ``AVAILABLE_DEVICES`` and the type of
``std::vector<std::string>``:

.. code-block:: sh

   static constexpr Property<std::vector<std::string>, PropertyMutability::RO> available_devices{"AVAILABLE_DEVICES"};


Refer to the :doc:`Hello Query Device C++ Sample <../../../learn-openvino/openvino-samples/hello-query-device>`
sources for an example of using the setting and getting properties in user applications.


Get a Set of Available Devices
###########################################################

Based on the ``ov::available_devices`` read-only property, OpenVINO Core collects information about currently available
devices enabled by OpenVINO plugins and returns information, using the ``ov::Core::get_available_devices`` method:


.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_properties_api.py
         :language: py
         :fragment: [get_available_devices]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_properties_api.cpp
         :language: cpp
         :fragment: [get_available_devices]


The function returns a list of available devices, for example:

.. code-block:: sh

   CPU
   GPU.0
   GPU.1

If there are multiple instances of a specific device, the devices are enumerated with a suffix comprising a full stop and
a unique string identifier, such as ``.suffix``. Each device name can then be passed to:

* ``ov::Core::compile_model`` to load the model to a specific device with specific configuration properties.
* ``ov::Core::get_property`` to get common or device-specific properties.
* All other methods of the ``ov::Core`` class that accept ``deviceName``.

Working with Properties in Your Code
###########################################################

The ``ov::Core`` class provides the following method to query device information, set or get different device configuration properties:

* ``ov::Core::get_property`` - Gets the current value of a specific property.
* ``ov::Core::set_property`` - Sets a new value for the property globally for specified ``device_name``.

The ``ov::CompiledModel`` class is also extended to support the properties:

* ``ov::CompiledModel::get_property``
* ``ov::CompiledModel::set_property``

For documentation about OpenVINO common device-independent properties, refer to
`properties.hpp (GitHub) <https://github.com/openvinotoolkit/openvino/blob/releases/2024/0/src/inference/include/openvino/runtime/properties.hpp>`__.
Device-specific configuration keys can be found in a corresponding device folders,
for example, ``openvino/runtime/intel_gpu/properties.hpp``.

Working with Properties via Core
###########################################################

Getting Device Properties
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The code below demonstrates how to query ``HETERO`` device priority of devices which will be used to infer the model:


.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_properties_api.py
         :language: py
         :fragment: [hetero_priorities]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_properties_api.cpp
         :language: cpp
         :fragment: [hetero_priorities]


.. note::

   All properties have a type, which is specified during property declaration. Based on this, actual type under ``auto`` is automatically deduced by C++ compiler.

To extract device properties such as available devices (``ov::available_devices``), device name (``ov::device::full_name``),
supported properties (``ov::supported_properties``), and others, use the ``ov::Core::get_property`` method:


.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_properties_api.py
         :language: py
         :fragment: [cpu_device_name]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_properties_api.cpp
         :language: cpp
         :fragment: [cpu_device_name]


A returned value appears as follows: ``Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz``.

.. note::

   In order to understand a list of supported properties on ``ov::Core`` or ``ov::CompiledModel`` levels, use ``ov::supported_properties``
   which contains a vector of supported property names. Properties which can be changed, has ``ov::PropertyName::is_mutable``
   returning the ``true`` value. Most of the properties which are changeable on ``ov::Core`` level, cannot be changed once the model is compiled,
   so it becomes immutable read-only property.

Configure a Work with a Model
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The ``ov::Core`` methods like:

* ``ov::Core::compile_model``
* ``ov::Core::import_model``
* ``ov::Core::query_model``

accept a selection of properties as last arguments. Each of the properties should be used as a function call to pass a property value with a specified property type.


.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_properties_api.py
         :language: py
         :fragment: [compile_model_with_property]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_properties_api.cpp
         :language: cpp
         :fragment: [compile_model_with_property]


The example below specifies hints that a model should be compiled to be inferred with multiple inference requests in parallel
to achieve best throughput, while inference should be performed without accuracy loss with FP32 precision.

Setting Properties Globally
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

``ov::Core::set_property`` with a given device name should be used to set global configuration properties,
which are the same across multiple ``ov::Core::compile_model``, ``ov::Core::query_model``, and other calls.
However, setting properties on a specific ``ov::Core::compile_model`` call applies properties only for the current call:


.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_properties_api.py
         :language: py
         :fragment: [core_set_property_then_compile]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_properties_api.cpp
         :language: cpp
         :fragment: [core_set_property_then_compile]


Properties on CompiledModel Level
###########################################################

Getting Property
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The ``ov::CompiledModel::get_property`` method is used to get property values the compiled model has been created with or a
compiled model level property such as ``ov::optimal_number_of_infer_requests``:


.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_properties_api.py
         :language: py
         :fragment: [optimal_number_of_infer_requests]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_properties_api.cpp
         :language: cpp
         :fragment: [optimal_number_of_infer_requests]


Or the number of threads that would be used for inference on ``CPU`` device:


.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_properties_api.py
         :language: py
         :fragment: [inference_num_threads]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_properties_api.cpp
         :language: cpp
         :fragment: [inference_num_threads]




