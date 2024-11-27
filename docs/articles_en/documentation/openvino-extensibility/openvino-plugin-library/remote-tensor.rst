Remote Tensor
=============


.. meta::
   :description: Use the ov::IRemoteTensor interface as a base class for device-specific remote tensors.


ov::RemoteTensor class functionality:

* Provides an interface to work with device-specific memory.

.. note::

   If plugin provides a public API for own Remote Tensor, the API should be header only and does not depend on the plugin library.


Device Specific Remote Tensor Public API
########################################

The public interface to work with device specific remote tensors should have header only implementation and doesn't depend on the plugin library.

.. doxygensnippet:: src/plugins/template/include/template/remote_tensor.hpp
   :language: cpp
   :fragment: [remote_tensor:public_header]

The implementation below has several methods:

type_check()
+++++++++++++++++++++++++

Static method is used to understand that some abstract remote tensor can be casted to this particular remote tensor type.

get_data()
+++++++++++++++++++++++++

The set of methods (specific for the example, other implementation can have another API) which are helpers to get an access to remote data.

Device-Specific Internal tensor implementation
##############################################

The plugin should have the internal implementation of remote tensor which can communicate with public API.
The example contains the implementation of remote tensor which wraps memory from stl vector.

OpenVINO Plugin API provides the interface ov::IRemoteTensor which should be used as a base class for remote tensors.

The example implementation have two remote tensor classes:

* Internal type dependent implementation which has as an template argument the vector type and create the type specific tensor.
* The type independent implementation which works with type dependent tensor inside.

Based on that, an implementation of a type independent remote tensor class can look as follows:

.. doxygensnippet:: src/plugins/template/src/remote_tensor.hpp
   :language: cpp
   :fragment: [vector_impl:implementation]

The implementation provides a helper to get wrapped stl tensor and overrides all important methods of ov::IRemoteTensor class and recall the type dependent implementation.

The type dependent remote tensor has the next implementation:

.. doxygensnippet:: src/plugins/template/src/remote_context.cpp
   :language: cpp
   :fragment: [vector_impl_t:implementation]

Class Fields
++++++++++++

The class has several fields:

* ``m_element_type`` - Tensor element type.
* ``m_shape`` - Tensor shape.
* ``m_strides`` - Tensor strides.
* ``m_data`` - Wrapped vector.
* ``m_dev_name`` - Device name.
* ``m_properties`` - Remote tensor specific properties which can be used to detect the type of the remote tensor.

VectorTensorImpl()
++++++++++++++++++

The constructor of remote tensor implementation. Creates a vector with data, initialize device name and properties, updates shape, element type and strides.

get_element_type()
++++++++++++++++++

The method returns tensor element type.

get_shape()
+++++++++++

The method returns tensor shape.

get_strides()
+++++++++++++

The method returns tensor strides.

set_shape()
+++++++++++

The method allows to set new shapes for the remote tensor.

get_properties()
++++++++++++++++

The method returns tensor specific properties.

get_device_name()
+++++++++++++++++

The method returns tensor specific device name.



