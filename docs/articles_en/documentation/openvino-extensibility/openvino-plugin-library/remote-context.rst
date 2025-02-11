Remote Context
==============


.. meta::
   :description: Use the ov::RemoteContext class as the base class for a plugin-specific remote context.


ov::RemoteContext class functionality:

* Represents device-specific inference context.
* Allows to create remote device specific tensor.

.. note::

   If plugin provides a public API for own Remote Context, the API should be header only and does not depend on the plugin library.


RemoteContext Class
###################

OpenVINO Plugin API provides the interface ov::IRemoteContext which should be used as a base class for a plugin specific remote context. Based on that, a declaration of an compiled model class can look as follows:

.. doxygensnippet:: src/plugins/template/src/remote_context.hpp
   :language: cpp
   :fragment: [remote_context:header]

Class Fields
++++++++++++

The example class has several fields:

* ``m_name`` - Device name.
* ``m_property`` - Device-specific context properties. It can be used to cast RemoteContext to device specific type.

RemoteContext Constructor
+++++++++++++++++++++++++

This constructor should initialize the remote context device name and properties.

.. doxygensnippet:: src/plugins/template/src/remote_context.cpp
   :language: cpp
   :fragment: [remote_context:ctor]

get_device_name()
++++++++++++++++++

The function returns the device name from the remote context.

.. doxygensnippet:: src/plugins/template/src/remote_context.cpp
   :language: cpp
   :fragment: [remote_context:get_device_name]

get_property()
+++++++++++++++

The implementation returns the remote context properties.

.. doxygensnippet:: src/plugins/template/src/remote_context.cpp
   :language: cpp
   :fragment: [remote_context:get_property]

create_tensor()
+++++++++++++++

The method creates device specific remote tensor.

.. doxygensnippet:: src/plugins/template/src/remote_context.cpp
   :language: cpp
   :fragment: [remote_context:create_tensor]

The next step to support device specific tensors is a creation of device specific :doc:`Remote Tensor <remote-tensor>` class.


