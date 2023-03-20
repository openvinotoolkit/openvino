# Remote Context {#openvino_docs_ov_plugin_dg_remote_context}

ov::RemoteContext class functionality:
- Represents device specific inference context.
- Allows to create remote device specific tensor.

> **NOTE**: If plugin provides a public API for own Remote Context, the API should be header only and doesn't depend on the plugin library.


RemoteContext Class
------------------------

OpenVINO Plugin API provides the interface ov::IRemoteContext which should be used as a base class for a plugin specific remote context. Based on that, a declaration of an compiled model class can look as follows: 

@snippet src/remote_context.hpp remote_context:header

### Class Fields

The example class has several fields:

- `m_name` - Device name.
- `m_property` - Device specific context properties. It can be used to cast RemoteContext to device specific type.

### RemoteContext Constructor

This constructor should initialize the remote context device name and properties.

@snippet src/remote_context.cpp remote_context:ctor

### get_device_name()

The function returns the device name from the remote context.

@snippet src/remote_context.cpp remote_context:get_device_name

### get_property()

The implementation returns the remote context properties.

@snippet src/remote_context.cpp remote_context:get_property


### create_tensor()

The method creates device specific remote tensor.

@snippet src/remote_context.cpp remote_context:create_tensor

The next step to support device specific tensors is a creation of device specific [Remote Tensor](@ref openvino_docs_ov_plugin_dg_remote_tensor) class.
