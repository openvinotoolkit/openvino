# OpenVINO Proxy Plugin

## Key Contacts

Please contact a member of [openvino-ie-maintainers](https://github.com/orgs/openvinotoolkit/teams/openvino-ie-maintainers) group, for assistance regarding Proxy plugin.

## Components

Proxy Plugin contains the following components:

* [include](./include/) - folder contains public plugin API.
* [src](./src/) - folder contains sources of the plugin.
* [tests](./tests/) - contains tests for the plugin.

## Motivation

 - OpenVINO may have multiple hardware plugins for similar device type from different vendors (e.g. Intel and NVidia GPUs) and currently user must address them with different names ("GPU" and "NVIDIA" respectively). Using same name for such cases seems to be more user-friendly approach.
 - Moreover, single physical device may have multiple plugin which support it. For example, Intel GPU plugin is OpenCL based, thus can run on other OCL-compatible devices including NVIDIA gpus. In that case we may have primary plugin ("NVIDIA") which provides best performance on target device and fallback plugin ("INTEL_GPU") which helps to improve models coverage for the cases when primary plugin has limited operation set support, and both plugins may be executed via HETERO mode.
 - Implicit HETERO plugin usage may be extended to run on different device types - HETERO:xPU,CPU

## Requirements

 - Do not provide additional libraries and don't affect load time (proxy plugin is a part of openvino library)
 - No overhead for load inference time
    - Fallback to hardware plugin if device is supported only by one plugin
    - Minimal overhead in case of multiple plugins for one device
        - Plus one call of query network if entire model can be executed on preferable plugin
        - Hetero execution in other case
 - Allow to configure device

## Proxy plugin properties

Proxy plugin cannot be created explicitly. In order to use proxy plugin under the real hardware plugins please use next properties for configuration:
 - `ov::proxy::configuration::alias` is an alias name for high level plugin.
 - `ov::proxy::configuration::priority` is a device priority under alias (lower value means the higher priority), this value allows to configure the device order.
 - `ov::proxy::configuration::fallback` the name of other hardware device for the fallback.
 - `ov::proxy::configuration::internal_name` the name which will be used as internal name in case if real device name has a collision with proxy name.

After the creation the proxy plugin has next properties:
 - `ov::device::priorities` is fallback order inside the proxy plugin.
 - `ov::proxy::device_priorities` is a device order under proxy plugin.
 - `ov::proxy::alias_for` the vector of low level plugins under the proxy alias.


## See also
 * [OpenVINO™ README](../../../README.md)
 * [OpenVINO Core Components](../../README.md)
 * [OpenVINO Plugins](../README.md)
 * [Developer documentation](../../../docs/dev/index.md)
 * [OpenVINO Plugin Developer Guide](https://docs.openvino.ai/2025/documentation/openvino-extensibility/openvino-plugin-library.html)

