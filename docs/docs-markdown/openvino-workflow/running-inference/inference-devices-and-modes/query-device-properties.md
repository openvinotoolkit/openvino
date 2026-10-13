---
sidebar_position: 7
sidebar_label: "Query Device Properties - Configuration"
description: "Learn the details on the process of querying different device properties and configuration values at runtime."
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Query Device Properties - Configuration

This article provides an overview of how to query different device properties
and configuration values at runtime.

OpenVINO runtime has two types of properties:

- **Read only properties** which provide information about devices, such as device
  name and execution capabilities, and information about configuration values
  used to compile the model - `ov::CompiledModel`.
- **Mutable properties**, primarily used to configure the `ov::Core::compile_model`
  process and affect final inference on a specific set of devices. Such properties
  can be set globally per device via `ov::Core::set_property` or locally for a
  particular model in the `ov::Core::compile_model` and `ov::Core::query_model`
  calls.

An OpenVINO property is represented as a named constexpr variable with a given string
name and a type. The following example represents a read-only property with the C++ name
of `ov::available_devices`, the string name of `AVAILABLE_DEVICES` and the type of
`std::vector<std::string>`:

```bash
static constexpr Property<std::vector<std::string>, PropertyMutability::RO> available_devices{"AVAILABLE_DEVICES"};
```

Refer to the [Hello Query Device C++ Sample](../../../get-started/learn-openvino/openvino-samples/hello-query-device.md)
sources for an example of using the setting and getting properties in user applications.

## Get a Set of Available Devices

Based on the `ov::available_devices` read-only property, OpenVINO Core collects information about currently available
devices enabled by OpenVINO plugins and returns information, using the `ov::Core::get_available_devices` method:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
      core = ov.Core()
      available_devices = core.available_devices
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  ov::Core core;
  std::vector<std::string> available_devices = core.get_available_devices();
  ```
  </TabItem>
</Tabs>

The function returns a list of available devices, for example:

```bash
CPU
GPU.0
GPU.1
```

If there are multiple instances of a specific device, the devices are enumerated with a suffix comprising a full stop and
a unique string identifier, such as `.suffix`. Each device name can then be passed to:

- `ov::Core::compile_model` to load the model to a specific device with specific configuration properties.
- `ov::Core::get_property` to get common or device-specific properties.
- All other methods of the `ov::Core` class that accept `deviceName`.

## Working with Properties in Your Code

The `ov::Core` class provides the following method to query device information, set or get different device configuration properties:

- `ov::Core::get_property` - Gets the current value of a specific property.
- `ov::Core::set_property` - Sets a new value for the property globally for specified `device_name`.

The `ov::CompiledModel` class is also extended to support the properties:

- `ov::CompiledModel::get_property`
- `ov::CompiledModel::set_property`

For documentation about OpenVINO common device-independent properties, refer to
[properties.hpp (GitHub)](https://github.com/openvinotoolkit/openvino/blob/releases/2026/0/src/inference/include/openvino/runtime/properties.hpp).
Device-specific configuration keys can be found in a corresponding device folders,
for example, `openvino/runtime/intel_gpu/properties.hpp`.

## Working with Properties via Core

### Getting Device Properties

The code below demonstrates how to query `HETERO` device priority of devices which will be used to infer the model:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
      device_priorites = core.get_property("HETERO", device.priorities)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  auto device_priorites = core.get_property("HETERO", ov::device::priorities);
  ```
  </TabItem>
</Tabs>

:::note
All properties have a type, which is specified during property declaration. Based on this, actual type under `auto` is automatically deduced by C++ compiler.
:::

To extract device properties such as available devices (`ov::available_devices`), device name (`ov::device::full_name`),
supported properties (`ov::supported_properties`), and others, use the `ov::Core::get_property` method:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
      cpu_device_name = core.get_property("CPU", device.full_name)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  auto cpu_device_name = core.get_property("CPU", ov::device::full_name);
  ```
  </TabItem>
</Tabs>

A returned value appears as follows: `Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz`.

:::note
In order to understand a list of supported properties on `ov::Core` or `ov::CompiledModel` levels, use `ov::supported_properties`
which contains a vector of supported property names. Properties which can be changed, has `ov::PropertyName::is_mutable`
returning the `true` value. Most of the properties which are changeable on `ov::Core` level, cannot be changed once the model is compiled,
so it becomes immutable read-only property.
:::

### Configure a Work with a Model

The `ov::Core` methods like:

- `ov::Core::compile_model`
- `ov::Core::import_model`
- `ov::Core::query_model`

accept a selection of properties as last arguments. Each of the properties should be used as a function call to pass a property value with a specified property type.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
      config = {hints.performance_mode: hints.PerformanceMode.THROUGHPUT,
              hints.inference_precision: ov.Type.f32}
      compiled_model = core.compile_model(model, "CPU", config)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  auto compiled_model = core.compile_model(model, "CPU",
      ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
      ov::hint::inference_precision(ov::element::f32));
  ```
  </TabItem>
</Tabs>

The example below specifies hints that a model should be compiled to be inferred with multiple inference requests in parallel
to achieve best throughput, while inference should be performed without accuracy loss with FP32 precision.

### Setting Properties Globally

`ov::Core::set_property` with a given device name should be used to set global configuration properties,
which are the same across multiple `ov::Core::compile_model`, `ov::Core::query_model`, and other calls.
However, setting properties on a specific `ov::Core::compile_model` call applies properties only for the current call:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
      # latency hint is a default for CPU
      core.set_property("CPU", {hints.performance_mode: hints.PerformanceMode.LATENCY})
      # compiled with latency configuration hint
      compiled_model_latency = core.compile_model(model, "CPU")
      # compiled with overridden performance hint value
      config = {hints.performance_mode: hints.PerformanceMode.THROUGHPUT}
      compiled_model_thrp = core.compile_model(model, "CPU", config)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  // set letency hint is a default for CPU
  core.set_property("CPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
  // compiled with latency configuration hint
  auto compiled_model_latency = core.compile_model(model, "CPU");
  // compiled with overridden ov::hint::performance_mode value
  auto compiled_model_thrp = core.compile_model(model, "CPU",
      ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
  ```
  </TabItem>
</Tabs>

## Properties on CompiledModel Level

### Getting Property

The `ov::CompiledModel::get_property` method is used to get property values the compiled model has been created with or a
compiled model level property such as `ov::optimal_number_of_infer_requests`:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
      compiled_model = core.compile_model(model, "CPU")
      nireq = compiled_model.get_property(props.optimal_number_of_infer_requests)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  auto compiled_model = core.compile_model(model, "CPU");
  auto nireq = compiled_model.get_property(ov::optimal_number_of_infer_requests);
  ```
  </TabItem>
</Tabs>

Or the number of threads that would be used for inference on `CPU` device:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
      compiled_model = core.compile_model(model, "CPU")
      nthreads = compiled_model.get_property(props.inference_num_threads)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  auto compiled_model = core.compile_model(model, "CPU");
  auto nthreads = compiled_model.get_property(ov::inference_num_threads);
  ```
  </TabItem>
</Tabs>
