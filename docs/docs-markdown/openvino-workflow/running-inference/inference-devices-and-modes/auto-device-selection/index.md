---
sidebar_position: 4
sidebar_label: "Automatic Device Selection"
description: "The Automatic Device Selection mode in OpenVINO™ Runtime detects available devices and selects the optimal processing unit for inference automatically."
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Automatic Device Selection


The Automatic Device Selection mode, or AUTO for short, uses a "virtual" or a "proxy" device,
which does not bind to a specific type of hardware, but rather selects the processing unit
for inference automatically. It detects available devices, picks the one best-suited for the
task, and configures its optimization settings. This way, you can write the application once
and deploy it anywhere.

The selection also depends on your performance requirements, defined by the “hints”
configuration API, as well as device priority list limitations, if you choose to exclude
some hardware from the process.

The logic behind the choice is as follows:

1. Check what supported devices are available.
2. Check precisions of the input model (for detailed information on precisions read more on the `ov::device::capabilities`).
3. Select the highest-priority device capable of supporting the given model, as listed in the table below.
4. If model's precision is FP32 but there is no device capable of supporting it, offload the model to a device supporting FP16.

| Device<br/>Priority | Supported<br/>Device | Supported<br/>model precision |
| --- | --- | --- |
| 1 | dGPU<br/>(e.g. Intel® Iris® Xe MAX) | FP32, FP16, INT8, BIN |
| 2 | iGPU<br/>(e.g. Intel® UHD Graphics 620 (iGPU)) | FP32, FP16, BIN |
| 3 | Intel® CPU<br/>(e.g. Intel® Core™ i7-1165G7) | FP32, FP16, INT8, BIN |
| 4 | Intel® NPU<br/>(e.g. Intel® Core™ Ultra) |  |

:::note
Note that NPU is currently excluded from the default priority list. To use it for inference, you
need to specify it explicitly
:::

## How AUTO Works

To put it simply, when loading the model to the first device on the list fails, AUTO will try to load it to the next device in line, until one of them succeeds.
What is important, **AUTO starts inference with the CPU of the system by default unless there is model cached for the best suited device**, as it provides very low latency and can start inference with no additional delays.
While the CPU is performing inference, AUTO continues to load the model to the device best suited for the purpose and transfers the task to it when ready.
This way, the devices which are much slower in compiling models, GPU being the best example, do not impact inference at its initial stages.
For example, if you use a CPU and a GPU, the first-inference latency of AUTO will be better than that of using GPU alone.

Note that if you choose to exclude CPU from the priority list or disable the initial
CPU acceleration feature via `ov::intel_auto::enable_startup_fallback`, it will be
unable to support the initial model compilation stage. For the models with
[stateful operations](../../inference-request/stateful-models/index.md),
CPU acceleration feature will also be disabled during model compilation.
These models will follow the normal flow and be loaded to the device based on priority.

![](autoplugin_accelerate.svg)

This mechanism can be easily observed in the [Using AUTO with Benchmark app sample](#using-auto-with-openvino-samples-and-benchmark-app)
section, showing how the first-inference latency (the time it takes to compile the
model and perform the first inference) is reduced when using AUTO. For example:

```bash
benchmark_app -m ../public/alexnet/FP32/alexnet.xml -d GPU -niter 128
```

```bash
benchmark_app -m ../public/alexnet/FP32/alexnet.xml -d AUTO -niter 128
```

:::note
The longer the process runs, the closer realtime performance will be to that of the best-suited device.
:::

:::note
**Testing accuracy with the AUTO device is not recommended.** Since the CPU and GPU (or other target devices) may produce slightly different accuracy numbers, using AUTO could lead to inconsistent accuracy results from run to run due to a different number of inferences on CPU and GPU. This is particularly true when testing with a small number of inputs. To achieve consistent inference on the GPU (or another target device), you can disable CPU acceleration by setting `ov::intel_auto::enable_startup_fallback` to false.
:::

## Using AUTO

Following the OpenVINO™ naming convention, the Automatic Device Selection mode is assigned the label of "AUTO".
It may be defined with no additional parameters, resulting in defaults being used, or configured further with
the following setup options:

| Property(C++ version) | Values and Description |
| --- | --- |
| `<device candidate list>` | **Values**:<br/><br/>empty<br/><br/>`AUTO`<br/><br/>`AUTO: <device names>` (comma-separated, no spaces)<br/><br/>Lists the devices available for selection.<br/>The device sequence will be taken as priority from high to low.<br/>If not specified, `AUTO` will be used as default,<br/>and all devices will be "viewed" as candidates. |
| `ov::device::priorities` | **Values**:<br/><br/>`<device names>` (comma-separated, no spaces)<br/><br/>Specifies the devices for AUTO to select.<br/>The device sequence will be taken as priority from high to low.<br/>This configuration is optional. |
| `ov::hint::performance_mode` | **Values**:<br/><br/>`ov::hint::PerformanceMode::LATENCY`<br/><br/>`ov::hint::PerformanceMode::THROUGHPUT`<br/><br/>`ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT`<br/><br/>Specifies the performance option preferred by the application. |
| `ov::hint::model_priority` | **Values**:<br/><br/>`ov::hint::Priority::HIGH`<br/><br/>`ov::hint::Priority::MEDIUM`<br/><br/>`ov::hint::Priority::LOW`<br/><br/>Indicates the priority for a model.<br/><br/>IMPORTANT: This property is not fully supported yet. |
| `ov::execution_devices` | Lists the runtime target devices on which the inferences are being<br/>executed.<br/><br/>Examples of returning results could be `(CPU)` (`CPU` is a<br/>temporary device, indicating that CPU is used for acceleration at<br/>the model compilation stage), `CPU`, `GPU`, `CPU GPU`,<br/>`GPU.0`, etc. |
| `ov::intel_auto::enable_startup_fallback` | **Values**:<br/><br/>`true`<br/><br/>`false`<br/><br/>Enables/disables CPU as acceleration (or the helper device) in the<br/>beginning. The default value is `true`, indicating that CPU is<br/>used as acceleration by default. |
| `ov::intel_auto::enable_runtime_fallback` | **Values**:<br/><br/>`true`<br/><br/>`false`<br/><br/>Enables/disables runtime fallback to other devices and performs<br/>the failed inference request again, if inference request fails on<br/>the currently selected device.<br/><br/>The default value is `true`. |
| `ov::intel_auto::schedule_policy` | **Values**:<br/><br/>`ROUND_ROBIN`<br/><br/>`DEVICE_PRIORITY`<br/><br/>Specify the schedule policy of infer request assigned to hardware<br/>plugin for AUTO cumulative mode.<br/><br/>The default value is `DEVICE_PRIORITY`. |

Inference with AUTO is configured similarly to when device plugins are used:
you compile the model on the plugin with configuration and execute inference.

The code samples on this page assume following import(Python)/using (C++) are included at the beginning of code snippets.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  import openvino as ov
  import openvino.properties as properties
  import openvino.properties.device as device
  import openvino.properties.hint as hints
  import openvino.properties.streams as streams
  import openvino.properties.intel_auto as intel_auto
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  #include <openvino/openvino.hpp>
  ```
  </TabItem>
</Tabs>

### Device Candidates and Priority

The device candidate list enables you to customize the priority and limit the choice of devices available to AUTO.

- If `<device candidate list>` is not specified, AUTO assumes all the devices present in the system can be used.
- If `AUTO` without any device names is specified, AUTO assumes all the devices present in the system can be used, and will load the network to all devices and run inference based on their default priorities, from high to low.

To specify the priority of devices, enter the device names in the priority order (from high to low) in `AUTO: <device names>`, or use the `ov::device::priorities` property.

See the following code for using AUTO and specifying devices:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
      core = ov.Core()

      #  compile a model on AUTO using the default list of device candidates.
      #  The following lines are equivalent:
      compiled_model = core.compile_model(model=model)
      compiled_model = core.compile_model(model=model, device_name="AUTO")

      # Optional
      # You can also specify the devices to be used by AUTO.
      # The following lines are equivalent:
      compiled_model = core.compile_model(model=model, device_name="AUTO:GPU,CPU")
      compiled_model = core.compile_model(
          model=model,
          device_name="AUTO",
          config={device.priorities: "GPU,CPU"},
      )

      # Optional
      # the AUTO plugin is pre-configured (globally) with the explicit option:
      core.set_property(
          device_name="AUTO", properties={device.priorities: "GPU,CPU"}
      )
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  ov::Core core;

  // Read a network in IR, PaddlePaddle, or ONNX format:
  std::shared_ptr<ov::Model> model = core.read_model("sample.xml");

  // compile a model on AUTO using the default list of device candidates.
  // The following lines are equivalent:
  ov::CompiledModel model0 = core.compile_model(model);
  ov::CompiledModel model1 = core.compile_model(model, "AUTO");

  // Optional
  // You can also specify the devices to be used by AUTO.
  // The following lines are equivalent:
  ov::CompiledModel model3 = core.compile_model(model, "AUTO:GPU,CPU");
  ov::CompiledModel model4 = core.compile_model(model, "AUTO", ov::device::priorities("GPU,CPU"));

  //Optional
  // the AUTO plugin is pre-configured (globally) with the explicit option:
  core.set_property("AUTO", ov::device::priorities("GPU,CPU"));
  ```
  </TabItem>
</Tabs>

Note that OpenVINO Runtime lets you use "GPU" as an alias for "GPU.0" in function calls.
More details on enumerating devices can be found in [Inference Devices and Modes](../index.md).

#### Checking Available Devices

To check what devices are present in the system, you can use Device API, as listed below. For information on how to use it, see [Query device properties and configuration](../query-device-properties.md).

<Tabs>
  <TabItem value="py" label="Python">
  ```bash
  openvino.Core.available_devices
  ```

  See the Hello Query Device Python Sample for reference.
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```bash
  ov::runtime::Core::get_available_devices()
  ```

  See the Hello Query Device C++ Sample for reference.
  </TabItem>
</Tabs>

#### Excluding Devices from Device Candidate List

You can also exclude hardware devices from AUTO, for example, to reserve CPU for other jobs. AUTO will not use the device for inference then. To do that, add a minus sign `(-)` before CPU in `AUTO: <device names>`, as in the following example:

<Tabs>
  <TabItem value="py" label="Python">
  ```bash
  compiled_model = core.compile_model(model=model, device_name="AUTO:-CPU")
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```bash
  ov::CompiledModel compiled_model = core.compile_model(model, "AUTO:-CPU");
  ```
  </TabItem>
</Tabs>

AUTO will then query all available devices and remove CPU from the candidate list.

Note that if you choose to exclude CPU from device candidate list, CPU will not be able to
support the initial model compilation stage. See more information in
[How AUTO Works](#auto-device-selection).

### Performance Hints for AUTO

The `ov::hint::performance_mode` property enables you to specify a performance option for AUTO to be more efficient for particular use cases. The default hint for AUTO is `LATENCY`.

The THROUGHPUT and CUMULATIVE\_THROUGHPUT hints below only improve performance in an
asynchronous inference pipeline. For information on asynchronous inference, see the
[Async API documentation](../../inference-request/index.md) .
The following notebooks provide examples of how to set up an asynchronous pipeline:

- [Image Classification Async Sample](../../../../get-started/learn-openvino/openvino-samples/image-classification-async.md)
- [Notebook - Asynchronous Inference with OpenVINO™](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/async-api)
- [Notebook - Automatic Device Selection with OpenVINO](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/auto-device)

#### LATENCY

This option prioritizes low latency, providing short response time for each inference job. It performs best for tasks where inference is required for a single input image, e.g. a medical analysis of an ultrasound scan image. It also fits the tasks of real-time or nearly real-time applications, such as an industrial robot's response to actions in its environment or obstacle avoidance for autonomous vehicles.

:::note
If no performance hint is set explicitly, AUTO will set LATENCY for devices that have not set `ov::device::properties`, for example, `ov::device::properties(<DEVICE_NAME>, ov::hint::performance_mode(ov::hint::LATENCY))`.
:::

(cumulative throughput)=

#### `THROUGHPUT`

This option prioritizes high throughput, balancing between latency and power. It is best suited for tasks involving multiple jobs, such as inference of video feeds or large numbers of images.

#### `CUMULATIVE_THROUGHPUT`

While `LATENCY` and `THROUGHPUT` can select one target device with your preferred performance option,
the `CUMULATIVE_THROUGHPUT` option enables running inference on multiple devices for higher throughput.
With `CUMULATIVE_THROUGHPUT`, AUTO loads the network model to all available devices (specified by AUTO)
in the candidate list, and then runs inference on them based on the default or specified priority.

If device priority is specified when using `CUMULATIVE_THROUGHPUT`, AUTO will run inference
requests on devices based on the priority. In the following example, AUTO will always
try to use GPU first, and then use CPU if GPU is busy:

<Tabs>
  <TabItem value="py" label="Python">
  ```bash
  compiled_model = core.compile_model(model, "AUTO:GPU,CPU", {hints.performance_mode: hints.PerformanceMode.CUMULATIVE_THROUGHPUT})
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```bash
  ov::CompiledModel compiled_model = core.compile_model(model, "AUTO:GPU,CPU", ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT));
  ```
  </TabItem>
</Tabs>

If AUTO is used without specifying any device names, and if there are multiple GPUs in the system, `CUMULATIVE_THROUGHPUT` mode will use all of the GPUs by default. If the system has more than two GPU devices, AUTO will remove CPU from the device candidate list to keep the GPUs running at full capacity. A full list of system devices and their unique identifiers can be queried using ov::Core::get\_available\_devices (for more information, see [Query Device Properties](../query-device-properties.md)). To explicitly specify which GPUs to use, set their priority when compiling with AUTO:

<Tabs>
  <TabItem value="py" label="Python">
  ```bash
  compiled_model = core.compile_model(model, "AUTO:GPU.1,GPU.0", {hints.performance_mode: hints.PerformanceMode.CUMULATIVE_THROUGHPUT})
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```bash
  ov::CompiledModel compiled_model = core.compile_model(model, "AUTO:GPU.1,GPU.0", ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT));
  ```
  </TabItem>
</Tabs>

#### Code Examples

To enable performance hints for your application, use the following code:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
      core = ov.Core()

      # Compile a model on AUTO with Performance Hints enabled:
      # To use the “THROUGHPUT” mode:
      compiled_model = core.compile_model(
          model=model,
          device_name="AUTO",
          config={
              hints.performance_mode: hints.PerformanceMode.THROUGHPUT
          },
      )
      # To use the “LATENCY” mode:
      compiled_model = core.compile_model(
          model=model,
          device_name="AUTO",
          config={
              hints.performance_mode: hints.PerformanceMode.LATENCY
          },
      )
      # To use the “CUMULATIVE_THROUGHPUT” mode:
      # To use the ROUND_ROBIN schedule policy:
      compiled_model = core.compile_model(
          model=model,
          device_name="AUTO",
          config={
              hints.performance_mode: hints.PerformanceMode.CUMULATIVE_THROUGHPUT,
              intel_auto.schedule_policy: intel_auto.SchedulePolicy.ROUND_ROBIN
          },
      )
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  ov::Core core;

  // Read a network in IR, PaddlePaddle, or ONNX format:
  std::shared_ptr<ov::Model> model = core.read_model("sample.xml");

  // Compile a model on AUTO with Performance Hint enabled:
  // To use the “THROUGHPUT” option:
  ov::CompiledModel compiled_model = core.compile_model(model, "AUTO",
      ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
  // To use the “LATENCY” option:
  ov::CompiledModel compiled_mode2 = core.compile_model(model, "AUTO",
      ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
  // To use the “CUMULATIVE_THROUGHPUT” option:
  ov::CompiledModel compiled_mode3 = core.compile_model(model, "AUTO",
      ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT));
  ```
  </TabItem>
</Tabs>

#### Disabling Auto-Batching for THROUGHPUT and CUMULATIVE\_THROUGHPUT

The `ov::hint::PerformanceMode::THROUGHPUT` mode and the `ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT` mode will trigger Auto-Batching (for example, for the GPU device) by default. You can disable it by setting `ov::hint::allow_auto_batching(false)`, or change the default timeout value to a large number, e.g. `ov::auto_batch_timeout(1000)`. See [Automatic Batching](../automatic-batching.md) for more details.

### Configuring Model Priority

The `ov::hint::model_priority` property enables you to control the priorities of models in the Auto-Device plugin. A high-priority model will be loaded to a supported high-priority device. A lower-priority model will not be loaded to a device that is occupied by a higher-priority model.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
      core = ov.Core()

      # Example 1
      compiled_model0 = core.compile_model(
          model=model,
          device_name="AUTO",
          config={hints.model_priority: hints.Priority.HIGH},
      )
      compiled_model1 = core.compile_model(
          model=model,
          device_name="AUTO",
          config={
              hints.model_priority: hints.Priority.MEDIUM
          },
      )
      compiled_model2 = core.compile_model(
          model=model,
          device_name="AUTO",
          config={hints.model_priority: hints.Priority.LOW},
      )
      # Assume that all the devices (CPU and GPUs) can support all the networks.
      # Result: compiled_model0 will use GPU.1, compiled_model1 will use GPU.0, compiled_model2 will use CPU.

      # Example 2
      compiled_model3 = core.compile_model(
          model=model,
          device_name="AUTO",
          config={hints.model_priority: hints.Priority.HIGH},
      )
      compiled_model4 = core.compile_model(
          model=model,
          device_name="AUTO",
          config={
              hints.model_priority: hints.Priority.MEDIUM
          },
      )
      compiled_model5 = core.compile_model(
          model=model,
          device_name="AUTO",
          config={hints.model_priority: hints.Priority.LOW},
      )
      # Assume that all the devices (CPU ang GPUs) can support all the networks.
      # Result: compiled_model3 will use GPU.1, compiled_model4 will use GPU.1, compiled_model5 will use GPU.0.
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  // Example 1
  ov::CompiledModel compiled_model0 = core.compile_model(model, "AUTO",
      ov::hint::model_priority(ov::hint::Priority::HIGH));
  ov::CompiledModel compiled_model1 = core.compile_model(model, "AUTO",
      ov::hint::model_priority(ov::hint::Priority::MEDIUM));
  ov::CompiledModel compiled_model2 = core.compile_model(model, "AUTO",
      ov::hint::model_priority(ov::hint::Priority::LOW));
  /************
    Assume that all the devices (CPU and GPUs) can support all the models.
    Result: compiled_model0 will use GPU.1, compiled_model1 will use GPU.0, compiled_model2 will use CPU.
   ************/

  // Example 2
  ov::CompiledModel compiled_model3 = core.compile_model(model, "AUTO",
      ov::hint::model_priority(ov::hint::Priority::LOW));
  ov::CompiledModel compiled_model4 = core.compile_model(model, "AUTO",
      ov::hint::model_priority(ov::hint::Priority::MEDIUM));
  ov::CompiledModel compiled_model5 = core.compile_model(model, "AUTO",
      ov::hint::model_priority(ov::hint::Priority::LOW));
  /************
    Assume that all the devices (CPU and GPUs) can support all the models.
    Result: compiled_model3 will use GPU.1, compiled_model4 will use GPU.1, compiled_model5 will use GPU.0.
   ************/
  ```
  </TabItem>
</Tabs>

### Checking Target Runtime Devices

To query the runtime target devices on which the inferences are being executed using AUTO, you can use the `ov::execution_devices` property. It must be used with `get_property`, for example:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
      core = ov.Core()

      # compile a model on AUTO and set log level to debug
      compiled_model = core.compile_model(model=model, device_name="AUTO")
      # query the runtime target devices on which the inferences are being executed
      execution_devices = compiled_model.get_property(properties.execution_devices)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  ov::Core core;

  // read a network in IR, PaddlePaddle, or ONNX format
  std::shared_ptr<ov::Model> model = core.read_model("sample.xml");

  // compile a model on AUTO and set log level to debug
  ov::CompiledModel compiled_model = core.compile_model(model, "AUTO");
  // query the runtime target devices on which the inferences are being executed
  ov::Any execution_devices = compiled_model.get_property(ov::execution_devices);
  ```
  </TabItem>
</Tabs>

## Configuring Individual Devices and Creating the Auto-Device plugin on Top

Although the methods described above are currently the preferred way to execute inference with AUTO, the following steps can be also used as an alternative. It is currently available as a legacy feature and used if AUTO is incapable of utilizing the Performance Hints option.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
      core = ov.Core()

      # gpu_config and cpu_config will load during compile_model()
      gpu_config = {
          hints.performance_mode: hints.PerformanceMode.THROUGHPUT,
          streams.num: 4
      }
      cpu_config = {
          hints.performance_mode: hints.PerformanceMode.LATENCY,
          streams.num: 8,
          properties.enable_profiling: True
      }
      compiled_model = core.compile_model(
          model=model,
          device_name="AUTO",
          config={
              device.priorities: "GPU,CPU",
              device.properties: {'CPU': cpu_config, 'GPU': gpu_config}
          }
      )
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  ov::Core core;

  // Read a network in IR, TensorFlow, TensorFlow Lite, PaddlePaddle, or ONNX format:
  std::shared_ptr<ov::Model> model = core.read_model("sample.xml");

  // Configure the CPU and the GPU devices when compiling model
  ov::CompiledModel compiled_model = core.compile_model(model, "AUTO",
      ov::device::properties("CPU", cpu_config),
      ov::device::properties("GPU", gpu_config));
  ```
  </TabItem>
</Tabs>


## Using AUTO with OpenVINO Samples and Benchmark app {#using-auto-with-openvino-samples-and-benchmark-app}

To see how the Auto-Device plugin is used in practice and test its performance, take a look at OpenVINO™ samples. All samples supporting the "-d" command-line option (which stands for "device") will accept the plugin out-of-the-box. The Benchmark Application will be a perfect place to start – it presents the optimal performance of the plugin without the need for additional settings, like the number of requests or CPU threads. To evaluate the AUTO performance, you can use the following commands:

For unlimited device choice:

```bash
benchmark_app –d AUTO –m <model> -i <input> -niter 1000
```

For limited device choice:

```bash
benchmark_app –d AUTO:CPU,GPU –m <model> -i <input> -niter 1000
```

For more information, refer to the [Benchmark Tool](../../../../get-started/learn-openvino/openvino-samples/benchmark-tool.md) article.

:::note
The default CPU stream is 1 if using “-d AUTO”.

You can use the FP16 IR to work with auto-device.

No demos are yet fully optimized for AUTO, by means of selecting the most suitable device, using the GPU streams/throttling, and so on.
:::

## Additional Resources

- [Automatic Device Selection with OpenVINO™ Notebook](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/auto-device/auto-device.ipynb)
- [Debugging AUTO](debugging-auto-device.md)
- [Inference Devices and Modes](../index.md)
