---
sidebar_position: 5
sidebar_label: "Optimizing for Throughput"
description: "Throughput-oriented approaches in OpenVINO involve execution of a large number of inference requests simultaneously which improves the device utilization."
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Optimizing for Throughput


As described in the section on the [latency-specific optimizations](../optimizing-latency/index.md), one of the possible use cases is delivering every single request with minimal delay.
Throughput, on the other hand, is about inference scenarios in which potentially **large numbers of inference requests are served simultaneously to improve resource use**.

The associated increase in latency is not linearly dependent on the number of requests executed in parallel.
A trade-off between overall throughput and serial performance of individual requests can be achieved with the right performance configuration of OpenVINO.

## Basic and Advanced Ways of Leveraging Throughput

There are two ways of leveraging throughput with individual devices:

- **Basic (high-level)** flow with [OpenVINO performance hints](../high-level-performance-hints.md) which is inherently **portable and future-proof**.
- **Advanced (low-level)** approach of explicit  **batching** and **streams**. For more details, see the [Advanced Throughput Options](advanced_throughput_options.md)

In both cases, the application should be designed to execute multiple inference requests in parallel, as described in the following section.


## Throughput-Oriented Application Design {#throughput-app-design}

In general, most throughput-oriented inference applications should:

- Expose substantial amounts of *input* parallelism (e.g. process multiple video- or audio- sources, text documents, etc).
- Decompose the data flow into a collection of concurrent inference requests that are aggressively scheduled to be executed in parallel:
  
  - Setup the configuration for the *device* (for example, as parameters of the `ov::Core::compile_model`) via either previously introduced [low-level explicit options](advanced_throughput_options.md) or [OpenVINO performance hints](../high-level-performance-hints.md) (**preferable**):
    
    <Tabs>
      <TabItem value="py" label="Python">
      ```python
          import openvino.properties as props
          import openvino.properties.hint as hints

          config = {hints.performance_mode: hints.PerformanceMode.THROUGHPUT}
          compiled_model = core.compile_model(model, "GPU", config)
      ```
      </TabItem>
      <TabItem value="cpp" label="C++">
      ```cpp
      auto compiled_model = core.compile_model(model, "GPU",
          ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
      ```
      </TabItem>
    </Tabs>
  - Query the `ov::optimal_number_of_infer_requests` from the `ov::CompiledModel` (resulted from a compilation of the model for the device) to create the number of the requests required to saturate the device.
- Use the Async API with callbacks, to avoid any dependency on the completion order of the requests and possible device starvation, as explained in the [common-optimizations section](../general-optimizations.md).

:::note
The [Automatic Device Selection](../../inference-devices-and-modes/auto-device-selection/index.md) allows configuration of all devices at once.
:::
