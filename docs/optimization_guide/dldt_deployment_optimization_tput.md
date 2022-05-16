# Optimizing for Throughput {#openvino_docs_deployment_optimization_guide_tput}

As described in the section on the [latency-specific considerations](./dldt_deployment_optimization_latency.md) one of the possible use-case is _delivering every single request at the minimal delay_.
Throughput on the other hand, is about inference scenarios in which potentially **large number of inference requests are served simultaneously to improve the device utilization**.

The associated increase in latency is not linearly dependent on the number of requests executed in parallel.
Here, a trade-off between overall throughput and serial performance of individual requests can be achieved with the right OpenVINO performance configuration.

##  Basic and Advanced Ways of Leveraging Throughput 
With the OpenVINO there are two means of leveraging the throughput with the individual device:
* **Basic (high-level)** flow with [OpenVINO performance hints](../OV_Runtime_UG/performance_hints.md) which is inherently **portable and future-proof**.
* **Advanced (low-level)** approach of explicit  **batching** and **streams** (for more details, see this [article](dldt_deployment_optimization_tput_advanced.md)).

In both cases application should be designed to execute multiple inference requests in parallel. Also, you should consider using the _automatic_ multi-device execution. In the next section You can find more details on those topics.

@anchor throughput_app_design
## Throughput-Oriented Application Design
In general, most throughput-oriented inference applications should:
* Expose substantial amounts of _inputs_ parallelism (e.g. process multiple video- or audio- sources, text documents, etc).
* Decompose the data flow into a collection of concurrent inference requests that are aggressively scheduled to be executed in parallel:
   * Setup the configuration for the _device_ (i.e., as parameters of the `ov::Core::compile_model`) via either introduced previously [low-level explicit options](dldt_deployment_optimization_tput_advanced.md) or [OpenVINO performance hints](../OV_Runtime_UG/performance_hints.md) (**preferable**):
   @sphinxdirective

   .. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_auto_batching.cpp
         :language: cpp
         :fragment: [compile_model]

   .. tab:: Python

      .. doxygensnippet:: docs/snippets/ov_auto_batching.py
         :language: python
         :fragment: [compile_model]

   @endsphinxdirective
   * Query the `ov::optimal_number_of_infer_requests` from the `ov::CompiledModel` (resulted from compilation of the model for a device) to create the number of the requests required to saturate the device.
* Use the Async API with callbacks, to avoid any dependency on the requests' completion order and possible device starvation.  explained in the [common-optimizations section](@ref openvino_docs_deployment_optimization_guide_common).

## Multi-Device Execution
OpenVINO offers automatic, scalable [multi-device inference](../OV_Runtime_UG/multi_device.md). This is simple _application-transparent_ way to improve the throughput. There is no need to re-architecture existing applications for any explicit multi-device support: no explicit network loading to each device, no separate per-device queues, no additional logic to balance the inference requests between devices, etc. From the application point of view, it is communicating to the single device that internally handles the actual machinery.
Just like with other throughput-oriented scenarios, there are two major pre-requisites for optimal multi-device performance:
*	Using the [Asynchronous API](@ref async_api) and [callbacks](../OV_Runtime_UG/ov_infer_request.md) in particular.
*	Providing the multi-device (and hence the underlying devices) with enough data to crunch. As the inference requests are naturally independent data pieces, the multi-device performs load-balancing at the “requests” (outermost) level to minimize the scheduling overhead.

Keep in mind that the resulting performance is usually a fraction of the “ideal” (plain sum) value, when the devices compete for a certain resources, like the memory-bandwidth which is shared between CPU and iGPU.

> **NOTE**: While the legacy approach of optimizing the parameters of each device separately works, the [OpenVINO performance hints](../OV_Runtime_UG/performance_hints.md) allow to configure all devices (that are part of the specific multi-device configuration) at once.
