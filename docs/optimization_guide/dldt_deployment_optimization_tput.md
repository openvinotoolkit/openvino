# Optimizing for Throughput {#openvino_docs_deployment_optimization_guide_tput}

As described in the section on the [latency-specific considerations](./dldt_deployment_optimization_latency.md), one of the possible use cases is *delivering every single request at the minimal delay*.
Throughput, on the other hand, is about inference scenarios in which potentially **large number of inference requests are served simultaneously to improve the device utilization**.

The associated increase in latency is not linearly dependent on the number of requests executed in parallel.
A trade-off between overall throughput and serial performance of individual requests can be achieved with the right performance configuration of OpenVINO.

##  Basic and Advanced Ways of Leveraging Throughput 
There are two ways of leveraging throughput with individual devices:
* **Basic (high-level)** flow with [OpenVINO performance hints](../OV_Runtime_UG/performance_hints.md) which is inherently **portable and future-proof**.
* **Advanced (low-level)** approach of explicit  **batching** and **streams**. For more details, see the [runtime inference optimizations](dldt_deployment_optimization_tput_advanced.md).

In both cases, the application should be designed to execute multiple inference requests in parallel, as described in the following section.

@anchor throughput_app_design
## Throughput-Oriented Application Design
In general, most throughput-oriented inference applications should:
* Expose substantial amounts of *input* parallelism (e.g. process multiple video- or audio- sources, text documents, etc).
* Decompose the data flow into a collection of concurrent inference requests that are aggressively scheduled to be executed in parallel:
   * Setup the configuration for the *device* (for example, as parameters of the `ov::Core::compile_model`) via either previously introduced [low-level explicit options](dldt_deployment_optimization_tput_advanced.md) or [OpenVINO performance hints](../OV_Runtime_UG/performance_hints.md) (**preferable**):
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
   * Query the `ov::optimal_number_of_infer_requests` from the `ov::CompiledModel` (resulted from a compilation of the model for the device) to create the number of the requests required to saturate the device.
* Use the Async API with callbacks, to avoid any dependency on the completion order of the requests and possible device starvation, as explained in the [common-optimizations section](@ref openvino_docs_deployment_optimization_guide_common).

## Multi-Device Execution
OpenVINO offers the automatic, scalable [multi-device inference mode](../OV_Runtime_UG/multi_device.md), which is a simple *application-transparent* way to improve throughput. There is no need to re-architecture existing applications for any explicit multi-device support: no explicit network loading to each device, no separate per-device queues, no additional logic to balance inference requests between devices, etc. For the application using it, multi-device is like any other device, as it manages all processes internally.
Just like with other throughput-oriented scenarios, there are several major pre-requisites for optimal multi-device performance:
*	Using the [Asynchronous API](@ref async_api) and [callbacks](../OV_Runtime_UG/ov_infer_request.md) in particular.
*	Providing the multi-device (and hence the underlying devices) with enough data to crunch. As the inference requests are naturally independent data pieces, the multi-device performs load-balancing at the "requests" (outermost) level to minimize the scheduling overhead.

Keep in mind that the resulting performance is usually a fraction of the "ideal" (plain sum) value, when the devices compete for certain resources such as the memory-bandwidth, which is shared between CPU and iGPU.

> **NOTE**: While the legacy approach of optimizing the parameters of each device separately works, the [OpenVINO performance hints](../OV_Runtime_UG/performance_hints.md) allow configuring all devices (that are part of the specific multi-device configuration) at once.
