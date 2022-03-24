# Optimizing for Throughput {#openvino_docs_deployment_optimization_guide_tput}

## General Throughput Considerations
As described in the section on the [latency-specific considerations](./dldt_deployment_optimization_latency.md) one possible use-case is delivering every single request at the minimal delay.
Throughput on the other hand, is about inference scenarios in which potentially large **number of inference requests are served simultaneously to improve the device utilization and total throughput**.

Also, the associated increase in latency is not linearly dependent on the number of requests executed in parallel.
Here, a trade-off between overall throughput and serial performance of individual requests can be achieved with the right OpenVINO performance configuration.

##  Basic and Advanced Ways of Leveraging Throughput 
With the OpenVINO there are two means of leveraging the throughput with the individual device:
* Basic flow with [OpenVINO performance hints](@ref throughput_hints) which is inherently portable and more future-proof
* Advanced (low-level) approach of  **batching** and **streams**, explained in the separate [document](./dldt_deployment_optimization_tput_advanced.md).

However, all possible design decisions, are predicated on the requirement that sufficient parallelism exists, as detailed in the [next section](@ref throughput_app_design).

Finally, consider the _automatic_ multi-device execution covered below.

@anchor throughput_app_design
## Throughput-Oriented Application Design
Most generally, throughput-oriented inference applications should:
* Expose substantial amounts of _inputs_ parallelism (e.g. process multiple video-sources)
* Decompose the data flow into a collection of concurrent inference requests that are aggressively scheduled to be executed in parallel
* Use the Async API with callbacks, to avoid any dependency on the requests' completion order, as explained in the [common-optimizations section](@ref openvino_docs_deployment_optimization_guide_common)

## Multi-Device Execution
OpenVINO offers automatic, [scalable multi-device inference](../OV_Runtime_UG/multi_device.md). This is simple _application-transparent_ way to improve the throughput. No need to re-architecture existing applications for any explicit multi-device support: no explicit network loading to each device, no separate per-device queues, no additional logic to balance the inference requests between devices, etc. From the application point of view, it is communicating to the single device that internally handles the actual machinery.
Just like with other throughput-oriented scenarios, there are two major pre-requisites for optimal multi-device performance:
*	Using the [Asynchronous API](@ref openvino_docs_deployment_optimization_guide_common) and [callbacks](../OV_Runtime_UG/ov_infer_request.md) in particular
*	Providing the multi-device (and hence the underlying devices) with enough data to crunch. As the inference requests are naturally independent data pieces, the multi-device performs load-balancing at the “requests” (outermost) level to minimize the scheduling overhead.

Notice that the resulting performance is usually a fraction of the “ideal” (plain sum) value, when the devices compete for a certain resources, like the memory-bandwidth which is shared between CPU and iGPU.
> **NOTE**: While the legacy approach of optimizing the parameters of each device separately works, the [OpenVINO performance hints](./dldt_deployment_optimization_hints.md) allow to configure all devices (that are part of the specific multi-device configuration) at once.
