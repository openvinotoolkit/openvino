# High-level Performance Hints (Presets) {#openvino_docs_deployment_optimization_guide_hints}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

@endsphinxdirective


Traditionally, each of the OpenVINO's [supported devices](../OV_Runtime_UG/supported_plugins/Supported_Devices.md) offers a bunch of low-level performance settings. 
Tweaking this detailed configuration requires deep architecture understanding.
Also, while the resulting performance may be optimal for the specific combination of the device and the model that is inferred, it is actually neither device/model nor future-proof:
- Even within a family of the devices (like various CPUs), things like different number of CPU cores would eventually result in different execution configuration to be optimal.
- Similarly the optimal batch size is very much specific to the particular instance of the GPU.
- Compute vs memory-bandwidth requirements for the model being inferenced, as well as inference precision, possible model's quantization and other factors add more unknowns to the resulting performance equation.
- Finally, the optimal execution parameters of one device do not transparently map to another device type, for example:
    - Both the CPU and GPU devices support the notion of the 'streams' (i.e. inference instances that are executed in parallel, please see `ov::num_streams`), yet the optimal number of the streams is deduced very differently.
 
Beyond execution _parameters_ there are potentially many device-specific details like _scheduling_ that greatly affect the performance. 
Specifically, GPU-oriented tricks like batching, which combines many (potentially tens) of input images to achieve optimal throughput, do not always map well to the CPU, as e.g. detailed in the next sections.
The hints allow to really hide _execution_ specifics required to saturate the device. For example, no need to explicitly combine multiple inputs into a batch to achieve good GPU performance.
Instead, it is possible to keep a separate infer request per camera or another source of input and process the requests in parallel using <a href="#ov-async-api">OpenVINO Async API</a>.

The only requirement for the application to leverage the throughput is about **running multiple inference requests in parallel**.
OpenVINO's device-specific implementation of the hints will take care of the rest. This allows a developer to greatly simplify the app-logic.

In summary, when the performance _portability_ is of concern, consider the [High-Level Performance Hints](../OV_Runtime_UG/performance_hints.md). 
Below you can find the implementation details (particularly how the OpenVINO implements the 'throughput' approach) for the specific devices. 
Keep in mind that while different throughput-oriented scheduling approaches ([like the batching or other means of executing individual inference requests](./dldt_deployment_optimization_tput.md)) can work together, the hints make these decisions to be transparent to the application.