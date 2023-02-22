# Inference Modes {#openvino_docs_Runtime_Inference_Modes_Overview}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_OV_UG_supported_plugins_AUTO
   openvino_docs_OV_UG_Running_on_multiple_devices
   openvino_docs_OV_UG_Hetero_execution
   openvino_docs_OV_UG_Automatic_Batching
 
@endsphinxdirective

OpenVINO Runtime offers multiple inference modes to allow optimum hardware utilization under different conditions. The most basic one is a single-device mode, which defines just one device responsible for the entire inference workload. It supports a range of Intel hardware by means of plugins embedded in the Runtime library, each set up to offer the best possible performance. For a complete list of supported devices and instructions on how to use them, refer to the [guide on inference devices](../OV_Runtime_UG/supported_plugins/Device_Plugins.md).

The remaining modes assume certain levels of automation in selecting devices for inference. Using them in the deployed solution may potentially increase its performance and portability. The automated modes are:

* [Automatic Device Selection (AUTO)](../OV_Runtime_UG/auto_device_selection.md)
* [Multi-Device Execution (MULTI)](../OV_Runtime_UG/multi_device.md)
* [Heterogeneous Execution (HETERO)](../OV_Runtime_UG/hetero_execution.md)
* [Automatic Batching Execution (Auto-batching)](../OV_Runtime_UG/automatic_batching.md)

