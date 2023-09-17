# Inference Modes {#openvino_docs_Runtime_Inference_Modes_Overview}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_OV_UG_supported_plugins_AUTO
   openvino_docs_OV_UG_Running_on_multiple_devices
   openvino_docs_OV_UG_Hetero_execution
   openvino_docs_OV_UG_Automatic_Batching


OpenVINO Runtime offers multiple inference modes to allow optimum hardware utilization under different conditions. The most basic one is a single-device mode, which defines just one device responsible for the entire inference workload. It supports a range of Intel hardware by means of plugins embedded in the Runtime library, each set up to offer the best possible performance. For a complete list of supported devices and instructions on how to use them, refer to the :doc:`guide on inference devices <openvino_docs_OV_UG_Working_with_devices>`.

The remaining modes assume certain levels of automation in selecting devices for inference. Using them in the deployed solution may potentially increase its performance and portability. The automated modes are:

* :doc:`Automatic Device Selection (AUTO) <openvino_docs_OV_UG_supported_plugins_AUTO>`
* :doc:`Multi-Device Execution (MULTI) <openvino_docs_OV_UG_Running_on_multiple_devices>`
* :doc:`Heterogeneous Execution (HETERO) <openvino_docs_OV_UG_Hetero_execution>`
* :doc:`Automatic Batching Execution (Auto-batching) <openvino_docs_OV_UG_Automatic_Batching>`

@endsphinxdirective
