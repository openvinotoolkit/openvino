# Post-training Quantization w/ POT {#pot_introduction}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   Quantizing Model <pot_default_quantization_usage>
   Quantizing Model with Accuracy Control <pot_accuracyaware_usage>
   Quantization Best Practices <pot_docs_BestPractices>
   API Reference <pot_compression_api_README>
   Command-line Interface <pot_compression_cli_README>
   Examples <pot_examples_description>
   pot_docs_FrequentlyAskedQuestions

@endsphinxdirective

For the needs of post-training optimization, OpenVINO&trade; provides a **Post-training Optimization Tool (POT)** which supports the **uniform integer quantization** method. This method allows moving from floating-point precision to integer precision (for example, 8-bit) for weights and activations during the inference time. It helps to reduce the model size, memory footprint and latency, as well as improve the computational efficiency, using integer arithmetic. During the quantization process the model undergoes the transformation process when additional operations, that contain quantization information, are inserted into the model. The actual transition to integer arithmetic happens at model inference.

The figure below shows the optimization workflow with POT:
![](./images/workflow_simple.png)

POT is distributed as a part of OpenVINO&trade; [Development Tools](@ref openvino_docs_install_guides_install_dev_tools) package and also available on [GitHub](https://github.com/openvinotoolkit/openvino/tree/master/tools/pot).

## Quantizing models with POT

Depending on your needs and requirements, POT provides two main quantization methods that can be used:

*  [Default Quantization](@ref pot_default_quantization_usage) -- a recommended method that provides fast and accurate results in most cases. It requires only an unannotated dataset for quantization. For more details, see the [Default Quantization algorithm](@ref pot_compression_algorithms_quantization_default_README) documentation.

*  [Accuracy-aware Quantization](@ref pot_accuracyaware_usage) -- an advanced method that allows keeping accuracy at a predefined range, at the cost of performance improvement, when `Default Quantization` cannot guarantee it. This method requires an annotated representative dataset and may require more time for quantization. For more details, see the
[Accuracy-aware Quantization algorithm](@ref accuracy_aware_README) documentation.

Different hardware platforms support different integer precisions and quantization parameters. For example, 8-bit is used by CPU, GPU, VPU, and 16-bit by GNA. POT abstracts this complexity by introducing a concept of the "target device" used to set quantization settings, specific to the device.

> **NOTE**: There is a special `target_device: "ANY"` which leads to portable quantized models compatible with CPU, GPU, and VPU devices. GNA-quantized models are compatible only with CPU.

For benchmarking results collected for the models optimized with the POT tool, refer to the [INT8 vs FP32 Comparison on Select Networks and Platforms](@ref openvino_docs_performance_int8_vs_fp32).

## Additional Resources

* [Performance Benchmarks](https://docs.openvino.ai/latest/openvino_docs_performance_benchmarks_openvino.html)
* [INT8 Quantization by Using Web-Based Interface of the DL Workbench](https://docs.openvino.ai/latest/workbench_docs_Workbench_DG_Int_8_Quantization.html)
