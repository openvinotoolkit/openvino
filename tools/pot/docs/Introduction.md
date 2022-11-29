# Quantizing Models Post-training {#pot_introduction}

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


Post-training quantization is a model compression technique where the values in a neural network are converted from a 32-bit or 16-bit format to an 8-bit integer format after the network has been fine-tuned on a training dataset. This helps to reduce the model’s latency by taking advantage of computationally efficient 8-bit integer arithmetic. It also reduces the model's size and memory footprint. 

Post-training quantization is easy to implement and is a quick way to boost model performance. It only requires a representative dataset, and it can be performed using the Post-training Optimization Tool (POT) in OpenVINO. POT is distributed as part of the [OpenVINO Development Tools](@ref openvino_docs_install_guides_install_dev_tools) package. To apply post-training quantization with POT, you need:

* A floating-point precision model, FP32 or FP16, converted into the OpenVINO Intermediate Representation (IR) format.
* A representative dataset (annotated or unannotated) of around 300 samples that depict typical use cases or scenarios.
* (Optional) An annotated validation dataset that can be used for checking the model’s accuracy.

The post-training quantization algorithm takes samples from the representative dataset, inputs them into the network, and calibrates the network based on the resulting weights and activation values. Once calibration is complete, values in the network are converted to 8-bit integer format.

While post-training quantization makes your model run faster and take less memory, it may cause a slight reduction in accuracy. If you performed post-training quantization on your model and find that it isn’t accurate enough, try using [Quantization-aware Training](@ref qat_introduction) to increase its accuracy.






The figure below shows the optimization workflow with POT:
![](./images/workflow_simple.png)

POT is distributed as a part of OpenVINO [Development Tools](@ref openvino_docs_install_guides_install_dev_tools) package and also available on [GitHub](https://github.com/openvinotoolkit/openvino/tree/master/tools/pot).

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
