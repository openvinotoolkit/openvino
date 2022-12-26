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

The post-training quantization algorithm takes samples from the representative dataset, inputs them into the network, and calibrates the network based on the resulting weights and activation values. Once calibration is complete, values in the network are converted to 8-bit integer format.

While post-training quantization makes your model run faster and take less memory, it may cause a slight reduction in accuracy. If you performed post-training quantization on your model and find that it isn’t accurate enough, try using [Quantization-aware Training](@ref qat_introduction) to increase its accuracy.


### Post-Training Quantization Quick Start Examples
Try out these interactive Jupyter Notebook examples to learn the POT API and see post-training quantization in action:

* [Quantization of Image Classification Models with POT](https://docs.openvino.ai/2022.2/notebooks/113-image-classification-quantization-with-output.html).
* [Object Detection Quantization with POT](https://docs.openvino.ai/2022.2/notebooks/111-detection-quantization-with-output.html).

## Quantizing Models with POT
The figure below shows the post-training quantization workflow with POT. In a typical workflow, a pre-trained model is converted to OpenVINO IR format using Model Optimizer. Then, the model is quantized with a representative dataset using POT.


![](./images/workflow_simple.svg)


### Post-training Quantization Methods
Depending on your needs and requirements, POT provides two quantization methods that can be used: Default Quantization and Accuracy-aware Quantization.

#### Default Quantization
Default Quantization uses an unannotated dataset to perform quantization. It uses representative dataset items to estimate the range of activation values in a network and then quantizes the network. This method is recommended to start with, because it results in a fast and accurate model in most cases. To quantize your model with Default Quantization, see the [Quantizing Models](@ref pot_default_quantization_usage) page.

#### Accuracy-aware Quantization
Accuracy-aware Quantization is an advanced method that maintains model accuracy within a predefined range by leaving some network layers unquantized. It uses a trade-off between speed and accuracy to meet user-specified requirements. This method requires an annotated dataset and may require more time for quantization. To quantize your model with Accuracy-aware Quantization, see the [Quantizing Models with Accuracy Control](@ref pot_accuracyaware_usage) page.

### Quantization Best Practices and FAQs
If you quantized your model and it isn’t accurate enough, visit the [Quantization Best Practices](@ref pot_docs_BestPractices) page for tips on improving quantized performance. Sometimes, older Intel CPU generations can encounter a saturation issue when running quantized models that can cause reduced accuracy: learn more on the [Saturation Issue Workaround](@ref pot_saturation_issue) page.

Have more questions about post-training quantization or encountering errors using POT? Visit the [POT FAQ](@ref pot_docs_FrequentlyAskedQuestions) page for answers to frequently asked questions and solutions to common errors.

## Additional Resources

* [Post-training Quantization Examples](@ref pot_examples_description)
* [Quantization Best Practices](@ref pot_docs_BestPractices)
* [Post-training Optimization Tool FAQ](@ref pot_docs_FrequentlyAskedQuestions)
* [Performance Benchmarks](@ref openvino_docs_performance_benchmarks_openvino)
