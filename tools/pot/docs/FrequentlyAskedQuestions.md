# Post-training Optimization Tool FAQ {#pot_docs_FrequentlyAskedQuestions}

If your question is not covered below, use the [OpenVINO™ Community Forum page](https://community.intel.com/t5/Intel-Distribution-of-OpenVINO/bd-p/distribution-openvino-toolkit),
where you can participate freely.

- <a href="#opensourced">Is the Post-training Optimization Tool opensourced?</a>
- <a href="#dataset">Can I quantize my model without a dataset?</a>
- <a href="#framework">Can a model in any framework be quantized by the POT?</a>
- <a href="#tradeoff">What is a tradeoff when you go to low precision?</a>
- <a href="#noac">I'd like to quantize a model and I've converted it to IR but I don't have the Accuracy Checker config. What can I do?</a>
- <a href="#nncf">I tried all recommendations from "Post-Training Optimization Best Practices" but either have a high accuracy drop or bad performance after quantization.
What else can I do?</a>
- <a href="#memory">I get “RuntimeError: Cannot get memory” and “RuntimeError: Output data was not allocated” when I quantize my model by the POT.</a>
- <a href="#quality">I have successfully quantized my model with a low accuracy drop and improved performance but the output video generated from the low precision model is much worse than from the full precision model. What could be the root cause?</a>
- <a href="#longtime">The quantization process of my model takes a lot of time. Can it be decreased somehow?</a>
- <a href="#import">I get "Import Error:... No such file or directory". How can I avoid it?</a>
- <a href="#python">When I execute POT CLI, I get "File "/workspace/venv/lib/python3.7/site-packages/nevergrad/optimization/base.py", line 35... SyntaxError: invalid syntax". What is wrong?</a>
- <a href="#nomodule">What does a message "ModuleNotFoundError: No module named 'some\_module\_name'" mean?</a>
- <a href="#dump">Is there a way to collect an intermidiate IR when the AccuracyAware mechanism fails?</a>
- <a name="#outputs"> What do the messages "Output name: <result_operation_name> not found" or "Output node with <result_operation_name> is not found in graph" mean?</a>


### <a name="opensourced">Is the Post-training Optimization Tool (POT) opensourced?</a>

Yes, POT is developed on GitHub as a part of [https://github.com/openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino) under Apache-2.0 License.

### <a name="dataset">Can I quantize my model without a dataset?</a>

In general, you should have a dataset. The dataset should be annotated if you want to validate the accuracy.
If your dataset is not annotated, you can use [Default Quantization](@ref pot_default_quantization_usage) to quantize the model or command-line interface with [Simplified mode](@ref pot_docs_simplified_mode).

### <a name="framework">Can a model in any framework be quantized by the POT?</a>

The POT accepts models in the OpenVINO&trade; Intermediate Representation (IR) format only. For that you need to convert your model to the IR format using
[Model Optimizer](@ref openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide).

### <a name="noac">I'd like to quantize a model and I've converted it to IR but I don't have the Accuracy Checker config. What can I do?</a>

1. Try quantization using Python API of the Post-training Optimization Tool. For more details see [Default Quantization](@ref pot_default_quantization_usage).
2. If you consider command-line usage only refer to [Accuracy Checker documentation](@ref omz_tools_accuracy_checker) to create the Accuracy Checker configuration file, and try to find the configuration file for your model among the ones available in the Accuracy Checker examples. 
3. An alternative way is to quantize the model in the [Simplified mode](#ref pot_docs_simplified_mode) but you will not be able to measure the accuracy.

### <a name="tradeoff">What is a tradeoff when you go to low precision?</a>

The tradeoff is between the accuracy drop and performance. When a model is in low precision, it is usually performed
compared to the same model in full precision but the accuracy might be worse. You can find some benchmarking results in
[INT8 vs FP32 Comparison on Select Networks and Platforms](@ref openvino_docs_performance_int8_vs_fp32).
The other benefit of having a model in low precision is its smaller size.

### <a name="nncf">I tried all recommendations from "Post-Training Optimization Best Practices" but either have a high accuracy drop or bad performance after quantization. What else can I do?</a>

First of all, you should validate the POT compression pipeline you are running, which can be done with the following steps:
1.	Make sure the accuracy of the original uncompressed model has the value you expect. Run your POT pipeline with an empty compression config and evaluate the resulting model metric. Compare this uncompressed model accuracy metric value with your reference.
2.	Run your compression pipeline with a single compression algorithm ([Default Quantization](@ref pot_default_quantization_usage) or [Accuracy-aware Quantization](@ref pot_accuracyaware_usage)) without any parameter values specified in the config (except for `preset` and `stat_subset_size`). Make sure you get the desirable accuracy drop/performance gain in this case.

Finally, if you have done the steps above and the problem persists, you could try to compress your model using the [Neural Network Compression Framework (NNCF)](https://github.com/openvinotoolkit/nncf_pytorch).
Note that NNCF usage requires you to have a PyTorch or TensorFlow 2 based training pipeline of your model to perform Quantization-aware Training. See [Model Optimization Guide](@ref openvino_docs_model_optimization_guide) for more details.

### <a name="memory">I get “RuntimeError: Cannot get memory” and “RuntimeError: Output data was not allocated” when I quantize my model by the POT.</a>

These issues happen due to insufficient available amount of memory for statistics collection during the quantization process of a huge model or
due to a very high resolution of input images in the quantization dataset. If you do not have a possibility to increase your RAM size, one of the following options can help:
- Set `inplace_statistics` parameters to "True". In that case the POT will change method collect statistics and use less memory. Note that such change might increase time required for quantization.
- Set `eval_requests_number` and `stat_requests_number` parameters to 1. In that case the POT will limit the number of infer requests by 1 and use less memory.
Note that such change might increase time required for quantization.
- Set `use_fast_bias` parameter to `false`. In that case the POT will switch from the FastBiasCorrection algorithm to the full BiasCorrection algorithm
which is usually more accurate and takes more time but requires less memory. See [Post-Training Optimization Best Practices](BestPractices.md) for more details.
- Reshape your model to a lower resolution and resize the size of images in the dataset. Note that such change might impact the accuracy.

### <a name="quality">I have successfully quantized my model with a low accuracy drop and improved performance but the output video generated from the low precision model is much worse than from the full precision model. What could be the root cause?</a>

It can happen due to the following reasons:
- A wrong or not representative dataset was used during the quantization and accuracy validation. Please make sure that your data and labels are correct and they sufficiently reflect the use case.
- If the command-line interface was used for quantization, a wrong Accuracy Checker configuration file could lead to this problem. Refer to [Accuracy Checker documentation](@ref omz_tools_accuracy_checker) for more information.
- If [Default Quantization](@ref pot_default_quantization_usage) was used for quantization you can also try [Accuracy-aware Quantization](@ref pot_accuracyaware_usage) method that allows controlling maximum accuracy deviation.

### <a name="longtime">The quantization process of my model takes a lot of time. Can it be decreased somehow?</a>

Quantization time depends on multiple factors such as the size of the model and the dataset. It also depends on the algorithm:
the [Default Quantization](@ref pot_default_quantization_usage) algorithm takes less time than the [Accuracy-aware Quantization](@ref pot_accuracyaware_usage) algorithm.
The following configuration parameters also impact the quantization time duration
(see details in [Post-Training Optimization Best Practices](BestPractices.md)):
- `use_fast_bias`: when set to `false`, it increases the quantization time
- `stat_subset_size`: the higher the value of this parameter, the more time will be required for the quantization
- `tune_hyperparams`: if set to `true` when the AccuracyAwareQuantization algorithm is used, it increases the quantization time
- `stat_requests_number`: the lower number, the more time might be required for the quantization
- `eval_requests_number`: the lower number, the more time might be required for the quantization
Note that higher values of `stat_requests_number` and `eval_requests_number` increase memory consumption by POT.

### <a name="python">When I execute POT CLI, I get "File "/workspace/venv/lib/python3.7/site-packages/nevergrad/optimization/base.py", line 35... SyntaxError: invalid syntax". What is wrong?</a>

This error is reported when you have a Python version older than 3.7 in your environment. Upgrade your Python version.

### <a name="nomodule">What does a message "ModuleNotFoundError: No module named 'some\_module\_name'" mean?</a>

It means that some required python module is not installed in your environment. To install it, run `pip install some_module_name`.

### <a name="dump">Is there a way to collect an intermidiate IR when the AccuracyAware mechanism fails?</a>

You can add `"dump_intermediate_model": true` to the POT configuration file and it will drop an intermidiate IR to `accuracy_aware_intermediate` folder. 

### <a name="outputs"> What do the messages "Output name: <result_operation_name> not found" or "Output node with <result_operation_name> is not found in graph" mean?</a>
Errors are caused by missing output nodes names in a graph when using the POT tool for model quantization. It might appear for some models only for IRs converted from ONNX models using new frontend (which is the default conversion path starting from 2022.1 release). To avoid such errors, use legacy MO frontend to convert a model to IR by passing the --use_legacy_frontend option. Then, use the produced IR for quantization.
