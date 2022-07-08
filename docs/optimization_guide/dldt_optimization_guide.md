# Introduction to Performance Optimization {#openvino_docs_optimization_guide_dldt_optimization_guide}
Even though inference performance should be defined as a combination of many factors, including accuracy and efficiency, it is most often described as the speed of execution. As the rate with which the model processes live data, it is based on two fundamentally interconnected metrics: latency and throughput.



![](../img/LATENCY_VS_THROUGHPUT.svg)

**Latency** measures inference time (in ms) required to process a single input. When it comes to executing multiple inputs simultaneously (for example, via batching), the overall throughput (inferences per second, or frames per second, FPS, in the specific case of visual processing) is usually more of a concern.
**Throughput** is calculated by dividing the number of inputs that were processed by the processing time.

## End-to-End Application Performance
It is important to separate the "pure" inference time of a neural network and the end-to-end application performance. For example, data transfers between the host and a device may unintentionally affect the performance when a host input tensor is processed on the accelerator such as dGPU.

Similarly, the input-preprocessing contributes significantly to the inference time. As described in the [getting performance numbers](../MO_DG/prepare_model/Getting_performance_numbers.md) section, when evaluating *inference* performance, one option is to measure all such items separately. 
For the **end-to-end scenario**, though, consider image pre-processing with OpenVINO and the asynchronous execution as a way to lessen the communication costs (like data transfers). For more details, see the [general optimizations guide](./dldt_deployment_optimization_common.md).

Another specific case is **first-inference latency** (for example, when a fast application start-up is required), where the resulting performance may be well dominated by the model loading time. [Model caching](../OV_Runtime_UG/Model_caching_overview.md) may be considered as a way to improve model loading/compilation time.

Finally, **memory footprint** restriction is another possible concern when designing an application. While this is a motivation for the use of the *model* optimization techniques, keep in mind that the throughput-oriented execution is usually much more memory consuming. For more details, see the [Runtime Inference Optimizations guide](../optimization_guide/dldt_deployment_optimization_guide.md). 


> **NOTE**: To get performance numbers for OpenVINO, along with the tips on how to measure and compare it with a native framework, see the [Getting performance numbers article](../MO_DG/prepare_model/Getting_performance_numbers.md).
 
## Improving Performance: Model vs Runtime Optimizations 

> **NOTE**: First, make sure that your model can be successfully inferred with OpenVINO Runtime. 

There are two primary optimization approaches to improving inference performance with OpenVINO: model- and runtime-level optimizations. They are **fully compatible** and can be done independently.

- **Model optimizations** include model modifications, such as quantization, pruning, optimization of preprocessing, etc. For more details, refer to this [document](./model_optimization_guide.md). 
    - The model optimizations directly improve the inference time, even without runtime parameters tuning (described below).

- **Runtime (Deployment) optimizations** includes tuning of model *execution* parameters. Fore more details, see the [Runtime Inference Optimizations guide](../optimization_guide/dldt_deployment_optimization_guide.md).

## Performance benchmarks
A wide range of public models for estimating performance and comparing the numbers (measured on various supported devices) are available in the [Performance benchmarks section](../benchmarks/performance_benchmarks.md).