# Introduction to Performance Optimization {#openvino_docs_optimization_guide_dldt_optimization_guide}
Before exploring possible optimization techniques, let us first define what the inference performance is and how to measure that.
Notice that reported inference performance often tends to focus on the speed of execution. 
In fact these are at least four connected factors of accuracy, throughput/latency and efficiency. The rest of the document discusses how to balance these key factors. 


## What Is Inference Performance
Generally, performance means how fast the model processes the live data. Two key metrics are used to measure the performance: latency and throughput are fundamentally interconnected. 

![](../img/LATENCY_VS_THROUGHPUT.svg)

**Latency** measures inference time (ms) required to process a single input. When it comes to the executing multiple inputs simultaneously (e.g. via batching) then the overall throughput (inferences per second, or frames per second, FPS, in the specific case of visual processing) is usually of more concern.
To calculate **throughput**, divide number of inputs that were processed by the processing time.

## End-to-End Application Performance
It is important to separate the "pure" inference time of a neural network and the end-to-end application performance. For example data transfers between the host and a device may unintentionally affect the performance when a host input tensor is processed on the accelerator like dGPU.

Similarly, the input-preprocessing contributes significantly to the to inference time. As detailed in the [getting performance numbers](../MO_DG/prepare_model/Getting_performance_numbers.md) section, when drilling into _inference_ performance, one option is to measure all such items separately. 
For the **end-to-end scenario** though, consider the image pre-processing thru the OpenVINO and the asynchronous execution as a way to amortize the communication costs like data transfers. You can find further details in the [general optimizations document](./dldt_deployment_optimization_common.md).

**First-inference latency** is another specific case (e.g. when fast application start-up is required) where the resulting performance may be well dominated by the model loading time. Consider [model caching](../OV_Runtime_UG/Model_caching_overview.md) as a way to improve model loading/compilation time.

Finally, **memory footprint** restrictions is another possible concern when designing an application. While this is a motivation for the _model_ optimization techniques referenced in the next section, notice that the the throughput-oriented execution is usually much more memory-hungry, as detailed in the [Runtime Inference Optimizations](../optimization_guide/dldt_deployment_optimization_guide.md). 


> **NOTE**: To get performance numbers for OpenVINO, as well as tips how to measure it and compare with native framework, check [Getting performance numbers](../MO_DG/prepare_model/Getting_performance_numbers.md) page.
 
## Improving the Performance: Model vs Runtime Optimizations 

> **NOTE**: Make sure that your model can be successfully inferred with OpenVINO Runtime. 

With the OpenVINO there are two primary ways of improving the inference performance, namely model- and runtime-level optimizations. **These two optimizations directions are fully compatible**. 

- **Model optimizations** includes model modification, such as quantization, pruning, optimization of preprocessing, etc. Fore more details, refer to this [document](./model_optimization_guide.md). 
    - Notice that the model optimizations directly improve the inference time, even without runtime parameters tuning, described below

- **Runtime (Deployment) optimizations**  includes tuning of model _execution_ parameters. To read more visit the [Runtime Inference Optimizations](../optimization_guide/dldt_deployment_optimization_guide.md).

## Performance benchmarks
To estimate the performance and compare performance numbers, measured on various supported devices, a wide range of public models are available at [Performance benchmarks](../benchmarks/performance_benchmarks.md) section.