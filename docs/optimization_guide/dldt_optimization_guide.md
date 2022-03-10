# Performance Optimization Guide {#openvino_docs_optimization_guide_dldt_optimization_guide}
Before exploring possible optimization techniques, let us first define what performance is and how it is measured.
Notice that reported inference performance often tends to focus on the speed of execution. 
In fact these are at least four connected factors of accuracy, throughput/latency and efficiency. The rest of the document discusses how to balance these key factors. 


## What Is Inference Performance
Generally, performance means how fast is the model. Two key metrics are used to measure the performance: latency and throughput  are fundamentally interconnected. 

![](../img/LATENCY_VS_THROUGHPUT.svg)

Latency measures inference time (ms) required to process a single input. When it comes to the executing multiple inputs executed simultaneously (e.g. via batching) then the overall throughput (inferences per second, or frames per second, FPS, in the specific case of visual processing) is usually of more concern.
To calculate throughput, divide number of frames that were processed by the processing time.   

> **NOTE**: To get performance numbers for OpenVINO, as well as tips how to measure it and compare with native framework, check [Getting performance numbers](../MO_DG/prepare_model/Getting_performance_numbers.md) page.
 
## Improving the Performance: Model vs Runtime Optimizations 

> **NOTE**: Make sure that your model can be successfully inferred with OpenVINO Runtime. 

With the OpenVINO there are two primary ways of improving the inference performance, namely model- and runtime-level optimizations. **These two optimizations directions are fully compatible**. 

- **Model optimization** includes model modification, such as quantization, pruning, optimization of preprocessing, etc. Fore more details, refer to this [document](./model_optimization_guide.md).

- **Runtime (Deployment) optimization**  includes tuning of model _execution_ parameters. To read more visit [Deployment Optimization Guide](../optimization_guide/dldt_deployment_optimization_guide.md).

## Performance benchmarks
To estimate the performance and compare performance numbers, measured on various supported devices, a wide range of public models are available at [Performance benchmarks](../benchmarks/performance_benchmarks.md) section.