# Performance Optimization Guide {#openvino_docs_optimization_guide_dldt_optimization_guide}


Before exploring optimization techniques, let us first define what performance is and how it is measured.

## What Is Performance 

Performance means how fast the model is in deployment. Two key metrics are used to measure performance: latency and throughput. 

![](../img/LATENCY_VS_THROUGHPUT.svg)

Latency measures inference time (ms) required to process a single input. When it comes to batch input need to measure throughput (images per second or frames per second, FPS). To calculate throughput, divide number of frames that were processed by the processing time.   

## How to measure performance
To get performance numbers for OpenVINO, as well as tips how to measure it and compare with native framework, check [Getting performance numbers](../MO_DG/prepare_model/Getting_performance_numbers.md) page.
 
## How to Improve Performance 

> **NOTE**: Make sure that your model can be successfully inferred with OpenVINO Inference Engine before diving into optimization topic. 

Inside OpenVINO there are two ways how to get better performance numbers: optimize the model that we call **model optimization** or tune parameters of execution which is called **deployment optimization**. Note, that it is possible to combine both types of optimizations. 

- **Model optimization** includes model modification, such as quantization, pruning, optimization of preprocessing, etc. Fore more details please refer to this [document](./model_optimization_guide.md).

- **Deployment optimization**  includes tuning inference parameters and optimizing model execution, to read more visit [Deployment Optimization Guide](../optimization_guide/dldt_deployment_optimization_guide.md).

## Performance benchmarks
To estimate the performance and for the sake of comparison performance numbers measured on various supported devices for a wide range of public model are available at [Perforance benchmarks](../benchmarks/performance_benchmarks.md) section.