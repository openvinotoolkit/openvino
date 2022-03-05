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

- **Deployment optimization**  includes tuning inference parameters and optimizing model execution. To read more visit [Deployment Optimization Guide](../optimization_guide/dldt_deployment_optimization_guide.md).

### **Runtime**/Deployment optimizations
Runtime optimizations include tuning of the inference parameters (e.g. optimal number of the inference requests executed simultaneously) and other means of how a model is _executed_. 

Here, possible optimization should start with defining the use-case. For example, whether the target scenario emphasizes throughput over latency. For example processing millions of samples by overnight jobs in the data centers.
In contrast, real-time usages would likely trade off the throughput to deliver the results at minimal latency. 
Often this is a combined scenario that targets highest possible throughput while maintaining a specific latency threshold.

Each of the [OpenVINO supported devices](../OV_Runtime_UG/supported_plugins/Device_Plugins.md) offers a bunch of performance settings. These settings provide really low-level control and allow to leverage the optimal model performance on the _specific_ device. At the same time, the detailed configuration require understanding of the device internals and is NOT portable between different types of the devices (like using GPU-optimal number of execution streams for the CPU). In fact the settings are not fully performance-portable even within a family of the devices (e.g. iGPU vs dGPU, or different versions of the iGPUs). Lastly, any hard-coded device performance settings are not future-proof​ and may require careful re-tuning when the model has changed.
**If the performance portability is of concern, consider using the [OpenVINO High-Level Performance Hints](../OV_Runtime_UG/performance_hints.md) first.**  

Finally, how the full-stack application uses the inference component _end-to-end_ is important.  
For example, what are the stages that needs to be orchestrated? In some cases a significant part of the workload time is spent on bringing and preparing the input data. Here the asynchronous inference should  increases performance by overlapping the compute with inputs population. Also, in many cases the (image) pre-processing can be offloaded to the OpenVINO. For variably-sized inputs, consider [dynamic shapes](../OV_Runtime_UG/ov_dynamic_shapes.md) to efficiently connect the data input pipeline and the model inference.

For further in-depth reading on the performance topics, please visit [Deployment Optimization Guide](./dldt_deployment_optimization_guide.md).

