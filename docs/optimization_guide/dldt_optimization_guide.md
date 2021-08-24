# Optimization Guide {#openvino_docs_optimization_guide_dldt_optimization_guide}

## Introduction

Before exploring optimization techniques, let us first define what performance is and how it is measured.

### What Is Performance 

Performance means how fast the model is in deployment. Two key metrics are used to measure performance: latency and throughput. 

![](img/Lat_Thr.png)

Latency measures inference time (ms) required to process a single input.  

When it comes to batch input need to measure throughput (images per second or frames per second, FPS). To calculate throughput, divide number of frames that were processed by the processing time.   

### How to Measure Performance 

-bench app 

-first inference latency 

-measure performance on frameworks (new page) 

Note: not measure pre and post processing 

### How to Improve Performance 

Inside OpenVINO there are two independent ways how to get better performance number: during developing and deployment your model. It is possible to combine both developing and deployment optimizations. 

developing step includes model modification. Inside developing optimization there are two ways to optimize your model:  

Post-training Optimization tool (POT) that includes quantization algorithms.  

Neural Network Compression Framework (NNCF) that includes quantization aware training (QAT), training with sparsity and compression	 

deployment step, that includes tuning inference parameters and optimizing model execution 

![](img/scheme1.png)

To understand when to use each development optimization tool follow this diagram: 

POT is the easiest way to get optimized models, while NNCF can be considered as an alternative or an addition when POT does not give accurate results. 

![](img/scheme1.png)

Future reading: POT, NNCF, Runtime optimization 

Runtime Optimization Guide 

To optimize your performance results during runtime step it is possible to play with  

batching (throughput mode),  

threading (async API) 

combination of devices 

device optimization 

To read more -> [Runtime optimization guide](../install_guides/Intro_to_Performance.md)   

