# Performance Optimization Guide {#openvino_docs_optimization_guide_dldt_optimization_guide}

## Introduction

Before exploring optimization techniques, let us first define what performance is and how it is measured.

### What Is Performance 

Performance means how fast the model is in deployment. Two key metrics are used to measure performance: latency and throughput. 

![](img/Latency_Throughput.png)

Latency measures inference time (ms) required to process a single input.  

When it comes to batch input need to measure throughput (images per second or frames per second, FPS). To calculate throughput, divide number of frames that were processed by the processing time.   

>**NOTE**: To get performance numbers for OpenVINO, tips how to measure it and compare with native framework check Getting performance numbers page
 
-first inference latency 

-Using Caching API for first inference latency optimization 

Since with the 2021.4 release, Inference Engine provides the ability to enable internal caching of loaded networks. This can significantly reduce load network latency for some devices at application startup. Internally caching uses plugin's Export/ImportNetwork flow, like it is done for Compile tool, using the regular ReadNetwork/LoadNetwork API. Refer to the Model Caching Overview for more detailed explanation. 

### How to Improve Performance 

NOTE: Be sure that your model can be successfully inferred with OpenVINO Inference Engine.   

Inside OpenVINO there are two ways how to get better performance number: during developing and deployment your model. It is possible to combine both developing and deployment optimizations. 

developing step includes model modification. Inside developing optimization there are two ways to optimize your model:  

Post-training Optimization tool (POT) is designed to optimize the inference of deep learning models by applying special methods without model retraining or fine-tuning, like post-training quantization.  

Neural Network Compression Framework (NNCF) provides a suite of advanced algorithms for Neural Networks inference optimization with minimal accuracy drop.  

Model Optimizer implement some optimization to a model, most of them added by default, but you can configure some parameters to speed-up preprocess of a model (Additional Optimization Use Cases) 

deployment step, that includes tuning inference parameters and optimizing model execution, to read more visit Deployment Optimization Guide 

More detailed flow: 

![](img/development_deployment.png)

To understand when to use what development optimization tool follow this diagram: 

POT is the easiest way to get optimized models and it is also really fast and usually takes several minutes depending on the model size and used HW. NNCF can be considered as an alternative or an addition when the first does not give accurate results. 


Future reading: POT, NNCF, [Runtime optimization guide](../install_guides/Intro_to_Performance.md)   

