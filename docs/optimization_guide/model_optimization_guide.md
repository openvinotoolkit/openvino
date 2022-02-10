 # Model Optimization Guide {#openvino_docs_model_optimization_guide}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:
   
   pot_README
   docs_nncf_introduction

@endsphinxdirective

 Model optimization assumes applying transformations to the model and relevant data flow to improve the inference performance. These transformations are basically offline and can require availability of training and validation data. This inlcudes suh methods as quantizaiton, pruning, preprocessing optmization, etc. OpenVINO provides several tools to optimize models at different steps of model development:

 - **Post-training Optimization tool [(POT)](../../tools/pot/README.md)** is designed to optimize the inference of deep learning models by applying post-training methods that do not require model retraining or fine-tuning, like post-training quantization. 

- **Neural Network Compression Framework [(NNCF)](./nncf_introduction.md)** provides a suite of advanced algorithms for Neural Networks inference optimization with minimal accuracy drop, for example, quantization, pruning algorithms.

- **Model Optimizer** implement some optimization to a model, most of them added by default, but you can configure mean/scale values, batch size RGB vs BGR input channels and other parameters to speed-up preprocess of a model ([Additional Optimization Use Cases](../MO_DG/prepare_model/Additional_Optimizations.md)) 


## Detailed workflow: 

![](../img/DEVELOPMENT_FLOW_V3_crunch.svg)

To understand when to use each development optimization tool, follow this diagram: 

POT is the easiest way to get optimized models and it is also really fast and usually takes several minutes depending on the model size and used HW. NNCF can be considered as an alternative or an addition when the first does not give accurate results. 

![](../img/WHAT_TO_USE.svg)

## See also
- [Deployment optimization](./dldt_deployment_optimization_guide.md)