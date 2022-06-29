 # Model Optimization Guide {#openvino_docs_model_optimization_guide}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:
   
   pot_introduction
   docs_nncf_introduction
   (Experimental) Protecting Model <pot_range_supervision_README>

@endsphinxdirective

 Model optimization is an optional offline step of improving final model performance by applying special optimization methods like quantization, pruning, preprocessing optimization, etc. OpenVINO provides several tools to optimize models at different steps of model development:

- **Model Optimizer** implements optimization to a model, most of them added by default, but you can configure mean/scale values, batch size, RGB vs BGR input channels, and other parameters to speed up preprocess of a model ([Embedding Preprocessing Computation](../MO_DG/prepare_model/Additional_Optimizations.md)).

- **Post-training Optimization tool** [(POT)](../../tools/pot/docs/Introduction.md) is designed to optimize the inference of deep learning models by applying post-training methods that do not require model retraining or fine-tuning, for example, post-training 8-bit quantization. 

- **Neural Network Compression Framework** [(NNCF)](./nncf_introduction.md) provides a suite of advanced methods for training-time model optimization within the DL framework, such as PyTorch and TensorFlow. It supports methods, like Quantization-aware Training and Filter Pruning. NNCF-optimized models can be inferred with OpenVINO using all the available workflows.


## Detailed workflow: 

![](../img/DEVELOPMENT_FLOW_V3_crunch.svg)

To understand which development optimization tool you need, refer to the diagram: 

Post-training methods are limited in terms of achievable accuracy and for challenging use cases accuracy might degrade. In this case, training-time optimization with NNCF is an option.

Once the model is optimized using the aforementioned tools it can be used for inference using the regular OpenVINO inference workflow. No changes to the code are required.

![](../img/WHAT_TO_USE.svg)

If you are not familiar with model optimization methods, we recommend starting from [post-training methods](@ref pot_introduction).

## See also
- [Deployment optimization](./dldt_deployment_optimization_guide.md)