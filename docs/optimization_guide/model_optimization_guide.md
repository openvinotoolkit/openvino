# Model Optimization Guide {#openvino_docs_model_optimization_guide}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   ptq_introduction
   tmo_introduction
   (Experimental) Protecting Model <pot_ranger_README>


Model optimization is an optional offline step of improving final model performance by applying special optimization methods, such as quantization, pruning, preprocessing optimization, etc. OpenVINO provides several tools to optimize models at different steps of model development:

- :doc:`Model Optimizer <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>` implements most of the optimization parameters to a model by default. Yet, you are free to configure mean/scale values, batch size, RGB vs BGR input channels, and other parameters to speed up preprocess of a model (:doc:`Embedding Preprocessing Computation <openvino_docs_MO_DG_Additional_Optimization_Use_Cases>`).

- :doc:`Post-training Quantization <pot_introduction>` is designed to optimize inference of deep learning models by applying post-training methods that do not require model retraining or fine-tuning, for example, post-training 8-bit integer quantization.

- :doc:`Training-time Optimization <nncf_ptq_introduction>`, a suite of advanced methods for training-time model optimization within the DL framework, such as PyTorch and TensorFlow 2.x. It supports methods, like Quantization-aware Training and Filter Pruning. NNCF-optimized models can be inferred with OpenVINO using all the available workflows.


Detailed workflow:
##################

To understand which development optimization tool you need, refer to the diagram:

.. image:: _static/images/DEVELOPMENT_FLOW_V3_crunch.svg

Post-training methods are limited in terms of achievable accuracy-performance trade-off for optimizing models. In this case, training-time optimization with NNCF is an option.

Once the model is optimized using the aforementioned tools it can be used for inference using the regular OpenVINO inference workflow. No changes to the inference code are required.

.. image:: _static/images/WHAT_TO_USE.svg

Post-training methods are limited in terms of achievable accuracy, which may degrade for certain scenarios.  In such cases, training-time optimization with NNCF may give better results.

Once the model has been optimized using the aforementioned tools, it can be used for inference using the regular OpenVINO inference workflow. No changes to the code are required.

If you are not familiar with model optimization methods, refer to :doc:`post-training methods <pot_introduction>`.

Additional Resources
####################

- :doc:`Deployment optimization <openvino_docs_deployment_optimization_guide_dldt_optimization_guide>`

@endsphinxdirective
