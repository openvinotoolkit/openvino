# OpenVINO™ Training Extensions {#ote_documentation}

@sphinxdirective 

OpenVINO™ Training Extensions provide a suite of advanced algorithms to train
Deep Learning models and convert them using the `OpenVINO™
toolkit <https://software.intel.com/en-us/openvino-toolkit>`__ for optimized
inference. It allows you to export and convert the models to the needed format. OpenVINO Training Extensions independently create and train the model. It is open-sourced and available on `GitHub <https://github.com/openvinotoolkit/training_extensions>`__. Read the OpenVINO Training Extensions `documentation <https://openvinotoolkit.github.io/training_extensions/stable/guide/get_started/introduction.html>`__ to learn more.

Detailed Workflow
#################

.. image:: ./_static/images/training_extensions_framework.png

1. To start working with OpenVINO Training Extensions, prepare and annotate your dataset. For example, on CVAT.

2. OpenVINO Training Extensions train the model, using training interface, and evaluate the model quality on your dataset, using evaluation and inference interfaces.

   .. note:: 
      Prepare a separate dataset or split the dataset you have for more accurate quality evaluation.

3. Having successful evaluation results received, you have an opportunity to deploy your model or continue optimizing it, using NNCF and POT. For more information about these frameworks, go to :doc:`Optimization Guide <openvino_docs_model_optimization_guide>`.

If the results are unsatisfactory, add datasets and perform the same steps, starting with dataset annotation.

OpenVINO Training Extensions Components
#######################################

- `OpenVINO Training Extensions SDK <https://github.com/openvinotoolkit/training_extensions/tree/master/ote_sdk>`__
- `OpenVINO Training Extensions CLI <https://github.com/openvinotoolkit/training_extensions/tree/master/ote_cli>`__
- `OpenVINO Training Extensions Algorithms <https://github.com/openvinotoolkit/training_extensions/tree/master/external>`__

Tutorials
#########

`Object Detection <https://github.com/openvinotoolkit/training_extensions/blob/master/ote_cli/notebooks/train.ipynb>`__

@endsphinxdirective 


