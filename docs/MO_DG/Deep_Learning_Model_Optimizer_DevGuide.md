# Model Optimizer Developer Guide {#openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:
   
   openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model
   openvino_docs_MO_DG_Additional_Optimization_Use_Cases
   openvino_docs_MO_DG_prepare_model_customize_model_optimizer_Customize_Model_Optimizer
   openvino_docs_MO_DG_prepare_model_Model_Optimizer_FAQ
   openvino_docs_MO_DG_Known_Issues_Limitations
   openvino_docs_MO_DG_Default_Model_Optimizer_Optimizations

@endsphinxdirective

## Introduction 

Model Optimizer is a cross-platform command-line tool that facilitates the transition between the training and deployment environment, performs static model analysis, and adjusts deep learning models for optimal execution on end-point target devices.

Model Optimizer process assumes you have a network model trained using supported deep learning frameworks: Caffe*, TensorFlow*, Kaldi*, MXNet* or converted to the ONNX* format. Model Optimizer produces an Intermediate Representation (IR) of the network, which can be inferred with the [Inference Engine](../IE_DG/Deep_Learning_Inference_Engine_DevGuide.md).

> **NOTE**: Model Optimizer does not infer models. Model Optimizer is an offline tool that runs before the inference takes place.

The scheme below illustrates the typical workflow for deploying a trained deep learning model: 

![](img/BASIC_FLOW_MO_simplified.svg)

The IR is a pair of files describing the model: 

*  <code>.xml</code> - Describes the network topology

*  <code>.bin</code> - Contains the weights and biases binary data.

> **TIP**: You also can work with the Model Optimizer inside the OpenVINO™ [Deep Learning Workbench](https://docs.openvinotoolkit.org/latest/workbench_docs_Workbench_DG_Introduction.html) (DL Workbench).
> [DL Workbench](https://docs.openvinotoolkit.org/latest/workbench_docs_Workbench_DG_Introduction.html) is a web-based graphical environment that enables you to optimize, fine-tune, analyze, visualize, and compare performance of deep learning models.

## Install Model Optimizer Pre-Requisites

Before running the Model Optimizer, you must install the Model Optimizer pre-requisites for the framework that was used to train the model.

@sphinxdirective
.. tab:: Using configuration scripts

   .. tab:: Linux

      .. tab:: All frameworks
      
         .. tab:: Install globally

            .. code-block:: sh

               cd <INSTALL_DIR>/deployment_tools/model_optimizer/install_prerequisites
               ./install_prerequisites.sh
         
         .. tab:: Install to virtualenv

            .. code-block:: sh

               cd <INSTALL_DIR>/deployment_tools/model_optimizer/install_prerequisites
               virtualenv --system-site-packages -p python3 ./venv
               source ./venv/bin/activate  # sh, bash, ksh, or zsh
               ./install_prerequisites.shs

      .. tab:: Caffe
      
         .. tab:: Install globally

            .. code-block:: sh

               cd <INSTALL_DIR>/deployment_tools/model_optimizer/install_prerequisitess
               install_prerequisites_caffe.sh
         
         .. tab:: Install to virtualenv

            .. code-block:: sh

               cd <INSTALL_DIR>/deployment_tools/model_optimizer/install_prerequisites
               install_prerequisites_caffe.shs

      .. tab:: Tensorflow 1.x
      
         .. tab:: Install globally

            test
         
         .. tab:: Install to virtualenv

            test

      .. tab:: Tensorflow 2.x
      
         .. tab:: Install globally

            test
         
         .. tab:: Install to virtualenv

            test

      .. tab:: MXNet
      
         .. tab:: Install globally

            test
         
         .. tab:: Install to virtualenv

            test

      .. tab:: ONNX
      
         .. tab:: Install globally

            test
         
         .. tab:: Install to virtualenv

            test

      .. tab:: Kaldi
      
         .. tab:: Install globally

            test
         
         .. tab:: Install to virtualenv

            test

   .. tab:: Windows

      .. tab:: All frameworks
      
         .. tab:: Install globally

            test
         
         .. tab:: Install to virtualenv

            test

      .. tab:: Caffe
      
         .. tab:: Install globally

            test
         
         .. tab:: Install to virtualenv

            test

      .. tab:: Tensorflow 1.x
      
         .. tab:: Install globally

            test
         
         .. tab:: Install to virtualenv

            test

      .. tab:: Tensorflow 2.x
      
         .. tab:: Install globally

            test
         
         .. tab:: Install to virtualenv

            tests

      .. tab:: MXNet
      
         .. tab:: Install globally

            test
         
         .. tab:: Install to virtualenv

            test

      .. tab:: ONNX
      
         .. tab:: Install globally

            test
         
         .. tab:: Install to virtualenv

            test

      .. tab:: Kaldi
      
         .. tab:: Install globally

            test
         
         .. tab:: Install to virtualenv

            test

   .. tab:: macOS

      .. tab:: All frameworks
      
         .. tab:: Install globally

            test
         
         .. tab:: Install to virtualenv

            test

      .. tab:: Caffe
      
         .. tab:: Install globally

            test
         
         .. tab:: Install to virtualenv

            test

      .. tab:: Tensorflow 1.x
      
         .. tab:: Install globally

            test
         
         .. tab:: Install to virtualenv

            test

      .. tab:: Tensorflow 2.x
      
         .. tab:: Install globally

            test
         
         .. tab:: Install to virtualenv

            tests

      .. tab:: MXNet
      
         .. tab:: Install globally

            test
         
         .. tab:: Install to virtualenv

            test

      .. tab:: ONNX
      
         .. tab:: Install globally

            test
         
         .. tab:: Install to virtualenv

            test

      .. tab:: Kaldi
      
         .. tab:: Install globally

            test
         
         .. tab:: Install to virtualenv

            test

.. tab:: Using manual configuration process

   .. tab:: Linux

      .. tab:: All frameworks
      
         test

      .. tab:: Caffe
      
         test

      .. tab:: Tensorflow 1.x
      
         test

      .. tab:: Tensorflow 2.x
      
         test

      .. tab:: MXNet
      
         test

      .. tab:: ONNX
      
         test

      .. tab:: Kaldi
      
         test


   .. tab:: Windows

      .. tab:: All frameworks
      
         test

      .. tab:: Caffe
      
         test

      .. tab:: Tensorflow 1.x
      
         test

      .. tab:: Tensorflow 2.x
      
         test

      .. tab:: MXNet
      
         test

      .. tab:: ONNX
      
         test

      .. tab:: Kaldi
      
         test

   .. tab:: macOS

      .. tab:: All frameworks
      
         test

      .. tab:: Caffe
      
         test

      .. tab:: Tensorflow 1.x
      
         test

      .. tab:: Tensorflow 2.x
      
         test

      .. tab:: MXNet
      
         test

      .. tab:: ONNX
      
         test

      .. tab:: Kaldi
      
         test

@endsphinxdirective

## Run Model Optimizer

To convert the model to the Intermediate Representation (IR), run Model Optimizer using the command for your type of OpenVINO™ installation:

@sphinxdirective
.. tab:: Package, Docker, open-source installation

   .. code-block:: sh

      python3 <INSTALL_DIR>/deployment_tools/model_optimizer/mo.py --input_model INPUT_MODEL --output_dir <OUTPUT_MODEL_DIR>

.. tab:: pip installation

    .. code-block:: sh

      mo --input_model INPUT_MODEL --output_dir <OUTPUT_MODEL_DIR>

@endsphinxdirective

You need to have have write permissions for an output directory.

> **NOTE**: Some models require using additional arguments to specify conversion parameters, such as `--input_shape`, `--scale`, `--scale_values`, `--mean_values`, `--mean_file`. To learn about when you need to use these parameters, refer to [Converting a Model to Intermediate Representation (IR)](Converting_Model.md).

To adjust the conversion process, you may use general parameters defined in the [Converting a Model to Intermediate Representation (IR)](Converting_Model.md) and 
framework-specific parameters for:
* [Caffe](Convert_Model_From_Caffe.md)
* [TensorFlow](Convert_Model_From_TensorFlow.md)
* [MXNet](Convert_Model_From_MxNet.md)
* [ONNX](Convert_Model_From_ONNX.md)
* [Kaldi](Convert_Model_From_Kaldi.md)

## Videos

@sphinxdirective

.. list-table::

   * - .. raw:: html

           <iframe width="220"
           src="https://www.youtube.com/embed/Kl1ptVb7aI8">
           </iframe>
    
     - .. raw:: html

           <iframe width="220"
           src="https://www.youtube.com/embed/BBt1rseDcy0">
           </iframe>

     - .. raw:: html

           <iframe width="220"
           src="https://www.youtube.com/embed/RF8ypHyiKrY">
           </iframe>

   * - **Model Optimizer Concept.**
     - **Model Optimizer Basic Operation.**
     - **Choosing the Right Precision.**

   * - Duration: 3:56
     - Duration: 2:57
     - Duration: 4:18

@endsphinxdirective
