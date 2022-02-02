# Model Optimizer Developer Guide {#openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:
   
   openvino_docs_MO_DG_IR_and_opsets
   openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model
   openvino_docs_MO_DG_Additional_Optimization_Use_Cases
   openvino_docs_MO_DG_prepare_model_customize_model_optimizer_Customize_Model_Optimizer
   openvino_docs_MO_DG_prepare_model_Model_Optimizer_FAQ
   openvino_docs_MO_DG_Known_Issues_Limitations
   openvino_docs_MO_DG_Default_Model_Optimizer_Optimizations

@endsphinxdirective

## Introduction 

Model Optimizer is a cross-platform command-line tool that facilitates the transition between the training and deployment environment, performs static model analysis, and adjusts deep learning models for optimal execution on end-point target devices.

Model Optimizer process assumes you have a network model trained using supported deep learning frameworks: Caffe*, TensorFlow*, Kaldi*, MXNet* or converted to the ONNX* format. Model Optimizer produces an Intermediate Representation (IR) of the network, which can be inferred with the [Inference Engine](../OV_Runtime_UG/Deep_Learning_Inference_Engine_DevGuide.md).

> **NOTE**: Model Optimizer does not infer models. Model Optimizer is an offline tool that runs before the inference takes place.

The scheme below illustrates the typical workflow for deploying a trained deep learning model: 

![](img/BASIC_FLOW_MO_simplified.svg)

The IR is a pair of files describing the model: 

*  <code>.xml</code> - Describes the network topology

*  <code>.bin</code> - Contains the weights and biases binary data.

> **TIP**: You also can work with the Model Optimizer inside the OpenVINO™ [Deep Learning Workbench](https://docs.openvino.ai/latest/workbench_docs_Workbench_DG_Introduction.html) (DL Workbench).
> [DL Workbench](https://docs.openvino.ai/latest/workbench_docs_Workbench_DG_Introduction.html) is a web-based graphical environment that enables you to optimize, fine-tune, analyze, visualize, and compare performance of deep learning models.

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
               virtualenv --system-site-packages -p python3 ./venv
               source ./venv/bin/activate  # sh, bash, ksh, or zsh
               install_prerequisites_caffe.shs

      .. tab:: Tensorflow 1.x
      
         .. tab:: Install globally

            .. code-block:: sh
               
               cd <INSTALL_DIR>/deployment_tools/model_optimizer/install_prerequisites
               install_prerequisites_tf.sh
         
         .. tab:: Install to virtualenv

            .. code-block:: sh
               
               cd <INSTALL_DIR>/deployment_tools/model_optimizer/install_prerequisites
               virtualenv --system-site-packages -p python3 ./venv
               source ./venv/bin/activate  # sh, bash, ksh, or zsh
               install_prerequisites_tf.sh

      .. tab:: Tensorflow 2.x
      
         .. tab:: Install globally

            .. code-block:: sh
               
               cd <INSTALL_DIR>/deployment_tools/model_optimizer/install_prerequisites
               install_prerequisites_tf2.sh
         
         .. tab:: Install to virtualenv

            .. code-block:: sh
               
               cd <INSTALL_DIR>/deployment_tools/model_optimizer/install_prerequisites
               virtualenv --system-site-packages -p python3 ./venv
               source ./venv/bin/activate  # sh, bash, ksh, or zsh
               install_prerequisites_tf2.sh

      .. tab:: MXNet
      
         .. tab:: Install globally

            .. code-block:: sh
               
               cd <INSTALL_DIR>/deployment_tools/model_optimizer/install_prerequisites
               install_prerequisites_mxnet.sh
         
         .. tab:: Install to virtualenv

            .. code-block:: sh
               
               cd <INSTALL_DIR>/deployment_tools/model_optimizer/install_prerequisites
               virtualenv --system-site-packages -p python3 ./venv
               source ./venv/bin/activate  # sh, bash, ksh, or zsh
               install_prerequisites_mxnet.sh

      .. tab:: ONNX
      
         .. tab:: Install globally

            .. code-block:: sh
               
               cd <INSTALL_DIR>/deployment_tools/model_optimizer/install_prerequisites
               install_prerequisites_onnx.sh
         
         .. tab:: Install to virtualenv

            .. code-block:: sh
               
               cd <INSTALL_DIR>/deployment_tools/model_optimizer/install_prerequisites
               virtualenv --system-site-packages -p python3 ./venv
               source ./venv/bin/activate  # sh, bash, ksh, or zsh
               install_prerequisites_onnx.sh

      .. tab:: Kaldi
      
         .. tab:: Install globally

            .. code-block:: sh
               
               cd <INSTALL_DIR>/deployment_tools/model_optimizer/install_prerequisites
               install_prerequisites_kaldi.sh
         
         .. tab:: Install to virtualenv

            .. code-block:: sh
               
               cd <INSTALL_DIR>/deployment_tools/model_optimizer/install_prerequisites
               virtualenv --system-site-packages -p python3 ./venv
               source ./venv/bin/activate  # sh, bash, ksh, or zsh
               install_prerequisites_kaldi.sh

   .. tab:: Windows

      .. tab:: All frameworks
      
         .. tab:: Install globally

            .. code-block:: sh
               
               cd <INSTALL_DIR>\deployment_tools\model_optimizer\install_prerequisites\
               install_prerequisites.bat
         
         .. tab:: Install to virtualenv

            .. code-block:: sh
               
               cd <INSTALL_DIR>\deployment_tools\model_optimizer\install_prerequisites
               virtualenv --system-site-packages -p python .\env
               env\Scripts\activate.bat
               install_prerequisites.bat

      .. tab:: Caffe
      
         .. tab:: Install globally

            .. code-block:: sh
               
               cd <INSTALL_DIR>\deployment_tools\model_optimizer\install_prerequisites\
               install_prerequisites_caffe.bat
         
         .. tab:: Install to virtualenv

            .. code-block:: sh
               
               cd <INSTALL_DIR>\deployment_tools\model_optimizer\install_prerequisites
               virtualenv --system-site-packages -p python .\env
               env\Scripts\activate.bat
               install_prerequisites_caffe.bat

      .. tab:: Tensorflow 1.x
      
         .. tab:: Install globally

            .. code-block:: sh
               
               cd <INSTALL_DIR>\deployment_tools\model_optimizer\install_prerequisites\
               install_prerequisites_tf.bat
         
         .. tab:: Install to virtualenv

            .. code-block:: sh
               
               cd <INSTALL_DIR>\deployment_tools\model_optimizer\install_prerequisites
               virtualenv --system-site-packages -p python .\env
               env\Scripts\activate.bat
               install_prerequisites_tf.bat

      .. tab:: Tensorflow 2.x
      
         .. tab:: Install globally

            .. code-block:: sh
               
               cd <INSTALL_DIR>\deployment_tools\model_optimizer\install_prerequisites\
               install_prerequisites_tf2.bat
         
         .. tab:: Install to virtualenv

            .. code-block:: sh
               
               cd <INSTALL_DIR>\deployment_tools\model_optimizer\install_prerequisites
               virtualenv --system-site-packages -p python .\env
               env\Scripts\activate.bat
               install_prerequisites_tf2.bat

      .. tab:: MXNet
      
         .. tab:: Install globally

            .. code-block:: sh
               
               cd <INSTALL_DIR>\deployment_tools\model_optimizer\install_prerequisites\
               install_prerequisites_mxnet.bat
         
         .. tab:: Install to virtualenv

            .. code-block:: sh
               
               cd <INSTALL_DIR>\deployment_tools\model_optimizer\install_prerequisites
               virtualenv --system-site-packages -p python .\env
               env\Scripts\activate.bat
               install_prerequisites_mxnet.bat

      .. tab:: ONNX
      
         .. tab:: Install globally

            .. code-block:: sh
               
               cd <INSTALL_DIR>\deployment_tools\model_optimizer\install_prerequisites\
               install_prerequisites_onnx.bat
         
         .. tab:: Install to virtualenv

            .. code-block:: sh
               
               cd <INSTALL_DIR>\deployment_tools\model_optimizer\install_prerequisites
               virtualenv --system-site-packages -p python .\env
               env\Scripts\activate.bat
               install_prerequisites_onnx.bat

      .. tab:: Kaldi
      
         .. tab:: Install globally

            .. code-block:: sh
               
               cd <INSTALL_DIR>\deployment_tools\model_optimizer\install_prerequisites\
               install_prerequisites_kaldi.bat
         
         .. tab:: Install to virtualenv

            .. code-block:: sh
               
               cd <INSTALL_DIR>\deployment_tools\model_optimizer\install_prerequisites
               virtualenv --system-site-packages -p python .\env
               env\Scripts\activate.bat
               install_prerequisites_kaldi.bat

   .. tab:: macOS

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
               virtualenv --system-site-packages -p python3 ./venv
               source ./venv/bin/activate  # sh, bash, ksh, or zsh
               install_prerequisites_caffe.shs

      .. tab:: Tensorflow 1.x
      
         .. tab:: Install globally

            .. code-block:: sh
               
               cd <INSTALL_DIR>/deployment_tools/model_optimizer/install_prerequisites
               install_prerequisites_tf.sh
         
         .. tab:: Install to virtualenv

            .. code-block:: sh
               
               cd <INSTALL_DIR>/deployment_tools/model_optimizer/install_prerequisites
               virtualenv --system-site-packages -p python3 ./venv
               source ./venv/bin/activate  # sh, bash, ksh, or zsh
               install_prerequisites_tf.sh

      .. tab:: Tensorflow 2.x
      
         .. tab:: Install globally

            .. code-block:: sh
               
               cd <INSTALL_DIR>/deployment_tools/model_optimizer/install_prerequisites
               install_prerequisites_tf2.sh
         
         .. tab:: Install to virtualenv

            .. code-block:: sh
               
               cd <INSTALL_DIR>/deployment_tools/model_optimizer/install_prerequisites
               virtualenv --system-site-packages -p python3 ./venv
               source ./venv/bin/activate  # sh, bash, ksh, or zsh
               install_prerequisites_tf2.sh

      .. tab:: MXNet
      
         .. tab:: Install globally

            .. code-block:: sh
               
               cd <INSTALL_DIR>/deployment_tools/model_optimizer/install_prerequisites
               install_prerequisites_mxnet.sh
         
         .. tab:: Install to virtualenv

            .. code-block:: sh
               
               cd <INSTALL_DIR>/deployment_tools/model_optimizer/install_prerequisites
               virtualenv --system-site-packages -p python3 ./venv
               source ./venv/bin/activate  # sh, bash, ksh, or zsh
               install_prerequisites_mxnet.sh

      .. tab:: ONNX
      
         .. tab:: Install globally

            .. code-block:: sh
               
               cd <INSTALL_DIR>/deployment_tools/model_optimizer/install_prerequisites
               install_prerequisites_onnx.sh
         
         .. tab:: Install to virtualenv

            .. code-block:: sh
               
               cd <INSTALL_DIR>/deployment_tools/model_optimizer/install_prerequisites
               virtualenv --system-site-packages -p python3 ./venv
               source ./venv/bin/activate  # sh, bash, ksh, or zsh
               install_prerequisites_onnx.sh

      .. tab:: Kaldi
      
         .. tab:: Install globally

            .. code-block:: sh
               
               cd <INSTALL_DIR>/deployment_tools/model_optimizer/install_prerequisites
               install_prerequisites_kaldi.sh
         
         .. tab:: Install to virtualenv

            .. code-block:: sh
               
               cd <INSTALL_DIR>/deployment_tools/model_optimizer/install_prerequisites
               virtualenv --system-site-packages -p python3 ./venv
               source ./venv/bin/activate  # sh, bash, ksh, or zsh
               install_prerequisites_kaldi.sh

.. tab:: Using manual configuration process

   .. tab:: Linux

      .. tab:: All frameworks
      
         .. code-block:: sh
               
            cd <INSTALL_DIR>/deployment_tools/model_optimizer/
            virtualenv --system-site-packages -p python3 ./venv
            source ./venv/bin/activate
            pip3 install -r requirements.txt

      .. tab:: Caffe
      
         .. code-block:: sh
               
            cd <INSTALL_DIR>/deployment_tools/model_optimizer/
            virtualenv --system-site-packages -p python3 ./venv
            source ./venv/bin/activate
            pip3 install -r requirements_caffe.txt

      .. tab:: Tensorflow 1.x
      
         .. code-block:: sh
               
            cd <INSTALL_DIR>/deployment_tools/model_optimizer/
            virtualenv --system-site-packages -p python3 ./venv
            source ./venv/bin/activate
            pip3 install -r requirements_tf.txt

      .. tab:: Tensorflow 2.x
      
         .. code-block:: sh
               
            cd <INSTALL_DIR>/deployment_tools/model_optimizer/
            virtualenv --system-site-packages -p python3 ./venv
            source ./venv/bin/activate
            pip3 install -r requirements_tf2.txt

      .. tab:: MXNet
      
         .. code-block:: sh
               
            cd <INSTALL_DIR>/deployment_tools/model_optimizer/
            virtualenv --system-site-packages -p python3 ./venv
            source ./venv/bin/activate
            pip3 install -r requirements_mxnet.txt

      .. tab:: ONNX
      
         .. code-block:: sh
               
            cd <INSTALL_DIR>/deployment_tools/model_optimizer/
            virtualenv --system-site-packages -p python3 ./venv
            source ./venv/bin/activate
            pip3 install -r requirements_onnx.txt

      .. tab:: Kaldi
      
         .. code-block:: sh
               
            cd <INSTALL_DIR>/deployment_tools/model_optimizer/
            virtualenv --system-site-packages -p python3 ./venv
            source ./venv/bin/activate
            pip3 install -r requirements_kaldi.txt

   .. tab:: Windows

      .. tab:: All frameworks
      
         .. code-block:: sh
               
            cd <INSTALL_DIR>\deployment_tools\model_optimizer
            virtualenv --system-site-packages -p python .\env
            env\Scripts\activate.bat
            pip install -r requirements.txt

      .. tab:: Caffe
      
         .. code-block:: sh
               
            cd <INSTALL_DIR>\deployment_tools\model_optimizer
            virtualenv --system-site-packages -p python .\env
            env\Scripts\activate.bat
            pip install -r requirements_caffe.txt

      .. tab:: Tensorflow 1.x
      
         .. code-block:: sh
               
            cd <INSTALL_DIR>\deployment_tools\model_optimizer
            virtualenv --system-site-packages -p python .\env
            env\Scripts\activate.bat
            pip install -r requirements_tf.txt

      .. tab:: Tensorflow 2.x
      
         .. code-block:: sh
               
            cd <INSTALL_DIR>\deployment_tools\model_optimizer
            virtualenv --system-site-packages -p python .\env
            env\Scripts\activate.bat
            pip install -r requirements_tf2.txt

      .. tab:: MXNet
      
         .. code-block:: sh
               
            cd <INSTALL_DIR>\deployment_tools\model_optimizer
            virtualenv --system-site-packages -p python .\env
            env\Scripts\activate.bat
            pip install -r requirements_mxnet.txt

      .. tab:: ONNX
      
         .. code-block:: sh
               
            cd <INSTALL_DIR>\deployment_tools\model_optimizer
            virtualenv --system-site-packages -p python .\env
            env\Scripts\activate.bat
            pip install -r requirements_onnx.txt

      .. tab:: Kaldi
      
         .. code-block:: sh
               
            cd <INSTALL_DIR>\deployment_tools\model_optimizer
            virtualenv --system-site-packages -p python .\env
            env\Scripts\activate.bat
            pip install -r requirements_kaldi.txt

   .. tab:: macOS

      .. tab:: All frameworks
      
         .. code-block:: sh
               
            cd <INSTALL_DIR>/deployment_tools/model_optimizer/
            virtualenv --system-site-packages -p python3 ./venv
            source ./venv/bin/activate
            pip3 install -r requirements.txt

      .. tab:: Caffe
      
         .. code-block:: sh
               
            cd <INSTALL_DIR>/deployment_tools/model_optimizer/
            virtualenv --system-site-packages -p python3 ./venv
            source ./venv/bin/activate
            pip3 install -r requirements_caffe.txt

      .. tab:: Tensorflow 1.x
      
         .. code-block:: sh
               
            cd <INSTALL_DIR>/deployment_tools/model_optimizer/
            virtualenv --system-site-packages -p python3 ./venv
            source ./venv/bin/activate
            pip3 install -r requirements_tf.txt

      .. tab:: Tensorflow 2.x
      
         .. code-block:: sh
               
            cd <INSTALL_DIR>/deployment_tools/model_optimizer/
            virtualenv --system-site-packages -p python3 ./venv
            source ./venv/bin/activate
            pip3 install -r requirements_tf2.txt

      .. tab:: MXNet
      
         .. code-block:: sh
               
            cd <INSTALL_DIR>/deployment_tools/model_optimizer/
            virtualenv --system-site-packages -p python3 ./venv
            source ./venv/bin/activate
            pip3 install -r requirements_mxnet.txt

      .. tab:: ONNX
      
         .. code-block:: sh
               
            cd <INSTALL_DIR>/deployment_tools/model_optimizer/
            virtualenv --system-site-packages -p python3 ./venv
            source ./venv/bin/activate
            pip3 install -r requirements_onnx.txt

      .. tab:: Kaldi
      
         .. code-block:: sh
               
            cd <INSTALL_DIR>/deployment_tools/model_optimizer/
            virtualenv --system-site-packages -p python3 ./venv
            source ./venv/bin/activate
            pip3 install -r requirements_kaldi.txt

@endsphinxdirective

## Run Model Optimizer

To convert the model to the Intermediate Representation (IR), run Model Optimizer:

```sh
mo --input_model INPUT_MODEL --output_dir <OUTPUT_MODEL_DIR>
```

You need to have have write permissions for an output directory.

> **NOTE**: Some models require using additional arguments to specify conversion parameters, such as `--input_shape`, `--scale`, `--scale_values`, `--mean_values`, `--mean_file`. To learn about when you need to use these parameters, refer to [Converting a Model to Intermediate Representation (IR)](prepare_model/convert_model/Converting_Model.md).

To adjust the conversion process, you may use general parameters defined in the [Converting a Model to Intermediate Representation (IR)](prepare_model/convert_model/Converting_Model.md) and 
framework-specific parameters for:
* [Caffe](prepare_model/convert_model/Convert_Model_From_Caffe.md)
* [TensorFlow](prepare_model/convert_model/Convert_Model_From_TensorFlow.md)
* [MXNet](prepare_model/convert_model/Convert_Model_From_MxNet.md)
* [ONNX](prepare_model/convert_model/Convert_Model_From_ONNX.md)
* [Kaldi](prepare_model/convert_model/Convert_Model_From_Kaldi.md)

## Videos

@sphinxdirective

.. list-table::

   * - .. raw:: html

           <iframe allowfullscreen mozallowfullscreen msallowfullscreen oallowfullscreen webkitallowfullscreen width="220"
           src="https://www.youtube.com/embed/Kl1ptVb7aI8">
           </iframe>
    
     - .. raw:: html

           <iframe allowfullscreen mozallowfullscreen msallowfullscreen oallowfullscreen webkitallowfullscreen width="220"
           src="https://www.youtube.com/embed/BBt1rseDcy0">
           </iframe>

     - .. raw:: html

           <iframe allowfullscreen mozallowfullscreen msallowfullscreen oallowfullscreen webkitallowfullscreen width="220"
           src="https://www.youtube.com/embed/RF8ypHyiKrY">
           </iframe>

   * - **Model Optimizer Concept.**
     - **Model Optimizer Basic Operation.**
     - **Choosing the Right Precision.**

   * - Duration: 3:56
     - Duration: 2:57
     - Duration: 4:18

@endsphinxdirective
