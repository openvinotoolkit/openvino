# Llama Sample C++ Sample {#openvino_inference_engine_samples_llama_sample_README}

@sphinxdirective

.. meta::
   :description: Learn how to do inference of llama models using Synchronous Inference Request 
                 (C++) API.


This sample demonstrates how to do inference of llama models using Synchronous Inference Request API. 

.. tab-set::

   .. tab-item:: Requirements 

      +-------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
      | Options                             | Values                                                                                                                                                                                |
      +=====================================+=======================================================================================================================================================================================+
      | Validated Models                    | :doc:`llama-2-7b-chat`, :doc:`llama-2-13b-chat`                                                                                                                                       |
      +-------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
      | Model Format                        | OpenVINO™ toolkit Intermediate Representation (\*.xml + \*.bin)                                                                                                                       |
      +-------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
      | Supported devices                   | :doc:`CPU`                                                                                                                                                                            |
      +-------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

   .. tab-item:: C++ API

      The following C++ API is used in the application:

      +-------------------------------------+----------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
      | Feature                             | API                                                            | Description                                                                                                                                                                             |
      +=====================================+================================================================+=========================================================================================================================================================================================+
      | OpenVINO Runtime Version            | ``ov::get_openvino_version``                                   | Get Openvino API version                                                                                                                                                                |
      +-------------------------------------+----------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
      | Basic Infer Flow                    | ``ov::Core::read_model``,                                      | Common API to do inference: read and compile a model, create an infer request, configure input and output tensors                                                                       |
      |                                     | ``ov::Core::compile_model``,                                   |                                                                                                                                                                                         |
      |                                     | ``ov::CompiledModel::create_infer_request``,                   |                                                                                                                                                                                         |
      |                                     | ``ov::InferRequest::set_tensor``,                              |                                                                                                                                                                                         |
      |                                     | ``ov::InferRequest::get_tensor``                               |                                                                                                                                                                                         |
      +-------------------------------------+----------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
      | Synchronous Infer                   | ``ov::InferRequest::infer``                                    | Do synchronous inference                                                                                                                                                                |
      +-------------------------------------+----------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
      | Model Operations                    | ``ov::Model::inputs``,                                         | Get inputs and outputs of a model                                                                                                                                                       |
      |                                     | ``ov::Model::outputs``                                         |                                                                                                                                                                                         |
      +-------------------------------------+----------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
      | Tensor Operations                   | ``ov::Tensor::get_shape``                                      | Get a tensor shape                                                                                                                                                                      |
      +-------------------------------------+----------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+


How It Works
############

At startup, the sample application reads command line parameters, load tokenier from ggml-vocab.bin with llama.cpp (https://github.com/ggerganov/llama.cpp), prepares input data, loads llama model to the OpenVINO™ Runtime plugin and performs synchronous inference. Then processes output data and decoder the output to text.

Building
########

To build the sample, please use instructions available at :doc:`Build the Sample Applications <openvino_docs_OV_UG_Samples_Overview>` section in OpenVINO™ Toolkit Samples guide.

Running
#######

.. code-block:: console
   
   llama_sampple <path_to_model> <path_to_vocab_file> <prompt_text> <max_sequence_length>

To run the sample, you need to specify llama model in IR format:

- You can download model from https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
- You can convert the modle to IR format.

Example
+++++++

1. Download the llama model and convert model:

2. Run inference with ``prompt`` and ``vocab_file``, for example:
   
   .. code-block:: console
      
      llama_sampple openvino_model.xml gglm-vocab.bin "What is OpenVINO?" 128

Sample Output
#############

The application outputs top-10 inference results.

.. code-block:: console
   
[ INFO ] Build ................................. <version>
[ INFO ]
llama.cpp: loading model from ggml-vocab.bin
[ INFO ] Tokenizer took 10.99 ms
[ INFO ] Loading model files: openvino_model.xml
[ INFO ] Read model took 231.80 ms
[ INFO ] Compile model took 9795.26 ms
[ INFO ] input_ids_len: 8
[ INFO ] Generate 1st token took 1218.92 ms
[ INFO ] Generate 128 new tokens took 35277.96 ms
[ INFO ] Lantacy: 275.61 ms/token

OpenVINO is an open-source software library for deep learning inference that is designed to optimize and run deep learning models on a variety of platforms, including CPUs, GPUs, and specialized accelerators like TPUs. OpenVINO is developed by Intel and is available under the Apache 2.0 license, which means that it is free and open-source.

OpenVINO provides a set of tools and APIs for developers to optimize, compile, and run deep learning models on different hardware platforms. The library supports popular deep learning frameworks like TensorFlow, PyTorch, and C

@endsphinxdirective

