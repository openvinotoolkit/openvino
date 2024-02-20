Accelerate Inference of Sparse Transformer Models with OpenVINO™ and 4th Gen Intel® Xeon® Scalable Processors
=============================================================================================================

This tutorial demonstrates how to improve performance of sparse
Transformer models with `OpenVINO <https://docs.openvino.ai/>`__ on 4th
Gen Intel® Xeon® Scalable processors.

The tutorial downloads `a BERT-base
model <https://huggingface.co/OpenVINO/bert-base-uncased-sst2-int8-unstructured80>`__
which has been quantized, sparsified, and tuned for `SST2
datasets <https://huggingface.co/datasets/sst2>`__ using
`Optimum-Intel <https://github.com/huggingface/optimum-intel>`__. It
demonstrates the inference performance advantage on 4th Gen Intel® Xeon®
Scalable Processors by running it with `Sparse Weight
Decompression <https://docs.openvino.ai/2023.3/openvino_docs_OV_UG_supported_plugins_CPU.html#sparse-weights-decompression-intel-x86-64>`__,
a runtime option that seizes model sparsity for efficiency. The notebook
consists of the following steps:

-  Install prerequisites
-  Download and quantize sparse public BERT model, using the OpenVINO
   integration with Hugging Face Optimum.
-  Compare sparse 8-bit vs. dense 8-bit inference performance.

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Prerequisites <#prerequisites>`__
-  `Imports <#imports>`__

   -  `Download, quantize and sparsify the model, using Hugging Face
      Optimum
      API <#download-quantize-and-sparsify-the-model-using-hugging-face-optimum-api>`__

-  `Benchmark quantized dense inference
   performance <#benchmark-quantized-dense-inference-performance>`__
-  `Benchmark quantized sparse inference
   performance <#benchmark-quantized-sparse-inference-performance>`__
-  `When this might be helpful <#when-this-might-be-helpful>`__

Prerequisites
-------------



.. code:: ipython3

    %pip install -q "openvino>=2023.1.0"
    %pip install -q "git+https://github.com/huggingface/optimum-intel.git" datasets onnx transformers>=4.33.0 --extra-index-url https://download.pytorch.org/whl/cpu


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


Imports
-------



.. code:: ipython3

    import shutil
    from pathlib import Path
    
    from optimum.intel.openvino import OVModelForSequenceClassification
    from transformers import AutoTokenizer, pipeline
    from huggingface_hub import hf_hub_download


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


.. parsed-literal::

    No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda'


.. parsed-literal::

    2024-02-09 23:02:05.779349: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-02-09 23:02:05.814537: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


.. parsed-literal::

    2024-02-09 23:02:06.378496: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


Download, quantize and sparsify the model, using Hugging Face Optimum API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



The first step is to download a quantized sparse transformers which has
been translated to OpenVINO IR. Then, it will be put through a
classification as a simple validation of a working downloaded model. To
find out how the model is being quantized and sparsified, refer to the
`OpenVINO/bert-base-uncased-sst2-int8-unstructured80 <https://huggingface.co/OpenVINO/bert-base-uncased-sst2-int8-unstructured80>`__
model card on Hugging Face.

.. code:: ipython3

    # The following model has been quantized, sparsified using Optimum-Intel 1.7 which is enabled by OpenVINO and NNCF
    # for reproducibility, refer https://huggingface.co/OpenVINO/bert-base-uncased-sst2-int8-unstructured80
    model_id = "OpenVINO/bert-base-uncased-sst2-int8-unstructured80"
    
    # The following two steps will set up the model and download them to HF Cache folder
    ov_model = OVModelForSequenceClassification.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Let's take the model for a spin!
    sentiment_classifier = pipeline("text-classification", model=ov_model, tokenizer=tokenizer)
    
    text = "He's a dreadful magician."
    outputs = sentiment_classifier(text)
    
    print(outputs)


.. parsed-literal::

    Compiling the model to CPU ...


.. parsed-literal::

    device must be of type <class 'str'> but got <class 'torch.device'> instead


.. parsed-literal::

    [{'label': 'negative', 'score': 0.9982142448425293}]


For benchmarking, we will use OpenVINO’s benchmark application and put
the IRs into a single folder.

.. code:: ipython3

    # create a folder
    quantized_sparse_dir = Path("bert_80pc_sparse_quantized_ir")
    quantized_sparse_dir.mkdir(parents=True, exist_ok=True)
    
    # following return path to specified filename in cache folder (which we've with the 
    ov_ir_xml_path = hf_hub_download(repo_id=model_id, filename="openvino_model.xml")
    ov_ir_bin_path = hf_hub_download(repo_id=model_id, filename="openvino_model.bin")
    
    # copy IRs to the folder
    shutil.copy(ov_ir_xml_path, quantized_sparse_dir)
    shutil.copy(ov_ir_bin_path, quantized_sparse_dir)                                




.. parsed-literal::

    'bert_80pc_sparse_quantized_ir/openvino_model.bin'



Benchmark quantized dense inference performance
-----------------------------------------------



Benchmark dense inference performance using parallel execution on four
CPU cores to simulate a small instance in the cloud infrastructure.
Sequence length is dependent on use cases, 16 is common for
conversational AI while 160 for question answering task. It is set to 64
as an example. It is recommended to tune based on your applications.

.. code:: ipython3

    # Dump benchmarking config for dense inference
    with (quantized_sparse_dir / "perf_config.json").open("w") as outfile:
        outfile.write(
            """
            {
                "CPU": {"NUM_STREAMS": 4, "INFERENCE_NUM_THREADS": 4}
            }
            """
        )

.. code:: ipython3

    !benchmark_app -m $quantized_sparse_dir/openvino_model.xml -shape "input_ids[1,64],attention_mask[1,64],token_type_ids[1,64]" -load_config $quantized_sparse_dir/perf_config.json


.. parsed-literal::

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2023.3.0-13775-ceeafaf64f3-releases/2023/3
    [ INFO ] 
    [ INFO ] Device info:


.. parsed-literal::

    [ INFO ] CPU
    [ INFO ] Build ................................. 2023.3.0-13775-ceeafaf64f3-releases/2023/3
    [ INFO ] 
    [ INFO ] 
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(CPU) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files


.. parsed-literal::

    [ INFO ] Read model took 62.38 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     input_ids (node: input_ids) : i64 / [...] / [?,?]
    [ INFO ]     attention_mask (node: attention_mask) : i64 / [...] / [?,?]
    [ INFO ]     token_type_ids (node: token_type_ids) : i64 / [...] / [?,?]
    [ INFO ] Model outputs:
    [ INFO ]     logits (node: logits) : f32 / [...] / [?,2]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [ INFO ] Reshaping model: 'input_ids': [1,64], 'attention_mask': [1,64], 'token_type_ids': [1,64]
    [ INFO ] Reshape model took 23.14 ms
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     input_ids (node: input_ids) : i64 / [...] / [1,64]
    [ INFO ]     attention_mask (node: attention_mask) : i64 / [...] / [1,64]
    [ INFO ]     token_type_ids (node: token_type_ids) : i64 / [...] / [1,64]
    [ INFO ] Model outputs:
    [ INFO ]     logits (node: logits) : f32 / [...] / [1,2]
    [Step 7/11] Loading the model to the device


.. parsed-literal::

    [ INFO ] Compile model took 1107.64 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: torch_jit
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 4
    [ INFO ]   NUM_STREAMS: 4
    [ INFO ]   AFFINITY: Affinity.CORE
    [ INFO ]   INFERENCE_NUM_THREADS: 4
    [ INFO ]   PERF_COUNT: NO
    [ INFO ]   INFERENCE_PRECISION_HINT: <Type: 'float32'>
    [ INFO ]   PERFORMANCE_HINT: THROUGHPUT
    [ INFO ]   EXECUTION_MODE_HINT: ExecutionMode.PERFORMANCE
    [ INFO ]   PERFORMANCE_HINT_NUM_REQUESTS: 0
    [ INFO ]   ENABLE_CPU_PINNING: True
    [ INFO ]   SCHEDULING_CORE_TYPE: SchedulingCoreType.ANY_CORE
    [ INFO ]   ENABLE_HYPER_THREADING: True
    [ INFO ]   EXECUTION_DEVICES: ['CPU']
    [ INFO ]   CPU_DENORMALS_OPTIMIZATION: False
    [ INFO ]   CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE: 1.0
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'input_ids'!. This input will be filled with random values!
    [ WARNING ] No input files were given for input 'attention_mask'!. This input will be filled with random values!
    [ WARNING ] No input files were given for input 'token_type_ids'!. This input will be filled with random values!
    [ INFO ] Fill input 'input_ids' with random values 
    [ INFO ] Fill input 'attention_mask' with random values 
    [ INFO ] Fill input 'token_type_ids' with random values 
    [Step 10/11] Measuring performance (Start inference asynchronously, 4 inference requests, limits: 60000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).


.. parsed-literal::

    [ INFO ] First inference took 30.14 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            8852 iterations
    [ INFO ] Duration:         60038.32 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        26.79 ms
    [ INFO ]    Average:       26.86 ms
    [ INFO ]    Min:           24.76 ms
    [ INFO ]    Max:           42.20 ms
    [ INFO ] Throughput:   147.44 FPS


Benchmark quantized sparse inference performance
------------------------------------------------



To enable sparse weight decompression feature, users can add it to
runtime config like below. ``CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE``
takes values between 0.5 and 1.0. It is a layer-level sparsity threshold
for which a layer will be enabled.

.. code:: ipython3

    # Dump benchmarking config for dense inference
    # "CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE" controls minimum sparsity rate for weights to consider 
    # for sparse optimization at the runtime.
    with (quantized_sparse_dir / "perf_config_sparse.json").open("w") as outfile:
        outfile.write(
            """
            {
                "CPU": {"NUM_STREAMS": 4, "INFERENCE_NUM_THREADS": 4, "CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE": 0.75}
            }
            """
        )

.. code:: ipython3

    !benchmark_app -m $quantized_sparse_dir/openvino_model.xml -shape "input_ids[1,64],attention_mask[1,64],token_type_ids[1,64]" -load_config $quantized_sparse_dir/perf_config_sparse.json


.. parsed-literal::

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2023.3.0-13775-ceeafaf64f3-releases/2023/3
    [ INFO ] 
    [ INFO ] Device info:


.. parsed-literal::

    [ INFO ] CPU
    [ INFO ] Build ................................. 2023.3.0-13775-ceeafaf64f3-releases/2023/3
    [ INFO ] 
    [ INFO ] 
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(CPU) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files


.. parsed-literal::

    [ INFO ] Read model took 71.12 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     input_ids (node: input_ids) : i64 / [...] / [?,?]
    [ INFO ]     attention_mask (node: attention_mask) : i64 / [...] / [?,?]
    [ INFO ]     token_type_ids (node: token_type_ids) : i64 / [...] / [?,?]
    [ INFO ] Model outputs:
    [ INFO ]     logits (node: logits) : f32 / [...] / [?,2]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [ INFO ] Reshaping model: 'input_ids': [1,64], 'attention_mask': [1,64], 'token_type_ids': [1,64]


.. parsed-literal::

    [ INFO ] Reshape model took 23.54 ms
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     input_ids (node: input_ids) : i64 / [...] / [1,64]
    [ INFO ]     attention_mask (node: attention_mask) : i64 / [...] / [1,64]
    [ INFO ]     token_type_ids (node: token_type_ids) : i64 / [...] / [1,64]
    [ INFO ] Model outputs:
    [ INFO ]     logits (node: logits) : f32 / [...] / [1,2]
    [Step 7/11] Loading the model to the device
    [ ERROR ] Exception from src/inference/src/core.cpp:99:
    [ GENERAL_ERROR ] Exception from src/plugins/intel_cpu/src/config.cpp:158:
    Wrong value for property key CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE. Expected only float numbers
    
    Traceback (most recent call last):
      File "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/openvino/tools/benchmark/main.py", line 408, in main
        compiled_model = benchmark.core.compile_model(model, benchmark.device, device_config)
      File "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/openvino/runtime/ie_api.py", line 547, in compile_model
        super().compile_model(model, device_name, {} if config is None else config),
    RuntimeError: Exception from src/inference/src/core.cpp:99:
    [ GENERAL_ERROR ] Exception from src/plugins/intel_cpu/src/config.cpp:158:
    Wrong value for property key CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE. Expected only float numbers
    
    


When this might be helpful
--------------------------



This feature can improve inference performance for models with sparse
weights in the scenarios when the model is deployed to handle multiple
requests in parallel asynchronously. It is especially helpful with a
small sequence length, for example, 32 and lower.

For more details about asynchronous inference with OpenVINO, refer to
the following documentation:

-  `Deployment Optimization
   Guide <https://docs.openvino.ai/2023.3/openvino_docs_deployment_optimization_guide_common.html>`__
-  `Inference Request
   API <https://docs.openvino.ai/2023.3/openvino_docs_OV_UG_Infer_request.html>`__
