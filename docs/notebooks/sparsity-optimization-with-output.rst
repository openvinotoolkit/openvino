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
Decompression <https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html#sparse-weights-decompression-intel-x86-64>`__,
a runtime option that seizes model sparsity for efficiency. The notebook
consists of the following steps:

-  Install prerequisites
-  Download and quantize sparse public BERT model, using the OpenVINO
   integration with Hugging Face Optimum.
-  Compare sparse 8-bit vs. dense 8-bit inference performance.

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Prerequisites <#Prerequisites>`__
-  `Imports <#Imports>`__

   -  `Download, quantize and sparsify the model, using Hugging Face
      Optimum
      API <#Download,-quantize-and-sparsify-the-model,-using-Hugging-Face-Optimum-API>`__

-  `Benchmark quantized dense inference
   performance <#Benchmark-quantized-dense-inference-performance>`__
-  `Benchmark quantized sparse inference
   performance <#Benchmark-quantized-sparse-inference-performance>`__
-  `When this might be helpful <#When-this-might-be-helpful>`__

Prerequisites
-------------

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    %pip install -q "openvino>=2023.1.0"
    %pip install -q "git+https://github.com/huggingface/optimum-intel.git" "torch>=2.1" datasets onnx transformers>=4.33.0 --extra-index-url https://download.pytorch.org/whl/cpu


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    Note: you may need to restart the kernel to use updated packages.
    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    Note: you may need to restart the kernel to use updated packages.


Imports
-------

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    import shutil
    from pathlib import Path
    
    from optimum.intel.openvino import OVModelForSequenceClassification
    from transformers import AutoTokenizer, pipeline
    from huggingface_hub import hf_hub_download


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


.. parsed-literal::

    2024-06-06 02:53:32.407670: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-06-06 02:53:32.442101: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-06-06 02:53:33.039105: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-697/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      torch.utils._pytree._register_pytree_node(


Download, quantize and sparsify the model, using Hugging Face Optimum API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

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

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-697/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/huggingface_hub/file_download.py:1132: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
      warnings.warn(
    Compiling the model to CPU ...


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

`back to top ⬆️ <#Table-of-contents:>`__

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
    [ INFO ] Build ................................. 2024.3.0-15599-de4d00a5970
    [ INFO ] 
    [ INFO ] Device info:
    [ INFO ] CPU
    [ INFO ] Build ................................. 2024.3.0-15599-de4d00a5970
    [ INFO ] 
    [ INFO ] 
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(CPU) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 72.11 ms
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
    [ INFO ] Reshape model took 30.85 ms
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     input_ids (node: input_ids) : i64 / [...] / [1,64]
    [ INFO ]     attention_mask (node: attention_mask) : i64 / [...] / [1,64]
    [ INFO ]     token_type_ids (node: token_type_ids) : i64 / [...] / [1,64]
    [ INFO ] Model outputs:
    [ INFO ]     logits (node: logits) : f32 / [...] / [1,2]
    [Step 7/11] Loading the model to the device
    [ INFO ] Compile model took 1277.07 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: torch_jit
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 4
    [ INFO ]   NUM_STREAMS: 4
    [ INFO ]   INFERENCE_NUM_THREADS: 4
    [ INFO ]   PERF_COUNT: NO
    [ INFO ]   INFERENCE_PRECISION_HINT: <Type: 'float32'>
    [ INFO ]   PERFORMANCE_HINT: THROUGHPUT
    [ INFO ]   EXECUTION_MODE_HINT: ExecutionMode.PERFORMANCE
    [ INFO ]   PERFORMANCE_HINT_NUM_REQUESTS: 0
    [ INFO ]   ENABLE_CPU_PINNING: True
    [ INFO ]   SCHEDULING_CORE_TYPE: SchedulingCoreType.ANY_CORE
    [ INFO ]   MODEL_DISTRIBUTION_POLICY: set()
    [ INFO ]   ENABLE_HYPER_THREADING: True
    [ INFO ]   EXECUTION_DEVICES: ['CPU']
    [ INFO ]   CPU_DENORMALS_OPTIMIZATION: False
    [ INFO ]   LOG_LEVEL: Level.NO
    [ INFO ]   CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE: 1.0
    [ INFO ]   DYNAMIC_QUANTIZATION_GROUP_SIZE: 0
    [ INFO ]   KV_CACHE_PRECISION: <Type: 'float16'>
    [ INFO ]   AFFINITY: Affinity.CORE
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'input_ids'!. This input will be filled with random values!
    [ WARNING ] No input files were given for input 'attention_mask'!. This input will be filled with random values!
    [ WARNING ] No input files were given for input 'token_type_ids'!. This input will be filled with random values!
    [ INFO ] Fill input 'input_ids' with random values 
    [ INFO ] Fill input 'attention_mask' with random values 
    [ INFO ] Fill input 'token_type_ids' with random values 
    [Step 10/11] Measuring performance (Start inference asynchronously, 4 inference requests, limits: 60000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 25.64 ms
    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            9508 iterations
    [ INFO ] Duration:         60047.81 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        25.04 ms
    [ INFO ]    Average:       25.08 ms
    [ INFO ]    Min:           23.07 ms
    [ INFO ]    Max:           36.95 ms
    [ INFO ] Throughput:   158.34 FPS


Benchmark quantized sparse inference performance
------------------------------------------------

`back to top ⬆️ <#Table-of-contents:>`__

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
    [ INFO ] Build ................................. 2024.3.0-15599-de4d00a5970
    [ INFO ] 
    [ INFO ] Device info:
    [ INFO ] CPU
    [ INFO ] Build ................................. 2024.3.0-15599-de4d00a5970
    [ INFO ] 
    [ INFO ] 
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(CPU) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 63.33 ms
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
    [ INFO ] Reshape model took 30.21 ms
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     input_ids (node: input_ids) : i64 / [...] / [1,64]
    [ INFO ]     attention_mask (node: attention_mask) : i64 / [...] / [1,64]
    [ INFO ]     token_type_ids (node: token_type_ids) : i64 / [...] / [1,64]
    [ INFO ] Model outputs:
    [ INFO ]     logits (node: logits) : f32 / [...] / [1,2]
    [Step 7/11] Loading the model to the device
    [ ERROR ] Exception from src/inference/src/cpp/core.cpp:104:
    Exception from src/inference/src/dev/plugin.cpp:53:
    Exception from src/plugins/intel_cpu/src/config.cpp:226:
    Wrong value for property key CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE. Expected only float numbers
    
    
    Traceback (most recent call last):
      File "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-697/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/openvino/tools/benchmark/main.py", line 408, in main
        compiled_model = benchmark.core.compile_model(model, benchmark.device, device_config)
      File "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-697/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/openvino/runtime/ie_api.py", line 543, in compile_model
        super().compile_model(model, device_name, {} if config is None else config),
    RuntimeError: Exception from src/inference/src/cpp/core.cpp:104:
    Exception from src/inference/src/dev/plugin.cpp:53:
    Exception from src/plugins/intel_cpu/src/config.cpp:226:
    Wrong value for property key CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE. Expected only float numbers
    
    
    


When this might be helpful
--------------------------

`back to top ⬆️ <#Table-of-contents:>`__

This feature can improve inference performance for models with sparse
weights in the scenarios when the model is deployed to handle multiple
requests in parallel asynchronously. It is especially helpful with a
small sequence length, for example, 32 and lower.

For more details about asynchronous inference with OpenVINO, refer to
the following documentation:

-  `Deployment Optimization
   Guide <https://docs.openvino.ai/2024/openvino-workflow/running-inference/optimize-inference/general-optimizations.html>`__
-  `Inference Request
   API <https://docs.openvino.ai/2024/openvino-workflow/running-inference/integrate-openvino-with-your-application/inference-request.html>`__
