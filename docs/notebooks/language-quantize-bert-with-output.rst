Quantize NLP models with Post-Training Quantization ​in NNCF
============================================================

This tutorial demonstrates how to apply ``INT8`` quantization to the
Natural Language Processing model known as
`BERT <https://en.wikipedia.org/wiki/BERT_(language_model)>`__, using
the `Post-Training Quantization
API <https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.html>`__
(NNCF library). A fine-tuned `HuggingFace
BERT <https://huggingface.co/transformers/model_doc/bert.html>`__
`PyTorch <https://pytorch.org/>`__ model, trained on the `Microsoft
Research Paraphrase Corpus
(MRPC) <https://www.microsoft.com/en-us/download/details.aspx?id=52398>`__,
will be used. The tutorial is designed to be extendable to custom models
and datasets. It consists of the following steps:

-  Download and prepare the BERT model and MRPC dataset.
-  Define data loading and accuracy validation functionality.
-  Prepare the model for quantization.
-  Run optimization pipeline.
-  Load and test quantized model.
-  Compare the performance of the original, converted and quantized
   models.


**Table of contents:**


-  `Imports <#imports>`__
-  `Settings <#settings>`__
-  `Prepare the Model <#prepare-the-model>`__
-  `Prepare the Dataset <#prepare-the-dataset>`__
-  `Optimize model using NNCF Post-training Quantization
   API <#optimize-model-using-nncf-post-training-quantization-api>`__
-  `Load and Test OpenVINO Model <#load-and-test-openvino-model>`__

   -  `Select inference device <#select-inference-device>`__

-  `Compare F1-score of FP32 and INT8
   models <#compare-f1-score-of-fp32-and-int8-models>`__
-  `Compare Performance of the Original, Converted and Quantized
   Models <#compare-performance-of-the-original-converted-and-quantized-models>`__

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

.. code:: ipython3

    %pip install -q "nncf>=2.5.0"
    %pip install -q torch transformers "torch>=2.1" datasets evaluate tqdm  --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install -q "openvino>=2023.1.0"


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.


Imports
-------



.. code:: ipython3

    import os
    import time
    from pathlib import Path
    from zipfile import ZipFile
    from typing import Iterable
    from typing import Any
    
    import datasets
    import evaluate
    import numpy as np
    import nncf
    from nncf.parameters import ModelType
    import openvino as ov
    import torch
    from transformers import BertForSequenceClassification, BertTokenizer
    
    # Fetch `notebook_utils` module
    import requests
    
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    
    open("notebook_utils.py", "w").write(r.text)
    from notebook_utils import download_file, device_widget


.. parsed-literal::

    2024-10-23 01:38:12.900514: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-10-23 01:38:12.934654: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-10-23 01:38:13.484434: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


Settings
--------



.. code:: ipython3

    # Set the data and model directories, source URL and the filename of the model.
    DATA_DIR = "data"
    MODEL_DIR = "model"
    MODEL_LINK = "https://download.pytorch.org/tutorial/MRPC.zip"
    FILE_NAME = MODEL_LINK.split("/")[-1]
    PRETRAINED_MODEL_DIR = os.path.join(MODEL_DIR, "MRPC")
    
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

Prepare the Model
-----------------



Perform the following:

-  Download and unpack pre-trained BERT model for MRPC by PyTorch.
-  Convert the model to the OpenVINO Intermediate Representation
   (OpenVINO IR)

.. code:: ipython3

    download_file(MODEL_LINK, directory=MODEL_DIR, show_progress=True)
    with ZipFile(f"{MODEL_DIR}/{FILE_NAME}", "r") as zip_ref:
        zip_ref.extractall(MODEL_DIR)



.. parsed-literal::

    model/MRPC.zip:   0%|          | 0.00/387M [00:00<?, ?B/s]


Convert the original PyTorch model to the OpenVINO Intermediate
Representation.

From OpenVINO 2023.0, we can directly convert a model from the PyTorch
format to the OpenVINO IR format using model conversion API. Following
PyTorch model formats are supported:

-  ``torch.nn.Module``
-  ``torch.jit.ScriptModule``
-  ``torch.jit.ScriptFunction``

.. code:: ipython3

    MAX_SEQ_LENGTH = 128
    input_shape = ov.PartialShape([1, -1])
    ir_model_xml = Path(MODEL_DIR) / "bert_mrpc.xml"
    core = ov.Core()
    
    torch_model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_DIR)
    torch_model.eval
    
    input_info = [
        ("input_ids", input_shape, np.int64),
        ("attention_mask", input_shape, np.int64),
        ("token_type_ids", input_shape, np.int64),
    ]
    default_input = torch.ones(1, MAX_SEQ_LENGTH, dtype=torch.int64)
    inputs = {
        "input_ids": default_input,
        "attention_mask": default_input,
        "token_type_ids": default_input,
    }
    
    # Convert the PyTorch model to OpenVINO IR FP32.
    if not ir_model_xml.exists():
        model = ov.convert_model(torch_model, example_input=inputs, input=input_info)
        ov.save_model(model, str(ir_model_xml))
    else:
        model = core.read_model(ir_model_xml)


.. parsed-literal::

    WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.base has been moved to tensorflow.python.trackable.base. The old module will be deleted in version 2.11.


.. parsed-literal::

    [ WARNING ]  Please fix your imports. Module %s has been moved to %s. The old module will be deleted in version %s.


.. parsed-literal::

    WARNING:nncf:NNCF provides best results with torch==2.4.*, while current torch version is 2.2.2+cpu. If you encounter issues, consider switching to torch==2.4.*


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:4713: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
      warnings.warn(


Prepare the Dataset
-------------------



We download the `General Language Understanding Evaluation
(GLUE) <https://gluebenchmark.com/>`__ dataset for the MRPC task from
HuggingFace datasets. Then, we tokenize the data with a pre-trained BERT
tokenizer from HuggingFace.

.. code:: ipython3

    def create_data_source():
        raw_dataset = datasets.load_dataset("glue", "mrpc", split="validation")
        tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_DIR)
    
        def _preprocess_fn(examples):
            texts = (examples["sentence1"], examples["sentence2"])
            result = tokenizer(*texts, padding="max_length", max_length=MAX_SEQ_LENGTH, truncation=True)
            result["labels"] = examples["label"]
            return result
    
        processed_dataset = raw_dataset.map(_preprocess_fn, batched=True, batch_size=1)
    
        return processed_dataset
    
    
    data_source = create_data_source()


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
      warnings.warn(


Optimize model using NNCF Post-training Quantization API
--------------------------------------------------------



`NNCF <https://github.com/openvinotoolkit/nncf>`__ provides a suite of
advanced algorithms for Neural Networks inference optimization in
OpenVINO with minimal accuracy drop. We will use 8-bit quantization in
post-training mode (without the fine-tuning pipeline) to optimize BERT.

The optimization process contains the following steps:

1. Create a Dataset for quantization
2. Run ``nncf.quantize`` for getting an optimized model
3. Serialize OpenVINO IR model using ``openvino.save_model`` function

.. code:: ipython3

    INPUT_NAMES = [key for key in inputs.keys()]
    
    
    def transform_fn(data_item):
        """
        Extract the model's input from the data item.
        The data item here is the data item that is returned from the data source per iteration.
        This function should be passed when the data item cannot be used as model's input.
        """
        inputs = {name: np.asarray([data_item[name]], dtype=np.int64) for name in INPUT_NAMES}
        return inputs
    
    
    calibration_dataset = nncf.Dataset(data_source, transform_fn)
    # Quantize the model. By specifying model_type, we specify additional transformer patterns in the model.
    quantized_model = nncf.quantize(model, calibration_dataset, model_type=ModelType.TRANSFORMER)



.. parsed-literal::

    Output()










.. parsed-literal::

    Output()










.. parsed-literal::

    Output()










.. parsed-literal::

    Output()









.. code:: ipython3

    compressed_model_xml = Path(MODEL_DIR) / "quantized_bert_mrpc.xml"
    ov.save_model(quantized_model, compressed_model_xml)

Load and Test OpenVINO Model
----------------------------



To load and test converted model, perform the following:

-  Load the model and compile it for selected device.
-  Prepare the input.
-  Run the inference.
-  Get the answer from the model output.

Select inference device
~~~~~~~~~~~~~~~~~~~~~~~



select device from dropdown list for running inference using OpenVINO

.. code:: ipython3

    device = device_widget()
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



.. code:: ipython3

    # Compile the model for a specific device.
    compiled_quantized_model = core.compile_model(model=quantized_model, device_name=device.value)
    output_layer = compiled_quantized_model.outputs[0]

The Data Source returns a pair of sentences (indicated by
``sample_idx``) and the inference compares these sentences and outputs
whether their meaning is the same. You can test other sentences by
changing ``sample_idx`` to another value (from 0 to 407).

.. code:: ipython3

    sample_idx = 5
    sample = data_source[sample_idx]
    inputs = {k: torch.unsqueeze(torch.tensor(sample[k]), 0) for k in ["input_ids", "token_type_ids", "attention_mask"]}
    
    result = compiled_quantized_model(inputs)[output_layer]
    result = np.argmax(result)
    
    print(f"Text 1: {sample['sentence1']}")
    print(f"Text 2: {sample['sentence2']}")
    print(f"The same meaning: {'yes' if result == 1 else 'no'}")


.. parsed-literal::

    Text 1: Wal-Mart said it would check all of its million-plus domestic workers to ensure they were legally employed .
    Text 2: It has also said it would review all of its domestic employees more than 1 million to ensure they have legal status .
    The same meaning: yes


Compare F1-score of FP32 and INT8 models
----------------------------------------



.. code:: ipython3

    def validate(model: ov.Model, dataset: Iterable[Any]) -> float:
        """
        Evaluate the model on GLUE dataset.
        Returns F1 score metric.
        """
        compiled_model = core.compile_model(model, device_name=device.value)
        output_layer = compiled_model.output(0)
    
        metric = evaluate.load("glue", "mrpc")
        for batch in dataset:
            inputs = [np.expand_dims(np.asarray(batch[key], dtype=np.int64), 0) for key in INPUT_NAMES]
            outputs = compiled_model(inputs)[output_layer]
            predictions = outputs[0].argmax(axis=-1)
            metric.add_batch(predictions=[predictions], references=[batch["labels"]])
        metrics = metric.compute()
        f1_score = metrics["f1"]
    
        return f1_score
    
    
    print("Checking the accuracy of the original model:")
    metric = validate(model, data_source)
    print(f"F1 score: {metric:.4f}")
    
    print("Checking the accuracy of the quantized model:")
    metric = validate(quantized_model, data_source)
    print(f"F1 score: {metric:.4f}")


.. parsed-literal::

    Checking the accuracy of the original model:
    F1 score: 0.9019
    Checking the accuracy of the quantized model:
    F1 score: 0.8969


Compare Performance of the Original, Converted and Quantized Models
-------------------------------------------------------------------



Compare the original PyTorch model with OpenVINO converted and quantized
models (``FP32``, ``INT8``) to see the difference in performance. It is
expressed in Sentences Per Second (SPS) measure, which is the same as
Frames Per Second (FPS) for images.

.. code:: ipython3

    # Compile the model for a specific device.
    compiled_model = core.compile_model(model=model, device_name=device.value)

.. code:: ipython3

    num_samples = 50
    sample = data_source[0]
    inputs = {k: torch.unsqueeze(torch.tensor(sample[k]), 0) for k in ["input_ids", "token_type_ids", "attention_mask"]}
    
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(num_samples):
            torch_model(torch.vstack(list(inputs.values())))
        end = time.perf_counter()
        time_torch = end - start
    print(f"PyTorch model on CPU: {time_torch / num_samples:.3f} seconds per sentence, " f"SPS: {num_samples / time_torch:.2f}")
    
    start = time.perf_counter()
    for _ in range(num_samples):
        compiled_model(inputs)
    end = time.perf_counter()
    time_ir = end - start
    print(f"IR FP32 model in OpenVINO Runtime/{device.value}: {time_ir / num_samples:.3f} " f"seconds per sentence, SPS: {num_samples / time_ir:.2f}")
    
    start = time.perf_counter()
    for _ in range(num_samples):
        compiled_quantized_model(inputs)
    end = time.perf_counter()
    time_ir = end - start
    print(f"OpenVINO IR INT8 model in OpenVINO Runtime/{device.value}: {time_ir / num_samples:.3f} " f"seconds per sentence, SPS: {num_samples / time_ir:.2f}")


.. parsed-literal::

    We strongly recommend passing in an `attention_mask` since your input_ids may be padded. See https://huggingface.co/docs/transformers/troubleshooting#incorrect-output-when-padding-tokens-arent-masked.


.. parsed-literal::

    PyTorch model on CPU: 0.068 seconds per sentence, SPS: 14.67
    IR FP32 model in OpenVINO Runtime/AUTO: 0.020 seconds per sentence, SPS: 49.46
    OpenVINO IR INT8 model in OpenVINO Runtime/AUTO: 0.009 seconds per sentence, SPS: 108.19


Finally, measure the inference performance of OpenVINO ``FP32`` and
``INT8`` models. For this purpose, use `Benchmark
Tool <https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html>`__
in OpenVINO.

   **Note**: The ``benchmark_app`` tool is able to measure the
   performance of the OpenVINO Intermediate Representation (OpenVINO IR)
   models only. For more accurate performance, run ``benchmark_app`` in
   a terminal/command prompt after closing other applications. Run
   ``benchmark_app -m model.xml -d CPU`` to benchmark async inference on
   CPU for one minute. Change ``CPU`` to ``GPU`` to benchmark on GPU.
   Run ``benchmark_app --help`` to see an overview of all command-line
   options.

.. code:: ipython3

    # Inference FP32 model (OpenVINO IR)
    !benchmark_app -m $ir_model_xml -shape [1,128],[1,128],[1,128] -d {device.value} -api sync


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ WARNING ] Default duration 120 seconds is used for unknown device AUTO
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2024.4.0-16579-c3152d32c9c-releases/2024/4
    [ INFO ] 
    [ INFO ] Device info:
    [ INFO ] AUTO
    [ INFO ] Build ................................. 2024.4.0-16579-c3152d32c9c-releases/2024/4
    [ INFO ] 
    [ INFO ] 
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(AUTO) performance hint will be set to PerformanceMode.LATENCY.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 18.86 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     input_ids (node: input_ids) : i64 / [...] / [1,?]
    [ INFO ]     63 , attention_mask (node: attention_mask) : i64 / [...] / [1,?]
    [ INFO ]     token_type_ids (node: token_type_ids) : i64 / [...] / [1,?]
    [ INFO ] Model outputs:
    [ INFO ]     logits (node: __module.classifier/aten::linear/Add) : f32 / [...] / [1,2]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [ INFO ] Reshaping model: 'input_ids': [1,128], '63': [1,128], 'token_type_ids': [1,128]
    [ INFO ] Reshape model took 5.46 ms
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     input_ids (node: input_ids) : i64 / [...] / [1,128]
    [ INFO ]     63 , attention_mask (node: attention_mask) : i64 / [...] / [1,128]
    [ INFO ]     token_type_ids (node: token_type_ids) : i64 / [...] / [1,128]
    [ INFO ] Model outputs:
    [ INFO ]     logits (node: __module.classifier/aten::linear/Add) : f32 / [...] / [1,2]
    [Step 7/11] Loading the model to the device
    [ INFO ] Compile model took 373.39 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: Model0
    [ INFO ]   EXECUTION_DEVICES: ['CPU']
    [ INFO ]   PERFORMANCE_HINT: PerformanceMode.LATENCY
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 1
    [ INFO ]   MULTI_DEVICE_PRIORITIES: CPU
    [ INFO ]   CPU:
    [ INFO ]     AFFINITY: Affinity.CORE
    [ INFO ]     CPU_DENORMALS_OPTIMIZATION: False
    [ INFO ]     CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE: 1.0
    [ INFO ]     DYNAMIC_QUANTIZATION_GROUP_SIZE: 32
    [ INFO ]     ENABLE_CPU_PINNING: True
    [ INFO ]     ENABLE_HYPER_THREADING: False
    [ INFO ]     EXECUTION_DEVICES: ['CPU']
    [ INFO ]     EXECUTION_MODE_HINT: ExecutionMode.PERFORMANCE
    [ INFO ]     INFERENCE_NUM_THREADS: 12
    [ INFO ]     INFERENCE_PRECISION_HINT: <Type: 'float32'>
    [ INFO ]     KV_CACHE_PRECISION: <Type: 'float16'>
    [ INFO ]     LOG_LEVEL: Level.NO
    [ INFO ]     MODEL_DISTRIBUTION_POLICY: set()
    [ INFO ]     NETWORK_NAME: Model0
    [ INFO ]     NUM_STREAMS: 1
    [ INFO ]     OPTIMAL_NUMBER_OF_INFER_REQUESTS: 1
    [ INFO ]     PERFORMANCE_HINT: LATENCY
    [ INFO ]     PERFORMANCE_HINT_NUM_REQUESTS: 0
    [ INFO ]     PERF_COUNT: NO
    [ INFO ]     SCHEDULING_CORE_TYPE: SchedulingCoreType.ANY_CORE
    [ INFO ]   MODEL_PRIORITY: Priority.MEDIUM
    [ INFO ]   LOADED_FROM_CACHE: False
    [ INFO ]   PERF_COUNT: False
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'input_ids'!. This input will be filled with random values!
    [ WARNING ] No input files were given for input '63'!. This input will be filled with random values!
    [ WARNING ] No input files were given for input 'token_type_ids'!. This input will be filled with random values!
    [ INFO ] Fill input 'input_ids' with random values 
    [ INFO ] Fill input '63' with random values 
    [ INFO ] Fill input 'token_type_ids' with random values 
    [Step 10/11] Measuring performance (Start inference synchronously, limits: 120000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 24.05 ms
    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            5730 iterations
    [ INFO ] Duration:         120000.60 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        20.64 ms
    [ INFO ]    Average:       20.84 ms
    [ INFO ]    Min:           19.84 ms
    [ INFO ]    Max:           31.70 ms
    [ INFO ] Throughput:   47.75 FPS


.. code:: ipython3

    # Inference INT8 model (OpenVINO IR)
    ! benchmark_app -m $compressed_model_xml -shape [1,128],[1,128],[1,128] -d {device.value} -api sync


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ WARNING ] Default duration 120 seconds is used for unknown device AUTO
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2024.4.0-16579-c3152d32c9c-releases/2024/4
    [ INFO ] 
    [ INFO ] Device info:
    [ INFO ] AUTO
    [ INFO ] Build ................................. 2024.4.0-16579-c3152d32c9c-releases/2024/4
    [ INFO ] 
    [ INFO ] 
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(AUTO) performance hint will be set to PerformanceMode.LATENCY.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 25.60 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     input_ids (node: input_ids) : i64 / [...] / [1,?]
    [ INFO ]     63 , attention_mask (node: attention_mask) : i64 / [...] / [1,?]
    [ INFO ]     token_type_ids (node: token_type_ids) : i64 / [...] / [1,?]
    [ INFO ] Model outputs:
    [ INFO ]     logits (node: __module.classifier/aten::linear/Add) : f32 / [...] / [1,2]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [ INFO ] Reshaping model: 'input_ids': [1,128], '63': [1,128], 'token_type_ids': [1,128]
    [ INFO ] Reshape model took 7.46 ms
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     input_ids (node: input_ids) : i64 / [...] / [1,128]
    [ INFO ]     63 , attention_mask (node: attention_mask) : i64 / [...] / [1,128]
    [ INFO ]     token_type_ids (node: token_type_ids) : i64 / [...] / [1,128]
    [ INFO ] Model outputs:
    [ INFO ]     logits (node: __module.classifier/aten::linear/Add) : f32 / [...] / [1,2]
    [Step 7/11] Loading the model to the device
    [ INFO ] Compile model took 1067.96 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: Model0
    [ INFO ]   EXECUTION_DEVICES: ['CPU']
    [ INFO ]   PERFORMANCE_HINT: PerformanceMode.LATENCY
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 1
    [ INFO ]   MULTI_DEVICE_PRIORITIES: CPU
    [ INFO ]   CPU:
    [ INFO ]     AFFINITY: Affinity.CORE
    [ INFO ]     CPU_DENORMALS_OPTIMIZATION: False
    [ INFO ]     CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE: 1.0
    [ INFO ]     DYNAMIC_QUANTIZATION_GROUP_SIZE: 32
    [ INFO ]     ENABLE_CPU_PINNING: True
    [ INFO ]     ENABLE_HYPER_THREADING: False
    [ INFO ]     EXECUTION_DEVICES: ['CPU']
    [ INFO ]     EXECUTION_MODE_HINT: ExecutionMode.PERFORMANCE
    [ INFO ]     INFERENCE_NUM_THREADS: 12
    [ INFO ]     INFERENCE_PRECISION_HINT: <Type: 'float32'>
    [ INFO ]     KV_CACHE_PRECISION: <Type: 'float16'>
    [ INFO ]     LOG_LEVEL: Level.NO
    [ INFO ]     MODEL_DISTRIBUTION_POLICY: set()
    [ INFO ]     NETWORK_NAME: Model0
    [ INFO ]     NUM_STREAMS: 1
    [ INFO ]     OPTIMAL_NUMBER_OF_INFER_REQUESTS: 1
    [ INFO ]     PERFORMANCE_HINT: LATENCY
    [ INFO ]     PERFORMANCE_HINT_NUM_REQUESTS: 0
    [ INFO ]     PERF_COUNT: NO
    [ INFO ]     SCHEDULING_CORE_TYPE: SchedulingCoreType.ANY_CORE
    [ INFO ]   MODEL_PRIORITY: Priority.MEDIUM
    [ INFO ]   LOADED_FROM_CACHE: False
    [ INFO ]   PERF_COUNT: False
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'input_ids'!. This input will be filled with random values!
    [ WARNING ] No input files were given for input '63'!. This input will be filled with random values!
    [ WARNING ] No input files were given for input 'token_type_ids'!. This input will be filled with random values!
    [ INFO ] Fill input 'input_ids' with random values 
    [ INFO ] Fill input '63' with random values 
    [ INFO ] Fill input 'token_type_ids' with random values 
    [Step 10/11] Measuring performance (Start inference synchronously, limits: 120000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 17.45 ms
    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            13316 iterations
    [ INFO ] Duration:         120006.19 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        8.92 ms
    [ INFO ]    Average:       8.92 ms
    [ INFO ]    Min:           7.66 ms
    [ INFO ]    Max:           14.16 ms
    [ INFO ] Throughput:   110.96 FPS

