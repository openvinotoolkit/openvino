Quantize NLP models with Post-Training Quantization ​in NNCF
============================================================

This tutorial demonstrates how to apply ``INT8`` quantization to the
Natural Language Processing model known as
`BERT <https://en.wikipedia.org/wiki/BERT_(language_model)>`__, using
the `Post-Training Quantization
API <https://docs.openvino.ai/latest/nncf_ptq_introduction.html>`__
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

.. code:: ipython3

    !pip install -q nncf datasets evaluate

Imports
-------

.. code:: ipython3

    import os
    import sys
    import time
    from pathlib import Path
    from zipfile import ZipFile
    from typing import Iterable
    from typing import Any
    
    import numpy as np
    import torch
    from openvino import runtime as ov
    from openvino.tools import mo
    from openvino.runtime import serialize, Model
    import nncf
    from nncf.parameters import ModelType
    from transformers import BertForSequenceClassification, BertTokenizer
    import datasets
    import evaluate
    
    sys.path.append("../utils")
    from notebook_utils import download_file


.. parsed-literal::

    /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/openvino/offline_transformations/__init__.py:10: FutureWarning: The module is private and following namespace `offline_transformations` will be removed in the future, use `openvino.runtime.passes` instead!
      warnings.warn(


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

Perform the following: - Download and unpack pre-trained BERT model for
MRPC by PyTorch. - Convert the model to the ONNX. - Run Model Optimizer
to convert the model from the ONNX representation to the OpenVINO
Intermediate Representation (OpenVINO IR)

.. code:: ipython3

    download_file(MODEL_LINK, directory=MODEL_DIR, show_progress=True)
    with ZipFile(f"{MODEL_DIR}/{FILE_NAME}", "r") as zip_ref:
        zip_ref.extractall(MODEL_DIR)



.. parsed-literal::

    model/MRPC.zip:   0%|          | 0.00/387M [00:00<?, ?B/s]


Convert the original PyTorch model to the ONNX representation.

.. code:: ipython3

    BATCH_SIZE = 1
    MAX_SEQ_LENGTH = 128
    
    
    def export_model_to_onnx(model, path):
        with torch.no_grad():
            default_input = torch.ones(1, MAX_SEQ_LENGTH, dtype=torch.int64)
            inputs = {
                "input_ids": default_input,
                "attention_mask": default_input,
                "token_type_ids": default_input,
            }
            torch.onnx.export(
                model,
                (inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]),
                path,
                opset_version=11,
                input_names=["input_ids", "attention_mask", "token_type_ids"],
                output_names=["output"]
            )
            print("ONNX model saved to {}".format(path))
    
    
    torch_model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_DIR)
    onnx_model_path = Path(MODEL_DIR) / "bert_mrpc.onnx"
    if not onnx_model_path.exists():
        export_model_to_onnx(torch_model, onnx_model_path)


.. parsed-literal::

    ONNX model saved to model/bert_mrpc.onnx


Convert the ONNX Model to OpenVINO IR
-------------------------------------

Use Model Optimizer Python API to convert the model to OpenVINO IR with
``FP32`` precision. For more information about Model Optimizer Python
API, see the `Model Optimizer Developer
Guide <https://docs.openvino.ai/latest/openvino_docs_MO_DG_Python_API.html>`__.

.. code:: ipython3

    ir_model_xml = onnx_model_path.with_suffix(".xml")
    
    # Convert the ONNX model to OpenVINO IR FP32.
    if not ir_model_xml.exists():
        model = mo.convert_model(onnx_model_path)
        serialize(model, str(ir_model_xml))

Prepare the Dataset
-------------------

We download the General Language Understanding Evaluation (GLUE) dataset
for the MRPC task from HuggingFace datasets. Then, we tokenize the data
with a pre-trained BERT tokenizer from HuggingFace.

.. code:: ipython3

    def create_data_source():
        raw_dataset = datasets.load_dataset('glue', 'mrpc', split='validation')
        tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_DIR)
    
        def _preprocess_fn(examples):
            texts = (examples['sentence1'], examples['sentence2'])
            result = tokenizer(*texts, padding='max_length', max_length=MAX_SEQ_LENGTH, truncation=True)
            result['labels'] = examples['label']
            return result
        processed_dataset = raw_dataset.map(_preprocess_fn, batched=True, batch_size=1)
    
        return processed_dataset
    
    
    data_source = create_data_source()


.. parsed-literal::

    [ WARNING ] Found cached dataset glue (/opt/home/k8sworker/.cache/huggingface/datasets/glue/mrpc/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad)


.. parsed-literal::

    [ WARNING ]  Found cached dataset glue (/opt/home/k8sworker/.cache/huggingface/datasets/glue/mrpc/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad)


.. parsed-literal::

    [ WARNING ] Loading cached processed dataset at /opt/home/k8sworker/.cache/huggingface/datasets/glue/mrpc/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad/cache-b5f4c739eb2a4a9f.arrow


.. parsed-literal::

    [ WARNING ]  Loading cached processed dataset at /opt/home/k8sworker/.cache/huggingface/datasets/glue/mrpc/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad/cache-b5f4c739eb2a4a9f.arrow


Optimize model using NNCF Post-training Quantization API
--------------------------------------------------------

`NNCF <https://github.com/openvinotoolkit/nncf>`__ provides a suite of
advanced algorithms for Neural Networks inference optimization in
OpenVINO with minimal accuracy drop. We will use 8-bit quantization in
post-training mode (without the fine-tuning pipeline) to optimize BERT.

   **Note**: NNCF Post-training Quantization is available as a preview
   feature in OpenVINO 2022.3 release. Fully functional support will be
   provided in the next releases.

The optimization process contains the following steps:

1. Create a Dataset for quantization
2. Run ``nncf.quantize`` for getting an optimized model
3. Serialize OpenVINO IR model using ``openvino.runtime.serialize``
   function

.. code:: ipython3

    # Load the network in OpenVINO Runtime.
    core = ov.Core()
    model = core.read_model(ir_model_xml)
    INPUT_NAMES = [x.any_name for x in model.inputs]
    
    
    def transform_fn(data_item):
        """
        Extract the model's input from the data item.
        The data item here is the data item that is returned from the data source per iteration.
        This function should be passed when the data item cannot be used as model's input.
        """
        inputs = {
            name: np.asarray(data_item[name], dtype=np.int64) for name in INPUT_NAMES
        }
        return inputs
    
    
    calibration_dataset = nncf.Dataset(data_source, transform_fn)
    # Quantize the model. By specifying model_type, we specify additional transformer patterns in the model.
    quantized_model = nncf.quantize(model, calibration_dataset,
                                    model_type=ModelType.TRANSFORMER)


.. parsed-literal::

    INFO:openvino.tools.pot.pipeline.pipeline:Inference Engine version:                2022.3.0-9052-9752fafe8eb-releases/2022/3
    INFO:openvino.tools.pot.pipeline.pipeline:Model Optimizer version:                 2022.3.0-9052-9752fafe8eb-releases/2022/3
    INFO:openvino.tools.pot.pipeline.pipeline:Post-Training Optimization Tool version: 2022.3.0-9052-9752fafe8eb-releases/2022/3
    INFO:openvino.tools.pot.statistics.collector:Start computing statistics for algorithms : DefaultQuantization
    INFO:openvino.tools.pot.statistics.collector:Computing statistics finished
    INFO:openvino.tools.pot.pipeline.pipeline:Start algorithm: DefaultQuantization
    INFO:openvino.tools.pot.algorithms.quantization.default.algorithm:Start computing statistics for algorithm : ActivationChannelAlignment
    INFO:openvino.tools.pot.algorithms.quantization.default.algorithm:Computing statistics finished
    INFO:openvino.tools.pot.algorithms.quantization.default.algorithm:Start computing statistics for algorithms : MinMaxQuantization,FastBiasCorrection
    INFO:openvino.tools.pot.algorithms.quantization.default.algorithm:Computing statistics finished
    INFO:openvino.tools.pot.pipeline.pipeline:Finished: DefaultQuantization
     ===========================================================================


.. code:: ipython3

    compressed_model_xml = 'quantized_bert_mrpc.xml'
    ov.serialize(quantized_model, compressed_model_xml)

Load and Test OpenVINO Model
----------------------------

To load and test converted model, perform the following: \* Load the
model and compile it for CPU. \* Prepare the input. \* Run the
inference. \* Get the answer from the model output.

.. code:: ipython3

    core = ov.Core()
    
    # Read the model from files.
    model = core.read_model(model=compressed_model_xml)
    
    # Assign dynamic shapes to every input layer.
    for input_layer in model.inputs:
        input_shape = input_layer.partial_shape
        input_shape[1] = -1
        model.reshape({input_layer: input_shape})
    
    # Compile the model for a specific device.
    compiled_model_int8 = core.compile_model(model=model, device_name="CPU")
    
    output_layer = compiled_model_int8.outputs[0]

The Data Source returns a pair of sentences (indicated by
``sample_idx``) and the inference compares these sentences and outputs
whether their meaning is the same. You can test other sentences by
changing ``sample_idx`` to another value (from 0 to 407).

.. code:: ipython3

    sample_idx = 5
    sample = data_source[sample_idx]
    inputs = {k: torch.unsqueeze(torch.tensor(sample[k]), 0) for k in ['input_ids', 'token_type_ids', 'attention_mask']}
    
    result = compiled_model_int8(inputs)[output_layer]
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

    def validate(model: Model, dataset: Iterable[Any]) -> float:
        """
        Evaluate the model on GLUE dataset. 
        Returns F1 score metric.
        """
        compiled_model = core.compile_model(model, device_name='CPU')
        output_layer = compiled_model.output(0)
    
        metric = evaluate.load('glue', 'mrpc')
        INPUT_NAMES = [x.any_name for x in compiled_model.inputs]
        for batch in dataset:
            inputs = [
                np.expand_dims(np.asarray(batch[key], dtype=np.int64), 0) for key in INPUT_NAMES
            ]
            outputs = compiled_model(inputs)[output_layer]
            predictions = outputs[0].argmax(axis=-1)
            metric.add_batch(predictions=[predictions], references=[batch['labels']])
        metrics = metric.compute()
        f1_score = metrics['f1']
    
        return f1_score
    
    
    print('Checking the accuracy of the original model:')
    metric = validate(model, data_source)
    print(f'F1 score: {metric:.4f}')
    
    print('Checking the accuracy of the quantized model:')
    metric = validate(quantized_model, data_source)
    print(f'F1 score: {metric:.4f}')


.. parsed-literal::

    Checking the accuracy of the original model:
    F1 score: 0.8927
    Checking the accuracy of the quantized model:
    F1 score: 0.9014


Compare Performance of the Original, Converted and Quantized Models
-------------------------------------------------------------------

Compare the original PyTorch model with OpenVINO converted and quantized
models (``FP32``, ``INT8``) to see the difference in performance. It is
expressed in Sentences Per Second (SPS) measure, which is the same as
Frames Per Second (FPS) for images.

.. code:: ipython3

    model = core.read_model(model=ir_model_xml)
    
    # Assign dynamic shapes to every input layer.
    dynamic_shapes = {}
    for input_layer in model.inputs:
        input_shape = input_layer.partial_shape
        input_shape[1] = -1
    
        dynamic_shapes[input_layer] = input_shape
    
    model.reshape(dynamic_shapes)
    
    # Compile the model for a specific device.
    compiled_model_fp32 = core.compile_model(model=model, device_name="CPU")

.. code:: ipython3

    num_samples = 50
    sample = data_source[0]
    inputs = {k: torch.unsqueeze(torch.tensor(sample[k]), 0) for k in ['input_ids', 'token_type_ids', 'attention_mask']}
    
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(num_samples):
            torch_model(torch.vstack(list(inputs.values())))
        end = time.perf_counter()
        time_torch = end - start
    print(
        f"PyTorch model on CPU: {time_torch / num_samples:.3f} seconds per sentence, "
        f"SPS: {num_samples / time_torch:.2f}"
    )
    
    start = time.perf_counter()
    for _ in range(num_samples):
        compiled_model_fp32(inputs)
    end = time.perf_counter()
    time_ir = end - start
    print(
        f"IR FP32 model in OpenVINO Runtime/CPU: {time_ir / num_samples:.3f} "
        f"seconds per sentence, SPS: {num_samples / time_ir:.2f}"
    )
    
    start = time.perf_counter()
    for _ in range(num_samples):
        compiled_model_int8(inputs)
    end = time.perf_counter()
    time_ir = end - start
    print(
        f"OpenVINO IR INT8 model in OpenVINO Runtime/CPU: {time_ir / num_samples:.3f} "
        f"seconds per sentence, SPS: {num_samples / time_ir:.2f}"
    )


.. parsed-literal::

    PyTorch model on CPU: 0.072 seconds per sentence, SPS: 13.80
    IR FP32 model in OpenVINO Runtime/CPU: 0.020 seconds per sentence, SPS: 50.10
    OpenVINO IR INT8 model in OpenVINO Runtime/CPU: 0.009 seconds per sentence, SPS: 113.99


Finally, measure the inference performance of OpenVINO ``FP32`` and
``INT8`` models. For this purpose, use `Benchmark
Tool <https://docs.openvino.ai/latest/openvino_inference_engine_tools_benchmark_tool_README.html>`__
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
    ! benchmark_app -m $ir_model_xml -d CPU -api sync


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2022.3.0-9052-9752fafe8eb-releases/2022/3
    [ INFO ] 
    [ INFO ] Device info:
    [ INFO ] CPU
    [ INFO ] Build ................................. 2022.3.0-9052-9752fafe8eb-releases/2022/3
    [ INFO ] 
    [ INFO ] 
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(CPU) performance hint will be set to LATENCY.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 170.14 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     input_ids (node: input_ids) : i64 / [...] / [1,128]
    [ INFO ]     attention_mask (node: attention_mask) : i64 / [...] / [1,128]
    [ INFO ]     token_type_ids (node: token_type_ids) : i64 / [...] / [1,128]
    [ INFO ] Model outputs:
    [ INFO ]     output (node: output) : f32 / [...] / [1,2]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     input_ids (node: input_ids) : i64 / [...] / [1,128]
    [ INFO ]     attention_mask (node: attention_mask) : i64 / [...] / [1,128]
    [ INFO ]     token_type_ids (node: token_type_ids) : i64 / [...] / [1,128]
    [ INFO ] Model outputs:
    [ INFO ]     output (node: output) : f32 / [...] / [1,2]
    [Step 7/11] Loading the model to the device
    [ INFO ] Compile model took 216.41 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: torch_jit
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 1
    [ INFO ]   NUM_STREAMS: 1
    [ INFO ]   AFFINITY: Affinity.CORE
    [ INFO ]   INFERENCE_NUM_THREADS: 12
    [ INFO ]   PERF_COUNT: False
    [ INFO ]   INFERENCE_PRECISION_HINT: <Type: 'float32'>
    [ INFO ]   PERFORMANCE_HINT: PerformanceMode.LATENCY
    [ INFO ]   PERFORMANCE_HINT_NUM_REQUESTS: 0
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'input_ids'!. This input will be filled with random values!
    [ WARNING ] No input files were given for input 'attention_mask'!. This input will be filled with random values!
    [ WARNING ] No input files were given for input 'token_type_ids'!. This input will be filled with random values!
    [ INFO ] Fill input 'input_ids' with random values 
    [ INFO ] Fill input 'attention_mask' with random values 
    [ INFO ] Fill input 'token_type_ids' with random values 
    [Step 10/11] Measuring performance (Start inference synchronously, limits: 60000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 30.61 ms
    [Step 11/11] Dumping statistics report
    [ INFO ] Count:            3076 iterations
    [ INFO ] Duration:         60012.39 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        19.42 ms
    [ INFO ]    Average:       19.42 ms
    [ INFO ]    Min:           18.70 ms
    [ INFO ]    Max:           22.31 ms
    [ INFO ] Throughput:   51.49 FPS


.. code:: ipython3

    # Inference INT8 model (OpenVINO IR)
    ! benchmark_app -m $compressed_model_xml -d CPU -api sync


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2022.3.0-9052-9752fafe8eb-releases/2022/3
    [ INFO ] 
    [ INFO ] Device info:
    [ INFO ] CPU
    [ INFO ] Build ................................. 2022.3.0-9052-9752fafe8eb-releases/2022/3
    [ INFO ] 
    [ INFO ] 
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(CPU) performance hint will be set to LATENCY.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 130.18 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     input_ids , input_ids:0 (node: input_ids) : i64 / [...] / [1,128]
    [ INFO ]     attention_mask (node: attention_mask) : i64 / [...] / [1,128]
    [ INFO ]     token_type_ids:0 , token_type_ids (node: token_type_ids) : i64 / [...] / [1,128]
    [ INFO ] Model outputs:
    [ INFO ]     output (node: output) : f32 / [...] / [1,2]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     input_ids , input_ids:0 (node: input_ids) : i64 / [...] / [1,128]
    [ INFO ]     attention_mask (node: attention_mask) : i64 / [...] / [1,128]
    [ INFO ]     token_type_ids:0 , token_type_ids (node: token_type_ids) : i64 / [...] / [1,128]
    [ INFO ] Model outputs:
    [ INFO ]     output (node: output) : f32 / [...] / [1,2]
    [Step 7/11] Loading the model to the device
    [ INFO ] Compile model took 387.50 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: torch_jit
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 1
    [ INFO ]   NUM_STREAMS: 1
    [ INFO ]   AFFINITY: Affinity.CORE
    [ INFO ]   INFERENCE_NUM_THREADS: 12
    [ INFO ]   PERF_COUNT: False
    [ INFO ]   INFERENCE_PRECISION_HINT: <Type: 'float32'>
    [ INFO ]   PERFORMANCE_HINT: PerformanceMode.LATENCY
    [ INFO ]   PERFORMANCE_HINT_NUM_REQUESTS: 0
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'input_ids'!. This input will be filled with random values!
    [ WARNING ] No input files were given for input 'attention_mask'!. This input will be filled with random values!
    [ WARNING ] No input files were given for input 'token_type_ids'!. This input will be filled with random values!
    [ INFO ] Fill input 'input_ids' with random values 
    [ INFO ] Fill input 'attention_mask' with random values 
    [ INFO ] Fill input 'token_type_ids' with random values 
    [Step 10/11] Measuring performance (Start inference synchronously, limits: 60000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 14.96 ms
    [Step 11/11] Dumping statistics report
    [ INFO ] Count:            6830 iterations
    [ INFO ] Duration:         60003.55 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        8.74 ms
    [ INFO ]    Average:       8.70 ms
    [ INFO ]    Min:           7.54 ms
    [ INFO ]    Max:           11.06 ms
    [ INFO ] Throughput:   114.43 FPS

