Quantize NLP models with Post-Training Quantization ​in NNCF
============================================================

This tutorial demonstrates how to apply ``INT8`` quantization to the
Natural Language Processing model known as
`BERT <https://en.wikipedia.org/wiki/BERT_(language_model)>`__, using
the `Post-Training Quantization
API <https://docs.openvino.ai/nightly/basic_quantization_flow.html>`__
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
-  `Load and Test OpenVINO
   Model <#load-and-test-openvino-model>`__

   -  `Select inference device <#select-inference-device>`__

-  `Compare F1-score of FP32 and INT8
   models <#compare-f-score-of-fp-and-int-models>`__
-  `Compare Performance of the Original, Converted and Quantized
   Models <#compare-performance-of-the-original-converted-and-quantized-models>`__

.. code:: ipython3

    %pip install -q "nncf>=2.5.0" 
    %pip install -q "transformers" datasets evaluate
    %pip install -q "openvino>=2023.1.0"


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.


Imports 
-------------------------------------------------

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
    import urllib.request
    urllib.request.urlretrieve(
        url='https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/utils/notebook_utils.py',
        filename='notebook_utils.py'
    )
    from notebook_utils import download_file


.. parsed-literal::

    2023-10-30 22:33:08.247649: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2023-10-30 22:33:08.281400: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-10-30 22:33:08.912908: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


Settings 
--------------------------------------------------

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
-----------------------------------------------------------

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
    
    input_info = [("input_ids", input_shape, np.int64),("attention_mask", input_shape, np.int64),("token_type_ids", input_shape, np.int64)]
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

    WARNING:nncf:NNCF provides best results with torch==2.0.1, while current torch version is 2.1.0+cpu. If you encounter issues, consider switching to torch==2.0.1


.. parsed-literal::

    No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda'
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/torch/jit/annotations.py:386: UserWarning: TorchScript will treat type annotations of Tensor dtype-specific subtypes as if they are normal Tensors. dtype constraints are not enforced in compilation either.
      warnings.warn(


Prepare the Dataset 
-------------------------------------------------------------

We download the `General Language Understanding Evaluation
(GLUE) <https://gluebenchmark.com/>`__ dataset for the MRPC task from
HuggingFace datasets. Then, we tokenize the data with a pre-trained BERT
tokenizer from HuggingFace.

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

    Map:   0%|          | 0/408 [00:00<?, ? examples/s]


Optimize model using NNCF Post-training Quantization API 
--------------------------------------------------------------------------------------------------

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
        inputs = {
            name: np.asarray([data_item[name]], dtype=np.int64) for name in INPUT_NAMES
        }
        return inputs
    
    calibration_dataset = nncf.Dataset(data_source, transform_fn)
    # Quantize the model. By specifying model_type, we specify additional transformer patterns in the model.
    quantized_model = nncf.quantize(model, calibration_dataset,
                                    model_type=ModelType.TRANSFORMER)


.. parsed-literal::

    Statistics collection: 100%|██████████| 300/300 [00:07<00:00, 39.50it/s]
    Applying Smooth Quant: 100%|██████████| 50/50 [00:00<00:00, 51.91it/s]


.. parsed-literal::

    INFO:nncf:36 ignored nodes was found by name in the NNCFGraph


.. parsed-literal::

    Statistics collection: 100%|██████████| 300/300 [00:25<00:00, 11.96it/s]
    Applying Fast Bias correction: 100%|██████████| 74/74 [00:25<00:00,  2.93it/s]


.. code:: ipython3

    compressed_model_xml = Path(MODEL_DIR) / "quantized_bert_mrpc.xml"
    ov.save_model(quantized_model, compressed_model_xml)

Load and Test OpenVINO Model 
----------------------------------------------------------------------

To load and test converted model, perform the following:

-  Load the model and compile it for selected device.
-  Prepare the input.
-  Run the inference.
-  Get the answer from the model output.

Select inference device 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

select device from dropdown list for running inference using OpenVINO

.. code:: ipython3

    import ipywidgets as widgets
    
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value='AUTO',
        description='Device:',
        disabled=False,
    )
    
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
    inputs = {k: torch.unsqueeze(torch.tensor(sample[k]), 0) for k in ['input_ids', 'token_type_ids', 'attention_mask']}
    
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
----------------------------------------------------------------------------------

.. code:: ipython3

    def validate(model: ov.Model, dataset: Iterable[Any]) -> float:
        """
        Evaluate the model on GLUE dataset. 
        Returns F1 score metric.
        """
        compiled_model = core.compile_model(model, device_name=device.value)
        output_layer = compiled_model.output(0)
    
        metric = evaluate.load('glue', 'mrpc')
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
    F1 score: 0.9019
    Checking the accuracy of the quantized model:
    F1 score: 0.8985


Compare Performance of the Original, Converted and Quantized Models 
-------------------------------------------------------------------------------------------------------------

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
        compiled_model(inputs)
    end = time.perf_counter()
    time_ir = end - start
    print(
        f"IR FP32 model in OpenVINO Runtime/{device.value}: {time_ir / num_samples:.3f} "
        f"seconds per sentence, SPS: {num_samples / time_ir:.2f}"
    )
    
    start = time.perf_counter()
    for _ in range(num_samples):
        compiled_quantized_model(inputs)
    end = time.perf_counter()
    time_ir = end - start
    print(
        f"OpenVINO IR INT8 model in OpenVINO Runtime/{device.value}: {time_ir / num_samples:.3f} "
        f"seconds per sentence, SPS: {num_samples / time_ir:.2f}"
    )


.. parsed-literal::

    We strongly recommend passing in an `attention_mask` since your input_ids may be padded. See https://huggingface.co/docs/transformers/troubleshooting#incorrect-output-when-padding-tokens-arent-masked.


.. parsed-literal::

    PyTorch model on CPU: 0.073 seconds per sentence, SPS: 13.72
    IR FP32 model in OpenVINO Runtime/AUTO: 0.022 seconds per sentence, SPS: 46.40
    OpenVINO IR INT8 model in OpenVINO Runtime/AUTO: 0.010 seconds per sentence, SPS: 98.65


Finally, measure the inference performance of OpenVINO ``FP32`` and
``INT8`` models. For this purpose, use `Benchmark
Tool <https://docs.openvino.ai/2023.0/openvino_inference_engine_tools_benchmark_tool_README.html>`__
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
    !benchmark_app -m $ir_model_xml -shape [1,128],[1,128],[1,128] -d device.value -api sync


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ WARNING ] Default duration 120 seconds is used for unknown device device.value
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2023.1.0-12185-9e6b00e51cd-releases/2023/1
    [ INFO ] 
    [ INFO ] Device info:
    [ ERROR ] Exception from src/inference/src/core.cpp:84:
    Exception from src/inference/src/dev/core_impl.cpp:565:
    Device with "device" name is not registered in the OpenVINO Runtime
    
    Traceback (most recent call last):
      File "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/openvino/tools/benchmark/main.py", line 102, in main
        benchmark.print_version_info()
      File "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/openvino/tools/benchmark/benchmark.py", line 48, in print_version_info
        for device, version in self.core.get_versions(self.device).items():
    RuntimeError: Exception from src/inference/src/core.cpp:84:
    Exception from src/inference/src/dev/core_impl.cpp:565:
    Device with "device" name is not registered in the OpenVINO Runtime
    
    


.. code:: ipython3

    # Inference INT8 model (OpenVINO IR)
    ! benchmark_app -m $compressed_model_xml -shape [1,128],[1,128],[1,128] -d device.value -api sync


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ WARNING ] Default duration 120 seconds is used for unknown device device.value
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2023.1.0-12185-9e6b00e51cd-releases/2023/1
    [ INFO ] 
    [ INFO ] Device info:
    [ ERROR ] Exception from src/inference/src/core.cpp:84:
    Exception from src/inference/src/dev/core_impl.cpp:565:
    Device with "device" name is not registered in the OpenVINO Runtime
    
    Traceback (most recent call last):
      File "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/openvino/tools/benchmark/main.py", line 102, in main
        benchmark.print_version_info()
      File "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/openvino/tools/benchmark/benchmark.py", line 48, in print_version_info
        for device, version in self.core.get_versions(self.device).items():
    RuntimeError: Exception from src/inference/src/core.cpp:84:
    Exception from src/inference/src/dev/core_impl.cpp:565:
    Device with "device" name is not registered in the OpenVINO Runtime
    
    

