Quantize NLP models with Post-Training Quantization ​in NNCF
============================================================

.. _top:

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

**Table of contents**:

- `Imports <#imports>`__
- `Settings <#settings>`__
- `Prepare the Model <#prepare-the-model>`__
- `Prepare the Dataset <#prepare-the-dataset>`__
- `Optimize model using NNCF Post-training Quantization API <#optimize-model-using-nncf-post-training-quantization-api>`__
- `Load and Test OpenVINO Model <#load-and-test-openvino-model>`__

  - `Select inference device <#select-inference-device>`__

- `Compare F1-score of FP32 and INT8 models <#compare-f1-score-of-fp32-and-int8-models>`__
- `Compare Performance of the Original, Converted and Quantized Models <#compare-performance-of-the-original,-converted-and-quantized-models>`__

.. code:: ipython3

    !pip install -q "nncf>=2.5.0" datasets evaluate

Imports `⇑ <#top>`__
###############################################################################################################################


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
    from openvino.runtime import serialize, Model, PartialShape
    import nncf
    from nncf.parameters import ModelType
    from transformers import BertForSequenceClassification, BertTokenizer
    from openvino.tools.mo import convert_model
    import datasets
    import evaluate
    
    sys.path.append("../utils")
    from notebook_utils import download_file


.. parsed-literal::

    2023-08-15 22:29:19.942802: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2023-08-15 22:29:19.975605: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-08-15 22:29:20.517786: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


Settings `⇑ <#top>`__
###############################################################################################################################


.. code:: ipython3

    # Set the data and model directories, source URL and the filename of the model.
    DATA_DIR = "data"
    MODEL_DIR = "model"
    MODEL_LINK = "https://download.pytorch.org/tutorial/MRPC.zip"
    FILE_NAME = MODEL_LINK.split("/")[-1]
    PRETRAINED_MODEL_DIR = os.path.join(MODEL_DIR, "MRPC")
    
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

Prepare the Model `⇑ <#top>`__
###############################################################################################################################


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
    input_shape = PartialShape([1, -1])
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
        model = convert_model(torch_model, example_input=inputs, input=input_info)
        serialize(model, str(ir_model_xml))
    else:
        model = core.read_model(ir_model_xml)


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-475/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/torch/jit/annotations.py:309: UserWarning: TorchScript will treat type annotations of Tensor dtype-specific subtypes as if they are normal Tensors. dtype constraints are not enforced in compilation either.
      warnings.warn("TorchScript will treat type annotations of Tensor "


Prepare the Dataset `⇑ <#top>`__
###############################################################################################################################

We download the `General Language Understanding Evaluation (GLUE) <https://gluebenchmark.com/>`__ dataset
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

Optimize model using NNCF Post-training Quantization API `⇑ <#top>`__
###############################################################################################################################


`NNCF <https://github.com/openvinotoolkit/nncf>`__ provides a suite of
advanced algorithms for Neural Networks inference optimization in
OpenVINO with minimal accuracy drop. We will use 8-bit quantization in
post-training mode (without the fine-tuning pipeline) to optimize BERT.

The optimization process contains the following steps:

1. Create a Dataset for quantization
2. Run ``nncf.quantize`` for getting an optimized model
3. Serialize OpenVINO IR model using ``openvino.runtime.serialize``
   function

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

    INFO:nncf:202 ignored nodes was found by types in the NNCFGraph
    INFO:nncf:24 ignored nodes was found by name in the NNCFGraph
    INFO:nncf:Not adding activation input quantizer for operation: 22 aten::rsub_16
    INFO:nncf:Not adding activation input quantizer for operation: 25 aten::rsub_17
    INFO:nncf:Not adding activation input quantizer for operation: 30 aten::mul_18
    INFO:nncf:Not adding activation input quantizer for operation: 11 aten::add_40
    INFO:nncf:Not adding activation input quantizer for operation: 14 aten::add__46
    INFO:nncf:Not adding activation input quantizer for operation: 17 aten::layer_norm_48
    20 aten::layer_norm_49
    23 aten::layer_norm_50
    
    INFO:nncf:Not adding activation input quantizer for operation: 36 aten::add_108
    INFO:nncf:Not adding activation input quantizer for operation: 55 aten::softmax_109
    INFO:nncf:Not adding activation input quantizer for operation: 74 aten::matmul_110
    INFO:nncf:Not adding activation input quantizer for operation: 26 aten::add_126
    INFO:nncf:Not adding activation input quantizer for operation: 31 aten::layer_norm_128
    47 aten::layer_norm_129
    66 aten::layer_norm_130
    
    INFO:nncf:Not adding activation input quantizer for operation: 85 aten::add_140
    INFO:nncf:Not adding activation input quantizer for operation: 103 aten::layer_norm_142
    133 aten::layer_norm_143
    171 aten::layer_norm_144
    
    INFO:nncf:Not adding activation input quantizer for operation: 38 aten::add_202
    INFO:nncf:Not adding activation input quantizer for operation: 57 aten::softmax_203
    INFO:nncf:Not adding activation input quantizer for operation: 76 aten::matmul_204
    INFO:nncf:Not adding activation input quantizer for operation: 209 aten::add_220
    INFO:nncf:Not adding activation input quantizer for operation: 236 aten::layer_norm_222
    250 aten::layer_norm_223
    267 aten::layer_norm_224
    
    INFO:nncf:Not adding activation input quantizer for operation: 287 aten::add_234
    INFO:nncf:Not adding activation input quantizer for operation: 316 aten::layer_norm_236
    342 aten::layer_norm_237
    364 aten::layer_norm_238
    
    INFO:nncf:Not adding activation input quantizer for operation: 39 aten::add_296
    INFO:nncf:Not adding activation input quantizer for operation: 58 aten::softmax_297
    INFO:nncf:Not adding activation input quantizer for operation: 77 aten::matmul_298
    INFO:nncf:Not adding activation input quantizer for operation: 221 aten::add_314
    INFO:nncf:Not adding activation input quantizer for operation: 242 aten::layer_norm_316
    259 aten::layer_norm_317
    279 aten::layer_norm_318
    
    INFO:nncf:Not adding activation input quantizer for operation: 300 aten::add_328
    INFO:nncf:Not adding activation input quantizer for operation: 326 aten::layer_norm_330
    348 aten::layer_norm_331
    370 aten::layer_norm_332
    
    INFO:nncf:Not adding activation input quantizer for operation: 40 aten::add_390
    INFO:nncf:Not adding activation input quantizer for operation: 59 aten::softmax_391
    INFO:nncf:Not adding activation input quantizer for operation: 78 aten::matmul_392
    INFO:nncf:Not adding activation input quantizer for operation: 223 aten::add_408
    INFO:nncf:Not adding activation input quantizer for operation: 243 aten::layer_norm_410
    260 aten::layer_norm_411
    280 aten::layer_norm_412
    
    INFO:nncf:Not adding activation input quantizer for operation: 302 aten::add_422
    INFO:nncf:Not adding activation input quantizer for operation: 328 aten::layer_norm_424
    350 aten::layer_norm_425
    372 aten::layer_norm_426
    
    INFO:nncf:Not adding activation input quantizer for operation: 41 aten::add_484
    INFO:nncf:Not adding activation input quantizer for operation: 60 aten::softmax_485
    INFO:nncf:Not adding activation input quantizer for operation: 79 aten::matmul_486
    INFO:nncf:Not adding activation input quantizer for operation: 225 aten::add_502
    INFO:nncf:Not adding activation input quantizer for operation: 244 aten::layer_norm_504
    261 aten::layer_norm_505
    281 aten::layer_norm_506
    
    INFO:nncf:Not adding activation input quantizer for operation: 304 aten::add_516
    INFO:nncf:Not adding activation input quantizer for operation: 330 aten::layer_norm_518
    352 aten::layer_norm_519
    374 aten::layer_norm_520
    
    INFO:nncf:Not adding activation input quantizer for operation: 42 aten::add_578
    INFO:nncf:Not adding activation input quantizer for operation: 61 aten::softmax_579
    INFO:nncf:Not adding activation input quantizer for operation: 80 aten::matmul_580
    INFO:nncf:Not adding activation input quantizer for operation: 227 aten::add_596
    INFO:nncf:Not adding activation input quantizer for operation: 245 aten::layer_norm_598
    262 aten::layer_norm_599
    282 aten::layer_norm_600
    
    INFO:nncf:Not adding activation input quantizer for operation: 306 aten::add_610
    INFO:nncf:Not adding activation input quantizer for operation: 332 aten::layer_norm_612
    354 aten::layer_norm_613
    376 aten::layer_norm_614
    
    INFO:nncf:Not adding activation input quantizer for operation: 43 aten::add_672
    INFO:nncf:Not adding activation input quantizer for operation: 62 aten::softmax_673
    INFO:nncf:Not adding activation input quantizer for operation: 81 aten::matmul_674
    INFO:nncf:Not adding activation input quantizer for operation: 229 aten::add_690
    INFO:nncf:Not adding activation input quantizer for operation: 246 aten::layer_norm_692
    263 aten::layer_norm_693
    283 aten::layer_norm_694
    
    INFO:nncf:Not adding activation input quantizer for operation: 308 aten::add_704
    INFO:nncf:Not adding activation input quantizer for operation: 334 aten::layer_norm_706
    356 aten::layer_norm_707
    378 aten::layer_norm_708
    
    INFO:nncf:Not adding activation input quantizer for operation: 44 aten::add_766
    INFO:nncf:Not adding activation input quantizer for operation: 63 aten::softmax_767
    INFO:nncf:Not adding activation input quantizer for operation: 82 aten::matmul_768
    INFO:nncf:Not adding activation input quantizer for operation: 231 aten::add_784
    INFO:nncf:Not adding activation input quantizer for operation: 247 aten::layer_norm_786
    264 aten::layer_norm_787
    284 aten::layer_norm_788
    
    INFO:nncf:Not adding activation input quantizer for operation: 310 aten::add_798
    INFO:nncf:Not adding activation input quantizer for operation: 336 aten::layer_norm_800
    358 aten::layer_norm_801
    380 aten::layer_norm_802
    
    INFO:nncf:Not adding activation input quantizer for operation: 45 aten::add_860
    INFO:nncf:Not adding activation input quantizer for operation: 64 aten::softmax_861
    INFO:nncf:Not adding activation input quantizer for operation: 83 aten::matmul_862
    INFO:nncf:Not adding activation input quantizer for operation: 233 aten::add_878
    INFO:nncf:Not adding activation input quantizer for operation: 248 aten::layer_norm_880
    265 aten::layer_norm_881
    285 aten::layer_norm_882
    
    INFO:nncf:Not adding activation input quantizer for operation: 312 aten::add_892
    INFO:nncf:Not adding activation input quantizer for operation: 338 aten::layer_norm_894
    360 aten::layer_norm_895
    382 aten::layer_norm_896
    
    INFO:nncf:Not adding activation input quantizer for operation: 46 aten::add_954
    INFO:nncf:Not adding activation input quantizer for operation: 65 aten::softmax_955
    INFO:nncf:Not adding activation input quantizer for operation: 84 aten::matmul_956
    INFO:nncf:Not adding activation input quantizer for operation: 235 aten::add_972
    INFO:nncf:Not adding activation input quantizer for operation: 249 aten::layer_norm_974
    266 aten::layer_norm_975
    286 aten::layer_norm_976
    
    INFO:nncf:Not adding activation input quantizer for operation: 314 aten::add_986
    INFO:nncf:Not adding activation input quantizer for operation: 340 aten::layer_norm_988
    362 aten::layer_norm_989
    384 aten::layer_norm_990
    
    INFO:nncf:Not adding activation input quantizer for operation: 35 aten::add_1048
    INFO:nncf:Not adding activation input quantizer for operation: 54 aten::softmax_1049
    INFO:nncf:Not adding activation input quantizer for operation: 73 aten::matmul_1050
    INFO:nncf:Not adding activation input quantizer for operation: 215 aten::add_1066
    INFO:nncf:Not adding activation input quantizer for operation: 240 aten::layer_norm_1068
    257 aten::layer_norm_1069
    277 aten::layer_norm_1070
    
    INFO:nncf:Not adding activation input quantizer for operation: 296 aten::add_1080
    INFO:nncf:Not adding activation input quantizer for operation: 322 aten::layer_norm_1082
    344 aten::layer_norm_1083
    366 aten::layer_norm_1084
    
    INFO:nncf:Not adding activation input quantizer for operation: 37 aten::add_1142
    INFO:nncf:Not adding activation input quantizer for operation: 56 aten::softmax_1143
    INFO:nncf:Not adding activation input quantizer for operation: 75 aten::matmul_1144
    INFO:nncf:Not adding activation input quantizer for operation: 218 aten::add_1160
    INFO:nncf:Not adding activation input quantizer for operation: 241 aten::layer_norm_1162
    258 aten::layer_norm_1163
    278 aten::layer_norm_1164
    
    INFO:nncf:Not adding activation input quantizer for operation: 298 aten::add_1174
    INFO:nncf:Not adding activation input quantizer for operation: 324 aten::layer_norm_1176
    346 aten::layer_norm_1177
    368 aten::layer_norm_1178
    


.. parsed-literal::

    Statistics collection: 100%|██████████| 300/300 [00:24<00:00, 12.04it/s]
    Biases correction: 100%|██████████| 74/74 [00:25<00:00,  2.95it/s]


.. code:: ipython3

    compressed_model_xml = Path(MODEL_DIR) / "quantized_bert_mrpc.xml"
    ov.serialize(quantized_model, compressed_model_xml)

Load and Test OpenVINO Model `⇑ <#top>`__
###############################################################################################################################


To load and test converted model, perform the following:

-  Load the model and compile it for selected device.
-  Prepare the input.
-  Run the inference.
-  Get the answer from the model output.

Select inference device `⇑ <#top>`__
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


Select device from dropdown list for running inference using OpenVINO:

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


Compare F1-score of FP32 and INT8 models `⇑ <#top>`__
###############################################################################################################################


.. code:: ipython3

    def validate(model: Model, dataset: Iterable[Any]) -> float:
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
    F1 score: 0.8995


Compare Performance of the Original, Converted and Quantized Models. `⇑ <#top>`__
###############################################################################################################################

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

    PyTorch model on CPU: 0.070 seconds per sentence, SPS: 14.22
    IR FP32 model in OpenVINO Runtime/AUTO: 0.021 seconds per sentence, SPS: 48.42
    OpenVINO IR INT8 model in OpenVINO Runtime/AUTO: 0.010 seconds per sentence, SPS: 98.01


Finally, measure the inference performance of OpenVINO ``FP32`` and
``INT8`` models. For this purpose, use 
`Benchmark Tool <https://docs.openvino.ai/2023.0/openvino_inference_engine_tools_benchmark_tool_README.html>`__
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
    ! benchmark_app -m $ir_model_xml -shape [1,128],[1,128],[1,128] -d device.value -api sync


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ WARNING ] Default duration 120 seconds is used for unknown device device.value
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2023.0.0-10926-b4452d56304-releases/2023/0
    [ INFO ] 
    [ INFO ] Device info:
    [ ERROR ] Check 'false' failed at src/inference/src/core.cpp:84:
    Device with "device" name is not registered in the OpenVINO Runtime
    Traceback (most recent call last):
      File "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-475/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/openvino/tools/benchmark/main.py", line 103, in main
        benchmark.print_version_info()
      File "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-475/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/openvino/tools/benchmark/benchmark.py", line 48, in print_version_info
        for device, version in self.core.get_versions(self.device).items():
    RuntimeError: Check 'false' failed at src/inference/src/core.cpp:84:
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
    [ INFO ] Build ................................. 2023.0.0-10926-b4452d56304-releases/2023/0
    [ INFO ] 
    [ INFO ] Device info:
    [ ERROR ] Check 'false' failed at src/inference/src/core.cpp:84:
    Device with "device" name is not registered in the OpenVINO Runtime
    Traceback (most recent call last):
      File "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-475/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/openvino/tools/benchmark/main.py", line 103, in main
        benchmark.print_version_info()
      File "/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-475/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/openvino/tools/benchmark/benchmark.py", line 48, in print_version_info
        for device, version in self.core.get_versions(self.device).items():
    RuntimeError: Check 'false' failed at src/inference/src/core.cpp:84:
    Device with "device" name is not registered in the OpenVINO Runtime
    

