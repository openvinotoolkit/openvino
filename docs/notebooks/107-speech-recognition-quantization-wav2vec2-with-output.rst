Quantize Speech Recognition Models using NNCF PTQ API
=====================================================

This tutorial demonstrates how to apply ``INT8`` quantization to the
speech recognition model, known as
`Wav2Vec2 <https://huggingface.co/docs/transformers/model_doc/wav2vec2>`__,
using the NNCF (Neural Network Compression Framework) 8-bit quantization
in post-training mode (without the fine-tuning pipeline). This notebook
uses a fine-tuned
`Wav2Vec2-Base-960h <https://huggingface.co/facebook/wav2vec2-base-960h>`__
`PyTorch <https://pytorch.org/>`__ model trained on the `LibriSpeech ASR
corpus <https://www.openslr.org/12>`__. The tutorial is designed to be
extendable to custom models and datasets. It consists of the following
steps:

-  Download and prepare the Wav2Vec2 model and LibriSpeech dataset.
-  Define data loading and accuracy validation functionality.
-  Model quantization.
-  Compare Accuracy of original PyTorch model, OpenVINO FP16 and INT8
   models.
-  Compare performance of the original and quantized models.

**Table of contents:**

- `Imports <#imports>`__
- `Settings <#settings>`__
- `Prepare the Model <#prepare-the-model>`__
- `Prepare LibriSpeech Dataset <#prepare-librispeech-dataset>`__
- `Define DataLoader <#define-dataloader>`__
- `Run Quantization <#run-quantization>`__
- `Model Usage Example with Inference Pipeline <#model-usage-example-with-inference-pipeline>`__
- `Validate model accuracy on dataset <#validate-model-accuracy-on-dataset>`__
- `Compare Performance of the Original and Quantized Models <#compare-performance-of-the-original-and-quantized-models>`__

.. code:: ipython3

    !pip install -q "openvino==2023.1.0.dev20230811" "nncf>=2.5.0"
    !pip install -q soundfile librosa transformers onnx

Imports
###############################################################################################################################

.. code:: ipython3

    import os
    import sys
    import re
    import numpy as np
    import openvino as ov
    import tarfile
    import torch
    from itertools import groupby
    import soundfile as sf
    import IPython.display as ipd
    
    from transformers import Wav2Vec2ForCTC
    
    sys.path.append("../utils")
    from notebook_utils import download_file


.. parsed-literal::

    2023-09-08 22:38:42.752981: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2023-09-08 22:38:42.787924: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-09-08 22:38:43.332490: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


Settings
###############################################################################################################################

.. code:: ipython3

    from pathlib import Path
    
    # Set the data and model directories, model source URL and model filename.
    MODEL_DIR = Path("model")
    DATA_DIR = Path("../data/datasets/librispeech")
    MODEL_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

Prepare the Model
###############################################################################################################################

Perform the following: - Download and unpack a pre-trained Wav2Vec2
model. - Convert the model to ONNX. - Run model conversion API to
convert the model from the ONNX representation to the OpenVINO
Intermediate Representation (OpenVINO IR).

.. code:: ipython3

    download_file("https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/pytorch_model.bin", directory=Path(MODEL_DIR) / 'pytorch', show_progress=True)
    download_file("https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/config.json", directory=Path(MODEL_DIR) / 'pytorch', show_progress=False)



.. parsed-literal::

    model/pytorch/pytorch_model.bin:   0%|          | 0.00/360M [00:00<?, ?B/s]




.. parsed-literal::

    PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/notebooks/107-speech-recognition-quantization/model/pytorch/config.json')



Import all dependencies to load the original PyTorch model and convert
it to the ONNX representation.

.. code:: ipython3

    BATCH_SIZE = 1
    MAX_SEQ_LENGTH = 30480
    
    
    def export_model_to_onnx(model, path):
        with torch.no_grad():
            default_input = torch.zeros([1, MAX_SEQ_LENGTH], dtype=torch.float)
            inputs = {
                "inputs": default_input
            }
            symbolic_names = {0: "batch_size", 1: "sequence_len"}
            torch.onnx.export(
                model,
                (inputs["inputs"]),
                path,
                opset_version=11,
                input_names=["inputs"],
                output_names=["logits"],
                dynamic_axes={
                    "inputs": symbolic_names,
                    "logits": symbolic_names,
                },
            )
            print("ONNX model saved to {}".format(path))
    
    
    torch_model = Wav2Vec2ForCTC.from_pretrained(Path(MODEL_DIR) / 'pytorch')
    onnx_model_path = Path(MODEL_DIR) / "wav2vec2_base.onnx"
    if not onnx_model_path.exists():
        export_model_to_onnx(torch_model, onnx_model_path)


.. parsed-literal::

    Some weights of Wav2Vec2ForCTC were not initialized from the model checkpoint at model/pytorch and are newly initialized: ['wav2vec2.masked_spec_embed']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/wav2vec2/modeling_wav2vec2.py:595: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/wav2vec2/modeling_wav2vec2.py:634: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):


.. parsed-literal::

    ONNX model saved to model/wav2vec2_base.onnx


.. code:: ipython3

    ov_model = ov.convert_model(onnx_model_path)
    
    ir_model_path = MODEL_DIR / "wav2vec2_base.xml"
    ov.save_model(ov_model, str(ir_model_path))

Prepare LibriSpeech Dataset
###############################################################################################################################

Use the code below to download and unpack the archives with ‘dev-clean’
and ‘test-clean’ subsets of LibriSpeech Dataset.

.. code:: ipython3

    download_file("http://openslr.elda.org/resources/12/dev-clean.tar.gz", directory=DATA_DIR, show_progress=True)
    download_file("http://openslr.elda.org/resources/12/test-clean.tar.gz", directory=DATA_DIR, show_progress=True)
    
    if not os.path.exists(f'{DATA_DIR}/LibriSpeech/dev-clean'):
        with tarfile.open(f"{DATA_DIR}/dev-clean.tar.gz") as tar:
            tar.extractall(path=DATA_DIR)
    if not os.path.exists(f'{DATA_DIR}/LibriSpeech/test-clean'):
        with tarfile.open(f"{DATA_DIR}/test-clean.tar.gz") as tar:
            tar.extractall(path=DATA_DIR)



.. parsed-literal::

    ../data/datasets/librispeech/dev-clean.tar.gz:   0%|          | 0.00/322M [00:00<?, ?B/s]



.. parsed-literal::

    ../data/datasets/librispeech/test-clean.tar.gz:   0%|          | 0.00/331M [00:00<?, ?B/s]


Define DataLoader
###############################################################################################################################

Wav2Vec2 model accepts a raw waveform of the speech signal as input and
produces vocabulary class estimations as output. Since the dataset
contains audio files in FLAC format, use the ``soundfile`` package to
convert them to waveform.

.. note::

   Consider increasing ``samples_limit`` to get more precise
   results. A suggested value is ``300`` or more, as it will take longer
   time to process.

.. code:: ipython3

    class LibriSpeechDataLoader:
    
        @staticmethod
        def read_flac(file_name):
            speech, samplerate = sf.read(file_name)
            assert samplerate == 16000, "read_flac: only 16kHz supported!"
            return speech
    
        # Required methods
        def __init__(self, config, samples_limit=300):
            """Constructor
            :param config: data loader specific config
            """
            self.samples_limit = samples_limit
            self._data_dir = config["data_source"]
            self._ds = []
            self._prepare_dataset()
    
        def __len__(self):
            """Returns size of the dataset"""
            return len(self._ds)
    
        def __getitem__(self, index):
            """
            Returns annotation, data and metadata at the specified index.
            Possible formats:
            (index, annotation), data
            (index, annotation), data, metadata
            """
            label = self._ds[index][0]
            inputs = {'inputs': np.expand_dims(self._ds[index][1], axis=0)}
            return label, inputs
    
        # Methods specific to the current implementation
        def _prepare_dataset(self):
            pattern = re.compile(r'([0-9\-]+)\s+(.+)')
            data_folder = Path(self._data_dir)
            txts = list(data_folder.glob('**/*.txt'))
            counter = 0
            for txt in txts:
                content = txt.open().readlines()
                for line in content:
                    res = pattern.search(line)
                    if not res:
                        continue
                    name = res.group(1)
                    transcript = res.group(2)
                    fname = txt.parent / name
                    fname = fname.with_suffix('.flac')
                    identifier = str(fname.relative_to(data_folder))
                    self._ds.append(((counter, transcript.upper()), LibriSpeechDataLoader.read_flac(os.path.join(self._data_dir, identifier))))
                    counter += 1
                    if counter >= self.samples_limit:
                        # Limit exceeded
                        return

Run Quantization
###############################################################################################################################

`NNCF <https://github.com/openvinotoolkit/nncf>`__ provides a suite of
advanced algorithms for Neural Networks inference optimization in
OpenVINO with minimal accuracy drop.

Create a quantized model from the pre-trained ``FP16`` model and the
calibration dataset. The optimization process contains the following
steps: 1. Create a Dataset for quantization. 2. Run ``nncf.quantize``
for getting an optimized model. The ``nncf.quantize`` function provides
an interface for model quantization. It requires an instance of the
OpenVINO Model and quantization dataset. Optionally, some additional
parameters for the configuration quantization process (number of samples
for quantization, preset, ignored scope, etc.) can be provided. For more
accurate results, we should keep the operation in the postprocessing
subgraph in floating point precision, using the ``ignored_scope``
parameter. ``advanced_parameters`` can be used to specify advanced
quantization parameters for fine-tuning the quantization algorithm. In
this tutorial we pass range estimator parameters for activations. For
more information see `Tune quantization
parameters <https://docs.openvino.ai/2023.0/basic_quantization_flow.html#tune-quantization-parameters>`__.
3. Serialize OpenVINO IR model using ``openvino.runtime.serialize``
function.

.. code:: ipython3

    import nncf
    from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters, RangeEstimatorParameters
    from nncf.quantization.range_estimator import StatisticsCollectorParameters, StatisticsType, AggregatorType
    from nncf.parameters import ModelType
    
    
    def transform_fn(data_item):
        """
        Extract the model's input from the data item.
        The data item here is the data item that is returned from the data source per iteration.
        This function should be passed when the data item cannot be used as model's input.
        """
        _, inputs = data_item
    
        return inputs["inputs"]
    
    
    dataset_config = {"data_source": os.path.join(DATA_DIR, "LibriSpeech/dev-clean")}
    data_loader = LibriSpeechDataLoader(dataset_config, samples_limit=300)
    calibration_dataset = nncf.Dataset(data_loader, transform_fn)
    
    
    quantized_model = nncf.quantize(
        ov_model,
        calibration_dataset,
        model_type=ModelType.TRANSFORMER,  # specify additional transformer patterns in the model
        ignored_scope=nncf.IgnoredScope(
            names=[
                '/wav2vec2/feature_extractor/conv_layers.1/conv/Conv',
                '/wav2vec2/feature_extractor/conv_layers.2/conv/Conv',
                '/wav2vec2/encoder/layers.7/feed_forward/output_dense/MatMul'
            ],
        ),
        advanced_parameters=AdvancedQuantizationParameters(
            activations_range_estimator_params=RangeEstimatorParameters(
                min=StatisticsCollectorParameters(
                    statistics_type=StatisticsType.MIN,
                    aggregator_type=AggregatorType.MIN
                ),
                max=StatisticsCollectorParameters(
                    statistics_type=StatisticsType.QUANTILE,
                    aggregator_type=AggregatorType.MEAN,
                    quantile_outlier_prob=0.0001
                ),
            )
        )
    )


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino
    INFO:nncf:3 ignored nodes was found by name in the NNCFGraph
    INFO:nncf:193 ignored nodes was found by types in the NNCFGraph
    INFO:nncf:24 ignored nodes was found by name in the NNCFGraph
    INFO:nncf:Not adding activation input quantizer for operation: 5 MVN_224
    INFO:nncf:Not adding activation input quantizer for operation: 7 /wav2vec2/feature_extractor/conv_layers.0/layer_norm/Mul
    8 /wav2vec2/feature_extractor/conv_layers.0/layer_norm/Add
    
    INFO:nncf:Not adding activation input quantizer for operation: 10 /wav2vec2/feature_extractor/conv_layers.1/conv/Conv
    INFO:nncf:Not adding activation input quantizer for operation: 12 /wav2vec2/feature_extractor/conv_layers.2/conv/Conv
    INFO:nncf:Not adding activation input quantizer for operation: 23 /wav2vec2/feature_projection/layer_norm/Div
    24 /wav2vec2/feature_projection/layer_norm/Mul
    25 /wav2vec2/feature_projection/layer_norm/Add_1
    
    INFO:nncf:Not adding activation input quantizer for operation: 28 /wav2vec2/encoder/Add
    INFO:nncf:Not adding activation input quantizer for operation: 30 /wav2vec2/encoder/layer_norm/Div
    32 /wav2vec2/encoder/layer_norm/Mul
    34 /wav2vec2/encoder/layer_norm/Add_1
    
    INFO:nncf:Not adding activation input quantizer for operation: 36 /wav2vec2/encoder/layers.0/Add
    INFO:nncf:Not adding activation input quantizer for operation: 42 /wav2vec2/encoder/layers.0/layer_norm/Div
    49 /wav2vec2/encoder/layers.0/layer_norm/Mul
    58 /wav2vec2/encoder/layers.0/layer_norm/Add_1
    
    INFO:nncf:Not adding activation input quantizer for operation: 66 /wav2vec2/encoder/layers.0/Add_1
    INFO:nncf:Not adding activation input quantizer for operation: 74 /wav2vec2/encoder/layers.0/final_layer_norm/Div
    79 /wav2vec2/encoder/layers.0/final_layer_norm/Mul
    82 /wav2vec2/encoder/layers.0/final_layer_norm/Add_1
    
    INFO:nncf:Not adding activation input quantizer for operation: 84 /wav2vec2/encoder/layers.1/Add
    INFO:nncf:Not adding activation input quantizer for operation: 90 /wav2vec2/encoder/layers.1/layer_norm/Div
    96 /wav2vec2/encoder/layers.1/layer_norm/Mul
    105 /wav2vec2/encoder/layers.1/layer_norm/Add_1
    
    INFO:nncf:Not adding activation input quantizer for operation: 113 /wav2vec2/encoder/layers.1/Add_1
    INFO:nncf:Not adding activation input quantizer for operation: 121 /wav2vec2/encoder/layers.1/final_layer_norm/Div
    126 /wav2vec2/encoder/layers.1/final_layer_norm/Mul
    129 /wav2vec2/encoder/layers.1/final_layer_norm/Add_1
    
    INFO:nncf:Not adding activation input quantizer for operation: 131 /wav2vec2/encoder/layers.2/Add
    INFO:nncf:Not adding activation input quantizer for operation: 137 /wav2vec2/encoder/layers.2/layer_norm/Div
    143 /wav2vec2/encoder/layers.2/layer_norm/Mul
    152 /wav2vec2/encoder/layers.2/layer_norm/Add_1
    
    INFO:nncf:Not adding activation input quantizer for operation: 160 /wav2vec2/encoder/layers.2/Add_1
    INFO:nncf:Not adding activation input quantizer for operation: 168 /wav2vec2/encoder/layers.2/final_layer_norm/Div
    173 /wav2vec2/encoder/layers.2/final_layer_norm/Mul
    176 /wav2vec2/encoder/layers.2/final_layer_norm/Add_1
    
    INFO:nncf:Not adding activation input quantizer for operation: 178 /wav2vec2/encoder/layers.3/Add
    INFO:nncf:Not adding activation input quantizer for operation: 184 /wav2vec2/encoder/layers.3/layer_norm/Div
    190 /wav2vec2/encoder/layers.3/layer_norm/Mul
    199 /wav2vec2/encoder/layers.3/layer_norm/Add_1
    
    INFO:nncf:Not adding activation input quantizer for operation: 207 /wav2vec2/encoder/layers.3/Add_1
    INFO:nncf:Not adding activation input quantizer for operation: 215 /wav2vec2/encoder/layers.3/final_layer_norm/Div
    220 /wav2vec2/encoder/layers.3/final_layer_norm/Mul
    223 /wav2vec2/encoder/layers.3/final_layer_norm/Add_1
    
    INFO:nncf:Not adding activation input quantizer for operation: 225 /wav2vec2/encoder/layers.4/Add
    INFO:nncf:Not adding activation input quantizer for operation: 231 /wav2vec2/encoder/layers.4/layer_norm/Div
    237 /wav2vec2/encoder/layers.4/layer_norm/Mul
    246 /wav2vec2/encoder/layers.4/layer_norm/Add_1
    
    INFO:nncf:Not adding activation input quantizer for operation: 254 /wav2vec2/encoder/layers.4/Add_1
    INFO:nncf:Not adding activation input quantizer for operation: 262 /wav2vec2/encoder/layers.4/final_layer_norm/Div
    267 /wav2vec2/encoder/layers.4/final_layer_norm/Mul
    270 /wav2vec2/encoder/layers.4/final_layer_norm/Add_1
    
    INFO:nncf:Not adding activation input quantizer for operation: 272 /wav2vec2/encoder/layers.5/Add
    INFO:nncf:Not adding activation input quantizer for operation: 278 /wav2vec2/encoder/layers.5/layer_norm/Div
    284 /wav2vec2/encoder/layers.5/layer_norm/Mul
    293 /wav2vec2/encoder/layers.5/layer_norm/Add_1
    
    INFO:nncf:Not adding activation input quantizer for operation: 301 /wav2vec2/encoder/layers.5/Add_1
    INFO:nncf:Not adding activation input quantizer for operation: 309 /wav2vec2/encoder/layers.5/final_layer_norm/Div
    314 /wav2vec2/encoder/layers.5/final_layer_norm/Mul
    317 /wav2vec2/encoder/layers.5/final_layer_norm/Add_1
    
    INFO:nncf:Not adding activation input quantizer for operation: 319 /wav2vec2/encoder/layers.6/Add
    INFO:nncf:Not adding activation input quantizer for operation: 325 /wav2vec2/encoder/layers.6/layer_norm/Div
    331 /wav2vec2/encoder/layers.6/layer_norm/Mul
    340 /wav2vec2/encoder/layers.6/layer_norm/Add_1
    
    INFO:nncf:Not adding activation input quantizer for operation: 348 /wav2vec2/encoder/layers.6/Add_1
    INFO:nncf:Not adding activation input quantizer for operation: 356 /wav2vec2/encoder/layers.6/final_layer_norm/Div
    361 /wav2vec2/encoder/layers.6/final_layer_norm/Mul
    364 /wav2vec2/encoder/layers.6/final_layer_norm/Add_1
    
    INFO:nncf:Not adding activation input quantizer for operation: 366 /wav2vec2/encoder/layers.7/Add
    INFO:nncf:Not adding activation input quantizer for operation: 372 /wav2vec2/encoder/layers.7/layer_norm/Div
    378 /wav2vec2/encoder/layers.7/layer_norm/Mul
    387 /wav2vec2/encoder/layers.7/layer_norm/Add_1
    
    INFO:nncf:Not adding activation input quantizer for operation: 412 /wav2vec2/encoder/layers.7/feed_forward/output_dense/MatMul
    418 /wav2vec2/encoder/layers.7/feed_forward/output_dense/Add
    
    INFO:nncf:Not adding activation input quantizer for operation: 395 /wav2vec2/encoder/layers.7/Add_1
    INFO:nncf:Not adding activation input quantizer for operation: 403 /wav2vec2/encoder/layers.7/final_layer_norm/Div
    408 /wav2vec2/encoder/layers.7/final_layer_norm/Mul
    411 /wav2vec2/encoder/layers.7/final_layer_norm/Add_1
    
    INFO:nncf:Not adding activation input quantizer for operation: 413 /wav2vec2/encoder/layers.8/Add
    INFO:nncf:Not adding activation input quantizer for operation: 419 /wav2vec2/encoder/layers.8/layer_norm/Div
    425 /wav2vec2/encoder/layers.8/layer_norm/Mul
    434 /wav2vec2/encoder/layers.8/layer_norm/Add_1
    
    INFO:nncf:Not adding activation input quantizer for operation: 442 /wav2vec2/encoder/layers.8/Add_1
    INFO:nncf:Not adding activation input quantizer for operation: 450 /wav2vec2/encoder/layers.8/final_layer_norm/Div
    455 /wav2vec2/encoder/layers.8/final_layer_norm/Mul
    458 /wav2vec2/encoder/layers.8/final_layer_norm/Add_1
    
    INFO:nncf:Not adding activation input quantizer for operation: 460 /wav2vec2/encoder/layers.9/Add
    INFO:nncf:Not adding activation input quantizer for operation: 466 /wav2vec2/encoder/layers.9/layer_norm/Div
    472 /wav2vec2/encoder/layers.9/layer_norm/Mul
    481 /wav2vec2/encoder/layers.9/layer_norm/Add_1
    
    INFO:nncf:Not adding activation input quantizer for operation: 489 /wav2vec2/encoder/layers.9/Add_1
    INFO:nncf:Not adding activation input quantizer for operation: 497 /wav2vec2/encoder/layers.9/final_layer_norm/Div
    502 /wav2vec2/encoder/layers.9/final_layer_norm/Mul
    505 /wav2vec2/encoder/layers.9/final_layer_norm/Add_1
    
    INFO:nncf:Not adding activation input quantizer for operation: 507 /wav2vec2/encoder/layers.10/Add
    INFO:nncf:Not adding activation input quantizer for operation: 513 /wav2vec2/encoder/layers.10/layer_norm/Div
    519 /wav2vec2/encoder/layers.10/layer_norm/Mul
    528 /wav2vec2/encoder/layers.10/layer_norm/Add_1
    
    INFO:nncf:Not adding activation input quantizer for operation: 536 /wav2vec2/encoder/layers.10/Add_1
    INFO:nncf:Not adding activation input quantizer for operation: 544 /wav2vec2/encoder/layers.10/final_layer_norm/Div
    549 /wav2vec2/encoder/layers.10/final_layer_norm/Mul
    552 /wav2vec2/encoder/layers.10/final_layer_norm/Add_1
    
    INFO:nncf:Not adding activation input quantizer for operation: 554 /wav2vec2/encoder/layers.11/Add
    INFO:nncf:Not adding activation input quantizer for operation: 560 /wav2vec2/encoder/layers.11/layer_norm/Div
    566 /wav2vec2/encoder/layers.11/layer_norm/Mul
    575 /wav2vec2/encoder/layers.11/layer_norm/Add_1
    
    INFO:nncf:Not adding activation input quantizer for operation: 583 /wav2vec2/encoder/layers.11/Add_1
    INFO:nncf:Not adding activation input quantizer for operation: 591 /wav2vec2/encoder/layers.11/final_layer_norm/Div
    596 /wav2vec2/encoder/layers.11/final_layer_norm/Mul
    599 /wav2vec2/encoder/layers.11/final_layer_norm/Add_1
    


.. parsed-literal::

    Statistics collection: 100%|██████████| 300/300 [02:51<00:00,  1.75it/s]
    Biases correction: 100%|██████████| 74/74 [00:25<00:00,  2.96it/s]


.. code:: ipython3

    MODEL_NAME = 'quantized_wav2vec2_base'
    quantized_model_path = Path(f"{MODEL_NAME}_openvino_model/{MODEL_NAME}_quantized.xml")
    ov.save_model(quantized_model, str(quantized_model_path))

Model Usage Example with Inference Pipeline
###############################################################################################################################

Both initial (``FP16``) and quantized (``INT8``) models are exactly the
same in use.

Start with taking one example from the dataset to show inference steps
for it.

Next, load the quantized model to the inference pipeline.

.. code:: ipython3

    audio = LibriSpeechDataLoader.read_flac(f'{DATA_DIR}/LibriSpeech/test-clean/121/127105/121-127105-0017.flac')
    
    ipd.Audio(audio, rate=16000)




.. raw:: html

    
    <audio  controls="controls" >
        <source src="data:audio/wav;base64,UklGRgRRAQBXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YeBQAQAGAP7//P/+//7//v/+/wAAAgACAAIAAAAAAP7//v/+//z//v/+/wAAAAACAAIAAgD+//z//v/+/wAA/v/8//r/+v8AAP7//P/6//7//v/+//7/AAAEAAIAAgAAAAAAAAAAAAAA/v8AAAAAAAACAAAAAAAAAAAA+v/+//7//v8AAAAAAAACAAIAAAD+/wAA/v/8//z//P8AAAAAAgACAPz//P/8//r/+v/4//z//v/+/wAAAAAAAPz/AAACAAAAAAACAAIAAAAAAAQABAAEAAQAAAD+//z//P8AAP7//v/+/wAAAAAAAPz/+v/4//j/9v/4//b/8f/z//P/9v/6/wAA/v/6//z/AAD+//7//v8AAAAAAAAAAAIA/v/+/wAAAgACAAAAAAAAAAAAAAAAAP7/AAD+//7//v/+//7/AAAAAAAAAAAAAP7//v/+//7//P/+//7/AAD8//7//v/6//j/+P/6//r/9v/z//P/9v/2//j/+v/+//r/8//2//b/9v/z//b/8//z//H/8//x/+//7//t/+v/7//t/+3/7f/t/+3/7f/t/+3/6f/r/+v/6//t/+//8f/2//r//v/+//z//v8CAAAAAgAEAAgACAAIAAgABgAEAAQABAAEAAIAAAAEAAIAAAACAAQABgACAAYABgACAP7/AAACAAIAAgD+/wAABAACAPz/AAAAAAAAAAAAAPz//P/+//7//v/+/wAA+v/6//j//P/6//r//P8AAAAA/v/8/wAAAAD+/wAAAAACAAQAAAD+//7/AAAAAAAAAAAAAAIAAgACAAIAAgAAAAAAAAAAAAAA/P/+/wIAAgACAAIAAAAAAAAAAgACAP7//P/8//r/+v/4//P/7//t/+v/7f/p/+n/7//x/+3/7f/t/+//7f/x/+//7f/r/+//8f/x/+//8f/2//b/9v/2//b/9v/z//b/+P/4//j//P/6//z/AAAAAP7/AAD+//7//v/+/wIABAD+//7//v8AAAQABgAGAAYACAAKAAYABgAEAAAAAAAAAAAAAAAAAAIAAgACAAIAAgAEAAIA/v/+/wAAAAACAAIAAgACAAIA/v/8/wAAAAAAAAAAAAACAAIABAAEAP7/AAACAAQABAAGAAgACgAKAAgACgAIAAoACAAGAAoABgAGAAYACgAKAAoABAACAAIAAgACAAIAAgACAAIABAD+//7//v8AAP7//v/+/wIAAAD+/wIAAAD+/wIAAgACAAAAAAAAAAIAAAAAAAIAAgACAAIAAgAAAAAABAAEAAgACAAKAA0ACgAIAAgABgAGAAYABgAKAAoACgAPABEAEwARABEAEQARAA8ADQANAA0ACgAKAAoADQAKAAoACgAKAAgABgAEAAAAAAACAAAA/P/8//7//P/+//7/AAAAAAIABAAEAAYACAAIAAoACgAKAAgABgAIAAoACgAKAAYABgAEAAIAAgACAAAA/v/+/wAAAAD+//7/AAAAAP7/AAAAAAAAAAACAAQABAAGAAYABgAEAAQAAAAAAP7//v8CAAIAAAAAAP7//v/8//z//v/+//7//v8AAAAAAgACAAIAAAAAAAAAAAAAAAAAAAACAAIAAgACAAAA/v/+//7//v/+//7/AAAAAAAAAgACAAIAAAAAAAAAAAAAAAAAAgACAAAAAAAAAP7//v8AAP7//v/+/wAAAAAAAAQAAgACAAAAAAAAAAAAAAAAAAIAAgACAAIAAgAAAP7//v/8//j/+P/4//j/+P/z//H/8f/x/+//7f/r/+v/6f/p/+n/5v/m/+n/5v/m/+b/6f/m/+n/6//t/+//8//z//j/+P/2//j/+P/4//b/+P/8//z//v/+/wIAAAD+//z//v8CAP7/AAAAAAAAAAAAAAIAAAACAAIAAAACAAIAAgACAAIAAAACAP7/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAP7//v/+//7//v/+/wAAAAAAAP7//v/8//z//P/8//z//P/8//z//P/8//z//P/6//r/+v/+//7//v8AAAAAAgAEAAIAAAAAAAAAAAAAAAAAAAACAAIAAAAAAP7//P/8//z//P/4//j/+P/4//r//v/8//r/+v/6//7/AAAAAAAAAgAAAAAAAgAAAAAA/v8CAP7/AAAAAAIAAgACAAIAAgACAPz//P/+//7/AAACAAQACAAIAAgACAAIAAIAAgAAAAAAAAAAAAIAAAACAAYAAAD8//z//P/+//7//v8AAAAAAgACAAIAAgAAAP7/AAAAAAAAAAACAAIAAgAEAAQAAAD+//7//v/+//7/AAACAAIAAgACAAQABAAGAAYAAgACAAIABAAGAAYAAgACAAQAAgD+//7//v8AAAAAAgACAAIAAgAAAPz//v/+//7//v/+/wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAgACAAIAAgD+//7/AAAAAP7/AAAAAAIAAgACAAIAAgAAAAAAAAAAAAAAAAACAAIAAgACAAIAAgAAAAAAAAAAAAAAAAACAAIAAgACAAQAAgACAAIAAgACAAIABAAGAAYABgAGAAYABAAEAAQABAACAAIABgAGAAIA/v/+//z//P/+//7//v/+//7/AAAAAAIAAgACAAAAAAAAAP7//v/+/wAAAAAAAAAAAAAAAP7//v/+//7//v/+/wIAAgACAAIAAgAAAAAAAAAAAP7//v8CAAIAAgAAAAAAAAD+//7//v/+/wAAAAACAAIAAgACAAIAAgAAAAAAAAAAAAAAAAACAAIAAAAAAAAA/v/+//7//v/+//7/AAAAAAAAAAAAAAAAAAAAAAAA/v/+//7/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD+/wAAAAAAAAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP7//v/+//7//v/+/wAAAAAAAAAAAAD+//7//v/+//7//v8AAAAAAAAAAAAAAAD+//7//v/+//7//v8AAAAAAAAAAAAA/v/+//7//v/+//7//v8AAAAAAAAAAAAA/v/+//7//v/+//7/AAAAAAAAAAAAAAAA/v/+//7//v/+//7/AAAAAAAAAAAAAPz//P/8//z//P/8//z//P/8//z/+v/6//r//P/8//z/+v/4//j//P/8//z//P/8//z//P/+/wAAAAAAAAIAAgAEAAQABAAAAAAAAAAAAAAAAAACAAIAAgAAAAAAAAD+//7//v8AAAAAAAAAAAAAAgACAAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/v/+//7/AAD+//7/AAAAAAAAAgACAAIAAAAAAAAAAAD+//7/AAAAAAAAAAAAAP7//v/+/wAAAAAAAAIAAAAAAAIAAgACAAAAAAAAAAAAAAAAAAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/v8CAAAAAgACAAAAAAAAAAAAAAAAAAAAAgACAAAAAAAAAAAAAAAAAPz/AAAAAAAAAAAAAAIAAAD+/wAA/v8AAAAAAAAAAP7/AAAAAAAAAAACAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/v/+/wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/v8AAAIAAAAAAAAAAgAAAP7//v/+/wAAAAAEAAAAAAAAAAAA/v8AAAAAAAACAAAAAgD+/wAAAAAAAAIA/v/8//7//v8AAP7//v8AAAIAAAAAAAIAAAAAAP7//v8AAAAAAAAAAAAAAAACAAAA/v/+//r/AAAAAAIAAgAAAAAA/P8AAAAAAAACAPz/AAD+/wAAAgAGAAIAAAAAAP7/AgAAAP7//v/8//7/AgACAAQA/v8AAAAA/v/+/wAA/v8EAP7/BAACAPz/+v8AAAIABgD+/wAABAACAAAABAD+/wAABAACAP7//P/+/wAA/P8CAAAAAgAAAAIA/v/8//7//v8CAPz//v8EAAIAAAAGAAQAAAAAAPz/AAD+//z/AgAGAAIAAAAEAP7//v/+//r//P8GAP7/AAAAAAQABAAAAAIA/v/8//7/AAD+/wIAAgACAAQAAgACAAAA/P8AAAAAAAACAP7/AgAGAAQA/P8EAAAA+v/8//7//P8CAAIA+v8AAP7//v/+/wAA/P/8//z/AAACAP7/7//6/wIAAAD+//7/AgD8//7//v8CAAYAAgD4/wAA/P/+/wYACgAAAP7/+P/8/wIABgAAAP7//v/+/wQAAAAAAPr/AgAGAP7//v8AAOb/IADZ/w8A6f8vAL//JwDz/+//EwAIAPr/EwACAPz/EwAEAAAABgATAA0ADwD8//7/AAAEAA0ABgAGAAQA+v/4/wQAAAAGAAAA8/8CAPP/7f/t/wQAAAACAPb/8//2/+b/5v8NAPP/9v8EAPj//P8XAP7/9v8AAAYA9v8IAA8A+v/g/+//7f8CAPr/+v/r/9f/0//Z/8//+P/2/wYA9v/V/+v/9v8AAAQA+v/k//7/CADv/wAA/v8CAPj/9v/x//b//P8RAA8ACgD6/+L/8//+/wQACAD8/wIA+P/i/+L/AAD4/woAFQAPAAYA4v/m//H/AgARABcAEQAIAAgABADv//z/CAD2/wgAAgD+/w8ALwAXAPj/6//6/wAAAAAKAAoABgAXAAoADQATABMAAAD4/xwAFQDm/8//5v/x//H/2f/R/8b/wv/X/9H/9v/v/8b/6//r/8z/5v/p//j//P+5/8T/vf+y/+D/6f/6//z//P/6/+n//P8RAA8ACgAnADoAKwAiABcAFwAvABMAOAAcAAQACAD6/xEAJwAVAPr/8//m/+//9v+7/6r/5v8iAEEACAD2/7f/yP/G/wAA/v8XADQA6/+5/+v/yv/z/woAEQANALn/sv/P/+L/EwDG//j/3v/R/yAA0f/e/6X/JwCS/8T/7f9BAOD/HgC7/6H/i//4/wQA0/84AAIAyP+j/4H/wv/P//P/+P/r/6H/kP+U/4v/3P/p/wAA3v8GANP//v/x/yQALQARAB4AHgARAEEAHABHAGUAUAAkALf/u/8NABEAcABWAGUAJAApABMABgAtAC8AaACVAHcAdwBoABoACgDr/yAAEQBlAIIAYQAnAAIA1f+7/+L/0//V/+L/BAAvACsAHAANACAAv/8gAA8ARQAgAEUA4P9SADQAFQArAML/v/8cAB4A6/9hADQAdP/g/9z/uf8+ABcADQDR/37/vf+7/xEAEQDt/7D/ZP+f/+b/JAAAAPz/AgAAAIP/T/9p/4H/3P/x/wYA7//z/wgAyP/+/0kABgDZ/6z/b//T/+T/gf/Z/7//qP/I/7v/0f+s/yIA1f/X/+D/0f8VAOn/BgAEAA0AzP/V/xEAMQA+AEEANgAiAPj/OgA8AEMALwAtAD4ANgAXACcAAgA2AB4AQwBoACsATgAnAFsAKwAnAFgAMQArAA8AQQAxAOT/vf82ALoAeQA6AFsASwA4AD4AKwB/AEEArQAnAFAAVgBUAEsAWwB3AJcAWwBFAAIANgBbAGoA7/8pAB4AFQAvAJkAUACl/4H//v9sAHcAiAB7AHkA+v9qAHUArv9a/8//KQAeAAgABADE/zP/qv5C/8b/wv9i/zv/Lv87/7n/4P/K/4v/T/9p/6P/+v/6/xwA7f/t/xoA9wB5APr/pf/v//H//P/v/0UACADx/4H/v/8NAJwApABUAAQAVgAPAF8AOgCMAGEAmQDp/67/3P9oAHkAVADNAO3/CgCvAAQBXwBv/7X/DQD+/8z/IABjAFsACgDt/7D/lP8PAI8AVgCh/+n/hgCmABwAZQCcAE4AzP8vAIYAkQCEAO3/jv/P/9QAYwA0AAIAmP96/xoAZQEZAUsAsv8W/7L/AAB9AN0AhgCgAOv/6f/M/9P/DQD8//P/fQBQAF8AzQDyAMj/rP+7/6AA8f9oAAoArQCzAL4ASQBlAGEASwCCAAYBiABLAFsA7/8aAKj/7/8O/+T/9QD1AKz/jv/g/wYA3P9oAJMARwCEADoAWAA8ALL/pf8kADwAAABc/4X7GPDF9xIJJxApAc39cABD8irsqgV4G48M/ftw/Bn51ewb/ssTChVOAEn6HPdZ9MP3PQgwDiYHmf2g+Mv4iPqbAmcEtwd9BQgCEPuM/QX/Gf7a/JcFeAoMBGX7hvm1+0L9YQHLBWsF5P7u+LX5dPu3/d7+7f2/8gDwNPai+632gftu+l7sj+UP9M0A7fuu/A78L/eu8A8AgwajCZECfghvBkgNNAr3C/YJLw/KCbYP5BtoH9MYeBf3HjQejiSFJlct8CRMDV3VkskD7AUWfRVBHEQKlMq2jpGwo/CfC2UVrSDn++O3DLMp5/8Rhh9FMpQs+ApE6v7yqgQTFZEdoiROGwsLOvgD6kXlN+uW8kL6bwZWAUDnrsp0yoXapOsK+v8CJPyJ7QblQeMG5kjvzQGbB/QIjgZBClYBS/sq/4INYxHcGOYdbRtTBy/7VgGBB5UPuBNjFP0EPPvO9Z367f9HEPMOvhIIEk8Z4hz9JvgsxSTwI+c2cSvH+anhjfIDA4j7HRgQKHT7YbPXq2DLTdwV6fMQWQ6q2oO6eMrY5xbx+gshHrsbegY+A/sCxgltB/UP1hM/IHcfeRJTBsP4lu1l5hP4cgWlCk//nvnO5RbeN9q46I3ym/48AnT7T+8Z7J3pS+ah7i8BVBGLCFUJewR5AsD6ywWfFvwZtRdgF0EQ6wRFAjkGxAl0AR8FKwL7AvD73vuZ+iT7dvqmAT8LphgJHv4fgiDdIe4jhiX+LLE14yeMAhDteOxA+Lz2jw4OGaABQc/qwUnFEMcLzJLv8vzp7HXi2uuc9KLqrfSBAQcKVQrxGVshsB+jDaUJaQJVBgkJ1BBFFEkQFQRP+Xn09fJM73zutPVB9RP0Je527/TqYepA66zsn+vN7eTvGvCj8/f3ivc99K35KP+rAAYCMAzqEMkQYxF1EDsLKAXiCUMLYgqlC6UMbQpwAk8ChACxAV8DQwtwEKQX6xvxHkof+STlJu0n7CWuKFcdOQeP9h71oPmT9hL+wAR/+TLfbNQq2MfWEdLV3iXk8+AY32ntFfgv+on/vgGQ/4X+6QS5CfAPJhUKF5wQNg+GDZsGfQCs/ib8avbQ+Cr/U//C+Uf4mPUf8SjseOva647tJ/CD7t7tZO8w78XlyeX67EPyu/Go/U4LMgyyCP4NVhJFDSMMpxCKEe0MdxFlFP8UihGtEcwNlArkCZgLtgy4D+UQXw6NDhkSsxafGmMi1CYGJoseqxHwA0T77PRm8EPyjPgK+FfvWuxp7SjpgOLB5dHpYeWc4dvlH+y57R7yA/o0APr/DgOoCH4M/gmLCGcGxQIU/xv/UACY/+n/HACZAPb+AgBT/vT8pPpP+Sv2w/Jx8dLyXfQL9Wz0OPSj86HwfO8376Hw9fBK9Lr5O/59AUQF8Qg/CzIN0g0vD0oOfg40DUwN8QvECocLyg1sDpENggwYDDAKlAjzCVELQw0/DlAR9BI8FMsUrxRAFaoUGRKnDVMJAwU1/8n58PbU9O/yefNs9t33vva29h72cfSH8Rbx/vCp8RPzUvWr9iL3cPdW+LP5ufsH/qz+hf47/Wv9Tfxh+6365/od+mP4e/g3+0L8Ffwq/DP8N/o4+P/5L/zA/PT7cPvp+gz5VPjd+CL5kfhU98v30PiZ+cf6FfxA/qX/HgGMA+kEvAWBBXIGYAgqCvML0gz8C/oJeAiHCZYL4g3JD3cQzA/HDhoPZhAxEr4TbBQzFPQTPhRCE/kQbg7RCq4GwgPpA0cEOAJ4/sD6Pve29OTzzPTk9CHzhfDR7vjube9G8Bbw3O7/7AfsVe3g7/nyxfQ49M7zHPSr9aj4Y/tC/QP8yfq6+nD7PvwS/bf9bf1C/Nz8jv21/iP//P3G/B/8VvxL/J/8nf23/Ef6Pvla+n38QP1T/v7+/P7p/jwAIAOFBYwGmQchCC4I8QgNCr8Kbwr+CSwKbQpeCl4LQQyCDOoLsAtdDB4NGA5mDqcOew/bD8MPog9hD24O9wsyCcQGTwRwAssA3v8j/kL7/fi29zb3Pvev9g32APTz8e3wyvDT8fXxWfFv8FPvKO/879nxEfQG9ZP10vXb9s34m/rw/IX+EP7n/fb9tf6l/4IALQGkAAX/2v0B/of+uf7+/rP+T/5y/uD+Of8b/xb/M/8z/7//4v8f/3/+dP5r/h//WACrARMC9AEmAroCgQPuA/YEzwQvBIUE8gTCBZkGUwYoBsQFFwYOB34I7wkjCkwKwQqsC54MZg0aDosOkQ22DCkMZAxODJQKSAjuBKsB6f8O/3r+rv3y+0/6m/jh9/v3+/dy95X2F/Zu9Vn1HvZj9rT1OvV59Ob0svVh9zH5pPl2+br55/rT/Jn+YQAiAfz/vf6q/hT/SP9e//b+2v0f/E/7yfuX/P/8//zw/Jn85fxa/R3+wP7k/sD+vf4z/4P/Wv9T/5D+SP3Y/In9z/5C/3z/FwAnACkA0gA+AloDPASfBN4E6wQqBagFTwbeBjUHIQdlB6oHNwjtCD8Jdgn2CX4K6wqnC/oLowsaC8EKHwrRCX4JNwngB74FHQOGAI7+Tf36/Fr8Z/v9+X34nPev90X4y/iV+Oz3TvcT91v3PviM+Oz3PPdf9wz4CPmx+vL7Pvws/Ar9H/6J/woBDgLjAVoBogBsAJ4ASwDC/wX/Dv5L/aH8n/zY/On8nfyb/Fz8Sfx//Fr9Of63/rP+Xv5g/i7+4P2x/aj9of0O/dH8Bf2f/Ev8Wvxg/c/+BABJAUsCoAK1AiIDSQRnBfoFbwZYBvQFmQXRBVwGtQasBl4HtQeUB7sH4gcbCCMIRAiOCIkIfgj+B6YHXAf2BmcGEAamBagEHQMIASj/xv3N/Dz88vuD+2H6T/nn+K34Lfnl+TX6PvrF+bP5qPnc+cD6a/uM+0L7Evuo+1j8rv38/sr/NgCMANQAxQDdAD4BPgFJAFr/Qv/K/rP9Mf3P/AX8Uftp+/L7+Pu++3L7Kvsm+x/7gfsk/En89vsZ/FH8XvxP/KT8wvyu/Mn8EP1t/bf9G/7C/m//FQC4AKIBTQKmAowDbgQmBYMF6wV2BrkGpgbrBnoH9AeSB4EHPQfEBtwG4AYdB0IHBweDBusFoQUoBd4EqgRsA8UCSQIiAjUCPAJLAtoBVgEiAYQAqv/0/sj9IvxE+4776/u3+y/7f/oq+gr6uvrt+5T8Ufz2++X7Cvyu/Af+5P7I/hT+wv3E/u3/YwFPAnICGwKzAcMB/QHNAS8BKwCW/+/+jP7E/ib+tfy8+0D7jPot+rf6lfry+XD5Ffmo+Tr6Z/rP+o761Pn5+XT65/pn+x/8xPxL/dj9s/67/zEApgDJAU0CRwNYA0kEigX6BfYFMwamBv8GkAeMB8YHFgiwB8AHZAhECLcHLge7BhQG4AXABX8FnQRyA38C+QGEAQgBUgDe/y7/0/2B/VH94P2F/on+9P5C/xj/EP9K/9f/5v98/zv+Kv1w/Mb7hfwu/dj9Cv3//IX9/P21/q7/iABDAC0A9v+vAB4BQwGiASACZwEeARcBGwIbA7ECsQLFAn8B3QDdAOMAvACB/1X+Yv3C/Of7H/zG+276QPmX+P/4FfkT+TP5mfl9+cf4CvkI+kL61vo++xn7gftP/JT8m/2O/rP+KP9YAV8CgQIXA2cDyQNhBA4G0QZ0B3gH+gf6B2cH2AbcBywIhQgoCDcHJAYvBSQFaQU5BTkFLAWbBDEE4wKRAmcBPgBg/w7/eP4H/o79Hf1A/e38Vf05/lz+Qv67/ZT9Hf1N/aj8eP3e/CT8qPzr+337afso/Nb8+vwq/iX/qv9i/9P+Pf9E/2v/qwBAAcz/z/+pAGEBYQJ3ArwD+QKvAewCdALlAncB5wFDAQoA6/7e/hj/Dv7R/cD9+vxn/Gf8LPxW/Ar6f/nN+nn6oPp0+aD4dvlf+mD8PPxl/On8n/so+x3+V/+KAQ4CMwOrAboCjgPyBNME3ARhBJcEKAYUBw4GEAY1B3QFRAZiBpIHKgeQBkIHtQbYBlMFaQQXBFoF/wPwA9wCBgPe/Mv6hABHBMkFpf8I+8P0VuoH/eMFBA8HDPcMuAEI+8v6OPi3+ob6Tf8CAGEEAQO+Ab79cPzG/PD7f/0D/cD4FfpD9vv6l/2+A6QE5QORAmn/CgDg/R4Cpf9sAbECYwOiA4oCCAE3+wP9g/5y/n8Ccv1n+hX45/cR98339fer+ED5YfZF93D2dfZl+Rv82vz9+V78Rv6h/l8ALv+q/Bv7efpP/iAEqANPBpsHGwSfCw8MKQ2bB04NAg31Ea0Nbg9ZDvEOURVPFawZphgrH04bvR0HHGkWPQev+vfx8e8I6Cztuvll/rb0rfeD+S31Te6b6drpO+BM37XqJfTU8xn8Tf/R/R370fvL/WX9B/3E/Vr+fQIhBucElAYoBzEDB/1l/az7jvjM9C/2X/fD93L7LQDR/5v4jfUJ9Ivz8/WM+6T62PlC+sj+Fv52/cr+pgCZ+tj4/fln+Tz3XP2+/DP+XwJ7BM0GugOiBFgE1gO3BGwDxgfVBgoDlwSqBVgGWQ4rD3ASDxFuD7MRow2iD0MRexPJFM4bNxkQGfMhhRzNFrERlP657Qjlq+fO4tDk1/T58xvuMeXe7dfffdjB3ljnkt3K4Z70MfxnBOAGLw+fCx8HQgkfCbUHZg3uEOUV3BMoGWEPyAhpA04Alfi69134WPmh8WDyhved7yjrWObd5EPkjOcx7PTs+Oz18Wjwsu+S8ZPzevJS9CL3Hf52/3L+jAH2/94FyAdTBioF+wQsBtMJowyoCGIJfAcdBgsKZAziC0EPvBXFEjYRkhhOIUwdVx2TIYMYjx90KFEsWiV2J2kmxxBSAt/zjuq85W3havPS+Jzzzehf50/bmdZF1PjYdtuw3N3mp/MPADH9LQE3/+MEB/8VA4YNphRWFjUXrBsUG5kVWBKqCQH/uPg4+ev+Yv0zAzMBsfxM8G3rk+KH3ondX+Qh7GnvvPax+A/1yPI+6YHqaezC68P4wwEIEXMJMRSyDk8J1gK8A8cA8AWSCyMMug5FDVcKAwVRBCgGrwNTCi8QCQ39EtoY4SAuHcoZdx3+GlMaNysHKwwqmi5aJ6UgBwmn9a3lleab3bDvKPyd+jft/erO3pDHZ8oW0ArUCtkO6qPxovtBAG3/0vlL+jP4dwDJD58cwRvQH1cehxh5E7YODwvWAy8BRP46BHIGIgCQ/s7y0eoG5yDjIOKg4+zlT+mu73btI/Dr7hfqb+w+61LnPfAK+4gDmwHSDiwL5Ae9C5YN6wWEDccR+xRIFzcV3RHnBw8Bm/9W+tX9IhKgAKoaihDYEAIRPBbKB2UTrxUDGlAheS84MH0myTHlFk0F3/Un5U7ie+Vi7Xv6oPfC7jHnfN9lyq/IUc4H0OXYEOkX+v/9cgIo/Sr6EfQ4+Zv/jw6rErIbuRuAH7cZqRCJCTMB/v8ZAYQBpAVgCTv/YPx58yrsCuZO5HPjDupC6n7x3O4T9MDtRu3/6nbuFO9E74zqtfly/vb/YgmWC+8IpACUBjILBQcbCbAO6g6+EQ0PuA3ZCaj/yfUq/ZL+jASUClwXcBINEScQzAvZCocLuRUqGyYkUB6qJ5EiNh10FuAJcvz67krx1fLJ9RL7UvXK8qDoZuLA2TPZ0ds13uLgKefM8NL0Pvgv+4P7bvRL+eMAIQefCi0RQhdyFHcRCQ4uCJAFwATdAKgF2AYVAssBRP4z+ffxrvK98Jvuz+vR7s70K/eL9cz0kfLX7wnyZPQx+gr5Uvn4/Bv9of5aAUQF3AYbAggExgZ4Bw8MTAwJDcAFlQQ6BG8GxQJrBqwESwevAgEGggzyA7sIqQxRCTMT+xJoDB8Y9gUoFhAXdhWFHDUWsRKbEyYHKwLQ9Yz3tPJ/+RECivr5+C/2Ru3S5lTmW+UK5lTnV+3q9T3zM/rn/Qj35vQx+Ef2YP/NA8wInAwNDogQwxLgCzUJZQBc/QgEMwSFCkIJkv+SBFf+3vrA+YD1JfBO8930NvcX+aD3zfeG+KD0bPSN9V31rflP+YH9/fzFAA8Aev5v/dH8bf/WAocG9QpaCHgLYgfxC00DdgYMAn8H0gFtBe4OAwiqBZ4OTwOPAB4S7PgfFysBFg0TE1wFphM4EdULBRf1DUoOFAqIEFcOPgHRClPw1Plf+YL2twV//Aj3M/4R8mbwH/CB7ZTwh/HI8n337f/d8uD/lfbL9vD2EP78/nQDev/hBH4LZwX8CLsIqwEpAd8BMP5lBlz7ZwJNAlP/Bf8F/4X8R/lW9Z73uPkh/c/6TgAN9i37BPfK9M/9DfQZ+n8B2PcmA635BAEk+oP/pPYXAMD7wvykBdYCQAQuCskArAmZAjD/vwij/iwGawkDB0sRt/5CGtLz0xjf+L8M2QgbBPMN3QsvAfwaEfaYHEv2ygyOB18Bmgx8CzwB9hRp+60T4fqb/YH/6PYx/RT+//kKBE73eAbh99P7bvXm8s7wmvaa8ZL9IvfJAIb4gf0w/k/5kP9a+2gAWPzVB2ED6wcyCe4EkwJc/8z/i/8i++MCWwC1/RUEvPsw/tL2y/uP9wfyXv9J9VL3ZfrA/SXwzf1F92bvMf2f8d/3PgL87n8F5faO/VP+AQKI9nYInPGFCDL0RgfW+jMHfQExBMEKcPoDCg4FMP+uCYQByguB/R8YcubuNrfLxDmj3yEZsP6wB9gEBQfA/KEZ8917MkLY/SE48q8QTPQvEqL4CBU65qcwWc+xM1TlDwAhBt7rQv+m/p73kQAd/owBZfekBJX2avfA/vHvEPww/1/3+wSeATn9XAMZA8P4vQgx/f8B9gh/AP4J2AQmAs0Fm/1p/a8BPPtn+sgHAvM3BuzzTwJb8vn3pfXq84/2YPw09S/8zPUs/Wb1CvgI98P1C/Vx/yPxZwOr8oMFZO/MDn/qSRP/6YwSZvA7CM0AlQHP+zIMc/KcDskACwhPA48Pe/juEJ38ww/2/r8NIQmM/nUMOwar9s4cmepjEiIDQ/fcCGAL5uQBKTLhWBXF9AEJ/+vUEiL2zQDvCr/1TwZLASAACAMhCEj9SAfR/Pj/j/KQB+jwYgjC+xb+bweT+dwC4P649iX/PPkx994G4fSuB3cA3wH0/rIIF/dUBBEAlPz0/YQN2/FJEtr89wEnAOz57fy89uv+a/uk+0MBxfJiBxfsiAKu7KID9eXiDd3jQgjR7AoFBe+HDAHq6QMl80b84/zz/3/36g4167MVavZUA2cELwL2+dYU4/WUCRoKpP1RB78J5P8jDVEEQghiBhgNwfOsHKHwBRek+OESJvp4CUgKWPmKEIYA+/sXEB73hQdJ+DkLsfrcBucGdfQREpHh7xvX4bcaQ+WBFtvz/gtE+hYOmvVqDsTwNQng7cIHMu7iDSD1MAcP+JUDMQDN+QoClQGZ7U8JgPAtAvb+HQcM+84PDPtR+3QGIPaVAssCfP7VB1P+BADZCHD5Kw7t7NkPSu0V+3D8SwQJ7qgVGvB/+kkAK/Mj7VwXud0ME53qXwN27wUL1e/iC0v6k/T6BWD+cfAWDTP4zQF2BH8GAAB/+igJa/wB/QcXJ/RVCvgEqxI09GAdLft6/YMEcw+g9IIg9/HzIMTptRWTAE37QggvFJDrIBMqCqbtjRCJChDuZA0w/uD9RwC8BYf/5fgPEPHzOBTg7sEdwOdrGW715AkH8kAFXfQY/zP8m/7h+CwG5PT8CKr7vPlA+lL5NPIbByv11whU+rsGi/ISCxf31gMJ/wgBqAJk/8ICjAdhABcGqAJYAcABFvECC3D1fP2oBaT6NPKsBTTz2u10Fgfc0wmW8+H2F/Yb+SkBC/IDBNT1hvVaAoL0dwEwDSHv+Q2G9ukDrfdYE2LwiwiWCCzvfwT2FsDpNxrRCUL5XAR5DXbtiRg/9C8QKAJAASYUbfxAAo4Z7evICokHmQL79CAjIPTqEtr73Baz6fcQWQko79cccACQ/KgGiw8L4uYfCPyIBEIGLwFt8aj+2e/X8+v9ZPL0/a8DFP9198T6GPGi+CX0m/nl+xIHWe+NDgcGTvNbDLH2MwZNBF4ErASDB679fQW1BdUH1vweAjH5h/wA9p747/3o84X/DfZ2B0DqZ/u++jns7vgf/S/5cPZRBY/wGQdz9db2OgPC+Rn+Kvxn/4wDDPjuAgMEvgIH8eYJUAAb/ngLKAkbA3cBFA298AYUogDL+LwTYP+t+ngWv//U+dEVZvVYA1kcqeWoJ3T+3/JxGpEB4fQ5HOf3BBMlCnMMwPskJlfwAhBeFUIE5fsZAz/z+vxw9qX1ogDL+U/vCvyH/H7v0PHI8sjsffa89dfyHf8N8r75ogQb/2AI7gFaB0cDigaxBi4N3PxBCvUMvALdDFwClQOU/KT8tPb/BMP2WfZSApf6R/c6+EH1sfqa9Tnvg/1F9TvwqwC3/1z7uAFd9vL4m/rz9If/kvz4/uH6DAIx+M39jAA+ANsATQLWBd7+JAQTAF4H3wAO/DsGbgEkAwsJ8gYDCPoGv//PB8gMAQYPC5UWpwzyErETNRUdFVwWbhF6GFYVUwwzFV0RRge1+Ej+C+686HHunPGQ7Xzw/vEu8gvv5OLl6jHq5uRv7H7y1Pi++HsAfgjcBQIBpAPjBV4EQAheBW8MoA9VC1QQThFrCQ4GiAIbAxH5UvcR9Yr8jfUT+E36Je/g77300/CU8h/xx/U4+bf7G/+tAcz/YP5LBIwCrv6+9oz6UfuB+3D89QAq/Wv8TQFhBMkBUAF7Ae4C//1YBM0BYQLaA/IGBgJHA+gAXgW5CuwNKQ/GGTUVChUQFn4aPRnbIBUi0CLJIqUZYBgQFOURw/dO9Hfye+d22uzjBORg3V/ipfO57z3hHuKX6bblleVR74L0+/llAcEOsRagEMEMlxFhEUYJCwqhDHEMcwwaDtcMAwX7+3/7s/rt79jrs+r75gbpGO+Y8t/0IfJU+Kn20vZu95f3IvmU/lYBEQJNBjsHpAS+BtUGh/1F+VX9zfxE+oz9SP/p/cL8vgMcAbf6nvd/+276wvpI/eEAjgXeBa4HvQ62DKUMQQ72FKIWhRc/HOUjoSsuKQUtXCuvIlMbXhcFHCIS2veL8bbibtjgz6HPM9ewz8nVtuL/6yDiveHn7eLxHPJU+aAAHwVxDUgY4yCPHKwavheBFoYUvQrnAaQFOALp/Hz9qPkL9IDvNe4M6zjkiubW5frrQfFm9Nj2I/8eADUDyAi1CSgDVANGCDsHAQRICQUHBQe1A7wDWALw++X4r/UE9cTw6/Au7oXxq/Pd9nL6RP6H/wECMwaBB6EJ8QsPEs0VsRWkFkIaRhyOGZgZUx3YIpclSiqbKd8lsB+yC7v+kv4I9lnhceLk4RfaM9U32lHbzOBE6izsRfMi+pv5rfNp/4wGZQDWBPURyRI+FC8RrBeVFAQO7QiQBgv/mfg78pjwxOz55ZXkGe0h7hLtlOsG8+Ty+u8E9un+WARvCAAN8A/SDhYNBA2UCKr/YP8IAMr/+Pra+zf7z/oS/Gf44/Wv9cPyNvb7+RL7z/nN+2oA/QNnA8QHYgiaCg4I4ApxC3MM2w3HEJ4OdQxxDIIOCQ2pDrgRmhz+G7caZhyVIVsfxBh+GSsQNg3ABQv1Wd0N09PKVsPJx6HQLNnW5Kf1NQKRAocFrQCBAlYC0QMtBMsFJw7NFHgXaxf9FVYS5AjN/b73XOyr41nhtdn1zRHX++d27BbvmwQNCvsDMQV/BWX6x/cMAvsC6wY0DtgRMxN/FbAOawRfAVH8x/NV8cHzofKP9L77KQDa/gj5R/kv+5f7wPry+w7/FQP8BrsKAAywDicP1A6GD4IM7wd0BDwB1AFNA1oGrArLESwc0iF7JMkhBCJFI70fQBacDuoNtA4QB6HxWeSs2l/SEssAzrzSXtm86ST72AOjB20IWwueEj8QbAOF/2MEWgRT/9ABgQWDBIMF+AV/A2X5Eu4o7vTsX+J027vdcOjn7ADxsfitAAUHIgVhBDsIUwVpAuUAoAG3AyADGQQ+BJsCrQHlANX++/VF85X0svUE9Ef1OPiM/c0AiAGDAzMFgwXjAzUG5QNWAG4CZwRaCNwJCgZJBfwHzwiVBO4CVgUzAx8F7Qp7DkATTxj+HZMfACIiI1sePRzLF80TgAzPCX32I+Iz2QHZEMtayinVz9ut5rz4u/8b/wIL9BSVEw0QUBAhCL8JjQomBjf/SQIWB/8ENgGX/Lb2WfOa8cDsLef15FDnPOu58Bjz4fN/+u4D1gTEAwwFjgftCYUJ9AMbAyYHEAhUBKQDdAJ4/lr9vPug9U7xU/CL8IXww/GR8/n3MP++AqsBNQMUBhYJAwgzBukEuwhMCo0KdgZvBoUGcQgoCAMFvADUAm4ExgXrBnwLjRB2GHMdQx3qHOwgXyAAHioZBhMVDwcNa/u44+naJNZCzNPIldZ83zfrJvqKBugM+Q5dEGMREROlCzcDVAIvAzv/H/zE+3j//wJFAp/9m/0K+kv2cfAB7jjnTuYG6RLtI+9H9Vr7NQOQCTYMEgqyC7AOVQnDAN8BqgT3AAP+hAARAWn9pvwR+S35XfbU9OHz2vek9r75FwB/BQoEZQd6C9IMpQv0BwwFaQVcBXAD+QLAAgwEtQYfB7UEdAPRBMQF6wXNBQkIOA53EcQVyhopHR4g2iVAJVYi7x7GF6AQLQ+l/7rlYNv03OjQCsVGzO3Zx+d68pkBpQtLE/gYXhh/FVcOUwd5AZkA+vyV9dXyovtdAIP7NfrP/mMB+/tx8xjuWOxC6DDjKeRE6yjuIfKk+hMEYAZVCOYLjw6YCoYD7/+tAMsAWPlJ97v84wDJ+mX4xPpN++741PcK99r4WvvY/HT/ZQQMBaEG+AoyDOAJ1QcdBlwDYQNFAG391/6xAWwByQOkBpQI7wk0CvwJrgpOC9MLIg/2FKgYhxwrIe4iSSL7IE4d+hY8Ew8Q6f+851nfa9zKzZ7DH88y33LrnvbaA0ENURrOG4wVFRR7EcgIv//6/Jz2YPGp8nL61vt2+pX6IgF5AoX8avQJ8evuY+t55LjjqOjG7LnwuPcTASQF0whZCxwMKAh/Afz+dwD4/A/1uvSK+ZP4OvQp9UX4qPr4+5365flT/d79dvxi/RUBdv/qAIUE5wUSBcAFMwekBocFswG8AVYB4QLuAmkDJgSfB/gH6QXaBgkLAA2CEPsUwBjdHM4g3yENIikjwR1nGUsXuhTxCZH2bObG4OLcTM0UyiTZWujy7J73dAf1ERkYBxjRGCYW3RBvBiIAg/qK9zbzEfOC93j7U/yz/vIEKgO7/X359/UJ8Bnq5uMA5HLo8Ouo7F/1RP/ABswJ9wx+DacMpwmUBc0BS/wX92z1oPbK81nzUvXA+u375/tE+zP8gf1h+3n5jvqX/Ur+5P9aAmUErwT9AksD/wPpBKsCrwItA3kDLQMOA4gEwgYUCAsJjQ11EOoRSxPEFRkYfhsHHu8eOiC6IHMcMRjaFToSNQOn8H3mPeGG0kvHa83W2ADhlehj98AEQxEbFY4WhRiVFcUMqAUTAHf4C/TK8yL1VveB+fj7qwExA5T/u/x0++j16e7C6e7lReby6FbrQu8n9lAAQAfTCswLEQ7FDIAJ3AXr/jH4UPbl9tXx8fDM86b5WvtN+/363P0CAEj+Tf5c/7//I/+9/5wA0gLNA9oCaQRAB9UHrgdeCH8HLgYjB3gG2ATYBc0HIweHBrUJow0yD7wPVhIDFx8bhRyHHQggdyILHiQXfBg9GRII8fCS6gznw9TLxwvOHdnZ3xfoxfRCBY8SnRVpFdoZwBfiDHYEZ/+K+JLwN+4q8PnzgvOr9UT9KgTJAh3+Hf4h/af23uyT6Uvq/ejP54PqnPLg+7MAvAPcCXoMEAdgBlsKqARe+xD6vvsx+MX01fKg9pv8dvse9hD6fwEpAAX98f8ZA00CVAARAssETQV7At8CfwUmBcADJAXLBvgFNQXYBIwGNQgLCM0EOQfTCU4LyAzJEPISqBZeGzAdyh6VIPcgThsxGPYWuhM+Ab3vKun+49nR+Mr30h/ddePT6+z4OQgBFUcVWBdzG8YY0A+yCh0DsfxL+QL21PSt95H3kfb0+97+zfsF+9L5q/fb8lHtKOkM7CzrJOpm7xP3kPxwA+QIKQuPDewNrAoqCrcEV/6k+tr6LfQn8T3x4vL39Tb3jfbf9wf+g/2o/I7/cgKkAWcCeQPuA+MF4AXcAqADSwVTBeUDswbTBZIGqgbgBzAIBwnRBrcG5AkrDUwOuBDlEogVMBmoGjId6B/HIAkcshnpFxAUQgaj8wbpQeP/1ozJ+s8d2mTiJOir9psE6BBCF3gYtBuOGS0UcwoSBbP+JPmY8jbzEfXx8lDyhvpHAIX+WPz4/LH7R/j+8CHsTe3v7UTr9Ovk8mP3t/yeAUsG0wkEDAsMTgtBCs0EggBI/mH6c/Uh8xbzSvTQ9Nfzc/VC+y7+Fv5AAVoFRgUbBKYEmwVCBRcFAwSQBBcFOQbcBnoINQnzCokLHwoyCuAK8QmwCLIK3QvbDIYOwxA1FDUXJhlxGt0dWx7KG7sYRhgoFioI+fXI7AXqfNsSz8PSi96n4nDl7+70/XEK2QwYDqQU6RdwEA0L9giZA6T8Dvr19xH28fS89bz4kPwM+4z4JPl5+WH0/+w16kDs8usp5q/mAO/k9BP1CvpYAksGmQfZCfwLAAqSBj4EdwL8/Y/3APWl9XP1pfLk8ob3Lv1l/bP81/9jBFEF/wJCA+EEMwaKA4gCAwQfBfsD0wRvBm0HzAhiCdcJ4gknCvQHPQk2CykLsgoGDtARlRSiFk8ZgBsLHkUfER5ZGzsZihdkC+H51+376SreFdMC0EfWu9ur4SDl9u/p/4cKVwt7EEQX0xaGEVcMMAhSA6H/cPnq9mH0avQW9KL4mfi++Nr3JPqO+ZP1h/Dp77/wKu0t6R3rTvCc85j1CPlk/9wEGwiYCIAKpwtOC/gHVgUrAd78Mfly+Rf4AvZ39yj7Zfzn+5f94v6GAXsCCAKrAKQE/wVtBdED4wWzBnYGvgVEBokHvQeOBzUHHQdrB1MIJQnmChENZg/hEpAXhxleGz8ewSECIAAdehqwGXYUUwVu9qzt1ue52nfS8dLH2Pbbp+DS5mTxz/tyAxYJsA5AEn0ROBBUDvwJ/wUqAzMBAf6++Qj41Pe1+Pn1UvSI9af2v/R18dXvGvDT7wvuu+yO7kHyYfW6+Wf90/6rAfgFqAgdCGIH4ghGCaEIswJt/7P+LP+b+0f67voi/LP7iPrn+R36ffv9+qr7QP3r/5kC4QRyBW0FaQb2B/wIyAgfCFUHFgj8B74G8AXrBVwHQgi1CTILoA4oEwwWOxhGG5EeTB+rHW8bOxr4F5MQ6QN5+WTzZerk4NHcA99Q4KvhjeW+6pbxpvfy/XADlgh1CuAKcQtoC0wJJgaqAzoCfP/9+4P5CvpN+Nb1OvQn9UH08fKP8cjxofHB8EjvVe988VvzEfWI9kn4Ffrg+xT+0gCzAm4DJATEBu8G2gQQBOMEJgUDAz4CMQFDAcT/hf0k+/36JvsS+1j7qvsz/Hz98f+pAeEC7gNyBRQHWgfNBgMHbwdAB5AGFgcJB6MIEAqHC8wLbA5dEWMT4RS8Fs8YVxmUGBkW0RQbFBERRggh/wj5TvQD7lHqYuy37mDuU+6p8S31nvZy+MD6XP3E/kD/Yv6b/gP/kP6z/ET9af6z/kj+G/8NAC7/V/7N/aT9Zfwx++z5zfg69xH2qfXu9Zj1xfSn9UH3E/hO9zr4+/k6+4P7Vvwo/kb/GP9n/lf/SwAeABL/ev+O/5/+y/xj/AX9wP2d/az9h/5I/8r/6ACMAhcE3ARnBXQFNQbIBuMFZwSgA50DOgO+A8kEIQYhB+AIWwpODYsPphGKE8AWqBhTGD4X2hVpFbUVtRONDZ0HOQX5Ac/9fP2RAEUA/v78//sBngAs/xT+wPzY+gz4pfRx8T/xeu9p7QPsAe637kLvb/AP8370avWn9iD4EfkX+br55fk3+l/5r/hW+HL4uvc69uj1dfY09mH0yvPB9DH2nvYr93T4Zfp/+5f8//01/1P/HgBwAaYBYQDP/w0AkQAXAYoBVgJjAy8FNwZ0B28IJwqdCtkK6wqlC5wLzApkCnMJNwiuBpIGfQbLBs8GAQfpB6EJjQuYDdIQnhPYFOUUIBS+EvIRRRGPDRsIIAOd/y37Yfkd+hD8NfxI/ef+EwAb/yj+2Pze+7r5Bvca89XwV+8F7hbtGO5c7yrwavIN9db24feD+tb76fw3/eD9/fxI/YH8jPsM+mP6AfrN+Vr5PPkb+Wf5w/gP+N/46fn2+en5dPu8+9P7ZfxV/rn+O/90/xMAz/92/+v+z/7k/jn/af/6/xUBKwJLA1EERAa3B5AI1Qg1CWYJBQkoCEQH4gZyBusFigX2BSMHfAgbCe8JBwzODtsQNRMBFgoXXhYzFXkUrxOPEt8O3ggdA9z//ftl+kT7m/0z/Qz+Vf/g/4n+xv3N/OX6Dvlh9mzzj/Cq73ruoe2S7SPvlvCh8gv1BveR+JX6Kvys/Af9af3G/Of7bvtp+vn44fhY+ef4OPj39633F/do9133ivcp+Gz50vmM+qj7h/yF/Gn9cv7T/rD+fv9YAEkAAACo/4v/4v+vAOwAQwF5ApcDLwQxBZkGSwc9B7kHOwhKCHQHtQY3BjcGnwVYBVgFZwaxBloHgQjiCj8NJw8TEZcT8hSKFCQTuhKrEuoSBhEuDJAFPAHy/U/7N/oM/IX9kP1//t7/yP+q/RT9wPuG+rb3aPXX8bDwAPBp74fuqu8p8cPyzPRY9zX5dvpw/E3+RP/4/mD+xP2H/eP8ovv0+Yz5nvmg+bX44/fj9kv2K/b99qD3WPig+WP6mfqZ+i/8H/0b/qz+FQBjAOEA8AAZAWEATgCkAM0A9QAkATwC7gJpBGMFYAezB2kIgwg5CU8I5wejBx0HvAUmBaYExAMtAy0EDgX6BU8HZgkWC+oMBg8vEf0S2BO8E8USPhLDEcERYQ80CigDqv6k+4b6zfpp/Q7+RP4o/7gAWv8U/ir9Nfzs+F/2hvOL8ETvzu/T71PuMO938Wz0Iva4+Cb6H/w7/Zv+tf5N/gf9Pvw6+5755fdY9yb4x/jH+OH3jPcp97r2kfa89xP5VPkt+Wz5uPlj+dr5l/tv/Uj+Tf9YALMAQwCEAAIBQAH1AG4BswE+AtIC9gMqBdgGLAhGCe0JDQtgC8YK8Qk1CUYItwahBcAEMQT9A4MEcAXPBlcIVQrODDYP3REmFNgVzxUzFdoTTRK8EfMQ6AstBDn/zfy8+a/3vvqF/Sr+yP5fATwBEv+b/Yz8efqi9xH0Ku+H7HjsTew56znsi+9i8rT0WPfU+eX6WPw7/U/9M/wi+4r5cPeE9qf1YfSr80z1iPZo9sz1Mfbb9Zr1R/bH9qD2NPfD+H/5kfnU+bf6vPtc/Sb+8f41/y8AUgCKABoAYQD5AIYC3AP/BNMF0QbrBxQJpQkWCp0KzgoQCvwIuwedBh8GtwWbBYEFegXjBfwGjAfiB9cIeArIC6sNERATEucT4RX2FgwWihT5EtgSQBL3DlEHfQEW/5n8ffr7+2T/qv/8/90AfQGF/wf+s/sm+XD2TPM57ybtAe6u7hvuo+4A8VDzWfbh+Df6Y/sU/XT+xP3l/Mv7Bvqb+N34GfnQ9+j2Zfg6+v/6Z/rn+dr5rflL+eP4VPjl93L4DPqM+0v8Qv14/h3+XP38/Tv/uf7P/tP+RP/V/n7/swD/AX8DpASxBPgEjgaZB4cHZQcQCGAH7wYOBjMG6wW+Bf8EswR2BJ8EMQbvB3oIFgl8CiULJwtxDDoO5g8eEloU9hS4E0kT9BLFE2cUABHCCCgCWv+b/BD7e/vC/bD+jv4O/3UA6/4z/WX7bvm+9tvzzPAO7YPsZO7m7nrt7e6H8Q/0gPb3+Cb6lfok/Hr89PvU+uH50Pdj9nP2ivag9v32k/f0+Jn6G/qV+bj5Pvph+kf5PvgG+Dr4IPrj+rf7EP7E/5/+nf0h/lP+H/0h/ar+jv8o//z/hgEKAvYD9gXjBTEGwgcJCU8I+gi5CG8I/Ab0BbMFhwXrBDMFuQUuBn8GlwZCB7kHpwlRChQLDwxSDaMOLRBfEmcUhRUMFVgTrxIvEwgT1w4kBk4AV/52+8X5f/sJ/sL9+PwY/6z/Vf8k/U38dvmR+FD1gPK17gDxC/FE8L/wv/P99ST5Hfxp/Vr+vf8AACj/U/9R/lb7//iR+Gf4NPfA92H4vPie+c35nvjn9633Z/it+CD5pPhL90n3q/jy+JX7MP5X/2MAEwEAABD/vf6Y/wYA2AAKAQ8B6gD2At4DzQK1A9EEmQV0Bh8GkgTRBMAEDgNNATgBDAGpAdQBCAT2BD4G/QVIBooFHwbnBs8HbwkyDJMOLxFyEywWCRjXGGcYmxZyFq0UvQzG/yn2GvN/7bznkOqT97r16vLb9A30HvQ6+F77m/uf//IBTf4Q/Gf+jv7n+8T75/65/3ICHwVeBPIFgQiUBmcCMQG9/335TPWs8J/rA+mX6Izo4ekU7qfw2/Fs9Yj5Zf0H/+cBYwPEBEAG9gPsAVwCWgQuBiEH9AjPB08HZQWs/5T8WPmX9s71Ffii9+z2X/ny+4P+1AH/A5kFkgcqCnoJygh4CSgJ0wh4CWsKZArtCusJXAhVCOsIsQa1BZQFrgY5BvwG+AgwC7ER5xRPGV4bjx31HB0YtRbPFR4Q5fi05Lvd+9jpzLnI0tCq3C3lRPCk/FoGzg73EkkUYRVuFFsKCgPlAakAyfsx+7X9sQFgBbkH4wVpBJkD+Pv18R/sS+lq48jf+eGa5RLqzu/l+HAAawdrC+sJIQkQCaEIaQR0AQv/RP4kAK8BaAD9ALoDywOL/wH+nfvj9rTzP/Fx7uDqkuv27vv12P3lAbUGuww/ENAQCA80DO8HGwmNCpsHFwW+BS4GZQSQBf0EkASbAw4DMwHz/9X/h/+9/7UCnwXAB+oLaBBrFqMboCBWJAwleyStH1gYeRRuEocGEuz03G/bONaoyvPNs9x35c/ur/tEBYsK9xEpEsUPMg23CS7/LP7EBPoGQgMhBtEKTAwjCckDePwx9x7ysep54+TgUOA74ADlLu3+9Ef5TfxUBGQLgg07CJ0FjAYzBk0FhgPjA9EDXgdVCm8KkgZ0AWT/rv8d/N30Fu9x74XwhPOA9Ab1bPZE+nj+WABaBFQEUwbPB7IJ4gaHBTsGGwfICLkJswduBEsE9AOMBFQDVAAk/B3+i/8f/0b+ZP+u/wEDMAhoCnULWw7SEkcV0xoTHyci0iH9IJgg2xsxF4wSFA1M9cHenNWc1WvKCsex1WjiYO/g/DgNWQ/HFJ8X+xRmEKMN0wSb+zn/2gdCBrcGqgl6C5QHzQQW/nfzzetS5jDhwd4h4a3ihuSd7G73JP0w/yICBQfMCNoH7ALX/lP9+QESBmkIXAhBCr0LYAySCU0Dzfz/+Gr2vfJG8UH0UPU++C37ofzn+Zv4dPrG+2X+Zf6O/8MBEgk/DQ0P+Q+nD6MNmAwOCTECRv9c/yP+y/yU/t7/LQC+AbwDrQGY/wz9pP0iABUDlwNyBkwMJBHuFFEZVCCPIO4fUxwJHQMV1A5IC4oELu1e2HfWDNZgz9POlt/W6fzyg/3zCYsLFgw3CaoJaAoCC30ChAAfBwINNgtkCykNFAnJA4z+PvlK71LnIeDV36XhHuQG5OHnsu+n9cD5jvtn/wYBcgJaA1MFZQWmBGcHVQwcDfEIkgbvB14Fkv8o+q/4Pvfu9Qj25/kS/eP8H/uU/dH+OvqI9hP34fmx+pb+FwM3CT8PChJBDwcNnA1tCJUDVgCd/yT8t/wd/4gC/QQxBCYCsQKDBWcCigF9AiYCQAJ6BfgJ6AotDjoS2BLfE+sVGxlxHOgczRn0Ff0SJhI4ETIOovhB4o7YH9kp1CDTJd3/55rxfP3ICSkQLREtDrcKqwwWDXAFEQBgBj0OOw3QDUMQPQwmAYz5Ovba7RXmueC94XPkeutg79PxVfNN+Cb4rfgk+xT+Sfwv+2EAFwbCCDsL0A+rEUMOsgo5BxkCFfuI9jb3sfrg+8T6dPo9/8//8v2v+/367vTb8S7znvj2+a7/RgewDDgQZA4pD7UJ2QgGA1QB0ABUAqgDcAOgAxUETwX0BFwC9v9r/u37Ffx4+8/+dAE1CLsKQQ0/DmMRyxD1EL4S9hVaGeUXzxYfFssU0hGvEYYQEwS+6kTdvtnl15XUst0M7E7yHPfYAfoLNgx2BdgAPAPwBSEH6wUhCwAQ2w8tDnsOYAmB/hj0FvFx72vsR+kI6KLqquzi79XyQfW09APv2fCT89/30Pjr/NoCbQUdCCULXQ13DJsFOgF7ACsCof4d+3T7Ev14/Af9zfx//CT4mvI08PzwCfIu8xD73wJAB28IuQpTCnQG3AV5AkkC5wQ1CKEHRghGCusIpAPlAOn9N/t/93D3Cvk1/FP++QDyA9oG4gY3CTQLIwnECY0O+RSKF5AbXhmHFpUSqxD3D/cSOBHy+4TmFOC9243UZ9kZ6CHtmO9L+iYC7ADV/jX8Afn0/iQF9gVXCI0QQxEWC0gLGwms/lj3efar9eLxTvDX79Pu6PEC8eTwP/KJ8rnupuyu8hr1F/k+/dICZQUcAT4DaQRPCNgEcAEXAwMFdgSBAyQGgQQCAXj9f/oI91f0Ku8S6gXulfZn+xkEYAorDXgIlAZLA6kAKQDr/9sAUwcsDKsNog/5DWYKkAaXBXkBGgA1AmUDiAQkBv8D6gDJAE0Bb/41AxAHQAiAChkSQBVNFMQUZRLbDMYJoQj0B/cM6A7BDAz+F/fL7P3mk+Nv7Ifyv/Ph+Dv9UgCB/SX/6fs8+S/7SP2M/twFowdTB7MEoQcGAtH9qP1p+xn30PY++X35Cvrj+Dj1I/Sp823y2e9K87/19fbV/ecBTwQkAZ3+ev47/i7/RQEkAyEGHQTsAf8BPAHP/0b+Sv8j/6b+YwHaAssC4QIRAsMBnAAeASsASwEtA08DHQRLBJAFYwRYBLcDjgPpBDcFAwYFB/oG1AOQA68DvAOMA9ICQAJjAEcBLQMQA5UCMQBa/4f+Vf+QA2AG8QhMDOIL+giZBxAF0QObAnQGkgaYCTIL7wfp/iT4i/SL8CL23vrr/+L/RQOVAr3/QP+Q+rH3W/dL+YH5KQDFAskBqP6z/vD8ivkF+/n5l/qU/TD+Cvv6/Bv6zvZ59VT3afk++Ub85fv5+YH55/oX99z6XPuz+8D8wP3yABD/h/+3/pf8WP2m/VIBvALRBBsGqgRcA5EByQFuANn/swJjAjwFbgTHAv8CpAB6/xj/jgK3A/YDvgVGBuUD6QIvA+kCewIQBEkElwRCBH8C9gItAhUBxv9YAP0A0gHnAo4EbgPAAXQB9AT9BdwE2AJn/977F/21/nkC7gFlA5kB1/5I/039QP0f/c36yfx4/hL/SQH3AfH/s/4CAYz9Tfx8/b78ufxk/6z9if6xABv+bf3k/l78DPw1/c/+a/3v/YX9iPt//Cb8OvkX/Bn8S/sD/8r+z/qQ/Fz6E/lJ++n9SP2w/jMBZ/2u/9r+5/qk/MT/hgGgA/gEjgRNBNH97/9r/Kz+tQRSAqsBwv+k/RD99wDwA2EExwEJB/ICPANVBvkCLPwzArX9Dv5jBWsHkwKQBm//8vyb/ckAKgOB/30FzQNFAQcHU//wAl8Aw/e8+5sC7gNhBKUJXP8tBP/9VPV0/6j9dv12BZ0HhALcBRcAIfOCALX+SfnsAKoJw/hyAxMCWPja+0cBc/FuA0UDffZcA+sDRfZ+/8MAm/pc/U0BEfkaAOECIvi8BfACY/eTACr98/Q+Bpr1gQHaAib95QFC+gP8ffsv+mv94P6KA3/8ywTT/xv9HQUX+ooCdP0XA0f7XAmd/B4CqwGMA8b8pvxiCK/3TQQ9CxD6lv9DDZH3IAAfB2n6uQWbBCj8XAiM/pD8GAmi+p4AiwkR9SQGtfwOCcr0ZxLF9AEC1wjQ8wgD4gZx8XwMrP/N/gUETf7GBPL4WgR79+MBxP6KAeUDRwE5/8v7igNj+Cr++PtTBun8ewBjBav0+wJFAu/zmAxP+0T6GQM+A//3ZgqH/pP5bQcg9ZAEKAIQ/nABLPwIAr75/AcG+WAIgfiHCLbzgweK+OsHqvxyBIj4mQR+/7X8mgwH7b0L4P7F9VQTOvSBBCIFhPZ/Bw77rv0NC/70PQ2T9ecBpAF/Acf2PQ7+7lwI9P4tAWf9QgTJAI79JgT0/l8BbPpMDyn3af3PCN/z5QOuCi/3qQHGBcryFAwz+kcAVQk8+UcDSQIV+wQADAXJ+df/xgZB9okM/fYqCqT4xv/T/jX8nwdO9UoMtvQLDXbumBxM3REfZ+7NBFwCqQDT/Fr9uAxn7uII6QnY5VUeoOjWAXsUC998HxLvFwEGAuX7fQBcB0r0bA0W8vQGYP21BIz73wF7A/PxxxBq9v8DmfvICwHosxGi9L0J9vpFAxsEHPZcBcL7HwXd9T8MgvUXAagIzPPqD0H1vAITA0zzOBJQ8xUAFAwJ7SEX2uYXEJb/s+srI5/cPRlz8v/4EBp84UQVUvlP+fwJ0vkCAegAyQHI/RsF3fUaDeDtExSN9Tf/lAVu9d8PsvNvBzj5Jgd7934J2P3V/+766gvk7oAKQgP18NgSVPSB/DALOPLUD1bqsRe562IGDwz663QXf+zNA2UEYfXHFIft2Qxc//jsZhwG5BsTqvyp9R8X2+FVGcrzkwCOA277qP7WA1gCT/uXBbr4KQDwAcf6VQbG/ir+hQVl+j4Buf7V/yIDlfqQBpn9igKx+QcKAPR8Bwr5QgkN9AQOh/J2BwP+sfleC/n2OQbT/0v8zwXU+ZcERfehClr5WAE3Bsv4CAETA9H7agH7APT4bQrW+IwGtfq8AdgDBvSMESPw2Q1y928GEP81+e0Mfu9WEGX30//WAhT/RQD9AUD8gQUB+/38VA9/6B0aS+ttCi8Bv/XkDi304gjL+pEBvgSG+bAIkfmzB4L2NwOIAxH5BA1S9ucEJf/t/2ED/fpPCMz1Rgm1+cAEcgF0/SYHW/U9DG/yyxC78AkOI/RGCyf3KgOFCD3xVhNv7FgR3/ISCZL/nfoDCTb4uwev+Q8MOPPVDEzyHQpJ+3EIXPmMA5//lfquChn4BQd4/oYB2f/X/oEFwvo3B2z5TwZJ+M8Hdv2kACgDPPz7A9z5WgTm/0AE4P0ZAUkCvvhvCqX1bwk1+usDYQH2+8QICPdrB4P7oAE6AOL9xgSO+qEFm/2OAzf9VAPe/VgCO/5dACQA0gEQ/xcAvgAIAOn/3AJP/2UByP4f/ir/1gAoAzv/gf/0AWn68ANsAOn+2AHR/iIAwv8rAaz+TQWV+60BSv9a/8//agHM/xj/6gF2/LwCLP7dAYP/aABHAYX9UgMH/aIDm/xFAir+pAGJ/tgBzfwbAqj93QEL/90A4v1LAqz97gJ2/3cAJf+s/kcA1/7WAar+lQEQ/qADvPuFBDP7MwKf/ecBrP+2AH8AxP8S/24BF/0IAzr7IgLi/1r+JgPg+74DePtuAoP+XwBc/2L/1ADx/hMCjv15AR3+KQG9/t0BSv5C/3z/KP+CAPz+KAKD/5/+IABN/9j9jgLe/SkA3wDR/sj+SwAz/04ASP9a/w8B1vzDANX/g/75AEMAg/3JAOD9DQCmAqz9aQIK/O//FP/V/5MATgDT/1X9uf/g/jEB1//4/zYAeP7i/w7/2f+h/n0Bv/9A/gQBNf38/j4Blv4eAHUAYP6s/0T/vgCKAO/+4wCo/aj/LQCw/wgAEwFP/T4B9P7n/eoBA/+q/08CnfvCAnb9EwERAMsAUf/8/+3+b/8cAKr/9P5FAf7+pv6pAUj+cAA5/4gA2wCf/iYBUf7g/1IBtf80APP/Wv8J/7YARP+TANQAtf2MART/KwA8AQX/Wv/z/4n/+v/HAMD+OgJ0/dIBRP7NAB//vACJ/54A9v+9/8r/VgC9/6QAh/9qAGT/iACkACH/SQCRADn/EwA6AOb/UgBg/8L/3P9p/54BjP6TAJv/RwAPAIYAtf95ADf/ewBK/6YAdP9lAA8Ab/+rAOv/RQBg/ysBg/95AF7/pgCJ/0cAi/9jAJj/OgBDAMb/Uf8CAeT+6ADC/2//rQDz/8T/+QDp/roA1/9QAL3/2AD4/8L//P+//+3/LwCiAHz/SwCs//7/YwCh/4gAbf9bACz/fQBg/3cAvf88ACIAvf/c/ysAHgC//zoAhf+rAEb/bgFt/wYAagDE/18ABgCCAMT/AABBAAoABgBqANH/3v9QACIAsQDT//b/ZQAs/xEB1//NABEAUgCq//z/qQCU/3kA9v9lAJMAhf/i/+T/fP/dAJb/1gD2//0Aa/9/AJL/xwDI/8kAEQAgAG4Aof+EAAAALQC4AHj/4v9YALf/oADr/9H/a/9FAIP/9QAPAEsAg/8TAEr/lwD4/24ASQBWAGT/LQAcABoAewB3APH/EwDt/wAAnADM/8cAjADa/pUA5P9UALMABgBjADoAPAATAB4A6gDV/x4AKwFx/6YA3P8nAGUAPABqAF0AGgDr/x4AGgD2/zYAPgDk/4oAgf91AJL/2f+1//j/DwDt/0MAhf+EAFH/SQCF/38A3v9BAMb/2f/i/xEAPgCl/7MAU/9OAJL/VADi/0cAbf8KAFgA7/99ACQAzP8XAB4A4v8XABoA6/8VAIf/qQDx/wQAjwAEABEAnf+//7L/G/+kAAAA5P/FAPT+8f+9/1QAt/9fANH/lP/C//z/6f8+ANf/bAAPAGT/lQAPALv/ewCD/5EAxP/5AJD/VABt/zgAfwDV/8UAdQAaALX/bAC3/1gANgDX//H/JAAGAIgAcf/C/0j/7/82AFQAEwCB/9X/u/87/y8AEQCD/y0AMP/m/woAmP8kAOv/v/9HANH/uf+d/7n/z/9WAN7/lwC//0MAdv9oAIP/u/9fAG//uf+7/w0Agf/I/w8Arv+q/9z/ggCo/xMA4v/4/4H/7f/V//b/b/8PAAAA6/8iAKX/7f+y/+b/OABJAM//dP/8/4f/CADZ/+b//v9+/7//b/9qAPr/rP8gABEA+v+H/4X/OAAf/wgAPgCj/y7/BgBc/8j/z/8gAFQA+P/v/8z/n/8IACkAIABbAHsAdv/g/woAAAAAAHIA6f/c/1IAv/+TAMz/FQAnAE3/HAB3AIYACADm/+v/DwDX/04AVAA0AGT/9v9I/ycABAAaAPP/b/+j/wAA1f9qAHT/EP98/8T/FwAvALoAb//p/xD/kwAXABcAOAARAHT/if9wAAAAVADT//b/DwAnAK0Ag/+CAFQA3P/k/7X/NgBfAC8ALwDE/+//GgCj/8T/xv/M/xEAFwDt/1f/aADk/7D/JAC7/9X/sv8cAAIAewA6AGwAz/+h/8b/v/94/9H/WAAKAEEAIgBv/3H/i/8GAFIABgAaANn/yP8GAJL/YwCB/+v/MQDG/0b/2f/R/woAewAkAMz//v98/0MA6/5yAIv/i/+F/6r/CACb/3r/i/9p/2D/ev/V/xUABgC//yIAv/98/w7/6f/x/wAA/v+7/xoAKQC7/0T/uf+h/8r/v/9oABcAh/+7/1AAa/97ALL/jwAAAB4AVf/e/4f/i//P/6IARQDX//z/yP6j/hoAPAA6AJcAuADe/zQAvf8B/5EAXwAXAUr/3P5v/yP/qQA6ANIAHgBdAEL+ZQAf/ykB4wD+/wYAkP8vAAAA8//z/xcBqwA5/3z/cABr/5UA9v/Z/6H/aAB1AOD/FwDg/7f/IgCeANX/kwAtAMsAEv8RAPb+UgBi/9sACACiAeUA8f9n/zYA5v8MAUsBtf9fAIIAagD2/1AAigFYAH0AHgA2AFYB+v8KAfz/rQD6/wQADQDNATYBtgCIAA0Avf/wAJwA1gFoAPUAIgCmANIAWwBaAd8A+QKU/+oBeP/i/9n/lwFFADEBCgJX/y8D9v6pAcr/MwGeAJcCUgBUAdYClv/uAeMAxQC4ARMASwC2AUD//QFLAUUBpgDg/6kAcACY/+oA2AC6AaX/eQGu/rX/YQB4/8UBsQDsAPADXv8w/2D/1/+s/vQDMP+VBD4APAAu/7D/kwFhAAQCdQAZAa7+IgGo//H/9v/l/VH/BgApACsBFQCpACz/HAA5/8D+Xv/wABT/ggA1AlAAEQEIAa7+uAD0+wgCIgDsAREAxv4PAMb+0ABlAMsAh/3I/3/9LQCCACQBjv90Aa8BBf5hAW//UAFUAYn/hf96/o8AM/6ZALwA8AHL/cACDv3UAG4A5f2tAmD+qAI1/TwA3vyw/wn+ZQKF//7+MQF8/2X7UgEQ+0ABPgCh/YoEkP9P/UUBXP61/e4BXv7wAboAvf7R+0IElfk2APj+pv6pAfH/4QH2/V79//y3/nABof6RArf9gQTw+CjqtwSu8SABIAHyBCYDGwMqAxMApAF3AZcEXgVEBMAF6QIRABb//P1t/0MAxv9HAMAAa/4b/vT8wv2u/vr9hf0z/w78BADr+4oCOAHc/aQBlP+MAnUA+P4F/jz8nAD5AggDYQBE/vsAvv3J/I75ogF2/7oACf/x/zD+KQDE/TQAH/5n/a7+wv++AH0Am/+OAaH9SfzX/jX9dAE4AmwAVgN0/0394v9N/1X/LwFYA2D/C/9NAoH+JgE4Adf+qwHp+00CKwFYALgB5v9I/zD/RP55Aen97gHv/jYAdwGM/uT+QQBFAEsArv/2/ikBbgAl/9P+Cf+3/jP9SwBE/If9iPgcAVb8kv9V/eD/gf9v/bgBcvxt/ab+kP8F/uEA9wDfANYBDgQ3/o8Ah/7I//IBtf6h/54AUgNwAcMB9ALDAFABKP6TAlX++wIgA1oDqQEkAcz/lQEAACACRQJWAigCkwB0ArL/u/+zAHUA/wJAAXACWwBuAN7/cv5YABoAVf/R/vv7fP4OAlr/VANA/qQAwPvn/v/8sP8TAW4Bb/+5/8b9PgB1AKQACABG/7L/KP8mA4H/HACIAD77jAIRAvH/BgJT//T86/4H/zn/Nf9NAbv/o/5c/eD+/P9P/uMARwGkAAAAjwB2/g8A+v7NAEkCswKzAToExwEOAuUDcgS8AdoB0/+9/qACmQT2AwcIKASrAhn+xv+GANYA3wDYBJIEMwLi/+4DGf3Z/0sAHgGBAjMDdATWATX/vvzg/dz+Gf4kAZEBsQCiAI4By/27/d/5DvtY/N77mQBT/gr8KPyo+Vz8M/pU+Vb5GfyV+Uv9bACm/HT9RPqV+sj+Fv+pAfIBuAG7/7oAyv+6AqsAIACBAqABswOvBcsGDAN3AoYB5wOHBb0I6QlTB6EHgQcwCN4KIw0/DNMKcwpoDTsMuhAADjAKpQswDLAJeRI0EBYIxfgc817qa+5v8HD86/029dD2svHj6lbn4u3T6uLsivjT/egAdvxR+0z0d/Ui98IC7weoCN8M8wlTCR8GrwSW/6r7DPp4/sICPAWVAEL++fQY8RLu/O8/81vwlvNH9ND1e/jy+d34cPeI+DX6kP2kAGcIwgc1B9UGdAjTBWMExAZNB58FwgnZCvoKnQh0B8QEAQOzBcIHUwgbB88FNQafBYkJawkYDjgPLw+zEbYQEw60CpoJnwZeCCMLIQueAd77IPRt8IfuvvZ/+RX7GvY085/qdumd6yrqEPCN81b5g/rc++P7+fkg+Tz7vAByBBQJbQl/BoUJ2ge3BEIGpgS8AzD/kgQGAx/+cvz7+zH4p/Wg+qT3svUE9CT2ffj/9nD7kfmt99T1k/ex/IH8M/7t/wIAt/tA/l8BN/9UAbEDRAbwA6sA+gUzAd79igLyAlgBqgNFAlwC4wBA/rf/U/9/AHsDSwNUBBMECgTJAmECYwIOBYUH6QeLC6gIGwnyBeAH8QpkC1kLlQSBBggEDAeWC2YM4Ar3+Mf3lvIw8332UAC3BI77VPeI9ZjyCOpg8Un2TvHf9jP+m/46+q78r/sP9oL37/5YAi8EyghcBmsGJAPFADn+EP4Q//7+3gPsAI7///xh+jT2bvcz+mr2r/a29Yz7PPmv+FT6ffYV9Un1Ffkx+qb83P6q/LP9FwFSAD3/kP+OAcAA2gL9BHcCEwMVBPsCvgChBv8B+v5CAg8A6QMkBe0FVwhNA5ADnQcY/8cBpQidAzcKMwgsCLUJAQaKBBsHRgheBUQJbwjXBy0AHQcqCNH/BQpCB/YCWPr593T7rfkz+3gGKARl+8j+Rv8i9Yb6U//P+bP5JAIiAKT9Ev+b/5X7//pR/jYBV/5NAagDn/9lAD4DVgFE/pn9wv/i/0ABKAMRAd79Yftu+3z9LP1y/vb9MP7W+rn74P8S/dH8sP6k+f35DAKX/Zv9Pf6bAvH+ivzWAK8DT/7M//YC8ARFAy0ECf6d/90BZwPG++MCRgn9A/T6cgCkBjoCYv1K/80FEwSVAUL/uQQ6Azn+0//P/dgEvgIcAQgCMQML/5D6PAF3AjQAf/7fAWMEqP2f/S8CAQPP/6b+0AHhApEC2wCd/44CigJbAM/+7gKdBMABLP/CAgEE4QCJ/iACIAQIAeEAyQD0AcsBkwAeAN8AjAAnAML+z/8KARb/t/4Y/6H/bf2f/BcAsP5N/IH+Fv8h/mL9Kv0Q/LP8+P07/bf9Z/zg/Sj9+Psd/Fz91f1t/eX8qvxi/Vr+yv7Y/U38Rv8nAOn9b/7UAJsB4P/X//8BXQHc/5sBcAHHAiYDfQNNBC0EmwH5AQIBEATFAjECQAN2BJ4CiAEbAncAOAFfAYIADwEGAmMBXP9i/p//4v1p/hL/BAAu/979g/2d/m39+vx2/bD+Lv7l/TH9hf7p/aH9Fv+J/kD+v/8s/7f+Sv/Z/8L/Nf6kAM///P7T/6IBfwEGAAgBIAKMAKz/1ACeAUABjgG8AloBBgF3AfP/yP+gAd8AxQACAakAQwBv/xT/DwCJ/6z/tf+j/yr/QP+S/uL+FP85/wn/3v7P/hv+yP7n/sr+i//C//j+6/6h/of/Nf/X/zX/Kv+o/8b/ev94/1gApf8W/+n/RQB/ALgAmQAeAO//KQAVAA0AqQB1AMz/VABuALD/7f8cAC0Avf9uAHAAt/9n/0L/lv9g/zP/Lv+H/xL/lP7A/of+5/6U/iz+m/65/uL+h/6D/nj+Lv5N/pv+4v4W/8D+qv6z/oz+af6B/s3+n/47/mn+Af/k/pn+b/4L/8D+mf6j/uT+z/6W/jf+9v1y/sL+pv65/rf+1/6O/sD+GP8b//7+/P4Y/xL/a/7e/k3/Yv/I/rv+dv89/w7/b/+q/4n/fv/n/u3+Sv9n/1f/m/+y/5b/GP96//r/CADK/53/2f8rAOb/2f8VAH0AAAC5/7v/AAATAFsAUgD+/w8ALQA4AFIAUgBhAHIAIAARAE4AKQAcABoAUACIAF0AYwByAKkAhABHAHAAkwC8AKAAlQAZAVIB6gCkAP0A4wDbAMcA8AD/ANQA8AC+AMkA6ACGAFYA8gAgAf8AwwAcAU0B+QDlAAQBBAHsAMAApgCCAKYAvgDAAPsA8ACIAG4AswDsAMsAkQC+AAgB3wDYALgA8gAPAd8AjwCTAPUAqQCGAPkAKQHbAGEAcACiAPAA9QAXAS0BbgFYAfsADwFAAVABDAEcAZEBqwHdABUBbgEzAQYBBgEIAV0BHAH5AAgBkQGZAWcBYwFaAU0BPgH3AHkBjgFdARwBcAF/AUkBFwFQAXIBJAHhABwBXwE8ARMBGQEPAQwBEwEgASQBIAEMAeEA7gARAfAAxQD7ABkBTQEkAVABkQEtAeEAEwEcASIBzQCCAM0AwwDQAMsAoAC2AMAAlwC2AOEAngCcAPIA4QDjAOEA6ADdAK0A0ADJAO4AtgCZAHUA2wDoAMUAhACKAGEAPgB/AIgAhACTALMAlwCCAHcAPgBdAEUAaAByAA0ABgAnAC8ANgAnADQAIgBOAEkAIgAcAC0AEQAiAO//CABSABUA3P8GAC0A5P/6//r/zP+b/7//sP+f/4H/o/+F/5b/gf/Z/5b/Qv9v/2L/Uf9g/8//ev92/3b/Wv98/9z/0/92/5v/dv+H/3z/n/+l/4v/af+b/67/yP8q/zX/if9C/1r/n/+s/4f/Wv9R/6H/XP9t/0T/O/+L/2v/Of92/37/g/9a/3j/if+Y/6z/vf+Q/0L/pf+h/woA4v92/4P/sv/r/+v/if9t//P/6f8VAPH/8f+h/1r/hf94/3T/h/9n/0L/RP8l/zP/Hf85/zP/B/8w/yX/Nf9i/0r/T//2/ij/fv8d/xb/GP9e/0T/Cf9t/yz/4P7c/vr+O/+///T+yv4Y/yX/QP81/xT/RP/V/6H/Hf81/wn/If/R/gv/pf/4/h//FP9P/4n+QP5K/1f/Af8b/+L/af8U/oP+3P+1/zn/f/7C/of/tf6k/T4Asv/R/VP8A/95AUD9m/32/L3/of5n/U/96/+u/HT7oAOzAAP7m/2U/LMA1ACCAJD8VALf9A4EZ/3RCF31Ovfa+CX1utKWDsMvRx9s9hH0DQBGB+nuK/TuAFIDAA339z3wDvq29oPqmN9y2MkCsSEDB9X9Jw2FGmIGjABJBSAQTxjXGNTo9t1RB3QYjvvk3JcQ9zXd9y3Bpv3zG17+T8qmAlgzbgNE2NQDS/rl7LT2vPdkCIkYcvqxBAfvh/9y+yb6ExIOF9jsQBcCCzX/SflIC/AFvuzRGyP08Pj2GOEDJse69G0ZEBq3yFz96jRh+5HCk/mxIsTY6cZJ+i0E4/UOBo8fRTHy5gT1Fh6QGRsHQw6YP28et92Q2co8rfQa0kg7nhMtx/ADBS3H+BjCf/7gLZ3oacpxKtksl9Oa1EAZKglp/xcAMwh+KuD9qt2AC6olWvnm4RgL0xYZFzjXgg4yC0gL193a+zYOuAz88EcCvf9M4V0jM9yF8AIi5Bqvx4b0RigI+UvnEh2+FC/8QtuEAmEimxRWAGrRjRzRGRfprNmUOgcq7NFv3oMYCCAz+QPf4fSnMWf6PthU+9Ay9fcvwSkLK1PQ8QnMbwtqHyjaUvUGFdccof1H1k72ryR5Auvb2PsWHIoDAuXlAakf1QjQ093n+CaYGzT0wd5JFIoSferN6FgTDBbs+fPyQwxcFwLzUf7s9tsAWADvC1UIxATq+FgBBf5QAH7xfPG6IMwIE/PB9JgNFfkK9z4B7QkVD0zzwOmHCM0YdPog93wJjAAmAqrvYwWDGBsDZuHb8uoRZROK+yPyFwM/DZHmovZhEGQOrPCZ+I8AQP9aBI4EQfZs+kHyLQS2EXwLyfv27bH5Lw9uA5n6qAXrCegAnvSrACIU4ggl9Oz4+QKzBwz+CvaVBG0FEvxSALkI5/0L7rf6uQsnDf/54u8oBncR5/tl+j4G0gLS+LH3fwW9Cdb8pP3r+sj/gQGtA1AOs/2w8jP/1wfHAlf+mQDr+tH/wgLyBP0By/pC+7gAs/ynCSoIxPsS+0/8dAN/Ap/+9P0hB9j4C/VKC04QbQXc+ij+cvw3/cD97gIdCOQJ/P6n9W73pPy3BgMG8/9s+kr+VgG6A3T95fo9/rv8wAQhCGUFmQRr+6brTfvvCywLCwvK/8D8RPrm8wX/cwrlALf8LQLuBK7+wPpi/ZEAoPex+s8CYgZNBY8AFfgi/Pr+Jv2rAsoOcAR7+En5uf0z/xsDqwGB/pP4EQBt/7EArwCIApX7VPh5+Z8E5P43+qj9PgOF/bb1hgL8CAIBwPgz+hT9Uf9//KIAmwIbBlP+SfgO/rUG4P+o+97/iAHy/Hv67AEdBLj5DfZ0/PsAr/fa+TUCCAUu///9N/+rAbP9NgCU/Zv8EvxNA1sAZfym/Vz9f/zY/aADwgWO/034yfne+1YAEwC8ASAB5wE7/RX6ZP+k/YH4jPnYADUE9AKZAYgBDvvz9ab8vgJgBQMDev9f+XT7Z/7hAbn/6f7UAe75S/lOAKwEl/w3+9H9Rv+tAQoDU/48+4X8qP93AK8APAVE+zL0nf2XBC7/LQBeBTwBAfrP+toB2ANQAAH/cv6U/rr6BACMA8T/l/uI+ZD97/+1/w4DHAGf/UkAif9jAAEEeAZFAWD8kP2X/Un8igG+A/0Bm/1a/rgAbABP/6kBcAK9/hT9VAFyAJb/dP68+jYACgPJBMUBcAHA/PT80/92BMADlwETAJL/xPvP/7MC//3k/n0CgQG3/KIC+QGV+2n88f+BA30CUAFJABf8effW/PkBPgTNBd7/Zfu++7X9qP8KAboChgOtAe/+bgErAcb+4PtT/aX/sQHfArMDpAFK/t/56fkFBKwEMwavAxkFqP8++s38/QCRAfIBAQMMBYMFLwJG/fv5Bvmq/C8DxAfMCiwFgfp797f6dP/NA00GRgfSAPb5hfxyAYQBSwFpAtf/z/5wApQFrwQQAx4Bqv+VAecCWgNqAUsAwP2H/nsAkAXfApb/kwGo/1P/SQHaBLUCdP9C/uoBqgTeBXkCg/+x/Cz8sP4oBokNBQrYAML7RP3l/KsAWAfeClwFm/88/Pz9//12/S7+mQEbBAMEdAKzBEQEfP3/+2wAigUMA6QDvAJ7ARL8ivtn/mEBfwKmA50EUwXlA9z8wPmM++D+UgF/B8ELkgfy/Xv4k/bn+O39/QXrCKoHZwM+/JP5sfo1/W3+jgJEB88FxQIzAmL+L/l99oz9fQNrB3gK5wf5AGz5j/ez+qz/3gP/A/IEjgHe/YP7yfw+AKAA/wIuBm8GEwHR+wX+af+4ALcDpgYQBaX/f/wZ/fH/LwGrAa8D+ASXBfIABgCKALH9AfpE/mAFwAX0A5cByv75+TT3//mrAOUFmQXeA1QDjv9f+or7igBHBOcCpgOrAhj/bvtp/JT8Hf1n/woBOALLAo4CDP6U/HL8wv6o/0UCSwHr/xD9cPyo/dX9n/+RAEABkwIVAsr/9Pru+DH4KPt7APIEWgV7ACz+sfrL++D96QJPAsAAkv9E/6b8a/2b/YH+XP02AN8C4QPjAkICbf1A/Wf8jv0O/+MCZwJ7ADwAFQGd/vz+JwB9AREAhAAtAmUF5QQBBkAGugO7/lb8F/xy/qsAwgaQCeAIUQS5/rX42/ZL+xwBEgcWCx0K/v85/Hn57Phw/MUCLAZCBO4Cjv+G+Sb4S/dH+Gf79AGDBMIDHAFC/tb2lvIA9pT8yfwiAWUASP+k96b37Piz+kn7n/1nAW4AlP1n+y/5jfUk94H7/QCDBToEMwIs/e74oveD/PcAVgX9BaEF6gF4/7P9BgAbArEEGQYuCkEK+AaVBNgDjgJWAAEEgwfeB5AHTwmHB2MEnQMeAgMDrQOFB/0FEgcjB8AFhQQDBHICJAEaAHr/Kv7v/8T/o/8Z/gX/Ivga9PPu2+9F8sf34/x/ALf9APaW8P7taPGt9Jf8If7A/a/6ovqe+en8uvmG+NL1//md+pj/UgFBAAr4WfX79BP22PdR/QH+Nf9E/2gAo/9fA+4CGQK2AEkCVAC6AkgGFgmfB44GCAQSBckDGwY9CRYMqwx+DXoM+An0B14HVQi9CQcLCQwfDOgOkw+9DsoKlgkbCdwIAQk7CnQHf/xk9LDvzO527JjxUPgi/Gn5e/fD+Br1j/bq9j75uvoK/Gv8Kv7yAdn/KvuB+fv6w/jC+4f/SwBX/mP7vPlU+Tz4aPfd9zH5uvae9qj5Tf3E+x38YfpH+Dr2AfpG/hUEhQWXBCAAKwCu/n/9m/4BBUAF2ALRAzkFtQRFAksA/P4K/Tf+qv+xBfULnA+uDLIKBQltCCEH5gokEfISuhCgDw8PCw0WDUEOcw0wCjIJuQefCWgL/Aw/Ccb/NvZT7a3piOg17tn1QvqX+c34HvY99Cn3BPnw+v/6uf6O/Df86f3aAu3/cv7c/QH/FP9p/24CTQOgAWv8wPfW97r3ovY8+Mn6pvu++gj8A/1e/ZX7mfkx+tz9qv5UANABmwRaAzoCFwCQ/pv82vzt/Z4CkASvBOkCQgPJAZD/yP7bAHQBswCXAOEC2gI1AkkCAQWsBkIHXAi5CWIKIQpbCn4L1wxFDp4Oww44DwQRVBF/EcMR1BAEDcQIGQckBs0HoQd6B3j9IPXW6WrltOL26JTu/vOT9hH5pfVF8uT0E/cR+R36sf07/Z3+LQTCBHIFbAJJAwH9+v2O/r3/DQCEAV0Ay/oI+Rn4E/ci9iL3NvZj9vf4Kvr0+u/9xP96/UT7pv0b/woBwAGkAj4CVgOMAu4AjAFSAl0BjP3g/U/9jv03/Wf/IABR/8L9hf3w/IP92Pyf/UD/PAMoBCIEvgWjCYAJqAi9CFcLNgsEDb4QihOsFX8WdBfyFOcTRRGaDrAKNwkWCMoIRgsLCJL9WfEK5jngyNqN4aTnKvA98871S/Yv9X70w/Ml9IL3lfiM/FAAwgcyChgKhweZBWUB5f1//eb/sP4H/Sr8afuk+Sb51vfD9UHzQ/K98JrxX/Tl9xX6b/3T/hj/Hf4eAH0B8AH7AAYCOAKBAuMCKgUxBnIG0QRUAzX/mf3W+4j7bPoi/Cj8Pv3V/i8CCgLjAf0BVAIMAiIDFQQfBqMI8Qs0DUMNmg8VD8gN9QwGD5cQ0BBSE6oVVhYkF74V0hK2DLAJYgYxBSQFggCe99HsHuig4ubgQ+Ri7Czw6vKa9Zz2OPZ79j72ovY8+OD8EP1aAgkI7wmwB9kIYggzAxT/dP2G+qv2Rfav9e72QvkT+hX3C/Wt9ODx8/AW88z1Q/YT9ib54fl2/ef+NwMTA9wEqAJLA4wDwgVEBJADpAG4AlIBTQJ7Aa0Bif+f/cL7f/qx+kT7dvyz/U//QwHAAaQDfQTYBekEjAYsB5YI9gm4DEMOsg51DzQRmRLJE/IU6xVaF2kXJBc3FeoS0g85C70IqAflBd79HPQ+65HlmuBz4Rzlluz27+H0HPcD+4r79vvj+Bz44/Zh+U37zQKZB2sL+AojDA0KOQeZAh3/9/gr9WjxrPCS73zxHPIR88rzP/U89ST2J/aN9sz1Q/ja+f/60fzQAM8EJga5BoEHOwf/BjMEhgIVAGD/hf7//VP/QwArAa7/yP/z/2f/Tf3l+3b8Bf3n/pUArwPrBYsIwggoCbkJkgoHCbkILglMCqoJpwsjDuEPbBCTEUcTwhTtFrMXDhY6E0EPSgvxB1MIxAaU/+r1DO3w5vrgd+FO5pLsU++98oD2EPuf/M/+zf2m/Iz7GfxR/mwBlAWMBtEFTwezB8YFdAKQ/yr8TvYu85TwP/Ch8t/z/fWC9ij6gfl9+XL6ePwx/NP8s/xT/U/8AABSARMENQV4B64GkAWfBMACMQHC/2D/9v3P/Yn+V/9DAN0AhgHwAFsAyP7E/mL+Nf9l/uL/SwHCA+kErAa9B5YI3geuB1EHJgjZCDUJbQtqDt8RsRRaGGgcUB2NHEYa1haPEcoLuQj/BhECR/lI76jp1uQp4/XlH+xv8dLzIvaG+U38Kv/c/ev85fuh/If8yfycADwCRQOVA2kGpAbcBL4BO/77+UX3kfOL8eTwLvI48h7yqfV39nX3gvaE+NT4Cvn9+cD5r/r7+xD/pAHPBBYINQhGBkkFnQTCA4IA2v47/Wf9Hf25/Z/+9v9LANf/dQBWAa8Aaf9P/lX/fP6s/5UByQRLB34IIQp1CpgK6wpIC0EM0AywDTYPExFlFBQXYBmwGXwZpBfpFP0PlgzICcQIlQKb+YPuwOa/4dPgzOMg6RDvnvLj9Yz50/yJ/hT9/fzi/Sb+rv2s/KH+nf/WAHAC0QR6BmsFbgGQ/DH3DfOu7h/sl+r/6hDrFO7v7zL0LfV/90L5AfuM/Nb7Nfr0+Yj6h/0S/oYBCARvB34IgwniCJYIAQZNA0sAyv6h/C/6TfnP+Tf7f/vr/Nj8LP+O/6AArQCzAQoDeQNcBUAHFAm9CjcKUQqJCpEMXQ1QDCEMQQzFDF8Nzg+VEUsUxxS5FVIUihIID+YLfgrmCg4F6/yu8Kbq1OQ25WPnkutT8OryOvZf+ar+4wA2AYn/DwBK/679EPwk/Vr+hAATAWkD4wNjBOEBKv5c+gL3v/Ky7iPtQu697yfx5vIT9FT2dfY4+AT5DPv4+gr5CPe1+AX7SP1hAN4Dowe5CNwIawdRBVQDwwE0AJD/Kv3a+xL7+PtT/Qf9cv6W/nkB3QFhA9EDEgWdBXoGGQaFBnIG/gfEB/wIhQr8C/cLTAycDTYP2BH2FC4XtRf8F2cY5xZhE0cQIQxRC14HRwA49Z/s2uYw45rjnOY+6s/tFvSO+az/KgP4BLMBEQA1/TH8L/nn+Vj63PxT/08CRwPgBZcFswId/vT5HPbB8LXscOtA62LsSO8s8W700vfJ+lz8If2h/fb7RPoX+hL8of6KAHsB7gJTBmAKpQo/CWcGcAQtAQP/Bf03/Mv6JPva+wf+Sv78/6kAoAHHArwDjAMVAzMD3gNnBeAH4AkHChILpwxUDmgORQ56DPMLMgwuDUwN+Q8QFaoYhRq5G1EaFBeZEUwOSAoSBmn9P/K16Hnlp+X3523rNvEN9q/6KP9FAIIASP96/RL79PoB+3v6bvl/+gX+cgLuBDEGdgMIAmf/4/ui9wLyge5a7SPu7+8h8anzOPam+Bf6qvpJ/Ev8f/t5+ZH5jvoB/ev/+AOdBu8JHwv1ClMIOQXfAoQAwP7I/c37Kvw5/dP+nf9JAAgAZP98/zX+Fv6+/cr+kP/3AHIDigZCCJAKMAxSDQsP9w62DU4NAg0eDZ4O3RBhFHIYLhtiHRgdrhxZGsQUiQ7GCj0IJgS++8btuORX4L/ilebu6XPvNvRF+Z38ePzy+6T7N/q6+Rn6Ovr2+qL6dvz8/u4CLwXEBZcBrv5A+1L3CPXe8HruV+6w8CfzdfSL9Or2S/jS+b75mfl9+Nb46vaI9rb3wPvAAXIFygeFCWQJ4Ah4BWUChf+j/7f/DQDa/Vr9b/6B/10A2wC7/xb/U//4/kT/af6y/xMA0ABuAzUGVQjxCPMJ6QhICVcKvwqwCr0KtAuCDVQPLxKvFNwWNxjtGVwZaxd0FPcP9w0RDSUMBwZC+gzsnOO/4FbkLeh07cbwivgu/ZT/rvz9+Mv39/e6+WH6L/gE+NH7UgEdBWIHAQZcBAwBV/9u+gj1gPEu8KPv7fDH82r2dPgi+Lb3aPU09Un10PU/9Sf2BPfl+XL9LQJpBoEILgnVCDcI6QX9A0MBKQFYAj4EeAS+AhUCfwHfAUABpAA9/q79EP6q/0MA0gFWA/AETweyCe0J9AfcB/gHgwhaCEAIzAgFCvEMbA/JEJcRtRO8FDcWHRcqF+MVbBK/DHwJrAeQCo0MIQtaARbyr+P03FPe9+Rt6/zwnvTc+f38kv32+Uf2C/Pm82j1Bvi8+tr94wDpArED/QKrANr9iPoM973ztvLZ8071uPaI9yv2+fUJ9MP0UvOy8vXyqfXU+Gn85/3l/Zf9mP/4A/4HgQgwBygETQNjA74EIgXJBAoEOgSXA1QByP6q/Pj7Ufwk/Of8N/x8/WT/jAGeAiQD3AKVA9QD0QSFBS4GwAftCOQJtwpgCqoKXgtIDDgNKw84EnQVChdCF7MVHhOtEJYOCwuhCTIKoQwRDZ8HIvpG6zvgxt6G5cruS/YO+c34oPhj95f2e/cc+Lz5xPoS/IH8kP3sANoESAYXBUkADPuN9tn1ffYr99b3wPgg+Tr59fZO9I/x0PEn8930j/UP90v5yP17AX0CewBy/i7+5P+tAYECPAOSBGIGmAjCB9YFWgIkAF0AJAGIAfcB8AFaAeoB8AFFASIArv/qALYB9AFYAV0BYwIkBR0HNweHBjcE5wPUA4oEyQQfBnoIyAuYDXsOQQ5UDo8OnBDoEY8SKxH7D98MTgvICtcMrQ3xCUsAR/XT6jrmiOjV7q/29vzNAWkDRP+X+K7xXO1C71T1b/10AzsGkAW1AkD9efil9JX0k/cV/KX/WADr/un6Cvcr9D3zu/J385r1oPYR9+z2Y/Z79Zj1Tvav9l32c/bD94z5Jvv0/IX+M/+KAFYC8ALHApkCsQKOAmUCxwIRAgECOgLHAggC6ACzAFsAngCEASgCAwO3A8IEBwZnBrcGtwZyBSwF0QUjB2cIbQonDLAN2w++EfAS3xPnFJkVLRWXFMkTSxPnE38UJBScD58G0f5h+E72ffjY+gP83Ppq+P32GPRG8bvvqO607xPzk/Yv+Lr3Y/bH9TT1bPRF9Ef0lfTS98D6F/yd/Df7KPnQ9wT3Lfeg9sf1ovVW9WP1J/W29EP08/MJ9Cv00vMj9AT1vPUk9wr53vqh/dX/xwEdA8IDFwQDBCgEsQQ3BdgFFgdnBxsHmQU3BLMDVgPEA7EEwgUQBsQFhwVhBMcD4QN2BI4ErwT4BM0E6QSXBUQHHQnmCu8MsA6RD6IQ/xEtExkUEBS4EwQT/RIrEyQSCw+LCFgBpvxa+lz7Dv0d/HL5Q/Z59LT0nPS/9Gz0kfNd9JH2jPce+ND3Hvck9/X27PYv9xH3uPiO+pf7CPzr+rX5F/ly+RH5f/hl94b2WfZf9kf2/vTH81nyu/Gc8ebxTPIa8yD0J/Ug9w75QPuq/YX+JwC2AGMCJgTPBWUG0QaQBgUGNQZVBmcGjAbGBh0H0wcBCCYIbQfYBsAGKgYKBgEGLwWSBPIDQATGBLUFXga1B7kIvQklC4IM9w1BDxoRwBL7E3kU/RMME5ESzRI4EvsOyAhFAjf9y/rw++f8U/yr+U726PTD87Dzj/Pq8hHzd/SP9aT2d/dF9n32HPak9kf39/em+Ib6z/tA/eD8jPvC+qL6qvpC+6/63Pmo+GX4cPi1+Ar4k/dq9rr1IvWv9NT0R/Xs9mr40vlL+/37N/1e/hUAswE+A1YEzQTWBYoFUQazBa8FPgUUBjsGBwchB+cG9AY9B70H/geHB5cGJgYmBQEF8ASXBBkEeARcBAwFfwXLBf0F+gb6B+AJ+gvIDXkPWRDYEfQS2hL5EhUTVBPjEz4TzA/GCXkDu/6Q/Zn9Rv7P/MX5BvZ99Uz1HvYv9jb1nvS49FT24fbN9xr27PUl9YL2AvcI+Or3zfhJ+Yj63Po6+mn57vjH+Jf5Z/mT+B74hvdY94b4F/hf9x72RfXk9Dz1v/W49kn30vcZ+U/6zfu1/Mb+xP+IAeMCWgRNBLwD0QM+A5sDRwTrBNgEUwUKBR0FqARpBJ0EQAVaBuAGSweUBrwFgQRJBJsDJgRnBHYEvAQFBdgEfwXnBcAGqgcQCW8KNgzBDQgPrRAiERURoBD1DwIQ4Q/fD7sM6wfUAp3/9P0M/pT+jvyT+v/3bvei9nv3vPbl9n32UPfl+D75zfiI97r2ivYE+IH4CPl0+GH40PeM+Ar4jvg091341PfD+KT45fgP+KT3HvcM9+r23fbf9sX1oPWG9Lb1K/YC+Bf4KPmo+Of5jPrL/OX9IAAeASgDkAPYAwgETQNCA4wDPARABY4FyQWdBZkFqAUuBucFcgX/BLUENwXwBCQFIgQ+BD4D6QOxA+UDeQPPA0IE1gTcBb4GYgdpCMwJvQvVDDAOTg64DdANrQ5ZDxUQhA4JC5cFsQHi/kL+7/2B/jX8CPuG+Nr3Ivb19av0HvTh80P1Yfev+HL5xfiM9x733faX9qv35/cO+fT4L/k6+Kn3Rfdl9wb2oPXU9ez1YfZq9/n3Xfjq9w/35vXk9Oz1//ak+NT5kPpH+x37Qvva+un7/fw5/ggA0gH4A80EmwU1Bc0EeARnBKIDQgOgAw4FTQasBr4GsQV2BCoDuAJwAxsFmQaFBx0HHQZnBRIF0QNPAyYDIgQ3BUgGSweUB6wH1welCB0JqgpZDEUOug5MDpEMAgssC7gM8w75DWAIGgBh+ND00vX7+lAAPgI7/0T7Yfam9yL4+P1J+Zf4tPOe9lT6yP8iAt0B6f0g+k72lfU1+h4CGAojDVwIrP7k82TuO++P9Y78LP/e/Jf3MvPZ8T3xZvE575DvSO9D8zb29vnR+kD7zfjs9SL1bPfv/SIETwmNCvwGSwNLAcUCyQX6B2YJKAhEBl4GXAcdByMH2gUQBK8BTQGMA58HrAnoCqgIhQXwApADyQQhB6oIsAp+CoMJkgjVB7sHuwfcCRQLwQ2yDjoQFRDHDzQOjQoZB1wGDgmpCyMJcv7C7cjhXN4i5+DxKPpP+vXzbe436lbr3O1Q88/5HAB6BSEGqgcOBhsH/AaXBfYDOAK+BacJIA7QDFEHpf/J+F32QfXs9oD2Nvcr9/f1uvTS8sXyLvPO8430NvWV91z6hgBnAtQDtgE8ATYAFwHhA+IG/Ak0C0QKDgY+AR3+Uf1R/r798f41/tH9T/1g/RL8qvqK+lb7l/2W/7MBNQKvAtYDdgV/BkQHygdICfgJvQrkCYkI1QZeBw4IOwlrCngKiwrxCQ8LsgpiC28K1wpKCLsHzwWOB5YIHwsuCHr/SvH358vlQOtW9az8n/9l/F34GPOF8PjwCPYm+un9If/uAs8FgAl+CDsHGQHr/gz+Jf+zAKACpgNsAq7+G/p99hr1IvXF9Un1HvQT9MX0AvcM+NH6NfkX+Tb3YfjC+jf/swFSA3YDZQN7AkABSQAmAQQCXwMvA2ECVAERAW4B8gBt/1j9Ufwq/J393P8TAR4BcAATACr/If9n/80ADgI1A1wEFwWdBcsEmwQTA1wDmQM3BVYF4AV0BCIElQMxBE0EZwVEBEcEDgV9BrAHUQgLCJsHzQbeBi4HeAifCQ8L0QlGBYf+bPn/93L4Gfz6/SIAYP82AET+Kv3E+sL61Pnh+OP4kvsS/owAQwHhANf+rv0s/Fz7R/tY/Ev9T/1v/TH9Uf3Y/Eb8xPq4+b74+fnE+gj7kvy3/Pb7F/v7+i37ufsx/bn+jP25/F79af69/pv/sQDDAPUAWgH7AbwBkQFjASIBRQByAU8CwAPAA74DoAJ7ASABhgErAl8DbAO1A38DUgMVAwMDmQKvAq8CrQG1AkcDCgMIA9gC8gGEAYEBhgIkAzMDpAIZAl8CPAJwAvsBbAF/AQwCkQJ7A6gDswMtBOkCBgPPAuUD3AI+BE8EwgMOAi0AUf/x/uT+0f+5/0//of5y/kr+Nf7C/Rf97fus+2X7jPt0+7z7r/vW+2D8evx9/Jf8wvxP/Cr7qPv0+9j8jP1e/d786/vE+xv8Kvyb/JT8lPzA/Pr8wvyM/Cj9Nf5y/jD/sP4O/yz/eP+h/9X/t//k/5EApgHlARECCAJAArgBewHWARsCigKoAlQCQgIRAh4CRQIMAgoCNQL9Af8BQgJWAiIC5QErAf0AoABfAKIA2AD5AI8AXwBdAD4AIgDg/5j/3P9k/5T/7/9E/xv/zf78/un+8f5e/7n/eP9k/3b/t/8IAE4APgBfABMBIgHdAB4BCAFLAREBrwDHAJMA3QCKAGwA+P8CAOL/PgCIAHIALQAcAGf/Af9V/9f+sP5g/1P/I/9T/xv/3P7K/vr+XP4Q/t79+v0S/jD++P3E/Xz9Cf7r/aT96/0f/iP+G/7v/U3+A/+F/s/+zf6m/tr+T/+s//z/JAD8/+v/qv/x/0UApACxAA8BwwCmAY4BqQGkAWcBiAE+AjwCOAKRAkUCdwITAucB8gH5AlQDDAKVAXABzQCCAJkBmwFSAbEB6gBjAOoANf97AGoA+v8o/3H/8f9i/5MAUADz/woAOgCh/U4ABAHm/yP/fwCZ/hv/VgCl/6j/0f9E/6z+MQDX/pEBDgJLAC0BfQEj/3H/7gCH/w8BFwHfAKsAswBwAK8A1ACy/3T/nf9G/7MAU/+y/1YAm/7C/tH+N//r/zP++v6x/bv9u/8IAGn/t/+7/iH/I/7i/hL+H//4/qX/RP/a/kr/ev+5/nj+iABG/vz+tf7+/tIAev9DAWoBrP+q/2//sv9WAh4AdwGMAHz/mQC8AIEBOgNWACABPAKH/zQAnADbAAoBJgEKAsQDuADDATD+Ev+s/1wD3wA9/xoAIACD/HcCTwJYANICtf7g/jn+/P6gAG4BwwDC/aH+o//x/2v9PgBHAVP+Sv8S/1f/N/95ACIBkv6l/+P83v5qAHb/fwB5AY4B3P50/ov/4P9CBDP/XP5E/z3+pv55ASsBfwLLALwA0/xwAHb9PgDSAJkAwv8zAfAClv65//L8t//2/fD8IACzBWMDt/2z/dz9t/2mANYBLQHe/8v9EwArAdz89v5UARwBOgByA0399vwcACABIgDg/JkBHAFbACj/UgAkAnABiAHE/XL9SwFR/owElwIB/YwA5/xHA5b/Sv/3AC8Ftf/5AZ4BEP4M/WD/hgNHAFoFmf6h/AYAmf5YBP/9+P4gAFoBdwJp/Qv/9Pvx/88FfQZL++v86f41/5n+XAO1/c0BewKo/4P/cv74/5v9rv6IADf/pgBwAk4Auf3UAT791f06AJD9VgHE/Av/Rv4bBWEACAKRAfT8hvokAb4BVgKBA0r+kQCu/ZL88AFpAp/+IQk1/lz94QAvAbf94wRlAqkBvvji/7P+yP8rAUAGyAgGAE3+XfjY+C7+KwKfChn8tQTA+ev/1gR2/9r5fP3JAJkFdwE6AX379P5P+hUCxQG+AO4BNAC1A8L+uvm5/3ICMQHLARn9jvwKAFQCV//0Bvv5V/82AJD+cgGF/hD+RP+4AaYHCPx39hL/3vr4BPsDywbc/ij7t/xlAV7/dAYm+dgBQP5a/3z/WAMV90cCM//sAkkDwvxJAqb80f8mAmD8jvpp/jEFEP1lAiYEjv3k/5f9Of8V+QH/hQStAmUCev8J/wQCqv1F9Sb9IAIxBRUFbAPh+IP/a/4f/aYE/Af6B/r8XPu89zn/mfoRAfIFMwOX/TD/ewED/5D/Af/t/hsE9vta+9r5hueF773xFQFoEFgR8hRCCdYAAAAcAagEbgAaANYFKAehBkcAFQLdATX6nvhC+1YBn/+6AYn9oPrp+Yf+Gf5y+zQADP57AkcB+gadCLEAcgN0/lwEJgXeBoYC4v5wAOP5Cf/AAHIAZQPr/OcFCgBG/tz6QPlj+u3/Bf3wBDoDRQPCBZ3+e/uH/gb58v32/xj/NwV2/bwBAf3W/Pj/4P9UAqz/+v5c/UQEjgYVBCYCiAQd/Fb7Hf+rAgP+qv4xA6P+If+h/Ej+RwHN/JwA3wFw+dzdr+jM8nb/Yg04ErAK/QPt/Vz8BQUQCbgC4wGzBn4Kfgy5CCIB2PaW7cHwG/l4/1YEqQGCAGf5DPhn/Pz/YwOiAQgA/v81ApkBwAOyCCoJbgQqCYAMMAlaAS7/Yfsz/1f/LwH/BOcEGwV6/PD6kPqx+xv6Af6qBMcDwgKiAX0AyQCKAiH/CgLz/xcANQJUBG4CAAA8AEL+DwGqBJcDugJl/dD4Mfkx/NgBnQRDAUEAXwH0/B3+PgAtAET/rP/5AagDJgJyAJUAEv5oAP7+Uf3E/zwBg/4rAW4C6QMrAZUAA/09/jf+kv5fAYgBqP82AH0B//2D/zD/Kv5y/ev+xQD/AnAEBAHt/jf+z//L/cT/Zf6H/kUBzQCzADgA9v6O/az9Cf4aAOcC2AEbA3IBogDyAisCo//T/0v9Rvxp/FYAQwHhAmMDugDdAGL/0/yM/UcA5P6q/XcAIAKQ/53/0AG2Afz+s/3n/HL+6//bAGkCxwKIAXz+Of50/rv+jP2D/qQA/wDnAdYC4QAB/+v9Jv4PAHsBqwDM/xT/hABwAOL+IAJHAcMAb/8NAFAABgL5ALEAcAH1AKAALP6f/h3/Uf8pAML/xP7N/ab+sQD6/3T+O/9n/yYCcgIqA8ADKwHe/rX8wvvV/WoAqwABA+MC+wAJ/vL7z/wM/df+YQCMATwAkwDk/wgCxQAF/c39qP1p/73/1gEGAZ//Wv3v/RT/Uf7r/ScAXwFLAPsAJAGxAO3/+vyG+n/6gfq++aQBJgSoBf0BwACz/VP9m/3e/rf/dAGf/CL8ZP8RAa7+Bf8U/eX9Ev36/SIAKQFc/6b99v6U/4oAngFJA/0CfP+3+2f6YP3e/hMC8AQKBhMDjAFJAeEA0f4U/rH9U/5A/hb+/v75AX0BGwKOAukCwAAcACsArwCl/24DvAAj/+n+yv8B/aj+VAHfAbL/SwF5AUcAFQAnAED+3v6f/rf+7/37ABUAG/1T/dz9qP97ASADuACZAOv8yfo5/ZUB2ALfAB4ANf1U+uf6zfuf/moASv/n/aj+0AFoANABRQInAO76x/nU+hb+fQBRBH8EVgNn/539wP16/un8A/4S/pj/xwCqA6QDUgJHAdX+Lv7p/B/8af3JAXsA/wJ/Ay0BDALhA+cBafwD/KH8+/pjA0gGhQX2AggDb/6o/hcAsQAXAFQBxwGZ/goALP+5/1QCjgLQAL3+tf9r/wP/dwEFBBkBewARAJD/FP5y/OP5ePvA/jP/1/7wAvICdwCKAkcEQQCb/Gn8n/1WANoBzQGCACP+4/pA+nj8DP5hAAMDGwVaAnkArwDX/6z/SP0S+yT7l/qQ+woANwOIAq0ADwE4Afj9Wv4b/sL9Sv+5/7YBMQR/BPkAuf5a/lr8kPti/5cAXQCZAkkCbAMxA50EYwOKAzUDbgLDADcDfQNCA1QE0gItAAP+FP2Z/pL+VgOdAxAFNwYdBuECagExATEAjPym/aYAswAvAHsDmwTDAYP9z/yX/Cr/zP8XAL3/GP/E/W381gMxBUcDkP3t/Of4sfa89rv8YwBQAeEB9AIkA7EAbf4H/mX6x/mI9xn9oAGxBGoBOgJ0/3/8zfmm/WgAvvzN/i8BBgJJAToCXgSMA2cBbgGtANAB4wHP/yICxwKKA6IEswXtBl8C+QDV/tAAXALJA30EtQWMA44DtgH0AWcBQwEKApcDAwewCKQGxgU+BPAA7/70AS0EKgOxALwBEP+f/iIBswPrBPb/zfv5+Yr40vh0+7v+QQCd/ML6uPls+UT6OvsD/5kCMQJNAjD/fv9W+ST4kfjU+mv7o/7g/Nr8l/qo+pn4xflC/Hj+Cf/R/y0CeQErAEj/CAAH/1f+eP6Q/1X+N//6/Yv/mwHTBIwG/QVjBYQCBgDx/8AB+wP7A2AFqAXpA4gBeQDhACcABAFNAzwFQAZrBssGtwesBjoEFwVvB5sGtQOpAZMB6gB3AmkF9gfeBBEAMfq6+MX20PZ99yn4A/ux+hn71f0h/pEAEP+F/Qr60/sM/pT/if+BAvD6PvaN9U36O/6+/Bb+Kvxg/NT56fwl/zoBh/1C+9r7wvtN/j4AqATYBHAAf/zc/QH+7f/1AE0CVgKGA3IGlwYMBbgBivwv/GL+ywJyBbAHQgkBCFUI/AcyCsIIBwiXBUUBg/9p/38CXAS3CYILzA38DdYPrgwfCn0Enf+J/hj/MP+B/ogAafuG+Lb3EPr2+sD9Vf+F/Az7y/dJ+K330fw++f/6QPoV+zP5wPjf+J716vO49yIAswH/AwQBg/0T89nu5vG99Fr5Wvwh/goAswBWAb4CCAR5AZn85/tX/zYATQHDAC8A/v56/sL9PgFjArgBgQEVA/IFDgVTBjAJuQgoBucEDAUOBsYG9AfiB3AFXARaAuUD/wZNCLkKjw2NDqcKRgmBCJsG0wQdAz4F1AJjAncArwEk/CP0t/Dr7YPqF+qy8RHzlfmJ/rEGAQRaB/8F3QGU/qD6OvdG8SPyxvBm8wb58PoI+mn6z/2H/eD90/+IAHb5qfSt8+P1oPmh/PUATQEx/W77G/xLAEcDFwK6AoYAPgJ/AYwCUQQkA2D/0/0vAP0CRQI8BQcH8AOEAP8BvAUFCD0KoQzXDA0M6wjVCbcIPQh/Br4FrAcsB08ImAnTC5wMtwoJCdkInQaDA+MBXP/5+JPzevH27QfrQfHs9/j7kQEKBjUJDgKy/0L+2ve69MXy1PXH9yL5+P1sANYBTf3w+tj2ffVB84v1dPnE+wn/T/6MAIr7XPvN93v4/fb59476SP7LAOcDTwadBawEKwJaAYIABAHYAPUAhgCS/hv+cf/2/5EAngC2AKYBAQNrBfYGTwi1CawI7wdXCO0IeAnBCxILOwrGCLsIPQcxBooGhQWSBRsGkAhVCL8IUQvXCawHiAJh+szzHe5l6qToA+919wAA3AYUDfMJmQMU/v359fap9ZH2BPc3+sT8jv4CAdIBl/2Z94r2kfQp9Lb14PwJ/nT+l/zi/sD8YfmC96n1Mfgm+Cj8af/jBE0EdAPFAjEEYQLR/iIARwF/AAX+0/6F/979//3e/pMBswDqAOEB1gMUBusGOQhICFMHawVGBjcHUQmlCSwMGA2fCzkLggurDdUMTAwwC5AKoQf/BoMFzQYOB/IEDv5m9WvwKOv46cDs0vZP+RD98gBjBEIDDgLhARn8E/e09uP1+fc3+/sA9v/SAFgAs/0T+pP5f/fB9Vn13/b7+eH5kP3E+j76vPcK96D3XPp//AP+QP9/ASgC1APYBaoEEAQkAQ8As/0m/tz+LP8KAWUDBgMOApsBYQK4AbEABAIGAvgDmQZaB4wH7QedCeQJkAqTDQQPjw/9EDoSGxTnEz4TyxAaC7kHCAMZAz4CkgUuB5IEEPoY88Lt/eqM5oPqWfA/8o/20/4mBe4D4QGzAQYAXvyD/RT99v1c/Ff+g/zi/r79Lv1o92X3w/Lz8TDzIvf0+eH4sfyG+lz8Ev1SAA7/y/0v/Kb7EPsu/R3/kwF7A+MEdAU1BcYEDATNAnkAeP0u/Xr9dv6VAJsBhgEiASsB/QCPAMUCqgPNBI4GTQjkCJIIWwq5CZILIw6NEEMQmRIzFdgUbBKPEOIK+AcIBaYHFwY1CH0ACfMR6BrgAuDv31Xw4fUk/Hj+RwOKAoIAFwNP/JP4w/Rf+OP3ngHNBbcH+AQIA0L+WvqM+9D3P/Li8BLxPfHz9Of6lPzS+C/5Sfgr+Vb7jP7E/LP7/fpN/l8A/wYfCfoHNwVdAfr+5/w1/vD8Bvq1+WP84P4VAYEDewJa/xf9m/zp/AX+MwGKAc0D4gdiCTAKWQoHC08JwghgDA8OnhF9FtUZ5BoUGasTGg3MCOsFZwRUAxUE3faS7L/kPeM94PDmevLF9bj1SfoPAIwA4wW3BxsGg/+h/3r+lP5uBAwGQgI2AML+Gfoi91H6ivmE80HxgvLZ70rxGvW1+Fj5OPkS+x36dvwK+uz5ZfqB/Sr+KwLEBfgG5QUTBHsDEwH9AO3+wP3R/PT7cPz2/icAHgD3AIEBzQGzAJADewKbAtICEAZTCKMJMAujC+YMjQwED8EQlxMXFagXehkFGcQWQxGjDXgHuwfPBQkI2f+a8//oP9/N3ETbr+fg78Hzcvcv+v/91vqF/nr+Mfv59pX69v4IA9oHWwzDCxIGDgVYAJ39Hvnf9w/yXO5a7nzvd/Fb9s/6pvrf+C36sfdZ9ZP1mfnS+d76sv+VA1gFYghKCnIHAwQ8BKgCDv5//p3+qP3E/BkBhgJLAQECdwLg/5v+7f4mAeoBSQUOB4kJ/gvmDb8PkxE6EtkPWw8PERAVGRiYHBMfqx3/F8sRqQyhCOAFvAQw/vjwUuUO3CTZJNc94q/qOfGL9Sr8b/8m/QYAywFT//36KPzA/uUALAYwCpgL3AfRBncAnf1C+5f4QfPE8LDy6vG68/n4uf3l+9P8tfxE/RH4Kvpa+qT85fyXAFYC7QUZCMQJowfTByMHRwQgART/rP0V+w79cgAQA5sCTQIMAon/A/4F/c/+bgH5AhkHVwgUDHMLvQ0NDyUQUhDFEMASlxV/GCEc4hzvGBAV+w58C2MFUwZPAvv4f+x141jbMdVN2uTiUenP7Db2+/vI/UcAwAKZAkT96fyO/NP/9APnCGsLaAx6Cr4G8AFK/sn4ZvMF8ZbuyO4Q8A31J/eb+U35wPmk+Nz5y/g4+dT5rPt//fAAKAVCB9MI2QgjCX8HywYbBZ4CyP/C/GX8M/2U/lAAtgCEAdf/b/+l/2EAHAGeAekDhQZxCNUKOQwgDnwMjw0lDWgPtA46EVQUTRmYHMgcqhprF3ARbQs1B+kDWvqK67/hNdrA1ZnVB9+G6Crt4PEE+Y77y/ty+sT9NfpL+Yz48/9nBIUKPQ5DEbEQ4guwB1oBHf3B9TvxJe+n8IvwW/K/86T2YfST9av29fcc9t31bPaZ92P4wP0mArMGlAmcCw8NjwswC2cH8gMvAe//G/9K/xEB1gHlAJUBmQEcAMb9A/5P/On6Bfvn/W3/eQKuB/UKOA1DDlkP5A3XDToPeQ/FELgTQhi5GaUZKhn0FcMREw69CmkD1fIK54Pbsde603TcWOZr6RvsDfKx+Zv5wPtR//D85fei+Zj/AQV8Bw8OZRGvEcEPxQ7xC0QGT/46+ATyT+/P7fbuC/EE9ND1QfaC9t325vSl8+HzwfX595X61/6OAukGlglvC3EMpQtTCc0EOAL8/z3+8f6gAEUDPAJUAuEBqwGf/7f+Xv2Z+4z6iPtc/rn+8gK8BRIJLgkADHkNtgxiDJgN6g2NDpcRDBajGIUYZBniF4oV8A+LDeQH1f657t/l4N0h2/bYo+FN6DnrKu8T90L9Ffsx/DH9iPts9/T5ZP+TArwFXguNDqIPzA9bEIIMMQaJ/1T5UPV383XyMPOL8631uPYZ+dr5M/iI9gT1zvSp9B73FfnE+z3+QgMfBnEI4AnXCTcIWAXYBDwD4wE+AasCaQLJAowCEwPdARcBMQDr/RX7Z/pp+2X9jv6pANYDkgXxBy4JaAqjCcoIuQkFCm0Ltg17Ef0V7xm7HHwcuRlJFmwRUQvaAYj34u2E5SXfwtyR4iDoruvz75729/gk9sH2Lflu9hP0hPYd/Nz+6wOEDK0RsRMVFYUVwxFMDHoHNQIV+634qfZQ9o/1yfd0+VT66fq++cf1ZvMu8X7wsu9V8hX1UPhC/j4EOwlZC7QNFg3vCpkHgwW8Ap4BogEGAtYCvgOOBSIFAwRwAmMAKv54+4r6jPmv+bX6XP3k/44DuwbtCIkJIQnICM0HKAgUCP4HawlBDPUPwBNpFv8XRBckFckRoA1rCHL+mvRL7JXmVeI446bng+o169XuI/Oe9Tb2Rfib+Ev2nPSM9z77RP92BOsJDw7zD8sSsxIeEaMNYggKA4X+cPtU+Pf2R/jw9nv2E/dN+Nv23fQ29YTzzvE78D/x3fLJ9bf6lv/PA8gGQgklCkwJ+gd/BgwFWAPFAq8BGQLsArwD2gPNA7wDoAGGALD++vzJ+pn6Mfqo+k/8DwAdAyEGhQjrCZ8KcQoaCuQJuwmSCusKAA3ODnIRSxMXFHYUlRKEEL8LfwTh+g/y2usV5hzk3+W86OfqQu5m9FD29/hj+tr7s/hQ91T4Tfm1+5//egW7CAcNeRBFEwwT/RErDoMIxwPt/338s/iR98X23fXb9m759PmV+Gz3X/bv86HxV/L88sfzC/Wz+ET94wDyBGcIaApcCSEI1wdNBzMFDgSxAq0BcABhAWUChgIIAgIBuf++/QH8Gfoo+ZX4uPhu+aj7I/5BAEkCbgT6BcgGhwd6CFUI6QeJBx8HYAfnB+8I/gjKCMgICwgfB9wFUQVuA+EBpgDDADYA2f/e/5L/gf28+tD4nPZ39JHynPMP9Fn1EfbS+JD6bfxa/tQA0gHJAV0BAgGEAJD/dQDYAIEBqQE6AuwCUgPCA4MDIAKrAMb+Jv3/+gz6e/k++er4JPly+Sj6G/uM+w78JPxe/LX7e/uQ/F78S/xr+8T80/xp/QX/4QFPBOkEjgYzBx8HcAWOBcAEtwOVAkcDCgTaBLsGsAg9CqULQw3vDd8NHg6lDjgOHg5dDpgOLQ40Dt8MdgdqAM35avSw7Znqr+rC6vTpLOtM8NfyIvWr99j6nvmk+Hv5ZfrA+Zn6vv2L//sBCgYcC9ANTA9WEP0PUg7IC7MHPAM7/0L8Ivmr9wj4zfig+P/4zfnN+ST53fct9yn2vvYg9zb4bPmk+3/+7gAdBG8HOQq/Cy4MBwwWC9UJnQg1B8sFxAQVBDUDrQL2ApUCgQGGANf/Wv7r/J37ivok+Z74w/iG+Z36Evy7/az+zP+ZABsCyQJHA/8CJAJnAfsAKQEvARUBJAGkAfkByQEeAtgB8ABG/8L+T/4d/YP8sfza/U/9yP38/ncAkQBUAVQCVAIPARUB8ABsAJ3/z/9bAC0AmQBlAcMBogF9AtQCTwKrAdIB0ADv/x3/1/4z/tj9G/4W/gX+If4l/+L/+v+b/+n/TgBUAEcACgDm/yz/1/7t/mT/M/83/5b/lP+//8z/RwAPABEAyP+Q/03/KP/G/jn+Hf5e/rv+V//t/18AyQD9AGcBigECAVIALQA6AFsAJwBfALgANgGvAYQCswJ5AmEChgJCAo4BCAHHAFsAGgAPAPz/5P/x/1gAoAB5ABoA0f+J/zP/pv5A/qT9g/2B/Y79dv1e/Yf9of3c/Vf+O/4u/sb9m/3i/bP9+P0w/l7+Nf5//lP/Yv+9/9z/KwAGAJEA6AArAckAmQCXAOwAJgFAAWoBcgFHAVQBngHlASICxwGEAVoBcAGvAUUB9wDSAJEAjwC8ANYA6gDfALwAugATAeUAsQCXAHUAUAAkAB4AJACo/yX/5P7I/lr+Tf5e/ij+wP1l/TP91vy+/E/8Wvwi/Az8//tt/LX8JP3A/fb9XP69/vr+Sv+Q/5//z/8NAFYAQQBhAKsA+wBlAbEB5QEZAk0CVALWAuUCmQKtAukC+wKoAv0CIgMxAxADNwNsA3sDYQNFAygD9AK8ApcCaQKzAfUAigAgAGv/of7//R/9QPy++6L7Y/tC+y37ufsQ/HD84Pxe/dj93v38/SP+RP5g/tX+3P4d/2//8f9BAH0AtgDJAOoA6ADUAH8ANgDi/7L/Z/8j/+3+1/63/nr+Zf6Z/sb+xP4F/2T/i//m/yAAfwDHAAYBSQFyAZEBsQHHAdQBvAHfAeEBxQGOAVIBJAE8AeMAhABWAEEAbAA2AIQA5QD1AO4AWgHhAboB0gFaAogCLwJAAqgCiAIrAmMCigLsAX8BcAFQAa0AIAAEAIX/t/5P/ij+of1L/XT9g/1E/TP9wP0Q/hD+of43/1f/Ev+J//b/vf+q/w8AcgB1ALgAMQFLAWUBbAGeAYgBWAFFASABuAB/AFsA/P/M/7D/hf9I/yj/H/8W/9P+4P7g/q7+rv7I/u3+B/8f/zf/SP9C/23/wv/r/+3/FwBbAH8AiACVALoA6gAZAT4BMQFSAVgBhAFuAWcBRwEkARUBDwHSAK8AswCTAHcAWwBlAEsACgDx/8//u/+W/4f/dv8o/yr/Dv/e/uL+2v7n/sr+uf7R/v7+Fv8S/2T/m/+H/6P/3v/+/+///P8iADgALQBSAFQAQQBOAGUAiABjABUAKwA+ABoA9v/z//b/1//P/97/+v8AAAQA9v/4/+//1//I/9z/7//i/9P/v/+3/9z/7//k//7/GgAaAPj/3v/p/+T/xP+//97/3P/g/+n/7//T/+b/+P8NAB4AKQA6AC0AJAAiAC0APABOAGwAbABDADoARQBBAEsAKQBYAFsAUgBOAEUAPgAtACQAEwDv/+3/5v/r//P/8f/z//P/BAAAAP7//P/i//j/CgACANn/2f/c/8r/1//T/8//yv/e/+L/xv+//7X/yv/e/+L/3P/V/+L/9v/m/+L/0f/C/8//6f/8//b/AgAiABwABgAKAAgALQBFAFIAaAB5AF8APgA4AEEAUgBbAEkAOgBLAE4AOAARAAAADQARABMABgAPAA8AEQAGAOb/wv/R/+b/AgD8/+T/3P/e/97/3P/Z/9f/yv/V/+//5v/+//j/7//k/+D/2f/t/+3/7//4/wIADQAkACsAJwAeACQAGgAeAAgAEwAtAB4AFQAeAC8AHAAaACIAFwD+/xMACgAIAAoAAAD2//7/7//z//z//v/z//j//P/6//P/DwAaAAYABAARACIALwA2ADoAJwAgAA8ADQAPACQALQAnABUA+v/i/+v/8f8AAAgADwD4/9n/xv+7/6r/qv/C/73/wv/M/8b/rv+5/7//xv/I/8z/1f/C/8j/1//X/+T/5v/4//r/8f/X/9H/zP/G/7v/tf+5/7L/tf+q/7L/tf/G/8r/v/+//9H/z//P/8b/0//X/9z/zP/G/9n/0f/M/9H/yv/R/9n/0f/G/6z/rv+9/8L/2f/g/9P/wv+y/7f/xP/X/+n/3v/Z/97/3v/4//z/AgD+/woAAAD6//H/1f/V/9z/5v/6//j/+v/z//P/7f/z//P/7//v//H/6//v/wQAEwAGAAQACgARABEAGgAKAPr//P8IABEABAD+//b/AAAIAAoABAD4//H/9v/2/wIACAACAAQA+v8EAAIACgAIAA0ACAAIAA8AGgAgACQAHgAeAA0AAgAXABcADQAGAAQABgAGAAQAAgAKAAIA/v/r/+//+v/6//z/7//x//b//v8EAAoACgANABcABAAVABUAGgAgAB4ADQAIAAQAAgD+/wIABAAAAAAAAgARAAQA/v/+//z//P8AAPr//P/+/wIA/P8EAPr/8f/4//r/9v/6/wIA/v8TAA8ADQAPABEACAACAAIABAAGAA0ACgAVABcAIgAeAA0ABgANABUAEQAeAB4AKQApACAAHAAaAC0ANAAxACsALwA+AEcARwBFADwAPAAnACIAIAAGAAYACAAGAAgAGgAgACsAKQAvAC8AJAATAA8AGgAXABwAHAAkABoAFwAVABUAEQAKAA0AAgAPACIAIgAaABUAGgAtAEMAPAA2ADEANgA4ADgAQwAvABwADQATACcAJAAiACAAIgApADQANAApADQANAA2ACcAKQAtADQANgA6ADQAJAAaAA0AAAD+//7/AgAAAAAA/v8TACcAIAAcABEACgAIAPj//P8CAPz//v/8//z/6//x//j/4v/Z/+n/6//+//z/5v/g/+n/9v8AAPH/8//8//j//P/8//b/+v8CAAQA8//2//P/+P/i//P/8/8AAPj//P8GAAYADQATAAoABgAVAB4AJAAiABwAFwAkACAAKwA4ADYAHgAKAAgA/P8GAPz/EwAkACcACgAEAA0AFwAaABcABgAAAAYACAD+/wYAGgATAAAAEQAXACcALQAaABMAHgAVABMAFQAPAP7/AAD6/wYADwD2/wQADQD6//b/+P/m/+D/3P/p/9f/0//x//P/7f8AABMA+v/x//H/3v/r/+v/3v/4/wIA/v8CAPP//v8CAPz/BgACAPz/BAAIAAIA9v/8/w0AFwAXABUAEwAAAOn/6//v/w0ABADr//j/CgAAAPb/6f/p/97/z//R/+v/4v/8/wQA+v/6/+3/AgAnABEAAgDv/+n/2f/e/9P/1//c/9f/z//c/+T/+v/V/8b/xv/M/9n/xP+//7//4P/P/8L/xP/E/+T/1f/p/+//+P/m/+//6f/6//r/6f8EAPj/AAAxAEUAMQAtAB4ANgAvABwALwA+ACQABgACAPz//P8IAAIABADr/9n/9v/X/+T//P/x/wYAFQAAABEA+P/k//r/6//E/8r/3P/P/7D/t//I/9P/zP/X/9P/0f/T/8T/1f/4/+T/pf/E/9n/wv/X/9X/1//g/73/wv/x//H/6//t//b/EwAAANf/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IgAVABcAHAAcAB4ABgDp//r/BgAGAAQAAgATABwAAgAIAA8A9v/8/wIA/P8CAAYAFQARABUABAAEABEADQD+/w8ABAAKAA8ABgANAB4AIAAiAC0AHAANAA0AEwATAAgAFQApACIAFwAxACkAGgAEABUADwAnABUAFQARABwAIAARABoANAAkABEAJAAaABcA/v8rACAAQQA0ADoAQQAnAC0AHgAeACIAFwAcAAgAJwAtABEACgARABEAGgAcABMA/v8CAAgAHAAgAAAAAAAIAAQAEQAPABMACAAIAA0AEwARABcADwANABwAHAARABMADQAVABoAEwAEAAoACAAAAAIADwAKAAgACgAPAAAA8/8AAA8ADQAGABUADwAXAA0ADwACAAIABAACAAQA/v/z/wAABgANAP7/+P8CAAYAAAD+/wAACgAAAAAACgAGAAoACAAEAAIAAAAKAAQABAATAAIAEwARAAAA9v8AAAQADQAIAAgADQAKAAIA9v8GAA0ABAACAAoAAAAEABEACgAKAAoAAgD+//j/AAAAAPz/BAACAPr//P8IAAgACgACAPP//P/z/wAAAAD2//b/AgDx/+v/+P/+//j/8f/t//j/+v8AAPH/5v/t//P/6f/m/+//6f/k/+v/5P/p/+n/7//t/+//7//2/+b/9v/2//b/AAD8//j/+P/2//j/+v/r//r/8f/4//r//v/2//j/6//v/+3/8//6//P/7//4//z/6//2//z/8f/2//H/7//t//b/7f/4/+3/3v/t/+//6//x/97/4v/p/9f/1f/Z/+D/6f/r//j/8//c/9f/2f/k/+T/4P/m/+n/5v/p/+T/4v/g/+n/5v/X/+n/3v/m/+T/7//v/+3/7f/k/9z/4v/i/97/4v/z//j/6f/m/9z/4v/k/+b/6//x/+n/6//e/+b/7//z/+v/7f/v//P/7f/m/+D/7//+//r/9v/v/+n/6//x//H/8//z//H//v/6/+3/+v/z/wAAAAAAAPz/AAD+/wAAAAACAAAABAAEAPj/8f/6//P/AAD+/wQAAgAAAAAAAgAAAPb/+P8AAP7/AAACAAIAAgAEAAQAAAAIAAQABAAAAA8ADQANAAIACAAGAA0AEQAIAAIAAgD+/wQACgAGAAAABAAIAAYA/v8AAP7//v8EAAoA/P8EAPr//v8AAP7//P8CAAQA/v/+//7/AgD+//7//P/+/wAAAAAAAP7/AAD+//7//v8EAAAAAAD+/wAACgAGAAAAAgACAAYABgAEAP7/CAAGAAgAEwAIAAQABgAPAAYACgACAAQAEQACAAQACAAAAAAABAACAP7/AAAGAAIABgACAAQAEQARAAYAAAAIAAgACAAAAAIABAD+//7//v8EAAgAAgACAAAAAAAAAAIA/v/+/wAA/v/8//z//v8AAAAAAgACAPz//P/8/wAA/v/+/wAAAAAAAAIAAgACAP7//v/6//z//P8AAAIAAAAAAAQABgAKAAIAAgACAP7/AAAAAAgAAgACAAgAEQAaABMABgD+/wgABgAAAAQAAgACAAIABgAEAAQACgAIAAQABAAIAAQACgACAAAAAAAAAAAAAAD+//7/AgACAAIAAgANAA8ACgAAAAQABgACAP7/AgAIAAQAAgACAAIAAAACAAQAAAAAAPz//v8CAAQAAgAPAAoACAAEAAAACAAIAAgAEQAKAA0ACAAKABEACgANAA8AEQAKAAoADwAKAA8ADQAXABUADQAGAAAAAgD+/wYACAAEAAoABAAAAAIAAgAAAAIA/v8AAAIABAAAAPz//v/+/wAAAAD8//r/8//2//z/+P/x/+//8f/z//b/+v/p//P/9v/z/+//8f/4//r//P/2/+//7f/x/+v/6//2//b/7//4/+v/8//8//P/9v/6//j/+P/8/wIAAAACAAAA/v/+//7//v/8//7//v/4//7/+P/+//z/+v/6//r/8//2//b//P/8//j/AAD+//H/9v8AAPr//P/8//z/AAAAAAAAAAAAAP7//v8AAP7//P/+//j//P8CAPr/+v/6//j/7//6//z/+P8AAPz/AAAAAAAAAgD8//r/+v/+/wAAAAACAP7/AgAAAAAA/v8AAAAA/v8AAAAAAgAAAAAAAAACAAAA/P8AAP7/AAAAAAAAAgACAAIAAAACAAIAAAAAAAAA/v/+/wAABgAEAAIABAAEAP7//v/8/wIAAgAAAAIABgAGAAYAAAAAAAQACAAIAAgAAAD+/wYABgAEAAIAAAAEAAQACAAGAAYABAAEAAIAAgACAAIAAAACAP7//v/+//7//v8AAAIAAAAAAAAA/v8AAAAA/v8AAAAAAAAAAAAAAAACAAAA/v8AAP7/AAAAAPr/AgAEAAIA/v8CAP7//v8CAAgABAD+/wYADQAIAAoABAAAAAIABAD+/wIABAAAAAYABAACAAAAAAACAAAA/v/+/wAA/v8AAP7/BAACAPr//P/+/wAA/P/8/wAAAAD+/wIA+v/8/wIABgACAPr//v/8/wAAAAAAAAAAAAAEAAIAAAACAAQAAgD+//7//v8EAAIAAAD+/wIAAgAAAP7/AgAAAAYA/P/z/wIABAAEAAQA/P/4//j//P8AAAAABAAEAAIAAgAAAPz//v8GAAYA/v/+/wAA+v/+/wgAAgAAAP7//v/6//7/BAAAAP7/BAD+//7/AAAAAAAA/P8TADEA0f/C/xMAJAAAAPr/EQAPAPb/5P8NAA8A8f8IAAIA9v/g/woADQD+/wIAAAD8//H//P8CAP7/BAAKAP7/+v8CAAAA7f8CABcABgDv//r/AAAAAAQADwAEAAQA/v/8//r/CAAEAAAAAAACAAgA+v/4/wIABAAAAAAACAD+/wIABgAAAPr/+v/+/wAAAAACAP7/AAACAAIA+v/4//7/AAAAAAIA+v8AAAQAAgACAP7/AgD+/wQAAgAAAP7/+v/+/wIAAAD6//r/BAAAAAAABAAAAP7/AAD+/wAA/v/+/wAA/v8AAAAAAAD+/wAA/v8AAAAAAAAEAAIAAAAAAAAAAAAAAP7//P8AAAgAAAAAAAAAAAAAAAoABgD6//r/AgACAAIAAAACAAAAAAAAAPz//v/6/wAAAgAAAP7/AAAAAP7//v8AAAAA/v8AAPz/AgAEAAgAAgD+//7/AAD6//z//v/+/wIAAgAAAAAAAAAAAAAAAAAAAP7//v8AAAIAAAACAAAAAAAAAP7/AAACAAgAAgD+/wIAAgACAAIAAgAAAP7/AAD8//7/AAAAAP7/AAAAAAAAAAAAAAAAAgACAP7//P/+//z//v8CAAIAAAD+//7//v/+//7/AgACAAIAAAAAAAAA/v8AAAAAAAD+/wAA/v8AAAAACAAIAAAA/v/8//z//v8CAAAAAAAAAAYABAAAAP7/AgAAAP7//v/+/wAACAACAAAAAAACAAAA/P/+//7/AAAAAAIAAAACAAIAAAAAAAAA/P/+//7/AAAAAAIAAgD+/wIA/v/+//z//v/+/wAABAACAAAAAAD+//7/AgACAAIAAgAAAPz/AgACAAQABgAIAAIAAAAAAAAAAAAAAAIABAAEAAAA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" type="audio/wav" />
        Your browser does not support the audio element.
    </audio>




.. code:: ipython3

    core = ov.Core()
    
    compiled_model = core.compile_model(model=quantized_model, device_name='CPU')
    
    input_data = np.expand_dims(audio, axis=0)
    output_layer = compiled_model.outputs[0]

Next, make a prediction.

.. code:: ipython3

    predictions = compiled_model([input_data])[output_layer]

Validate model accuracy on dataset
###############################################################################################################################

The code below is used for running model inference on a single sample
from the dataset. It contains the following steps:

-  Define ``MetricWER`` class to calculate Word Error Rate.
-  Define dataloader for test dataset.
-  Define functions to get inference for PyTorch and OpenVINO models.
-  Define functions to compute Word Error Rate.

.. code:: ipython3

    class MetricWER:
        alphabet = [
            "<pad>", "<s>", "</s>", "<unk>", "|",
            "e", "t", "a", "o", "n", "i", "h", "s", "r", "d", "l", "u",
            "m", "w", "c", "f", "g", "y", "p", "b", "v", "k", "'", "x", "j", "q", "z"]
        words_delimiter = '|'
        pad_token = '<pad>'
    
        # Required methods
        def __init__(self):
            self._name = "WER"
            self._sum_score = 0
            self._sum_words = 0
            self._cur_score = 0
            self._decoding_vocab = dict(enumerate(self.alphabet))
    
        @property
        def value(self):
            """Returns accuracy metric value for the last model output."""
            return {self._name: self._cur_score}
    
        @property
        def avg_value(self):
            """Returns accuracy metric value for all model outputs."""
            return {self._name: self._sum_score / self._sum_words if self._sum_words != 0 else 0}
    
        def update(self, output, target):
            """
            Updates prediction matches.
    
            :param output: model output
            :param target: annotations
            """
            decoded = [decode_logits(i) for i in output]
            target = [i.lower() for i in target]
            assert len(output) == len(target), "sizes of output and target mismatch!"
            for i in range(len(output)):
                self._get_metric_per_sample(decoded[i], target[i])
    
        def reset(self):
            """
            Resets collected matches
            """
            self._sum_score = 0
            self._sum_words = 0
    
        def get_attributes(self):
            """
            Returns a dictionary of metric attributes {metric_name: {attribute_name: value}}.
            Required attributes: 'direction': 'higher-better' or 'higher-worse'
                                 'type': metric type
            """
            return {self._name: {"direction": "higher-worse", "type": "WER"}}
    
        # Methods specific to the current implementation
        def _get_metric_per_sample(self, annotation, prediction):
            cur_score = self._editdistance_eval(annotation.split(), prediction.split())
            cur_words = len(annotation.split())
    
            self._sum_score += cur_score
            self._sum_words += cur_words
            self._cur_score = cur_score / cur_words
    
            result = cur_score / cur_words if cur_words != 0 else 0
            return result
    
        def _editdistance_eval(self, source, target):
            n, m = len(source), len(target)
    
            distance = np.zeros((n + 1, m + 1), dtype=int)
            distance[:, 0] = np.arange(0, n + 1)
            distance[0, :] = np.arange(0, m + 1)
    
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    cost = 0 if source[i - 1] == target[j - 1] else 1
    
                    distance[i][j] = min(distance[i - 1][j] + 1,
                                         distance[i][j - 1] + 1,
                                         distance[i - 1][j - 1] + cost)
            return distance[n][m]

Now, you just need to decode predicted probabilities to text, using
tokenizer ``decode_logits``.

Alternatively, use a built-in ``Wav2Vec2Processor`` tokenizer from the
``transformers`` package.

.. code:: ipython3

    def decode_logits(logits):
        decoding_vocab = dict(enumerate(MetricWER.alphabet))
        token_ids = np.squeeze(np.argmax(logits, -1))
        tokens = [decoding_vocab[idx] for idx in token_ids]
        tokens = [token_group[0] for token_group in groupby(tokens)]
        tokens = [t for t in tokens if t != MetricWER.pad_token]
        res_string = ''.join([t if t != MetricWER.words_delimiter else ' ' for t in tokens]).strip()
        res_string = ' '.join(res_string.split(' '))
        res_string = res_string.lower()
        return res_string
    
    
    predicted_text = decode_logits(predictions)
    predicted_text




.. parsed-literal::

    'it was almost the tone of hope  everybody will stay'



.. code:: ipython3

    from tqdm.notebook import tqdm
    
    import numpy as np
    
    
    dataset_config = {"data_source": os.path.join(DATA_DIR, "LibriSpeech/test-clean")}
    test_data_loader = LibriSpeechDataLoader(dataset_config, samples_limit=300)
    
    
    # inference function for pytorch
    def torch_infer(model, sample):
        output = model(torch.Tensor(sample[1]['inputs'])).logits
        output = output.detach().cpu().numpy()
    
        return output
    
    
    # inference function for openvino
    def ov_infer(model, sample):
        output = model.output(0)
        output = model(np.array(sample[1]['inputs']))[output]
    
        return output
    
    
    def compute_wer(dataset, model, infer_fn):
        wer = MetricWER()
        for sample in tqdm(dataset):
            # run infer function on sample
            output = infer_fn(model, sample)
            # update metric on sample result
            wer.update(output, [sample[0][1]])
    
        return wer.avg_value

Now, compute WER for the original PyTorch model, OpenVINO IR model and
quantized model.

.. code:: ipython3

    compiled_fp32_ov_model = core.compile_model(ov_model)
    
    pt_result = compute_wer(test_data_loader, torch_model, torch_infer)
    ov_fp32_result = compute_wer(test_data_loader, compiled_fp32_ov_model, ov_infer)
    quantized_result = compute_wer(test_data_loader, compiled_model, ov_infer)
    
    print(f'[PyTorch]   Word Error Rate: {pt_result["WER"]:.4f}')
    print(f'[OpenVino]  Word Error Rate: {ov_fp32_result["WER"]:.4f}')
    print(f'[Quantized OpenVino]  Word Error Rate: {quantized_result["WER"]:.4f}')



.. parsed-literal::

      0%|          | 0/300 [00:00<?, ?it/s]



.. parsed-literal::

      0%|          | 0/300 [00:00<?, ?it/s]



.. parsed-literal::

      0%|          | 0/300 [00:00<?, ?it/s]


.. parsed-literal::

    [PyTorch]   Word Error Rate: 0.0292
    [OpenVino]  Word Error Rate: 0.0292
    [Quantized OpenVino]  Word Error Rate: 0.0422


Compare Performance of the Original and Quantized Models
###############################################################################################################################

Finally, use `Benchmark
Tool <https://docs.openvino.ai/latest/openvino_inference_engine_tools_benchmark_tool_README.html>`__
to measure the inference performance of the ``FP16`` and ``INT8``
models.

.. note::

   For more accurate performance, it is recommended to run
   ``benchmark_app`` in a terminal/command prompt after closing other
   applications. Run ``benchmark_app -m model.xml -d CPU`` to benchmark
   async inference on CPU for one minute. Change ``CPU`` to ``GPU`` to
   benchmark on GPU. Run ``benchmark_app --help`` to see an overview of
   all command-line options.

.. code:: ipython3

    # Inference FP16 model (OpenVINO IR)
    ! benchmark_app -m $ir_model_path -shape [1,30480] -d CPU -api async


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2023.1.0-12050-e33de350633
    [ INFO ] 
    [ INFO ] Device info:
    [ INFO ] CPU
    [ INFO ] Build ................................. 2023.1.0-12050-e33de350633
    [ INFO ] 
    [ INFO ] 
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(CPU) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 61.48 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     inputs (node: inputs) : f32 / [...] / [?,?]
    [ INFO ] Model outputs:
    [ INFO ]     logits (node: logits) : f32 / [...] / [?,?,32]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [ INFO ] Reshaping model: 'inputs': [1,30480]
    [ INFO ] Reshape model took 28.87 ms
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     inputs (node: inputs) : f32 / [...] / [1,30480]
    [ INFO ] Model outputs:
    [ INFO ]     logits (node: logits) : f32 / [...] / [1,95,32]
    [Step 7/11] Loading the model to the device
    [ INFO ] Compile model took 644.15 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: torch_jit
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 6
    [ INFO ]   NUM_STREAMS: 6
    [ INFO ]   AFFINITY: Affinity.CORE
    [ INFO ]   INFERENCE_NUM_THREADS: 24
    [ INFO ]   PERF_COUNT: False
    [ INFO ]   INFERENCE_PRECISION_HINT: <Type: 'float32'>
    [ INFO ]   PERFORMANCE_HINT: PerformanceMode.THROUGHPUT
    [ INFO ]   EXECUTION_MODE_HINT: ExecutionMode.PERFORMANCE
    [ INFO ]   PERFORMANCE_HINT_NUM_REQUESTS: 0
    [ INFO ]   ENABLE_CPU_PINNING: True
    [ INFO ]   SCHEDULING_CORE_TYPE: SchedulingCoreType.ANY_CORE
    [ INFO ]   ENABLE_HYPER_THREADING: True
    [ INFO ]   EXECUTION_DEVICES: ['CPU']
    [ INFO ]   CPU_DENORMALS_OPTIMIZATION: False
    [ INFO ]   CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE: 1.0
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'inputs'!. This input will be filled with random values!
    [ INFO ] Fill input 'inputs' with random values 
    [Step 10/11] Measuring performance (Start inference asynchronously, 6 inference requests, limits: 60000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 69.35 ms
    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            2748 iterations
    [ INFO ] Duration:         60151.82 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        131.23 ms
    [ INFO ]    Average:       131.13 ms
    [ INFO ]    Min:           67.66 ms
    [ INFO ]    Max:           145.43 ms
    [ INFO ] Throughput:   45.68 FPS


.. code:: ipython3

    # Inference INT8 model (OpenVINO IR)
    ! benchmark_app -m $quantized_model_path -shape [1,30480] -d CPU -api async


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2023.1.0-12050-e33de350633
    [ INFO ] 
    [ INFO ] Device info:
    [ INFO ] CPU
    [ INFO ] Build ................................. 2023.1.0-12050-e33de350633
    [ INFO ] 
    [ INFO ] 
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(CPU) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 81.97 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     inputs (node: inputs) : f32 / [...] / [?,?]
    [ INFO ] Model outputs:
    [ INFO ]     logits (node: logits) : f32 / [...] / [?,?,32]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [ INFO ] Reshaping model: 'inputs': [1,30480]
    [ INFO ] Reshape model took 35.47 ms
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     inputs (node: inputs) : f32 / [...] / [1,30480]
    [ INFO ] Model outputs:
    [ INFO ]     logits (node: logits) : f32 / [...] / [1,95,32]
    [Step 7/11] Loading the model to the device
    [ INFO ] Compile model took 920.18 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: torch_jit
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 6
    [ INFO ]   NUM_STREAMS: 6
    [ INFO ]   AFFINITY: Affinity.CORE
    [ INFO ]   INFERENCE_NUM_THREADS: 24
    [ INFO ]   PERF_COUNT: False
    [ INFO ]   INFERENCE_PRECISION_HINT: <Type: 'float32'>
    [ INFO ]   PERFORMANCE_HINT: PerformanceMode.THROUGHPUT
    [ INFO ]   EXECUTION_MODE_HINT: ExecutionMode.PERFORMANCE
    [ INFO ]   PERFORMANCE_HINT_NUM_REQUESTS: 0
    [ INFO ]   ENABLE_CPU_PINNING: True
    [ INFO ]   SCHEDULING_CORE_TYPE: SchedulingCoreType.ANY_CORE
    [ INFO ]   ENABLE_HYPER_THREADING: True
    [ INFO ]   EXECUTION_DEVICES: ['CPU']
    [ INFO ]   CPU_DENORMALS_OPTIMIZATION: False
    [ INFO ]   CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE: 1.0
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'inputs'!. This input will be filled with random values!
    [ INFO ] Fill input 'inputs' with random values 
    [Step 10/11] Measuring performance (Start inference asynchronously, 6 inference requests, limits: 60000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 52.31 ms
    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            4500 iterations
    [ INFO ] Duration:         60105.34 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        79.88 ms
    [ INFO ]    Average:       79.99 ms
    [ INFO ]    Min:           47.16 ms
    [ INFO ]    Max:           106.32 ms
    [ INFO ] Throughput:   74.87 FPS

