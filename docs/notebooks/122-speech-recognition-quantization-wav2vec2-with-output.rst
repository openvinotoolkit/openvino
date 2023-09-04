Quantize Speech Recognition Models with accuracy control using NNCF PTQ API
===========================================================================



This tutorial demonstrates how to apply ``INT8`` quantization with
accuracy control to the speech recognition model, known as
`Wav2Vec2 <https://huggingface.co/docs/transformers/model_doc/wav2vec2>`__,
using the NNCF (Neural Network Compression Framework) 8-bit quantization
with accuracy control in post-training mode (without the fine-tuning
pipeline). This notebook uses a fine-tuned
`Wav2Vec2-Base-960h <https://huggingface.co/facebook/wav2vec2-base-960h>`__
`PyTorch <https://pytorch.org/>`__ model trained on the `LibriSpeech ASR
corpus <https://www.openslr.org/12>`__. The tutorial is designed to be
extendable to custom models and datasets. It consists of the following
steps:

-  Download and prepare the Wav2Vec2 model and LibriSpeech dataset.
-  Define data loading and accuracy validation functionality.
-  Model quantization with accuracy control.
-  Compare Accuracy of original PyTorch model, OpenVINO FP16 and INT8
   models.
-  Compare performance of the original and quantized models.

The advanced quantization flow allows to apply 8-bit quantization to the
model with control of accuracy metric. This is achieved by keeping the
most impactful operations within the model in the original precision.
The flow is based on the `Basic 8-bit
quantization <https://docs.openvino.ai/2023.0/basic_quantization_flow.html>`__
and has the following differences:

-  Besides the calibration dataset, a validation dataset is required to
   compute the accuracy metric. Both datasets can refer to the same data
   in the simplest case.
-  Validation function, used to compute accuracy metric is required. It
   can be a function that is already available in the source framework
   or a custom function.
-  Since accuracy validation is run several times during the
   quantization process, quantization with accuracy control can take
   more time than the Basic 8-bit quantization flow.
-  The resulted model can provide smaller performance improvement than
   the Basic 8-bit quantization flow because some of the operations are
   kept in the original precision.

.. note::

   Currently, 8-bit quantization with accuracy control in NNCF
   is available only for models in OpenVINO representation.

The steps for the quantization with accuracy control are described
below.



.. _top:

**Table of contents**:

- `Imports <#imports>`__
- `Prepare the Model <#prepare-the-model>`__
- `Prepare LibriSpeech Dataset <#prepare-librispeech-dataset>`__
- `Prepare calibration and validation datasets <#prepare-calibration-and-validation-datasets>`__
- `Prepare validation function <#prepare-validation-function>`__
- `Run quantization with accuracy control <#run-quantization-with-accuracy-control>`__
- `Model Usage Example <#model-usage-example>`__
- `Compare Accuracy of the Original and Quantized Models <#compare-accuracy-of-the-original-and-quantized-models>`__


.. code:: ipython2

    # !pip install -q "openvino-dev>=2023.1.0" "nncf>=2.6.0"
    !pip install -q "openvino==2023.1.0.dev20230811"
    !pip install git+https://github.com/openvinotoolkit/nncf.git@develop
    !pip install -q soundfile librosa transformers torch datasets torchmetrics

Imports `⇑ <#top>`__
###############################################################################################################################

.. code:: ipython2

    import numpy as np
    import torch

    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

Prepare the Model `⇑ <#top>`__
###############################################################################################################################

For instantiating PyTorch model class,
we should use ``Wav2Vec2ForCTC.from_pretrained`` method with providing
model ID for downloading from HuggingFace hub. Model weights and
configuration files will be downloaded automatically in first time
usage. Keep in mind that downloading the files can take several minutes
and depends on your internet connection.

Additionally, we can create processor class which is responsible for
model specific pre- and post-processing steps.

.. code:: ipython2

    BATCH_SIZE = 1
    MAX_SEQ_LENGTH = 30480


    torch_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", ctc_loss_reduction="mean")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

Convert it to the OpenVINO Intermediate Representation (OpenVINO IR)

.. code:: ipython2

    import openvino


    default_input = torch.zeros([1, MAX_SEQ_LENGTH], dtype=torch.float)
    ov_model = openvino.convert_model(torch_model, example_input=default_input)

Prepare LibriSpeech Dataset `⇑ <#top>`__
###############################################################################################################################

For demonstration purposes, we will use short dummy version of
LibriSpeech dataset - ``patrickvonplaten/librispeech_asr_dummy`` to
speed up model evaluation. Model accuracy can be different from reported
in the paper. For reproducing original accuracy, use ``librispeech_asr``
dataset.

.. code:: ipython2

    from datasets import load_dataset


    dataset = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
    test_sample = dataset[0]["audio"]


    # define preprocessing function for converting audio to input values for model
    def map_to_input(batch):
        preprocessed_signal = processor(batch["audio"]["array"], return_tensors="pt", padding="longest", sampling_rate=batch['audio']['sampling_rate'])
        input_values = preprocessed_signal.input_values
        batch['input_values'] = input_values
        return batch


    # apply preprocessing function to dataset and remove audio column, to save memory as we do not need it anymore
    dataset = dataset.map(map_to_input, batched=False, remove_columns=["audio"])

Prepare calibration dataset `⇑ <#top>`__
###############################################################################################################################

.. code:: ipython2

    import nncf


    def transform_fn(data_item):
        """
        Extract the model's input from the data item.
        The data item here is the data item that is returned from the data source per iteration.
        This function should be passed when the data item cannot be used as model's input.
        """
        return np.array(data_item["input_values"])


    calibration_dataset = nncf.Dataset(dataset, transform_fn)

Prepare validation function `⇑ <#top>`__
###############################################################################################################################

Define the validation function.

.. code:: ipython2

    from torchmetrics import WordErrorRate
    from tqdm.notebook import tqdm


    def validation_fn(model, dataset):
        """
        Calculate and returns a metric for the model.
        """
        wer = WordErrorRate()
        for sample in tqdm(dataset):
            # run infer function on sample
            output = model.output(0)
            logits = model(np.array(sample['input_values']))[output]
            predicted_ids = np.argmax(logits, axis=-1)
            transcription = processor.batch_decode(torch.from_numpy(predicted_ids))

            # update metric on sample result
            wer.update(transcription, [sample['text']])

        result = wer.compute()

        return 1 - result

Run quantization with accuracy control `⇑ <#top>`__
###############################################################################################################################

You should provide
the calibration dataset and the validation dataset. It can be the same
dataset. - parameter ``max_drop`` defines the accuracy drop threshold.
The quantization process stops when the degradation of accuracy metric
on the validation dataset is less than the ``max_drop``. The default
value is 0.01. NNCF will stop the quantization and report an error if
the ``max_drop`` value can’t be reached. - ``drop_type`` defines how the
accuracy drop will be calculated: ABSOLUTE (used by default) or
RELATIVE. - ``ranking_subset_size`` - size of a subset that is used to
rank layers by their contribution to the accuracy drop. Default value is
300, and the more samples it has the better ranking, potentially. Here
we use the value 25 to speed up the execution.

.. note::

   Execution can take tens of minutes and requires up to 10 GB
   of free memory


.. code:: ipython2

    from nncf.quantization.advanced_parameters import AdvancedAccuracyRestorerParameters
    from nncf.parameters import ModelType

    quantized_model = nncf.quantize_with_accuracy_control(
        ov_model,
        calibration_dataset=calibration_dataset,
        validation_dataset=calibration_dataset,
        validation_fn=validation_fn,
        max_drop=0.01,
        drop_type=nncf.DropType.ABSOLUTE,
        model_type=ModelType.TRANSFORMER,
        advanced_accuracy_restorer_parameters=AdvancedAccuracyRestorerParameters(
            ranking_subset_size=25
        ),
    )

Model Usage Example `⇑ <#top>`__
###############################################################################################################################

.. code:: ipython2

    import IPython.display as ipd


    ipd.Audio(test_sample["array"], rate=16000)

.. code:: ipython2

    core = openvino.Core()

    compiled_quantized_model = core.compile_model(model=quantized_model, device_name='CPU')

    input_data = np.expand_dims(test_sample["array"], axis=0)

Next, make a prediction.

.. code:: ipython2

    predictions = compiled_quantized_model([input_data])[0]
    predicted_ids = np.argmax(predictions, axis=-1)
    transcription = processor.batch_decode(torch.from_numpy(predicted_ids))
    transcription

Compare Accuracy of the Original and Quantized Models `⇑ <#top>`__
###############################################################################################################################

-  Define dataloader for test dataset.
-  Define functions to get inference for PyTorch and OpenVINO models.
-  Define functions to compute Word Error Rate.

.. code:: ipython2

    # inference function for pytorch
    def torch_infer(model, sample):
        logits = model(torch.Tensor(sample['input_values'])).logits
        # take argmax and decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        return transcription


    # inference function for openvino
    def ov_infer(model, sample):
        output = model.output(0)
        logits = model(np.array(sample['input_values']))[output]
        predicted_ids = np.argmax(logits, axis=-1)
        transcription = processor.batch_decode(torch.from_numpy(predicted_ids))
        return transcription


    def compute_wer(dataset, model, infer_fn):
        wer = WordErrorRate()
        for sample in tqdm(dataset):
            # run infer function on sample
            transcription = infer_fn(model, sample)
            # update metric on sample result
            wer.update(transcription, [sample['text']])
        # finalize metric calculation
        result = wer.compute()
        return result

Now, compute WER for the original PyTorch model and quantized model.

.. code:: ipython2

    pt_result = compute_wer(dataset, torch_model, torch_infer)
    quantized_result = compute_wer(dataset, compiled_quantized_model, ov_infer)

    print(f'[PyTorch]   Word Error Rate: {pt_result:.4f}')
    print(f'[Quantized OpenVino]  Word Error Rate: {quantized_result:.4f}')
