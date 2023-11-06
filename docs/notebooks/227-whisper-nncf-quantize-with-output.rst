Post-Training Quantization of OpenAI Whisper model with NNCF
============================================================

The goal of this tutorial is to demonstrate how to speed up the model by
applying 8-bit post-training quantization from
`NNCF <https://github.com/openvinotoolkit/nncf/>`__ (Neural Network
Compression Framework) and infer quantized model via OpenVINO™ Toolkit.
The optimization process contains the following steps:

1. Quantize the converted OpenVINO model from `227-whisper-convert
   notebook <227-whisper-convert.ipynb>`__ with NNCF.
2. Check model result for the demo video.
3. Compare model size, performance and accuracy of FP32 and quantized
   INT8 models.

..

   **NOTE**: you should run
   `227-whisper-convert <227-whisper-convert.ipynb>`__ notebook first to
   generate OpenVINO IR model that is used for quantization.

**Table of contents:**

- `Prerequisites <#prerequisites>`__
- `Create and initialize quantization <#create-and-initialize-quantization>`__
- `Prepare calibration datasets <#prepare-calibration-datasets>`__
- `Quantize Whisper encoder and decoder models <#quantize-whisper-encoder-and-decoder-models>`__
- `Transcribe video with quantized OpenVINO model <#transcribe-video-with-quantized-openvino-model>`__
- `Compare performance and accuracy of the FP32 and INT8 IRs <#compare-performance-and-accuracy-of-the-fp-and-int-irs>`__

Prerequisites 
-------------------------------------------------------

Install dependencies.

.. code:: ipython3

    %pip install -q "openvino>=2023.1.0"
    %pip install -q "nncf>=2.6.0"
    %pip install -q datasets librosa soundfile
    %pip install -q evaluate jiwer

Select device from dropdown list for running inference using OpenVINO.

.. code:: ipython3

    import ipywidgets as widgets
    
    from openvino import Core
    core = Core()
    
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value='AUTO',
        description='Device:',
        disabled=False,
    )
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=4, options=('CPU', 'GPU.0', 'GPU.1', 'GPU.2', 'AUTO'), value='AUTO')



Select the task for the model:

-  **transcribe** - generate audio transcription in the source language
   (automatically detected).
-  **translate** - generate audio transcription with translation to
   English language.

.. code:: ipython3

    task = widgets.Select(
        options=["transcribe", "translate"],
        value="translate",
        description="Select task:",
        disabled=False
    )
    task




.. parsed-literal::

    Select(description='Select task:', index=1, options=('transcribe', 'translate'), value='translate')



Create and initialize quantization
------------------------------------------------

`NNCF <https://github.com/openvinotoolkit/nncf/>`__ enables
post-training quantization by adding the quantization layers into the
model graph and then using a subset of the training dataset to
initialize the parameters of these additional quantization layers. The
framework is designed so that modifications to your original training
code are minor. Quantization is the simplest scenario and requires a few
modifications.

The optimization process contains the following steps:

1. Create a calibration dataset for quantization.
2. Run ``nncf.quantize`` to obtain quantized models.
3. Serialize the ``INT8`` model using ``openvino.runtime.serialize``
   function.

Set paths to the model converted in
`227-whisper-convert <227-whisper-convert.ipynb>`__ notebook and the
paths where quantized models will be saved.

.. code:: ipython3

    from pathlib import Path
    
    WHISPER_ENCODER_OV = Path("whisper_encoder.xml")
    WHISPER_DECODER_OV = Path("whisper_decoder.xml")
    
    WHISPER_ENCODER_OV_INT8 = Path("whisper_encoder_int8.xml")
    WHISPER_DECODER_OV_INT8 = Path("whisper_decoder_int8.xml")

Load FP32 model IR.

.. code:: ipython3

    import whisper
    from utils import patch_whisper_for_ov_inference, OpenVINOAudioEncoder, OpenVINOTextDecoder
    
    model_id = "base"
    model_fp32 = whisper.load_model(model_id).to("cpu").eval()
    patch_whisper_for_ov_inference(model_fp32)
    
    model_fp32.encoder = OpenVINOAudioEncoder(core, WHISPER_ENCODER_OV, device=device.value)
    model_fp32.decoder = OpenVINOTextDecoder(core, WHISPER_DECODER_OV, device=device.value)

Prepare calibration datasets 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Whisper consists of an encoder and a decoder models. We need to collect
calibration data for both of them.

Below we overwrite encoder/decoder forward methods in order to collect
calibration samples.

.. code:: ipython3

    from contextlib import contextmanager
    from functools import partial
    import openvino as ov
    from typing import Optional
    import torch
    
    COLLECT_CALIBRATION_DATA = False
    encoder_calibration_data = []
    decoder_calibration_data = []
    
    @contextmanager
    def calibration_data_collection():
        global COLLECT_CALIBRATION_DATA
        try:
            COLLECT_CALIBRATION_DATA = True
            yield
        finally:
            COLLECT_CALIBRATION_DATA = False
    
    
    def encoder_forward(self, mel: torch.Tensor):
        if COLLECT_CALIBRATION_DATA:
            encoder_calibration_data.append(mel)
        return torch.from_numpy(self.compiled_model(mel)[self.output_blob])
    
    def decoder_forward(self, x: torch.Tensor, xa: torch.Tensor, kv_cache: Optional[dict] = None):
        feed_dict = {'x': ov.Tensor(x.numpy()), 'xa': ov.Tensor(xa.numpy())}
        feed_dict = (self.preprocess_kv_cache_inputs(feed_dict, kv_cache))
        if COLLECT_CALIBRATION_DATA:
            decoder_calibration_data.append(feed_dict)
        res = self.compiled_model(feed_dict)
        return self.postprocess_outputs(res)
    
    model_fp32.encoder.forward = partial(encoder_forward, model_fp32.encoder)
    model_fp32.decoder.forward = partial(decoder_forward, model_fp32.decoder)

We use a portion of validation
`librispeech_asr <https://huggingface.co/datasets/librispeech_asr>`__
dataset from Hugging Face as calibration data.

.. code:: ipython3

    from datasets import load_dataset
    from tqdm.notebook import tqdm
    
    CALIBRATION_DATASET_SIZE = 30
    
    calibration_dataset = load_dataset("librispeech_asr", "clean", split="validation", streaming=True).take(CALIBRATION_DATASET_SIZE)
    
    with calibration_data_collection():
        for data_item in tqdm(calibration_dataset, desc="Collecting calibration data", total=CALIBRATION_DATASET_SIZE):
            model_fp32.transcribe(data_item["audio"]["array"].astype("float32"), task=task.value)



.. parsed-literal::

    Collecting calibration data:   0%|          | 0/30 [00:00<?, ?it/s]


Quantize Whisper encoder and decoder models 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Quantize both encoder and decoder models using ``nncf.quantize()`` API
and save the quantized IRs after that.

.. code:: ipython3

    import nncf
    from openvino.runtime import serialize
    
    print("Quantizing encoder...")
    quantized_encoder = nncf.quantize(
        model=model_fp32.encoder.model,
        calibration_dataset=nncf.Dataset(encoder_calibration_data),
        subset_size=len(encoder_calibration_data),
        model_type=nncf.ModelType.TRANSFORMER,
        advanced_parameters=nncf.AdvancedQuantizationParameters(
            smooth_quant_alpha=0.5      # Smooth Quant algorithm reduces activation quantization error; optimal alpha value was obtained through grid search
        )
    )
    serialize(quantized_encoder, WHISPER_ENCODER_OV_INT8)
    print(f"Saved quantized encoder at ./{WHISPER_ENCODER_OV_INT8}")
    
    print("Quantizing decoder...")
    quantized_decoder = nncf.quantize(
        model=model_fp32.decoder.model,
        calibration_dataset=nncf.Dataset(decoder_calibration_data),
        subset_size=len(decoder_calibration_data),
        model_type=nncf.ModelType.TRANSFORMER,
        advanced_parameters=nncf.AdvancedQuantizationParameters(
            smooth_quant_alpha=0.95     # Smooth Quant algorithm reduces activation quantization error; optimal alpha value was obtained through grid search
        )
    )
    serialize(quantized_decoder, WHISPER_DECODER_OV_INT8)
    print(f"Saved quantized decoder at ./{WHISPER_DECODER_OV_INT8}")


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino
    Quantizing encoder...


.. parsed-literal::

    2023-08-30 19:38:10.314501: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2023-08-30 19:38:10.347770: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-08-30 19:38:10.917857: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    Statistics collection: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [00:04<00:00, 12.26it/s]
    Applying Smooth Quant: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 24/24 [00:00<00:00, 60.29it/s]


.. parsed-literal::

    INFO:nncf:18 ignored nodes was found by name in the NNCFGraph


.. parsed-literal::

    Statistics collection: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [00:14<00:00,  4.14it/s]
    Applying Fast Bias correction: 100%|████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:06<00:00,  5.22it/s]


.. parsed-literal::

    Saved quantized encoder at ./whisper_encoder_int8.xml
    Quantizing decoder...


.. parsed-literal::

    Statistics collection: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 664/664 [00:12<00:00, 54.92it/s]
    Applying Smooth Quant: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 38/38 [00:00<00:00, 39.37it/s]


.. parsed-literal::

    INFO:nncf:36 ignored nodes was found by name in the NNCFGraph


.. parsed-literal::

    Statistics collection: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 664/664 [00:34<00:00, 19.20it/s]
    Applying Fast Bias correction: 100%|████████████████████████████████████████████████████████████████████████████████████████| 48/48 [00:07<00:00,  6.30it/s]


.. parsed-literal::

    Saved quantized decoder at ./whisper_decoder_int8.xml


Transcribe video with quantized OpenVINO model 
----------------------------------------------------------------------------------------

Load ``INT8`` models saved above into a new instance of Whisper model.

.. code:: ipython3

    model_int8 = whisper.load_model(model_id).to("cpu").eval()
    patch_whisper_for_ov_inference(model_int8)
    
    model_int8.encoder = OpenVINOAudioEncoder(core, WHISPER_ENCODER_OV_INT8, device=device.value)
    model_int8.decoder = OpenVINOTextDecoder(core, WHISPER_DECODER_OV_INT8, device=device.value)

Select a video for transcription as in
`227-whisper-convert <227-whisper-convert.ipynb>`__ notebook.

.. code:: ipython3

    VIDEO_LINK = "https://youtu.be/kgL5LBM-hFI"
    link = widgets.Text(
        value=VIDEO_LINK,
        placeholder="Type link for video",
        description="Video:",
        disabled=False
    )
    link




.. parsed-literal::

    Text(value='https://youtu.be/kgL5LBM-hFI', description='Video:', placeholder='Type link for video')



.. code:: ipython3

    from pytube import YouTube
    
    print(f"Downloading video {link.value} started")
    
    output_file = Path("downloaded_video.mp4")
    yt = YouTube(link.value)
    yt.streams.get_highest_resolution().download(filename=output_file)
    print(f"Video saved to {output_file}")


.. parsed-literal::

    Downloading video https://youtu.be/kgL5LBM-hFI started
    Video saved to downloaded_video.mp4


.. code:: ipython3

    from utils import get_audio
    
    audio = get_audio(output_file)

Run transcription by the quantized model.

.. code:: ipython3

    transcription = model_int8.transcribe(audio, task=task.value)

.. code:: ipython3

    from utils import prepare_srt
    
    srt_lines = prepare_srt(transcription)
    # save transcription
    with output_file.with_suffix(".srt").open("w") as f:
        f.writelines(srt_lines)

Now let us see the results.

.. code:: ipython3

    widgets.Video.from_file(output_file, loop=False, width=800, height=800)




.. parsed-literal::

    Video(value=b'\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00isommp42\x00\x00Aimoov\x00\x00\x00lmvhd...', height='800…



.. code:: ipython3

    print("".join(srt_lines))


.. parsed-literal::

    1
    00:00:00,000 --> 00:00:07,000
     What's that? Oh, wow.
    
    2
    00:00:09,000 --> 00:00:11,000
     Hello humans.
    
    3
    00:00:14,000 --> 00:00:15,000
     Focus on me.
    
    4
    00:00:15,000 --> 00:00:16,000
     Focus on the guard.
    
    5
    00:00:18,000 --> 00:00:20,000
     Don't tell anyone what you've seen in here.
    
    6
    00:00:22,000 --> 00:00:24,000
     Have you seen what's in there?
    
    7
    00:00:24,000 --> 00:00:25,000
     They have intel.
    
    8
    00:00:25,000 --> 00:00:27,000
     This is where it all changes.
    
    


As you can see the result is almost the same.

Compare performance and accuracy of the FP32 and INT8 IRs 
---------------------------------------------------------------------------------------------------

Compare model file size.

.. code:: ipython3

    def calculate_compression_rate(model_path_ov, model_path_ov_int8):
        model_size_fp32 = model_path_ov.with_suffix(".bin").stat().st_size / 1024
        model_size_int8 = model_path_ov_int8.with_suffix(".bin").stat().st_size / 1024
        print(f"Model: {model_path_ov.stem}")
        print(f"    * FP32 IR model size: {model_size_fp32:.2f} KB")
        print(f"    * INT8 IR model size: {model_size_int8:.2f} KB")
        print(f"    * Model compression rate: {model_size_fp32 / model_size_int8:.3f}")
    
    calculate_compression_rate(WHISPER_ENCODER_OV, WHISPER_ENCODER_OV_INT8)
    calculate_compression_rate(WHISPER_DECODER_OV, WHISPER_DECODER_OV_INT8)


.. parsed-literal::

    Model: whisper_encoder
        * FP32 IR model size: 40216.07 KB
        * INT8 IR model size: 21092.37 KB
        * Model compression rate: 1.907
    Model: whisper_decoder
        * FP32 IR model size: 101961.09 KB
        * INT8 IR model size: 78058.77 KB
        * Model compression rate: 1.306


To measure the inference performance of the ``FP32`` and ``INT8``
encoder/decoder models, we use median inference time on calibration
dataset. So we can approximately estimate the speed-up of the dynamic
quantized models.

   **NOTE**: For the most accurate performance estimation, it is
   recommended to run ``benchmark_app`` with static shapes in a
   terminal/command prompt after closing other applications.

.. code:: ipython3

    import time
    import numpy as np
    
    def calculate_call_inference_time(model, dataset):
        inference_time = []
        for data_item in tqdm(dataset[:100], desc="Measuring performance"):
            start = time.perf_counter()
            model(data_item)
            end = time.perf_counter()
            delta = end - start
            inference_time.append(delta)
        return np.median(inference_time)
    
    
    encoder_time_fp32 = calculate_call_inference_time(model_fp32.encoder.compiled_model, encoder_calibration_data)
    encoder_time_int8 = calculate_call_inference_time(model_int8.encoder.compiled_model, encoder_calibration_data)
    print(f"Encoder performance speedup: {encoder_time_fp32 / encoder_time_int8:.3f}")
    
    decoder_time_fp32 = calculate_call_inference_time(model_fp32.decoder.compiled_model, decoder_calibration_data)
    decoder_time_int8 = calculate_call_inference_time(model_int8.decoder.compiled_model, decoder_calibration_data)
    print(f"Decoder performance speedup: {decoder_time_fp32 / decoder_time_int8:.3f}")



.. parsed-literal::

    Measuring performance:   0%|          | 0/60 [00:00<?, ?it/s]



.. parsed-literal::

    Measuring performance:   0%|          | 0/60 [00:00<?, ?it/s]


.. parsed-literal::

    Encoder performance speedup: 1.325



.. parsed-literal::

    Measuring performance:   0%|          | 0/100 [00:00<?, ?it/s]



.. parsed-literal::

    Measuring performance:   0%|          | 0/100 [00:00<?, ?it/s]


.. parsed-literal::

    Decoder performance speedup: 1.609


We measure the whole transcription performance separately, because a
single Whisper ``transcribe()`` call triggers multiple encoder and
decoder inference calls. And the number of these calls is dynamic
depending on the model accuracy. In this experiment we use the mean time
instead of the median because the model transcription time is less
uniform.

We also compare accuracy values of the ``FP32`` and ``INT8`` models on a
subset of
`librispeech_asr <https://huggingface.co/datasets/librispeech_asr>`__
test dataset. We rely on the Word Error Rate (WER) metric and compute
accuracy as ``(1 - WER)``.

.. code:: ipython3

    from evaluate import load
    from transformers import WhisperProcessor
    
    wer = load("wer")
    
    TEST_DATASET_SIZE = 100
    test_dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True).take(TEST_DATASET_SIZE)
    
    def calculate_transcription_time_and_accuracy(model, dataset):
        processor = WhisperProcessor.from_pretrained("openai/whisper-large")
    
        ground_truths = []
        predictions = []
        inference_time = []
        for data_item in tqdm(dataset, desc="Measuring performance and accuracy", total=TEST_DATASET_SIZE):
            audio = data_item["audio"]["array"].astype("float32")
    
            start_time = time.perf_counter()
            transcription = model.transcribe(audio, task=task.value)
            end_time = time.perf_counter()
            delta_time = end_time - start_time
    
            reference = processor.tokenizer._normalize(data_item["text"])
            prediction = processor.tokenizer._normalize(transcription["text"])
            ground_truths.append(reference)
            predictions.append(prediction)
            inference_time.append(delta_time)
    
        word_accuracy = (1 - wer.compute(references=ground_truths, predictions=predictions)) * 100
        mean_inference_time = np.mean(inference_time)
        return mean_inference_time, word_accuracy
    
    transcription_time_fp32, accuracy_fp32 = calculate_transcription_time_and_accuracy(model_fp32, test_dataset)
    transcription_time_int8, accuracy_int8 = calculate_transcription_time_and_accuracy(model_int8, test_dataset)
    print(f"Whisper transcription performance speedup: {transcription_time_fp32 / transcription_time_int8:.3f}")
    print(f"Whisper transcription word accuracy. FP32: {accuracy_fp32:.2f}%. INT8: {accuracy_int8:.2f}%. Accuracy drop :{accuracy_fp32 - accuracy_int8:.2f}%.")



.. parsed-literal::

    Measuring performance and accuracy:   0%|          | 0/100 [00:00<?, ?it/s]



.. parsed-literal::

    Measuring performance and accuracy:   0%|          | 0/100 [00:00<?, ?it/s]


.. parsed-literal::

    Whisper transcription performance speedup: 1.446
    Whisper transcription word accuracy. FP32: 95.61%. INT8: 94.23%. Accuracy drop :1.38%.


   **NOTE**: Accuracy drop can generally be improved by increasing
   calibration dataset size.
