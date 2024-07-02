Video Subtitle Generation using Whisper and OpenVINO™
=====================================================

`Whisper <https://openai.com/blog/whisper/>`__ is an automatic speech
recognition (ASR) system trained on 680,000 hours of multilingual and
multitask supervised data collected from the web. It is a multi-task
model that can perform multilingual speech recognition as well as speech
translation and language identification.

.. figure:: https://user-images.githubusercontent.com/29454499/204536347-28976978-9a07-416c-acff-fc1214bbfbe0.svg
   :alt: asr-training-data-desktop.svg

   asr-training-data-desktop.svg

You can find more information about this model in the `research
paper <https://cdn.openai.com/papers/whisper.pdf>`__, `OpenAI
blog <https://openai.com/blog/whisper/>`__, `model
card <https://github.com/openai/whisper/blob/main/model-card.md>`__ and
GitHub `repository <https://github.com/openai/whisper>`__.

In this notebook, we will use Whisper with OpenVINO to generate
subtitles in a sample video. Additionally, we will use
`NNCF <https://github.com/openvinotoolkit/nncf>`__ improving model
performance by INT8 quantization. Notebook contains the following steps:
1. Download the model. 2. Instantiate the PyTorch model pipeline. 3.
Convert model to OpenVINO IR, using model conversion API. 4. Run the
Whisper pipeline with OpenVINO models. 5. Quantize the OpenVINO model
with NNCF. 6. Check quantized model result for the demo video. 7.
Compare model size, performance and accuracy of FP32 and quantized INT8
models. 8. Launch Interactive demo for video subtitles generation.

**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Instantiate model <#instantiate-model>`__

   -  `Convert model to OpenVINO Intermediate Representation (IR)
      format. <#convert-model-to-openvino-intermediate-representation-ir-format->`__

-  `Prepare inference pipeline <#prepare-inference-pipeline>`__

   -  `Select inference device <#select-inference-device>`__

-  `Run video transcription
   pipeline <#run-video-transcription-pipeline>`__
-  `Quantization <#quantization>`__

   -  `Prepare calibration datasets <#prepare-calibration-datasets>`__
   -  `Quantize Whisper encoder and decoder
      models <#quantize-whisper-encoder-and-decoder-models>`__
   -  `Run quantized model inference <#run-quantized-model-inference>`__
   -  `Compare performance and accuracy of the original and quantized
      models <#compare-performance-and-accuracy-of-the-original-and-quantized-models>`__

-  `Interactive demo <#interactive-demo>`__

Prerequisites
-------------



Install dependencies.

.. code:: ipython3

    %pip install -q "openvino>=2024.1.0" "nncf>=2.10.0"
    %pip install -q "python-ffmpeg<=1.0.16" moviepy transformers onnx "git+https://github.com/huggingface/optimum-intel.git" "peft==0.6.2" --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install -q "git+https://github.com/garywu007/pytube.git" soundfile librosa jiwer
    %pip install -q  "gradio>=4.19"

Instantiate model
-----------------



Whisper is a Transformer based encoder-decoder model, also referred to
as a sequence-to-sequence model. It maps a sequence of audio spectrogram
features to a sequence of text tokens. First, the raw audio inputs are
converted to a log-Mel spectrogram by action of the feature extractor.
Then, the Transformer encoder encodes the spectrogram to form a sequence
of encoder hidden states. Finally, the decoder autoregressively predicts
text tokens, conditional on both the previous tokens and the encoder
hidden states.

You can see the model architecture in the diagram below:

.. figure:: https://user-images.githubusercontent.com/29454499/204536571-8f6d8d77-5fbd-4c6d-8e29-14e734837860.svg
   :alt: whisper_architecture.svg

   whisper_architecture.svg

There are several models of different sizes and capabilities trained by
the authors of the model. In this tutorial, we will use the ``tiny``
model, but the same actions are also applicable to other models from
Whisper family.

.. code:: ipython3

    import ipywidgets as widgets
    
    MODELS = [
        "openai/whisper-large-v3",
        "openai/whisper-large-v2",
        "openai/whisper-large",
        "openai/whisper-medium",
        "openai/whisper-small",
        "openai/whisper-base",
        "openai/whisper-tiny",
    ]
    
    model_id = widgets.Dropdown(
        options=list(MODELS),
        value="openai/whisper-tiny",
        description="Model:",
        disabled=False,
    )
    
    model_id




.. parsed-literal::

    Dropdown(description='Model:', index=6, options=('openai/whisper-large-v3', 'openai/whisper-large-v2', 'openai…



Convert model to OpenVINO Intermediate Representation (IR) format using Optimum-Intel.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



The Hugging Face Optimum API is a high-level API that enables us to
convert and quantize models from the Hugging Face Transformers library
to the OpenVINO™ IR format. For more details, refer to the `Hugging Face
Optimum
documentation <https://huggingface.co/docs/optimum/intel/inference>`__.

Optimum Intel can be used to load optimized models from the `Hugging
Face Hub <https://huggingface.co/docs/optimum/intel/hf.co/models>`__ and
create pipelines to run an inference with OpenVINO Runtime using Hugging
Face APIs. The Optimum Inference models are API compatible with Hugging
Face Transformers models. This means we just need to replace the
``AutoModelForXxx`` class with the corresponding ``OVModelForXxx``
class.

Below is an example of the whisper-tiny model

.. code:: diff

   -from transformers import AutoModelForSpeechSeq2Seq
   +from optimum.intel.openvino import OVModelForSpeechSeq2Seq
   from transformers import AutoTokenizer, pipeline

   model_id = "openai/whisper-tiny"
   -model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
   +model = OVModelForSpeechSeq2Seq.from_pretrained(model_id, export=True)

Model class initialization starts with calling the ``from_pretrained``
method. When downloading and converting the Transformers model, the
parameter ``export=True`` should be added. We can save the converted
model for the next usage with the ``save_pretrained`` method.
Alternatively, model conversion can be performed using Optimum-CLI
interface. You can find more details about Optimum-Intel and Optimum CLI
usage in this `tutorial <hugging-face-hub-with-output.html>`__.
The command bellow illustrates how to convert whisper using optimum cli.

.. code:: ipython3

    from pathlib import Path
    
    model_dir = model_id.value.split("/")[-1]
    
    if not Path(model_dir).exists():
        !optimum-cli export openvino -m {model_id.value} {model_dir} --weight-format fp16

Prepare inference pipeline
--------------------------



The image below illustrates the pipeline of video transcribing using the
Whisper model.

.. figure:: https://user-images.githubusercontent.com/29454499/204536733-1f4342f7-2328-476a-a431-cb596df69854.png
   :alt: whisper_pipeline.png

   whisper_pipeline.png

Preprocessing and post-processing are important in this model use.
``transformers.AutoProcessor`` class used for initialization
``WhisperProcessor`` is responsible for preparing audio input data for
the PyTorch model, converting it to Mel-spectrogram and decoding
predicted output token_ids into string using tokenizer. Tokenizers and
Processors are distributed with models also compatible with the OpenVINO
model.

Like the original PyTorch model, the OpenVINO model is also compatible
with HuggingFace
`pipeline <https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline>`__
interface for ``automatic-speech-recognition``. Pipeline can be used for
long audio transcription. Distil-Whisper uses a chunked algorithm to
transcribe long-form audio files. In practice, this chunked long-form
algorithm is 9x faster than the sequential algorithm proposed by OpenAI
in the Whisper paper. To enable chunking, pass the chunk_length_s
parameter to the pipeline. For Distil-Whisper, a chunk length of 15
seconds is optimal. To activate batching, pass the argument batch_size.

Select inference device
~~~~~~~~~~~~~~~~~~~~~~~



select device from dropdown list for running inference using OpenVINO

.. code:: ipython3

    import openvino as ov
    
    core = ov.Core()

.. code:: ipython3

    import ipywidgets as widgets
    
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value="AUTO",
        description="Device:",
        disabled=False,
    )
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=3, options=('CPU', 'GPU.0', 'GPU.1', 'AUTO'), value='AUTO')



.. code:: ipython3

    from optimum.intel.openvino import OVModelForSpeechSeq2Seq
    from transformers import AutoProcessor, pipeline
    
    ov_model = OVModelForSpeechSeq2Seq.from_pretrained(model_dir, device=device.value)
    
    processor = AutoProcessor.from_pretrained(model_dir)
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model=ov_model,
        chunk_length_s=30,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
    )


.. parsed-literal::

    2024-06-10 09:43:58.190233: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-06-10 09:43:58.192258: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2024-06-10 09:43:58.228701: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-06-10 09:43:58.903562: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    WARNING[XFORMERS]: xFormers can't load C++/CUDA extensions. xFormers was built for:
        PyTorch 2.0.1+cu118 with CUDA 1108 (you have 2.3.0+cu121)
        Python  3.8.18 (you have 3.8.10)
      Please reinstall xformers (see https://github.com/facebookresearch/xformers#installing-xformers)
      Memory-efficient attention, SwiGLU, sparse and more won't be available.
      Set XFORMERS_MORE_DETAILS=1 for more details
    /home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      torch.utils._pytree._register_pytree_node(
    Compiling the encoder to AUTO ...
    Compiling the decoder to AUTO ...
    Compiling the decoder to AUTO ...
    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.


Run video transcription pipeline
--------------------------------



Now, we are ready to start transcription. We select a video from YouTube
that we want to transcribe. Be patient, as downloading the video may
take some time.

.. code:: ipython3

    import ipywidgets as widgets
    
    VIDEO_LINK = "https://youtu.be/kgL5LBM-hFI"
    link = widgets.Text(
        value=VIDEO_LINK,
        placeholder="Type link for video",
        description="Video:",
        disabled=False,
    )
    
    link




.. parsed-literal::

    Text(value='https://youtu.be/kgL5LBM-hFI', description='Video:', placeholder='Type link for video')



.. code:: ipython3

    from pathlib import Path
    from pytube import YouTube
    
    print(f"Downloading video {link.value} started")
    
    output_file = Path("downloaded_video.mp4")
    yt = YouTube(link.value)
    yt.streams.get_highest_resolution().download(filename=output_file)
    print(f"Video saved to {output_file}")


.. parsed-literal::

    Downloading video https://youtu.be/kgL5LBM-hFI started
    Video saved to downloaded_video.mp4


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
        disabled=False,
    )
    task




.. parsed-literal::

    Select(description='Select task:', index=1, options=('transcribe', 'translate'), value='translate')



.. code:: ipython3

    from moviepy.editor import VideoFileClip
    from transformers.pipelines.audio_utils import ffmpeg_read
    
    
    def get_audio(video_file):
        """
        Extract audio signal from a given video file, then convert it to float,
        then mono-channel format and resample it to the expected sample rate
    
        Parameters:
            video_file: path to input video file
        Returns:
          resampled_audio: mono-channel float audio signal with 16000 Hz sample rate
                           extracted from video
          duration: duration of video fragment in seconds
        """
        input_video = VideoFileClip(str(video_file))
        duration = input_video.duration
        audio_file = video_file.stem + ".wav"
        input_video.audio.write_audiofile(audio_file, verbose=False, logger=None)
        with open(audio_file, "rb") as f:
            inputs = f.read()
        audio = ffmpeg_read(inputs, pipe.feature_extractor.sampling_rate)
        return {"raw": audio, "sampling_rate": pipe.feature_extractor.sampling_rate}, duration

.. code:: ipython3

    inputs, duration = get_audio(output_file)
    
    transcription = pipe(inputs, generate_kwargs={"task": task.value}, return_timestamps=True)["chunks"]

.. code:: ipython3

    import math
    
    
    def format_timestamp(seconds: float):
        """
        format time in srt-file expected format
        """
        assert seconds >= 0, "non-negative timestamp expected"
        milliseconds = round(seconds * 1000.0)
    
        hours = milliseconds // 3_600_000
        milliseconds -= hours * 3_600_000
    
        minutes = milliseconds // 60_000
        milliseconds -= minutes * 60_000
    
        seconds = milliseconds // 1_000
        milliseconds -= seconds * 1_000
    
        return (f"{hours}:" if hours > 0 else "00:") + f"{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    
    
    def prepare_srt(transcription, filter_duration=None):
        """
        Format transcription into srt file format
        """
        segment_lines = []
        for idx, segment in enumerate(transcription):
            # for the case where the model could not predict an ending timestamp, which can happen if audio is cut off in the middle of a word.
            if segment["timestamp"][1] is None:
                segment["timestamp"] = (segment["timestamp"][0], filter_duration)
    
            if filter_duration is not None and (segment["timestamp"][0] >= math.floor(filter_duration) or segment["timestamp"][1] > math.ceil(filter_duration) + 1):
                break
            segment_lines.append(str(idx + 1) + "\n")
            time_start = format_timestamp(segment["timestamp"][0])
            time_end = format_timestamp(segment["timestamp"][1])
            time_str = f"{time_start} --> {time_end}\n"
            segment_lines.append(time_str)
            segment_lines.append(segment["text"] + "\n\n")
        return segment_lines

"The results will be saved in the ``downloaded_video.srt`` file. SRT is
one of the most popular formats for storing subtitles and is compatible
with many modern video players. This file can be used to embed
transcription into videos during playback or by injecting them directly
into video files using ``ffmpeg``.

.. code:: ipython3

    srt_lines = prepare_srt(transcription, filter_duration=duration)
    # save transcription
    with output_file.with_suffix(".srt").open("w") as f:
        f.writelines(srt_lines)

Now let us see the results.

.. code:: ipython3

    widgets.Video.from_file(output_file, loop=False, width=800, height=800)




.. parsed-literal::

    Video(value=b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00isommp42\x00\x00:'moov\x00\x00\x00lmvhd...", height='800…



.. code:: ipython3

    print("".join(srt_lines))


.. parsed-literal::

    1
    00:00:00,000 --> 00:00:05,000
     Oh, what's that?
    
    2
    00:00:05,000 --> 00:00:08,000
     Oh, wow.
    
    3
    00:00:08,000 --> 00:00:10,000
     Hello, humans.
    
    4
    00:00:13,000 --> 00:00:15,000
     Focus on me.
    
    5
    00:00:15,000 --> 00:00:17,000
     Focus on the guard.
    
    6
    00:00:17,000 --> 00:00:20,000
     Don't tell anyone what you're seeing in here.
    
    7
    00:00:22,000 --> 00:00:24,000
     Have you seen what's in there?
    
    8
    00:00:24,000 --> 00:00:25,000
     They have intel.
    
    9
    00:00:25,000 --> 00:00:27,000
     This is where it all changes.
    
    


Quantization
------------



`NNCF <https://github.com/openvinotoolkit/nncf/>`__ enables
post-training quantization by adding the quantization layers into the
model graph and then using a subset of the training dataset to
initialize the parameters of these additional quantization layers. The
framework is designed so that modifications to your original training
code are minor.

The optimization process contains the following steps:

1. Create a calibration dataset for quantization.
2. Run ``nncf.quantize`` to obtain quantized encoder and decoder models.
3. Serialize the ``INT8`` model using ``openvino.save_model`` function.

..

   **Note**: Quantization is time and memory consuming operation.
   Running quantization code below may take some time.

Please select below whether you would like to run Whisper quantization.

.. code:: ipython3

    to_quantize = widgets.Checkbox(
        value=True,
        description="Quantization",
        disabled=False,
    )
    
    to_quantize




.. parsed-literal::

    Checkbox(value=True, description='Quantization')



.. code:: ipython3

    # Fetch `skip_kernel_extension` module
    import requests
    
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/skip_kernel_extension.py",
    )
    open("skip_kernel_extension.py", "w").write(r.text)
    
    ov_quantized_model = None
    
    %load_ext skip_kernel_extension

Prepare calibration datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~



First step is to prepare calibration datasets for quantization. Since we
quantize whisper encoder and decoder separately, we need to prepare a
calibration dataset for each of the models. We import an
``InferRequestWrapper`` class that will intercept model inputs and
collect them to a list. Then we run model inference on some small amount
of audio samples. Generally, increasing the calibration dataset size
improves quantization quality.

.. code:: ipython3

    %%skip not $to_quantize.value
    
    from itertools import islice
    from optimum.intel.openvino.quantization import InferRequestWrapper
    
    
    def collect_calibration_dataset(ov_model: OVModelForSpeechSeq2Seq, calibration_dataset_size: int):
        # Overwrite model request properties, saving the original ones for restoring later
        encoder_calibration_data = []
        decoder_calibration_data = []
        ov_model.encoder.request = InferRequestWrapper(ov_model.encoder.request, encoder_calibration_data, apply_caching=True)
        ov_model.decoder_with_past.request = InferRequestWrapper(ov_model.decoder_with_past.request,
                                                                 decoder_calibration_data,
                                                                 apply_caching=True)
    
        pipe = pipeline(
          "automatic-speech-recognition",
          model=ov_model,
          chunk_length_s=30,
          tokenizer=processor.tokenizer,
          feature_extractor=processor.feature_extractor)
        try:
            calibration_dataset = dataset = load_dataset("openslr/librispeech_asr", "clean", split="validation", streaming=True, trust_remote_code=True)
            for sample in tqdm(islice(calibration_dataset, calibration_dataset_size), desc="Collecting calibration data",
                               total=calibration_dataset_size):
                pipe(sample["audio"], generate_kwargs={"task": task.value}, return_timestamps=True)
        finally:
            ov_model.encoder.request = ov_model.encoder.request.request
            ov_model.decoder_with_past.request = ov_model.decoder_with_past.request.request
    
        return encoder_calibration_data, decoder_calibration_data

Quantize Whisper encoder and decoder models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Below we run the ``quantize`` function which calls ``nncf.quantize`` on
Whisper encoder and decoder-with-past models. We don’t quantize
first-step-decoder because its share in whole inference time is
negligible.

.. code:: ipython3

    %%skip not $to_quantize.value
    
    import gc
    import shutil
    import nncf
    from datasets import load_dataset
    from tqdm.notebook import tqdm
    
    def extract_input_features(sample):
        input_features = processor(
            sample["audio"]["array"],
            sampling_rate=sample["audio"]["sampling_rate"],
            return_tensors="pt",
        ).input_features
        return input_features
    
    
    
    CALIBRATION_DATASET_SIZE = 50
    quantized_model_path = Path(f"{model_dir}_quantized")
    
    
    def quantize(ov_model: OVModelForSpeechSeq2Seq, calibration_dataset_size: int):
        if not quantized_model_path.exists():
            encoder_calibration_data, decoder_calibration_data = collect_calibration_dataset(
                ov_model, calibration_dataset_size
            )
            print("Quantizing encoder")
            quantized_encoder = nncf.quantize(
                ov_model.encoder.model,
                nncf.Dataset(encoder_calibration_data),
                subset_size=len(encoder_calibration_data),
                model_type=nncf.ModelType.TRANSFORMER,
                # Smooth Quant algorithm reduces activation quantization error; optimal alpha value was obtained through grid search
                advanced_parameters=nncf.AdvancedQuantizationParameters(smooth_quant_alpha=0.50)
            )
            ov.save_model(quantized_encoder, quantized_model_path / "openvino_encoder_model.xml")
            del quantized_encoder
            del encoder_calibration_data
            gc.collect()
    
            print("Quantizing decoder with past")
            quantized_decoder_with_past = nncf.quantize(
                ov_model.decoder_with_past.model,
                nncf.Dataset(decoder_calibration_data),
                subset_size=len(decoder_calibration_data),
                model_type=nncf.ModelType.TRANSFORMER,
                # Smooth Quant algorithm reduces activation quantization error; optimal alpha value was obtained through grid search
                advanced_parameters=nncf.AdvancedQuantizationParameters(smooth_quant_alpha=0.96)
            )
            ov.save_model(quantized_decoder_with_past, quantized_model_path / "openvino_decoder_with_past_model.xml")
            del quantized_decoder_with_past
            del decoder_calibration_data
            gc.collect()
    
            # Copy the config file and the first-step-decoder manually
            model_path = Path(model_dir)
            shutil.copy(model_path / "config.json", quantized_model_path / "config.json")
            shutil.copy(model_path / "generation_config.json", quantized_model_path / "generation_config.json")
            shutil.copy(model_path / "openvino_decoder_model.xml", quantized_model_path / "openvino_decoder_model.xml")
            shutil.copy(model_path / "openvino_decoder_model.bin", quantized_model_path / "openvino_decoder_model.bin")
    
        quantized_ov_model = OVModelForSpeechSeq2Seq.from_pretrained(quantized_model_path, compile=False)
        quantized_ov_model.to(device.value)
        quantized_ov_model.compile()
        return quantized_ov_model
    
    
    ov_quantized_model = quantize(ov_model, CALIBRATION_DATASET_SIZE)



.. parsed-literal::

    Collecting calibration data:   0%|          | 0/50 [00:00<?, ?it/s]



.. parsed-literal::

    Output()


.. parsed-literal::

    Quantizing encoder



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>




.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>



.. parsed-literal::

    INFO:nncf:12 ignored nodes were found by name in the NNCFGraph
    INFO:nncf:16 ignored nodes were found by name in the NNCFGraph



.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>




.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>




.. parsed-literal::

    Output()


.. parsed-literal::

    Quantizing decoder with past



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>




.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>



.. parsed-literal::

    INFO:nncf:24 ignored nodes were found by name in the NNCFGraph
    INFO:nncf:24 ignored nodes were found by name in the NNCFGraph



.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>




.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>



.. parsed-literal::

    Compiling the encoder to AUTO ...
    Compiling the decoder to AUTO ...
    Compiling the decoder to AUTO ...


Run quantized model inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Let’s compare the transcription results for original and quantized
models.

.. code:: ipython3

    if ov_quantized_model is not None:
        int8_pipe = pipeline(
            "automatic-speech-recognition",
            model=ov_quantized_model,
            chunk_length_s=30,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
        )
        inputs, duration = get_audio(output_file)
        transcription = int8_pipe(inputs, generate_kwargs={"task": task.value}, return_timestamps=True)["chunks"]
        srt_lines = prepare_srt(transcription, filter_duration=duration)
        print("".join(srt_lines))
        widgets.Video.from_file(output_file, loop=False, width=800, height=800)


.. parsed-literal::

    1
    00:00:00,000 --> 00:00:05,000
     What's that?
    
    2
    00:00:05,000 --> 00:00:07,000
     Oh, wow.
    
    3
    00:00:09,000 --> 00:00:11,000
     Hello humans.
    
    4
    00:00:14,000 --> 00:00:15,000
     Focus on me.
    
    5
    00:00:15,000 --> 00:00:16,000
     Focus on the guard.
    
    6
    00:00:18,000 --> 00:00:20,000
     Don't tell anyone what you're seen in here.
    
    7
    00:00:22,000 --> 00:00:24,000
     Have you seen what's in there?
    
    8
    00:00:24,000 --> 00:00:25,000
     They have intel.
    
    9
    00:00:25,000 --> 00:00:27,000
     This is where it all changes.
    
    


Compare performance and accuracy of the original and quantized models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Finally, we compare original and quantized Whisper models from accuracy
and performance stand-points.

To measure accuracy, we use ``1 - WER`` as a metric, where WER stands
for Word Error Rate.

When measuring inference time, we do it separately for encoder and
decoder-with-past model forwards, and for the whole model inference too.

.. code:: ipython3

    %%skip not $to_quantize.value
    
    import time
    from contextlib import contextmanager
    from jiwer import wer, wer_standardize
    
    
    TEST_DATASET_SIZE = 50
    MEASURE_TIME = False
    
    @contextmanager
    def time_measurement():
        global MEASURE_TIME
        try:
            MEASURE_TIME = True
            yield
        finally:
            MEASURE_TIME = False
    
    def time_fn(obj, fn_name, time_list):
        original_fn = getattr(obj, fn_name)
    
        def wrapper(*args, **kwargs):
            if not MEASURE_TIME:
                return original_fn(\*args, \*\*kwargs)
            start_time = time.perf_counter()
            result = original_fn(\*args, \*\*kwargs)
            end_time = time.perf_counter()
            time_list.append(end_time - start_time)
            return result
    
        setattr(obj, fn_name, wrapper)
    
    def calculate_transcription_time_and_accuracy(ov_model, test_samples):
        encoder_infer_times = []
        decoder_with_past_infer_times = []
        whole_infer_times = []
        time_fn(ov_model, "generate", whole_infer_times)
        time_fn(ov_model.encoder, "forward", encoder_infer_times)
        time_fn(ov_model.decoder_with_past, "forward", decoder_with_past_infer_times)
    
        ground_truths = []
        predictions = []
        for data_item in tqdm(test_samples, desc="Measuring performance and accuracy"):
            input_features = extract_input_features(data_item)
    
            with time_measurement():
                predicted_ids = ov_model.generate(input_features)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    
            ground_truths.append(data_item["text"])
            predictions.append(transcription[0])
    
        word_accuracy = (1 - wer(ground_truths, predictions, reference_transform=wer_standardize,
                                 hypothesis_transform=wer_standardize)) * 100
        mean_whole_infer_time = sum(whole_infer_times)
        mean_encoder_infer_time = sum(encoder_infer_times)
        mean_decoder_with_time_infer_time = sum(decoder_with_past_infer_times)
        return word_accuracy, (mean_whole_infer_time, mean_encoder_infer_time, mean_decoder_with_time_infer_time)
    
    test_dataset = load_dataset("openslr/librispeech_asr", "clean", split="validation", streaming=True, trust_remote_code=True)
    test_dataset = test_dataset.shuffle(seed=42).take(TEST_DATASET_SIZE)
    test_samples = [sample for sample in test_dataset]
    
    accuracy_original, times_original = calculate_transcription_time_and_accuracy(ov_model, test_samples)
    accuracy_quantized, times_quantized = calculate_transcription_time_and_accuracy(ov_quantized_model, test_samples)
    print(f"Encoder performance speedup: {times_original[1] / times_quantized[1]:.3f}")
    print(f"Decoder with past performance speedup: {times_original[2] / times_quantized[2]:.3f}")
    print(f"Whole pipeline performance speedup: {times_original[0] / times_quantized[0]:.3f}")
    print(f"Whisper transcription word accuracy. Original model: {accuracy_original:.2f}%. Quantized model: {accuracy_quantized:.2f}%.")
    print(f"Accuracy drop: {accuracy_original - accuracy_quantized:.2f}%.")



.. parsed-literal::

    Measuring performance and accuracy:   0%|          | 0/50 [00:00<?, ?it/s]



.. parsed-literal::

    Measuring performance and accuracy:   0%|          | 0/50 [00:00<?, ?it/s]


.. parsed-literal::

    Encoder performance speedup: 1.352
    Decoder with past performance speedup: 1.342
    Whole pipeline performance speedup: 1.350
    Whisper transcription word accuracy. Original model: 81.67%. Quantized model: 83.67%.
    Accuracy drop: -1.99%.


Interactive demo
----------------



.. code:: ipython3

    import gradio as gr
    
    
    def transcribe(url, task, use_int8):
        output_file = Path("downloaded_video.mp4")
        yt = YouTube(url)
        yt.streams.get_highest_resolution().download(filename=output_file)
        inputs, duration = get_audio(output_file)
        m_pipe = int8_pipe if use_int8 else pipe
        transcription = m_pipe(inputs, generate_kwargs={"task": task.lower()}, return_timestamps=True)["chunks"]
        srt_lines = prepare_srt(transcription, duration)
        with output_file.with_suffix(".srt").open("w") as f:
            f.writelines(srt_lines)
        return [str(output_file), str(output_file.with_suffix(".srt"))]
    
    
    demo = gr.Interface(
        transcribe,
        [
            gr.Textbox(label="YouTube URL"),
            gr.Radio(["Transcribe", "Translate"], value="Transcribe"),
            gr.Checkbox(value=ov_quantized_model is not None, visible=ov_quantized_model is not None, label="Use INT8"),
        ],
        "video",
        examples=[["https://youtu.be/kgL5LBM-hFI", "Transcribe"]],
        allow_flagging="never",
    )
    try:
        demo.launch(debug=False)
    except Exception:
        demo.launch(share=True, debug=False)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/


.. parsed-literal::

    Running on local URL:  http://127.0.0.1:7860
    
    To create a public link, set `share=True` in `launch()`.








.. parsed-literal::

    Keyboard interruption in main thread... closing server.

