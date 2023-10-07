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
subtitles in a sample video. Notebook contains the following steps: 

1. Download the model. 
2. Instantiate the PyTorch model pipeline. 
3. Convert model to OpenVINO IR, using model conversion API. 
4. Run the Whisper pipeline with OpenVINO models.

**Table of contents:**

- `Prerequisites <#Prerequisites>`__
- `Instantiate model <#Instantiate-model>`__

  - `Convert model to OpenVINO Intermediate Representation (IR) format. <#convert-model-to-openvino-intermediate-representation-ir-format>`__
  - `Convert Whisper Encoder to OpenVINO IR <#convert-whisper-encoder-to-openvino-ir>`__
  - `Convert Whisper decoder to OpenVINO IR <#convert-whisper-decoder-to-openvino-ir>`__

- `Prepare inference pipeline <#prepare-inference-pipeline>`__

  - `Select inference device <#select-inference-device>`__

- `Run video transcription pipeline <#run-video-transcription-pipeline>`__
.. - `Interactive demo <#interactive-demo>`__

Prerequisites
###############################################################################################################################

Install dependencies.

.. code:: ipython3

    %pip install -q "openvino==2023.1.0.dev20230811"
    %pip install -q "python-ffmpeg<=1.0.16" moviepy transformers onnx
    %pip install -q -I "git+https://github.com/garywu007/pytube.git"
    %pip install -q -U gradio
    %pip install -q -I "git+https://github.com/openai/whisper.git@e8622f9afc4eba139bf796c210f5c01081000472"


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 23.3 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    Note: you may need to restart the kernel to use updated packages.
    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 23.3 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    ppgan 2.1.0 requires imageio==2.9.0, but you have imageio 2.31.3 which is incompatible.
    ppgan 2.1.0 requires librosa==0.8.1, but you have librosa 0.9.2 which is incompatible.
    ppgan 2.1.0 requires opencv-python<=4.6.0.66, but you have opencv-python 4.8.0.76 which is incompatible.
    Note: you may need to restart the kernel to use updated packages.
    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 23.3 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    Note: you may need to restart the kernel to use updated packages.
    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 23.3 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    Note: you may need to restart the kernel to use updated packages.
    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 23.3 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    black 21.7b0 requires tomli<2.0.0,>=0.2.6, but you have tomli 2.0.1 which is incompatible.
    google-auth 2.22.0 requires urllib3<2.0, but you have urllib3 2.0.4 which is incompatible.
    nncf 2.5.0.dev0+90a1e860 requires networkx<=2.8.2,>=2.6, but you have networkx 3.1 which is incompatible.
    onnxconverter-common 1.14.0 requires protobuf==3.20.2, but you have protobuf 4.24.3 which is incompatible.
    paddleclas 2.5.1 requires faiss-cpu==1.7.1.post2, but you have faiss-cpu 1.7.4 which is incompatible.
    paddleclas 2.5.1 requires gast==0.3.3, but you have gast 0.4.0 which is incompatible.
    ppgan 2.1.0 requires imageio==2.9.0, but you have imageio 2.31.3 which is incompatible.
    ppgan 2.1.0 requires librosa==0.8.1, but you have librosa 0.9.2 which is incompatible.
    ppgan 2.1.0 requires opencv-python<=4.6.0.66, but you have opencv-python 4.8.0.76 which is incompatible.
    pyannote-audio 2.0.1 requires networkx<3.0,>=2.6, but you have networkx 3.1 which is incompatible.
    pytorch-lightning 1.6.5 requires protobuf<=3.20.1, but you have protobuf 4.24.3 which is incompatible.
    tensorflow 2.12.0 requires numpy<1.24,>=1.22, but you have numpy 1.24.4 which is incompatible.
    tf2onnx 1.15.1 requires protobuf~=3.20.2, but you have protobuf 4.24.3 which is incompatible.
    torchaudio 0.13.1+cpu requires torch==1.13.1, but you have torch 2.0.1 which is incompatible.
    torchvision 0.14.1+cpu requires torch==1.13.1, but you have torch 2.0.1 which is incompatible.
    Note: you may need to restart the kernel to use updated packages.


Instantiate model
###############################################################################################################################

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
the authors of the model. In this tutorial, we will use the ``base``
model, but the same actions are also applicable to other models from
Whisper family.

.. code:: ipython3

    import whisper
    
    model = whisper.load_model("base")
    model.to("cpu")
    model.eval()
    pass

Convert model to OpenVINO Intermediate Representation (IR) format.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

For best results with OpenVINO, it is recommended to convert the model
to OpenVINO IR format. We need to provide initialized model object and
example of inputs for shape inference. We will use ``ov.convert_model``
functionality to convert models. The ``ov.convert_model`` Python
function returns an OpenVINO model ready to load on device and start
making predictions. We can save it on disk for next usage with
``ov.save_model``.

Convert Whisper Encoder to OpenVINO IR
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. code:: ipython3

    from pathlib import Path
    
    WHISPER_ENCODER_OV = Path("whisper_encoder.xml")
    WHISPER_DECODER_OV = Path("whisper_decoder.xml")

.. code:: ipython3

    import torch
    import openvino as ov
    
    mel = torch.zeros((1, 80, 3000))
    audio_features = model.encoder(mel)
    encoder_model = ov.convert_model(model.encoder, example_input=mel)
    ov.save_model(encoder_model, WHISPER_ENCODER_OV)


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/whisper/model.py:166: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"


Convert Whisper decoder to OpenVINO IR
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

To reduce computational complexity, the decoder uses cached key/value
projections in attention modules from the previous steps. We need to
modify this process for correct tracing.

There are 2 types of attention modules in Whisper Decoder -
self-attention, that makes projection for internal decoder state and
cross-attention, that uses internal state of encoder for calculating
attention. Decoder model runs autoregressively, it means that each new
step uses prediction from previous step as input and in the same time it
conditioned by encoder hidden state calculated before decoding start. To
sum up, it is enough calculate cross-attention once on first step and
reuse it for next steps for reducing computational complexity.
Self-attention hidden state for sequence that generated on previous
steps remains without changes, so it is possible to calculate it only
for current token and then join it to previously generated.

.. code:: ipython3

    import torch
    from typing import Optional, Tuple
    from functools import partial
    
    
    def attention_forward(
            attention_module,
            x: torch.Tensor,
            xa: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
            kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """
        Override for forward method of decoder attention module with storing cache values explicitly.
        Parameters:
          attention_module: current attention module
          x: input token ids.
          xa: input audio features (Optional).
          mask: mask for applying attention (Optional).
          kv_cache: dictionary with cached key values for attention modules.
          idx: idx for search in kv_cache.
        Returns:
          attention module output tensor
          updated kv_cache
        """
        q = attention_module.query(x)
    
        if xa is None:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = attention_module.key(x)
            v = attention_module.value(x)
            if kv_cache is not None:
                k = torch.cat((kv_cache[0], k), dim=1)
                v = torch.cat((kv_cache[1], v), dim=1)
            
        else:
            if kv_cache is None or kv_cache[0].shape[1] == 0:
                # for cross-attention, calculate keys and values once and reuse in subsequent calls.
                k = attention_module.key(xa)
                v = attention_module.value(xa)
            else:
                k, v = kv_cache
        
        kv_cache_new = (k, v)
    
        wv, qk = attention_module.qkv_attention(q, k, v, mask)
        return attention_module.out(wv), kv_cache_new
    
    
    def block_forward(
        residual_block,
        x: torch.Tensor,
        xa: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """
        Override for residual block forward method for providing kv_cache to attention module.
          Parameters:
            residual_block: current residual block.
            x: input token_ids.
            xa: input audio features (Optional).
            mask: attention mask (Optional).
            kv_cache: cache for storing attention key values.
          Returns:
            x: residual block output
            kv_cache: updated kv_cache
    
        """
        x0, kv_cache_self = residual_block.attn(residual_block.attn_ln(
            x), mask=mask, kv_cache=kv_cache[0])
        x = x + x0
        if residual_block.cross_attn:
            x1, kv_cache_cross = residual_block.cross_attn(
                residual_block.cross_attn_ln(x), xa, kv_cache=kv_cache[1])
            x = x + x1
        x = x + residual_block.mlp(residual_block.mlp_ln(x))
        return x, (kv_cache_self, kv_cache_cross)
    
    class CrossAttnKVGetter(torch.nn.Module):
        """
        Helper class for scripting approach of caching cross attention key values.
        The main idea that they should be calculated once and reused for next steps.
        Tracing can not correctly catch condition for that, that is why we need to use scripting for this part of model.
        """
        def __init__(self, attn):
            super().__init__()
            self.attn_key = attn.key
            self.attn_value = attn.value
    
        def forward(self, xa: torch.Tensor, kv_cache: Tuple[torch.Tensor, torch.Tensor]):
            if kv_cache is None or kv_cache[0].shape[1] == 0:
                # for cross-attention, calculate keys and values once and reuse in subsequent calls.
                k = self.attn_key(xa)
                v = self.attn_value(xa)
            else:
                k, v = kv_cache
            return k, v
        
    def crossattention_forward(
            attention_module,
            x: torch.Tensor,
            xa: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
            kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """
        Override for forward method of decoder cross attention module with storing cache values explicitly.
        Parameters:
          attention_module: current attention module
          x: input token ids.
          xa: input audio features (Optional).
          mask: mask for applying attention (Optional).
          kv_cache: dictionary with cached key values for attention modules.
          idx: idx for search in kv_cache.
        Returns:
          attention module output tensor
          updated kv_cache
        """
        q = attention_module.query(x)
    
        if xa is None:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = attention_module.key(x)
            v = attention_module.value(x)        
        else:
            k, v = attention_module.kv_getter(xa, kv_cache)
        kv_cache_new = (k, v)
    
        wv, qk = attention_module.qkv_attention(q, k, v, mask)
        return attention_module.out(wv), kv_cache_new
    
    
    # update forward functions
    for _, block in enumerate(model.decoder.blocks):
        block.forward = partial(block_forward, block)
        block.attn.forward = partial(attention_forward, block.attn)
        if block.cross_attn:
            kv_getter = CrossAttnKVGetter(block.cross_attn)
            block.cross_attn.kv_getter = torch.jit.script(kv_getter)
            block.cross_attn.forward = partial(crossattention_forward, block.cross_attn)
    
    
    def decoder_forward(decoder, x: torch.Tensor, xa: torch.Tensor, kv_cache: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor]]] = None):
        """
        Override for decoder forward method.
        Parameters:
          x: torch.LongTensor, shape = (batch_size, <= n_ctx) the text tokens
          xa: torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
               the encoded audio features to be attended on
          kv_cache: Dict[str, torch.Tensor], attention modules hidden states cache from previous steps 
        """
        if kv_cache is not None:
            offset = kv_cache[0][0][0].shape[1]
        else:
            offset = 0
            kv_cache = [(None, None) for _ in range(len(decoder.blocks))]
        x = decoder.token_embedding(
            x) + decoder.positional_embedding[offset: offset + x.shape[-1]]
        x = x.to(xa.dtype)
        kv_cache_upd = []
    
        for block, kv_block_cache in zip(decoder.blocks, kv_cache):
            x, kv_block_cache_upd = block(x, xa, mask=decoder.mask, kv_cache=kv_block_cache)
            kv_cache_upd.append(tuple(kv_block_cache_upd))
    
        x = decoder.ln(x)
        logits = (
            x @ torch.transpose(decoder.token_embedding.weight.to(x.dtype), 1, 0)).float()
    
        return logits, tuple(kv_cache_upd)
    
    
    # override decoder forward
    model.decoder.forward = partial(decoder_forward, model.decoder)

.. code:: ipython3

    encoder_hidden_size = audio_features.shape[2]
    kv_cache_init = [((torch.zeros((5, 0, encoder_hidden_size)), torch.zeros((5, 0, encoder_hidden_size))), (torch.zeros((1, 0, encoder_hidden_size)), torch.zeros((1, 0, encoder_hidden_size)))) for _ in range(len(model.decoder.blocks))] 

.. code:: ipython3

    tokens = torch.ones((5, 3), dtype=torch.int64)
    logits, kv_cache = model.decoder(tokens, audio_features, kv_cache=kv_cache_init)
    
    tokens = torch.ones((5, 1), dtype=torch.int64)
    decoder_model = ov.convert_model(model.decoder, example_input=(tokens, audio_features, kv_cache))
    decoder_cache_input = decoder_model.inputs[2:]
    for i in range(2, len(decoder_cache_input), 4):
        decoder_cache_input[i].get_node().set_partial_shape(ov.PartialShape([-1, -1, encoder_hidden_size]))
        decoder_cache_input[i + 1].get_node().set_partial_shape(ov.PartialShape([-1, -1, encoder_hidden_size]))
       
    decoder_model.validate_nodes_and_infer_types()
    ov.save_model(decoder_model, WHISPER_DECODER_OV)
    del decoder_model


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/torch/jit/_trace.py:154: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations. (Triggered internally at aten/src/ATen/core/TensorBody.h:486.)
      if a.grad is not None:


The decoder model autoregressively predicts the next token guided by
encoder hidden states and previously predicted sequence. This means that
the shape of inputs which depends on the previous step (inputs for
tokens and attention hidden states from previous step) are dynamic. For
efficient utilization of memory, you define an upper bound for dynamic
input shapes.

Prepare inference pipeline
###############################################################################################################################

The image below illustrates the pipeline of video transcribing using the
Whisper model.

.. figure:: https://user-images.githubusercontent.com/29454499/204536733-1f4342f7-2328-476a-a431-cb596df69854.png
   :alt: whisper_pipeline.png

   whisper_pipeline.png

To run the PyTorch Whisper model, we just need to call the
``model.transcribe(audio, **parameters)`` function. We will try to reuse
original model pipeline for audio transcribing after replacing the
original models with OpenVINO IR versions.

Select inference device
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Select device from dropdown list for running inference using OpenVINO:

.. code:: ipython3

    core = ov.Core()

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

    from utils import patch_whisper_for_ov_inference, OpenVINOAudioEncoder, OpenVINOTextDecoder
    
    patch_whisper_for_ov_inference(model)
    
    model.encoder = OpenVINOAudioEncoder(core, WHISPER_ENCODER_OV, device=device.value)
    model.decoder = OpenVINOTextDecoder(core, WHISPER_DECODER_OV, device=device.value)

Run video transcription pipeline
###############################################################################################################################

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



.. code:: ipython3

    transcription = model.transcribe(audio, task=task.value)

"The results will be saved in the ``downloaded_video.srt`` file. SRT is
one of the most popular formats for storing subtitles and is compatible
with many modern video players. This file can be used to embed
transcription into videos during playback or by injecting them directly
into video files using ``ffmpeg``.

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
    00:00:00,000 --> 00:00:05,000
     Oh, what's that?
    
    2
    00:00:05,000 --> 00:00:09,000
     Oh, wow.
    
    3
    00:00:09,000 --> 00:00:10,000
     Hello, humans.
    
    4
    00:00:13,000 --> 00:00:15,000
     Focus on me.
    
    5
    00:00:15,000 --> 00:00:18,000
     Focus on the guard.
    
    6
    00:00:18,000 --> 00:00:22,000
     Don't tell anyone what you've seen in here.
    
    7
    00:00:22,000 --> 00:00:23,000
     Oh, my.
    
    8
    00:00:23,000 --> 00:00:24,000
     Have you seen what's in there?
    
    9
    00:00:24,000 --> 00:00:25,000
     They have intel.
    
    10
    00:00:25,000 --> 00:00:27,000
     This is where it all changes.
    
    


.. Interactive demo
.. ###############################################################################################################################

.. .. code:: ipython3

..     import gradio as gr
    
    
..     def transcribe(url, task):
..         output_file = Path("downloaded_video.mp4")
..         yt = YouTube(url)
..         yt.streams.get_highest_resolution().download(filename=output_file)
..         audio = get_audio(output_file)
..         transcription = model.transcribe(audio, task=task.lower())
..         srt_lines = prepare_srt(transcription)
..         with output_file.with_suffix(".srt").open("w") as f:
..             f.writelines(srt_lines)
..         return [str(output_file), str(output_file.with_suffix(".srt"))]
    
    
..     demo = gr.Interface(
..         transcribe,
..         [gr.Textbox(label="YouTube URL"), gr.Radio(["Transcribe", "Translate"], value="Transcribe")],
..         "video",
..         examples=[["https://youtu.be/kgL5LBM-hFI", "Transcribe"]],
..         allow_flagging="never"
..     )
..     try:
..         demo.launch(debug=False)
..     except Exception:
..         demo.launch(share=True, debug=False)
..     # if you are launching remotely, specify server_name and server_port
..     # demo.launch(server_name='your server name', server_port='server port in int')
..     # Read more in the docs: https://gradio.app/docs/


.. .. parsed-literal::

..     Running on local URL:  http://127.0.0.1:7860
    
..     To create a public link, set `share=True` in `launch()`.



.. .. raw:: html

..     <div><iframe src="http://127.0.0.1:7860/" width="100%" height="500" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>

