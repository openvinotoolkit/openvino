SpeechBrain Emotion Recognition with OpenVINO
=============================================

`SpeechBrain <https://github.com/speechbrain/speechbrain>`__ is an
open-source PyTorch toolkit that accelerates Conversational AI
development, i.e., the technology behind speech assistants, chatbots,
and large language models.

Lear more in `GitHub
repo <https://github.com/speechbrain/speechbrain>`__ and
`paper <https://arxiv.org/pdf/2106.04624>`__

This notebook tutorial demonstrates optimization and inference of
speechbrain emotion recognition model with OpenVINO.


**Table of contents:**


-  `Installations <#installations>`__
-  `Imports <#imports>`__
-  `Prepare base model <#prepare-base-model>`__
-  `Initialize model <#initialize-model>`__
-  `PyTorch inference <#pytorch-inference>`__
-  `SpeechBrain model optimization with Intel
   OpenVINO <#speechbrain-model-optimization-with-intel-openvino>`__

   -  `Step 1: Prepare input tensor <#step-1-prepare-input-tensor>`__
   -  `Step 2: Convert model to OpenVINO
      IR <#step-2-convert-model-to-openvino-ir>`__
   -  `Step 3: OpenVINO model
      inference <#step-3-openvino-model-inference>`__

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

Installations
~~~~~~~~~~~~~



.. code:: ipython3

    %pip install -q "speechbrain>=1.0.0" --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install -q --upgrade --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    %pip install -q "transformers>=4.30.0" "huggingface_hub>=0.8.0" "SoundFile"
    %pip install -q "openvino>=2024.1.0"

Imports
~~~~~~~



.. code:: ipython3

    import torch
    import torchaudio
    from speechbrain.inference.interfaces import foreign_class
    from huggingface_hub import hf_hub_download
    
    import openvino as ov
    
    import requests
    from pathlib import Path
    
    if not Path("notebook_utils.py").exists():
        r = requests.get(
            url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
        )
        open("notebook_utils.py", "w").write(r.text)
    
    # Read more about telemetry collection at https://github.com/openvinotoolkit/openvino_notebooks?tab=readme-ov-file#-telemetry
    from notebook_utils import collect_telemetry
    
    collect_telemetry("speechbrain-emotion-recognition.ipynb")

Prepare base model
~~~~~~~~~~~~~~~~~~



The foreign_class function in SpeechBrain is a utility that allows you
to load and use custom PyTorch models within the SpeechBrain ecosystem.
It provides a convenient way to integrate external or custom-built
models into SpeechBrain’s inference pipeline without modifying the core
SpeechBrain codebase.

1. source: This argument specifies the source or location of the
   pre-trained model checkpoint. In this case,
   “speechbrain/emotion-recognition-wav2vec2-IEMOCAP” refers to a
   pre-trained model checkpoint available on the Hugging Face Hub.
2. pymodule_file: This argument is the path to a Python file containing
   the definition of your custom PyTorch model class. In this example,
   “custom_interface.py” is the name of the Python file that defines the
   CustomEncoderWav2vec2Classifier class.
3. classname: This argument specifies the name of the custom PyTorch
   model class defined in the pymodule_file. In this case,
   “CustomEncoderWav2vec2Classifier” is the name of the class that
   extends SpeechBrain’s Pretrained class and implements the necessary
   methods for inference.

.. code:: ipython3

    classifier = foreign_class(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier"
    )


.. parsed-literal::

    INFO:speechbrain.utils.fetching:Fetch hyperparams.yaml: Fetching from HuggingFace Hub 'speechbrain/emotion-recognition-wav2vec2-IEMOCAP' if not cached
    INFO:speechbrain.utils.fetching:Fetch custom_interface.py: Fetching from HuggingFace Hub 'speechbrain/emotion-recognition-wav2vec2-IEMOCAP' if not cached
    2024-11-05 19:56:33.843411: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-11-05 19:56:33.855243: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    E0000 00:00:1730822193.869707   99152 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    E0000 00:00:1730822193.873697   99152 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    2024-11-05 19:56:33.888531: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    


.. parsed-literal::

    config.json:   0%|          | 0.00/1.84k [00:00<?, ?B/s]


.. parsed-literal::

    /home/ea/work/py311/lib/python3.11/site-packages/transformers/configuration_utils.py:306: UserWarning: Passing `gradient_checkpointing` to a config initialization is deprecated and will be removed in v5 Transformers. Using `model.gradient_checkpointing_enable()` instead, or if you are using the `Trainer` API, pass `gradient_checkpointing=True` in your `TrainingArguments`.
      warnings.warn(
    


.. parsed-literal::

    pytorch_model.bin:   0%|          | 0.00/380M [00:00<?, ?B/s]


.. parsed-literal::

    WARNING:speechbrain.lobes.models.huggingface_transformers.huggingface:speechbrain.lobes.models.huggingface_transformers.huggingface - Wav2Vec2Model is frozen.
    


.. parsed-literal::

    preprocessor_config.json:   0%|          | 0.00/159 [00:00<?, ?B/s]


.. parsed-literal::

    INFO:speechbrain.utils.fetching:Fetch wav2vec2.ckpt: Fetching from HuggingFace Hub 'speechbrain/emotion-recognition-wav2vec2-IEMOCAP' if not cached
    


.. parsed-literal::

    wav2vec2.ckpt:   0%|          | 0.00/378M [00:00<?, ?B/s]



.. parsed-literal::

    model.safetensors:   0%|          | 0.00/380M [00:00<?, ?B/s]


.. parsed-literal::

    INFO:speechbrain.utils.fetching:Fetch model.ckpt: Fetching from HuggingFace Hub 'speechbrain/emotion-recognition-wav2vec2-IEMOCAP' if not cached
    


.. parsed-literal::

    model.ckpt:   0%|          | 0.00/13.2k [00:00<?, ?B/s]


.. parsed-literal::

    INFO:speechbrain.utils.fetching:Fetch label_encoder.txt: Fetching from HuggingFace Hub 'speechbrain/emotion-recognition-wav2vec2-IEMOCAP' if not cached
    


.. parsed-literal::

    label_encoder.txt:   0%|          | 0.00/83.0 [00:00<?, ?B/s]


.. parsed-literal::

    INFO:speechbrain.utils.parameter_transfer:Loading pretrained files for: wav2vec2, model, label_encoder
    /home/ea/work/py311/lib/python3.11/site-packages/speechbrain/utils/checkpoints.py:200: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      state_dict = torch.load(path, map_location=device)
    

Initialize model
~~~~~~~~~~~~~~~~



.. code:: ipython3

    # wav2vec2 torch model
    torch_model = classifier.mods["wav2vec2"].model

PyTorch inference
~~~~~~~~~~~~~~~~~



Perform emotion recognition on the sample audio file.

1. out_prob: Tensor or list containing the predicted probabilities or
   log probabilities for each emotion class.
2. score: Scalar value representing the predicted probability or log
   probability of the most likely emotion class.
3. index: Integer value representing the index of the most likely
   emotion class in the out_prob tensor or list.
4. text_lab: String or list of strings containing the textual labels
   corresponding to the predicted emotion classes ([“anger”,
   “happiness”, “sadness”, “neutrality”]).

.. code:: ipython3

    hf_hub_download(repo_id="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", filename="anger.wav", local_dir="data")
    
    
    out_prob, score, index, text_lab = classifier.classify_file("data/anger.wav")
    print(f"Emotion Recognition with SpeechBrain PyTorch model: {text_lab}")



.. parsed-literal::

    anger.wav:   0%|          | 0.00/201k [00:00<?, ?B/s]


.. parsed-literal::

    WARNING:speechbrain.dataio.encoder:CategoricalEncoder.expect_len was never called: assuming category count of 4 to be correct! Sanity check your encoder using `.expect_len`. Ensure that downstream code also uses the correct size. If you are sure this does not apply to you, use `.ignore_len`.
    

.. parsed-literal::

    Emotion Recognition with SpeechBrain PyTorch model: ['ang']
    

SpeechBrain model optimization with Intel OpenVINO
--------------------------------------------------



Step 1: Prepare input tensor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    # Using sample audio file
    signals = []
    batch_size = 1
    signal, sr = torchaudio.load(str("data/anger.wav"), channels_first=False)
    norm_audio = classifier.audio_normalizer(signal, sr)
    signals.append(norm_audio)
    
    sequence_length = norm_audio.shape[-1]
    
    wavs = torch.stack(signals, dim=0)
    wav_len = torch.tensor([sequence_length] * batch_size).unsqueeze(0)

Step 2: Convert model to OpenVINO IR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    # Model optimization process
    input_tensor = wavs.float()
    ov_model = ov.convert_model(torch_model, example_input=input_tensor)


.. parsed-literal::

    /home/ea/work/py311/lib/python3.11/site-packages/transformers/modeling_utils.py:5006: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
      warnings.warn(
    `loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
    /home/ea/work/py311/lib/python3.11/site-packages/transformers/models/wav2vec2/modeling_wav2vec2.py:872: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
    

Step 3: OpenVINO model inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    from notebook_utils import device_widget
    
    device = device_widget()
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



.. code:: ipython3

    core = ov.Core()
    
    # OpenVINO Compiled model
    compiled_model = core.compile_model(ov_model, device.value)
    
    # Perform model inference
    output_tensor = compiled_model(wavs)[0]
    output_tensor = torch.from_numpy(output_tensor)
    
    # output post-processing
    outputs = classifier.mods.avg_pool(output_tensor, wav_len)
    outputs = outputs.view(outputs.shape[0], -1)
    outputs = classifier.mods.output_mlp(outputs).squeeze(1)
    ov_out_prob = classifier.hparams.softmax(outputs)
    score, index = torch.max(ov_out_prob, dim=-1)
    text_lab = classifier.hparams.label_encoder.decode_torch(index)
    
    print(f"Emotion Recognition with OpenVINO Model: {text_lab}")


.. parsed-literal::

    Emotion Recognition with OpenVINO Model: ['ang']
    
