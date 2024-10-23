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


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.
    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    altair 5.4.1 requires typing-extensions>=4.10.0; python_version < "3.13", but you have typing-extensions 4.9.0 which is incompatible.
    descript-audiotools 0.7.2 requires protobuf<3.20,>=3.9.2, but you have protobuf 3.20.3 which is incompatible.
    detectron2 0.6 requires iopath<0.1.10,>=0.1.7, but you have iopath 0.1.10 which is incompatible.
    mobileclip 0.1.0 requires torchvision==0.14.1, but you have torchvision 0.19.1+cpu which is incompatible.
    modelscope-studio 0.5.0 requires gradio<5.0,>=4.0, but you have gradio 3.43.1 which is incompatible.
    openvino-dev 2024.4.0 requires openvino==2024.4.0, but you have openvino 2024.5.0.dev20241014 which is incompatible.
    parler-tts 0.2 requires transformers<=4.43.3,>=4.43.0, but you have transformers 4.45.2 which is incompatible.
    tensorflow 2.12.0 requires numpy<1.24,>=1.22, but you have numpy 1.24.4 which is incompatible.
    typeguard 4.3.0 requires typing-extensions>=4.10.0, but you have typing-extensions 4.9.0 which is incompatible.
    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.


Imports
~~~~~~~



.. code:: ipython3

    import torch
    import torchaudio
    from speechbrain.inference.interfaces import foreign_class
    
    import openvino as ov

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

    config.json:   0%|          | 0.00/1.84k [00:00<?, ?B/s]


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/configuration_utils.py:302: UserWarning: Passing `gradient_checkpointing` to a config initialization is deprecated and will be removed in v5 Transformers. Using `model.gradient_checkpointing_enable()` instead, or if you are using the `Trainer` API, pass `gradient_checkpointing=True` in your `TrainingArguments`.
      warnings.warn(



.. parsed-literal::

    pytorch_model.bin:   0%|          | 0.00/380M [00:00<?, ?B/s]


.. parsed-literal::

    speechbrain.lobes.models.huggingface_transformers.huggingface - Wav2Vec2Model is frozen.



.. parsed-literal::

    preprocessor_config.json:   0%|          | 0.00/159 [00:00<?, ?B/s]


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/speechbrain/utils/checkpoints.py:194: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
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

    out_prob, score, index, text_lab = classifier.classify_file("speechbrain/emotion-recognition-wav2vec2-IEMOCAP/anger.wav")
    print(f"Emotion Recognition with SpeechBrain PyTorch model: {text_lab}")


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
    signal, sr = torchaudio.load(str("./anger.wav"), channels_first=False)
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

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:4779: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
      warnings.warn(
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/wav2vec2/modeling_wav2vec2.py:871: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):


Step 3: OpenVINO model inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    import requests
    
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    open("notebook_utils.py", "w").write(r.text)
    
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

