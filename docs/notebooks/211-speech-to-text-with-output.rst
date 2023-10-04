Speech to Text with OpenVINO™
=============================

This tutorial demonstrates speech-to-text recognition with OpenVINO.

This tutorial uses the `QuartzNet
15x5 <https://docs.openvino.ai/2021.4/omz_models_model_quartznet_15x5_en.html>`__
model. QuartzNet performs automatic speech recognition. Its design is
based on the Jasper architecture, which is a convolutional model trained
with Connectionist Temporal Classification (CTC) loss. The model is
available from `Open Model
Zoo <https://github.com/openvinotoolkit/open_model_zoo/>`__.

**Table of contents:**

- `Imports <#imports>`__
- `Settings <#settings>`__
- `Download and Convert Public Model <#download-and-convert-public-model>`__

  - `Download Model <#download-model>`__
  - `Convert Model <#convert-model>`__

- `Audio Processing <#audio-processing>`__

  - `Define constants <#define-constants>`__
  - `Available Audio Formats <#available-audio-formats>`__
  - `Load Audio File <#load-audio-file>`__
  - `Visualize Audio File <#visualize-audio-file>`__
  - `Change Type of Data <#change-type-of-data>`__
  - `Convert Audio to Mel Spectrum <#convert-audio-to-mel-spectrum>`__
  - `Run Conversion from Audio to Mel Format <#run-conversion-from-audio-to-mel-format>`__
  - `Visualize Mel Spectrogram <#visualize-mel-spectrogram>`__
  - `Adjust Mel scale to Input <#adjust-mel-scale-to-input>`__

- `Load the Model <#load-the-model>`__

  - `Do Inference <#do-inference>`__
  - `Read Output <#read-output>`__
  - `Implementation of Decoding <#implementation-of-decoding>`__
  - `Run Decoding and Print Output <#run-decoding-and-print-output>`__

Imports
###############################################################################################################################

.. code:: ipython3

    !pip install -q "librosa>=0.8.1" "openvino-dev==2023.1.0.dev20230811" "onnx"

.. code:: ipython3

    from pathlib import Path
    import sys
    
    import torch
    import torch.nn as nn
    import IPython.display as ipd
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display
    import numpy as np
    import scipy
    import openvino as ov

Settings
###############################################################################################################################

In this part, all variables used in the notebook are set.

.. code:: ipython3

    model_folder = "model"
    download_folder = "output"
    data_folder = "../data"
    
    precision = "FP16"
    model_name = "quartznet-15x5-en"

Download and Convert Public Model
###############################################################################################################################

If it is your first run, models will be downloaded and converted here.
It my take a few minutes. Use ``omz_downloader`` and ``omz_converter``,
which are command-line tools from the ``openvino-dev`` package.

Download Model
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The ``omz_downloader`` tool automatically creates a directory structure
and downloads the selected model. This step is skipped if the model is
already downloaded. The selected model comes from the public directory,
which means it must be converted into OpenVINO Intermediate
Representation (OpenVINO IR).

.. code:: ipython3

    # Check if a model is already downloaded (to the download directory).
    path_to_model_weights = Path(f'{download_folder}/public/{model_name}/models')
    downloaded_model_file = list(path_to_model_weights.glob('*.pth'))
    
    if not path_to_model_weights.is_dir() or len(downloaded_model_file) == 0:
        download_command = f"omz_downloader --name {model_name} --output_dir {download_folder} --precision {precision}"
        ! $download_command

Convert Model
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

In previous step, model was downloaded in PyTorch format. Currently,
PyTorch models supported in OpenVINO via ONNX exporting,
``torch.onnx.export`` function helps to trace PyTorch model to ONNX and
save it on disk. It is also recommended to convert model to OpenVINO
Intermediate Representation format for applying optimizations.

.. code:: ipython3

    def convert_model(model_path:Path, converted_model_path:Path):
        """
        helper function for converting QuartzNet model to IR
        The function accepts path to directory with dowloaded packages, weights and configs using OMZ downloader, 
        initialize model, export it to ONNX and then convert to OpenVINO model and serialize it to IR.
        Params:
          model_path: path to model modules, weights and configs downloaded via omz_downloader
          converted_model_path: path for saving converted model
        Returns:
          None
        """
        # add model path to PYTHONPATH for access to downloaded modules
        sys.path.append(str(model_path))
        
        # import necessary classes
        from ruamel.yaml import YAML
    
        from nemo.collections.asr import JasperEncoder, JasperDecoderForCTC
        from nemo.core import NeuralModuleFactory, DeviceType
    
        YAML = YAML(typ='safe')
    
        # utility fornction fr replacing 1d convolutions to 2d for better efficiency
        def convert_to_2d(model):
            for name, l in model.named_children():
                layer_type = l.__class__.__name__
                if layer_type == 'Conv1d':
                    new_layer = nn.Conv2d(l.in_channels, l.out_channels,
                                          (1, l.kernel_size[0]), (1, l.stride[0]),
                                          (0, l.padding[0]), (1, l.dilation[0]),
                                          l.groups, False if l.bias is None else True, l.padding_mode)
                    params = l.state_dict()
                    params['weight'] = params['weight'].unsqueeze(2)
                    new_layer.load_state_dict(params)
                    setattr(model, name, new_layer)
                elif layer_type == 'BatchNorm1d':
                    new_layer = nn.BatchNorm2d(l.num_features, l.eps)
                    new_layer.load_state_dict(l.state_dict())
                    new_layer.eval()
                    setattr(model, name, new_layer)
                else:
                    convert_to_2d(l)
        
        # model class
        class QuartzNet(torch.nn.Module):
            def __init__(self, model_config, encoder_weights, decoder_weights):
                super().__init__()
                with open(model_config, 'r') as config:
                    model_args = YAML.load(config)
                _ = NeuralModuleFactory(placement=DeviceType.CPU)
    
                encoder_params = model_args['init_params']['encoder_params']['init_params']
                self.encoder = JasperEncoder(**encoder_params)
                self.encoder.load_state_dict(torch.load(encoder_weights, map_location='cpu'))
    
                decoder_params = model_args['init_params']['decoder_params']['init_params']
                self.decoder = JasperDecoderForCTC(**decoder_params)
                self.decoder.load_state_dict(torch.load(decoder_weights, map_location='cpu'))
    
                self.encoder._prepare_for_deployment()
                self.decoder._prepare_for_deployment()
                convert_to_2d(self.encoder)
                convert_to_2d(self.decoder)
    
            def forward(self, input_signal):
                input_signal = input_signal.unsqueeze(axis=2)
                i_encoded = self.encoder(input_signal)
                i_log_probs = self.decoder(i_encoded)
    
                shape = i_log_probs.shape
                return i_log_probs.reshape(shape[0], shape[1], shape[3])
        
        # path to configs and weights for creating model instane
        model_config = model_path / ".nemo_tmp/module.yaml"
        encoder_weights = model_path / ".nemo_tmp/JasperEncoder.pt"
        decoder_weights = model_path / ".nemo_tmp/JasperDecoderForCTC.pt"
        # create model instance
        model = QuartzNet(model_config, encoder_weights, decoder_weights)
        # turn model to inference mode
        model.eval()
        # export model to ONNX with preserving dynamic shapes
        onnx_model_path = model_path / "quartznet.onnx"
        torch.onnx.export(
            model, 
            torch.zeros([1, 64, 128]), 
            onnx_model_path, 
            opset_version=11, 
            input_names=["audio_signal"], 
            output_names=['output'], 
            dynamic_axes={"audio_signal": {0: "batch_size", 2: "wave_len"}, "output": {0: "batch_size", 2: "wave_len"}}
        )
        # convert model to OpenVINO Model using model conversion API
        ov_model = ov.convert_model(str(onnx_model_path))
        # save model in IR format for next usage
        ov.save_model(ov_model, str(converted_model_path))

.. code:: ipython3

    # Check if a model is already converted (in the model directory).
    path_to_converted_weights = Path(f'{model_folder}/public/{model_name}/{precision}/{model_name}.bin')
    path_to_converted_model = Path(f'{model_folder}/public/{model_name}/{precision}/{model_name}.xml')
    
    if not path_to_converted_weights.is_file():
        downloaded_model_path = Path("output/public/quartznet-15x5-en/models")
        convert_model(downloaded_model_path, path_to_converted_model)

Audio Processing
###############################################################################################################################

Now that the model is converted, load an audio file.

Define constants
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

First, locate an audio file and define the alphabet used by the model.
This tutorial uses the Latin alphabet beginning with a space symbol and
ending with a blank symbol. In this case it will be ``~``, but that
could be any other character.

.. code:: ipython3

    audio_file_name = "edge_to_cloud.ogg"
    alphabet = " abcdefghijklmnopqrstuvwxyz'~"

Available Audio Formats
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

There are multiple supported audio formats that can be used with the
model:

``AIFF``, ``AU``, ``AVR``, ``CAF``, ``FLAC``, ``HTK``, ``SVX``,
``MAT4``, ``MAT5``, ``MPC2K``, ``OGG``, ``PAF``, ``PVF``, ``RAW``,
``RF64``, ``SD2``, ``SDS``, ``IRCAM``, ``VOC``, ``W64``, ``WAV``,
``NIST``, ``WAVEX``, ``WVE``, ``XI``

Load Audio File
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Load the file after checking a file extension. Pass ``sr`` (stands for a
``sampling rate``) as an additional parameter. The model supports files
with a ``sampling rate`` of 16 kHz.

.. code:: ipython3

    audio, sampling_rate = librosa.load(path=f'{data_folder}/audio/{audio_file_name}', sr=16000)

Now, you can play your audio file.

.. code:: ipython3

    ipd.Audio(audio, rate=sampling_rate)




.. raw:: html

    
    <audio  controls="controls" >
        <source src="data:audio/wav;base64,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" type="audio/wav" />
        Your browser does not support the audio element.
    </audio>




Visualize Audio File
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

You can visualize how your audio file presents on a wave plot and
spectrogram.

.. code:: ipython3

    plt.figure()
    librosa.display.waveshow(y=audio, sr=sampling_rate, max_points=50000, x_axis='time', offset=0.0);
    plt.show()
    specto_audio = librosa.stft(audio)
    specto_audio = librosa.amplitude_to_db(np.abs(specto_audio), ref=np.max)
    print(specto_audio.shape)
    librosa.display.specshow(specto_audio, sr=sampling_rate, x_axis='time', y_axis='hz');


.. parsed-literal::

    /tmp/ipykernel_2115037/2518307745.py:2: FutureWarning: waveshow() keyword argument 'x_axis' has been renamed to 'axis' in version 0.10.0.
    	This alias will be removed in version 1.0.
      librosa.display.waveshow(y=audio, sr=sampling_rate, max_points=50000, x_axis='time', offset=0.0);



.. image:: 211-speech-to-text-with-output_files/211-speech-to-text-with-output_21_1.png


.. parsed-literal::

    (1025, 51)



.. image:: 211-speech-to-text-with-output_files/211-speech-to-text-with-output_21_3.png


Change Type of Data
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The file loaded in the previous step may contain data in ``float`` type
with a range of values between -1 and 1. To generate a viable input,
multiply each value by the max value of ``int16`` and convert it to
``int16`` type.

.. code:: ipython3

    if max(np.abs(audio)) <= 1:
        audio = (audio * (2**15 - 1))
    audio = audio.astype(np.int16)

Convert Audio to Mel Spectrum
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Next, convert the pre-pre-processed audio to `Mel
Spectrum <https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53>`__.
For more information on why it needs to be done, refer to `this
article <https://towardsdatascience.com/audio-deep-learning-made-simple-part-2-why-mel-spectrograms-perform-better-aad889a93505>`__.

.. code:: ipython3

    def audio_to_mel(audio, sampling_rate):
        assert sampling_rate == 16000, "Only 16 KHz audio supported"
        preemph = 0.97
        preemphased = np.concatenate([audio[:1], audio[1:] - preemph * audio[:-1].astype(np.float32)])
    
        # Calculate the window length.
        win_length = round(sampling_rate * 0.02)
    
        # Based on the previously calculated window length, run short-time Fourier transform.
        spec = np.abs(librosa.core.spectrum.stft(preemphased, n_fft=512, hop_length=round(sampling_rate * 0.01),
                      win_length=win_length, center=True, window=scipy.signal.windows.hann(win_length), pad_mode='reflect'))
    
        # Create mel filter-bank, produce transformation matrix to project current values onto Mel-frequency bins.
        mel_basis = librosa.filters.mel(sr=sampling_rate, n_fft=512, n_mels=64, fmin=0.0, fmax=8000.0, htk=False)
        return mel_basis, spec
    
    
    def mel_to_input(mel_basis, spec, padding=16):
        # Convert to a logarithmic scale.
        log_melspectrum = np.log(np.dot(mel_basis, np.power(spec, 2)) + 2 ** -24)
    
        # Normalize the output.
        normalized = (log_melspectrum - log_melspectrum.mean(1)[:, None]) / (log_melspectrum.std(1)[:, None] + 1e-5)
    
        # Calculate padding.
        remainder = normalized.shape[1] % padding
        if remainder != 0:
            return np.pad(normalized, ((0, 0), (0, padding - remainder)))[None]
        return normalized[None]

Run Conversion from Audio to Mel Format
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

In this step, convert a current audio file into `Mel
scale <https://en.wikipedia.org/wiki/Mel_scale>`__.

.. code:: ipython3

    mel_basis, spec = audio_to_mel(audio=audio.flatten(), sampling_rate=sampling_rate)

Visualize Mel Spectrogram
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

For more information about Mel spectrogram, refer to this
`article <https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0>`__.
The first image visualizes Mel frequency spectrogram, the second one
presents filter bank for converting Hz to Mels.

.. code:: ipython3

    librosa.display.specshow(data=spec, sr=sampling_rate, x_axis='time', y_axis='log');
    plt.show();
    librosa.display.specshow(data=mel_basis, sr=sampling_rate, x_axis='linear');
    plt.ylabel('Mel filter');



.. image:: 211-speech-to-text-with-output_files/211-speech-to-text-with-output_29_0.png



.. image:: 211-speech-to-text-with-output_files/211-speech-to-text-with-output_29_1.png


Adjust Mel scale to Input
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Before reading the network, make sure that the input is ready.

.. code:: ipython3

    audio = mel_to_input(mel_basis=mel_basis, spec=spec)

Load the Model
###############################################################################################################################

Now, you can read and load the network.

.. code:: ipython3

    core = ov.Core()

You may run the model on multiple devices. By default, it will load the
model on CPU (you can choose manually CPU, GPU etc.) or let the engine
choose the best available device (AUTO).

To list all available devices that can be used, run
``print(core.available_devices)`` command.

.. code:: ipython3

    print(core.available_devices)


.. parsed-literal::

    ['CPU', 'GPU']


Select device from dropdown list

.. code:: ipython3

    import ipywidgets as widgets
    
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value='AUTO',
        description='Device:',
        disabled=False,
    )
    
    device

.. code:: ipython3

    model = core.read_model(
        model=f"{model_folder}/public/{model_name}/{precision}/{model_name}.xml"
    )
    model_input_layer = model.input(0)
    shape = model_input_layer.partial_shape
    shape[2] = -1
    model.reshape({model_input_layer: shape})
    compiled_model = core.compile_model(model=model, device_name=device.value)

Do Inference
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Everything is set up. Now, the only thing that remains is passing input
to the previously loaded network and running inference.

.. code:: ipython3

    output_layer_ir = compiled_model.output(0)
    
    character_probabilities = compiled_model([ov.Tensor(audio)])[output_layer_ir]

Read Output
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

After inference, you need to reach out the output. The default output
format for ``QuartzNet 15x5`` are per-frame probabilities (after
LogSoftmax) for every symbol in the alphabet, name - output, shape -
1x64x29, output data format is ``BxNxC``, where:

-  B - batch size
-  N - number of audio frames
-  C - alphabet size, including the Connectionist Temporal
   Classification (CTC) blank symbol

You need to make it in a more human-readable format. To do this you, use
a symbol with the highest probability. When you hold a list of indexes
that are predicted to have the highest probability, due to limitations
given by `Connectionist Temporal Classification
Decoding <https://towardsdatascience.com/beam-search-decoding-in-ctc-trained-neural-networks-5a889a3d85a7>`__
you will remove concurrent symbols and then remove all the blanks.

The last step is getting symbols from corresponding indexes in charlist.

.. code:: ipython3

    # Remove unnececery dimension
    character_probabilities = np.squeeze(character_probabilities)
    
    # Run argmax to pick most possible symbols
    character_probabilities = np.argmax(character_probabilities, axis=1)

Implementation of Decoding
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

To decode previously explained output, you need the `Connectionist
Temporal Classification (CTC)
decode <https://towardsdatascience.com/beam-search-decoding-in-ctc-trained-neural-networks-5a889a3d85a7>`__
function. This solution will remove consecutive letters from the output.

.. code:: ipython3

    def ctc_greedy_decode(predictions):
        previous_letter_id = blank_id = len(alphabet) - 1
        transcription = list()
        for letter_index in predictions:
            if previous_letter_id != letter_index != blank_id:
                transcription.append(alphabet[letter_index])
            previous_letter_id = letter_index
        return ''.join(transcription)

Run Decoding and Print Output
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. code:: ipython3

    transcription = ctc_greedy_decode(character_probabilities)
    print(transcription)


.. parsed-literal::

    from the edge to the cloud

