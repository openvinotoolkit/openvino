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
---

- `Imports <#imports>`__

-  `Settings <#settings>`__
-  `Download and Convert Public
   Model <#download-and-convert-public-model>`__

   -  `Download Model <#download-model>`__
   -  `Convert Model <#convert-model>`__

-  `Audio Processing <#audio-processing>`__

   -  `Define constants <#define-constants>`__
   -  `Available Audio Formats <#available-audio-formats>`__
   -  `Load Audio File <#load-audio-file>`__
   -  `Visualize Audio File <#visualize-audio-file>`__
   -  `Change Type of Data <#change-type-of-data>`__
   -  `Convert Audio to Mel
      Spectrum <#convert-audio-to-mel-spectrum>`__
   -  `Run Conversion from Audio to Mel
      Format <#run-conversion-from-audio-to-mel-format>`__
   -  `Visualize Mel Spectrogram <#visualize-mel-spectrogram>`__
   -  `Adjust Mel scale to Input <#adjust-mel-scale-to-input>`__

-  `Load the Model <#load-the-model>`__

   -  `Do Inference <#do-inference>`__
   -  `Read Output <#read-output>`__
   -  `Implementation of
      Decoding <#implementation-of-decoding>`__
   -  `Run Decoding and Print
      Output <#run-decoding-and-print-output>`__

Imports 
-------------------------------------------------

.. code:: ipython3

    %pip install -q "librosa>=0.8.1" "openvino-dev>=2023.1.0" "numpy<1.24"

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
    # Fetch `notebook_utils` module
    import urllib.request
    urllib.request.urlretrieve(
        url='https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/utils/notebook_utils.py',
        filename='notebook_utils.py'
    )
    from notebook_utils import download_file

Settings 
--------------------------------------------------

In this part, all variables used in the notebook are set.

.. code:: ipython3

    model_folder = "model"
    download_folder = "output"
    data_folder = "data"
    
    precision = "FP16"
    model_name = "quartznet-15x5-en"

Download and Convert Public Model 
---------------------------------------------------------------------------

If it is your first run, models will be downloaded and converted here.
It my take a few minutes. Use ``omz_downloader`` and ``omz_converter``,
which are command-line tools from the ``openvino-dev`` package.

Download Model 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
    
    sys.path.insert(0, str(path_to_model_weights))

.. code:: ipython3

    def convert_model(model_path:Path, converted_model_path:Path):
        """
        helper function for converting QuartzNet model to IR
        The function accepts path to directory with dowloaded packages, weights and configs using OMZ downloader, 
        initialize model and convert to OpenVINO model and serialize it to IR.
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
        # convert model to OpenVINO Model using model conversion API
        ov_model = ov.convert_model(model, example_input=torch.zeros([1, 64, 128]))
        # save model in IR format for next usage
        ov.save_model(ov_model, converted_model_path)

.. code:: ipython3

    # Check if a model is already converted (in the model directory).
    path_to_converted_weights = Path(f'{model_folder}/public/{model_name}/{precision}/{model_name}.bin')
    path_to_converted_model = Path(f'{model_folder}/public/{model_name}/{precision}/{model_name}.xml')
    
    if not path_to_converted_weights.is_file():
        downloaded_model_path = Path("output/public/quartznet-15x5-en/models")
        convert_model(downloaded_model_path, path_to_converted_model)


.. parsed-literal::

    [NeMo W 2023-09-11 15:01:17 jasper:148] Turned off 170 masked convolutions


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


.. parsed-literal::

    [NeMo W 2023-09-11 15:01:18 deprecated:66] Function ``local_parameters`` is deprecated. It is going to be removed in the 0.11 version.


Audio Processing 
----------------------------------------------------------

Now that the model is converted, load an audio file.

Define constants 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, locate an audio file and define the alphabet used by the model.
This tutorial uses the Latin alphabet beginning with a space symbol and
ending with a blank symbol. In this case it will be ``~``, but that
could be any other character.

.. code:: ipython3

    audio_file_name = "edge_to_cloud.ogg"
    alphabet = " abcdefghijklmnopqrstuvwxyz'~"

Available Audio Formats 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are multiple supported audio formats that can be used with the
model:

``AIFF``, ``AU``, ``AVR``, ``CAF``, ``FLAC``, ``HTK``, ``SVX``,
``MAT4``, ``MAT5``, ``MPC2K``, ``OGG``, ``PAF``, ``PVF``, ``RAW``,
``RF64``, ``SD2``, ``SDS``, ``IRCAM``, ``VOC``, ``W64``, ``WAV``,
``NIST``, ``WAVEX``, ``WVE``, ``XI``

Load Audio File 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Load the file after checking a file extension. Pass ``sr`` (stands for a
``sampling rate``) as an additional parameter. The model supports files
with a ``sampling rate`` of 16 kHz.

.. code:: ipython3

    # Download the audio from the openvino_notebooks storage 
    file_name = download_file(
        "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/audio/" + audio_file_name,
        directory=data_folder
    )
    
    audio, sampling_rate = librosa.load(path=str(file_name), sr=16000)

Now, you can play your audio file.

.. code:: ipython3

    ipd.Audio(audio, rate=sampling_rate)




.. raw:: html

    
    <audio  controls="controls" >
        <source src="data:audio/wav;base64,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" type="audio/wav" />
        Your browser does not support the audio element.
    </audio>




Visualize Audio File 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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



.. image:: 211-speech-to-text-with-output_files/211-speech-to-text-with-output_20_0.png


.. parsed-literal::

    (1025, 51)



.. image:: 211-speech-to-text-with-output_files/211-speech-to-text-with-output_20_2.png


Change Type of Data 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The file loaded in the previous step may contain data in ``float`` type
with a range of values between -1 and 1. To generate a viable input,
multiply each value by the max value of ``int16`` and convert it to
``int16`` type.

.. code:: ipython3

    if max(np.abs(audio)) <= 1:
        audio = (audio * (2**15 - 1))
    audio = audio.astype(np.int16)

Convert Audio to Mel Spectrum 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this step, convert a current audio file into `Mel
scale <https://en.wikipedia.org/wiki/Mel_scale>`__.

.. code:: ipython3

    mel_basis, spec = audio_to_mel(audio=audio.flatten(), sampling_rate=sampling_rate)

Visualize Mel Spectrogram 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For more information about Mel spectrogram, refer to this
`article <https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0>`__.
The first image visualizes Mel frequency spectrogram, the second one
presents filter bank for converting Hz to Mels.

.. code:: ipython3

    librosa.display.specshow(data=spec, sr=sampling_rate, x_axis='time', y_axis='log');
    plt.show();
    librosa.display.specshow(data=mel_basis, sr=sampling_rate, x_axis='linear');
    plt.ylabel('Mel filter');



.. image:: 211-speech-to-text-with-output_files/211-speech-to-text-with-output_28_0.png



.. image:: 211-speech-to-text-with-output_files/211-speech-to-text-with-output_28_1.png


Adjust Mel scale to Input 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before reading the network, make sure that the input is ready.

.. code:: ipython3

    audio = mel_to_input(mel_basis=mel_basis, spec=spec)

Load the Model 
--------------------------------------------------------

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




.. parsed-literal::

    Dropdown(description='Device:', index=2, options=('CPU', 'GPU', 'AUTO'), value='AUTO')



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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Everything is set up. Now, the only thing that remains is passing input
to the previously loaded network and running inference.

.. code:: ipython3

    character_probabilities = compiled_model([ov.Tensor(audio)])[0]

Read Output 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    transcription = ctc_greedy_decode(character_probabilities)
    print(transcription)


.. parsed-literal::

    from the edge to the cloud

