# Offline Speech Recognition Demo {#openvino_inference_engine_samples_speech_libs_and_demos_Offline_speech_recognition_demo}

This demo provides a command-line interface for automatic speech recognition using OpenVINO&trade;.  
Components used by this executable:

* `lspeech_s5_ext` model     - Example pretrained LibriSpeech DNN
* `speech_library.dll` (`.so`) - Open source speech recognition library that uses OpenVINO&trade; Inference Engine, Intel&reg; Speech Feature Extraction and Intel&reg; Speech Decoder libraries

## How It Works

The application transcribes speech from a given WAV file and outputs the text to the console.

## Running

The application requires two command-line parameters, which point to an audio file with speech to transcribe and a configuration file describing the resources to use for transcription.

### Parameters for Executable

* `-wave` - Path to input WAV to process. WAV file needs to be in the following format: RIFF WAVE PCM 16bit, 16kHz, 1 channel, with header.
* `-c`, `--config` - Path to configuration file with paths to resources and other parameters.

Example usage:

```sh
offline_speech_recognition_app.exe -wave="<path_to_audio>/inputAudio.wav" -c="<path_to_config>/configFile.cfg"
```

### Configuration File Description

The configuration file is an ASCII text file where:
* Parameter name and its value are separated with the space character
* Parameter and value pair ends with the end of line character

#### Parameter Description

| Parameter | Description | Value used for demo |
| --- | --- | --- |
| `-fe:rt:numCeps` | Number of MFCC cepstrums | *13* |
| `-fe:rt:contextLeft` | Numbers of past frames that are stacked to form input vector for neural network inference | *5* |
| `-fe:rt:contextRight` | Numbers of future frames that are stacked to form input vector for neural network inference | *5* |
| `-fe:rt:hpfBeta` | High pass filter beta coefficient, where 0.0f means no filtering | *0.0f* |
| `-fe:rt:inputDataType` | Feature extraction input format description | *INT16_16KHZ* |
| `-fe:rt:cepstralLifter` | Lifting factor | *22.0f* |
| `-fe:rt:noDct` | Flag: use DCT as final step or not | *0* |
| `-fe:rt:featureTransform` | [Kaldi](https://kaldi-asr.org/) feature transform file that normalizes stacked features for neural network inference | |
| `-dec:wfst:acousticModelFName` | Full path to the acoustic model file, for example `openvino_ir.xml`| |
| `-dec:wfst:acousticScaleFactor` | The acoustic log likelihood scaling factor | *0.1f* |
| `-dec:wfst:beamWidth` | Viterbi search beam width | *14.0f* |
| `-dec:wfst:latticeWidth` | Lattice beam width (extends the beam width) | *0.0f* |
| `-dec:wfst:nbest` | Number of n-best hypothesis to be generated | *1* |
| `-dec:wfst:confidenceAcousticScaleFactor` | Scaling parameter to factor in acoustic scores in confidence computations | *1.0f* |
| `-dec:wfst:confidenceLMScaleFactor` | Scaling parameter to factor in language model in confidence computations | *1.0f* |
| `-dec:wfst:hmmModelFName` | Full path to HMM model | |
| `-dec:wfst:fsmFName` | Full path to pronunciation model or full statically composed LM, if static composition is used | |
| `-dec:wfstotf:gramFsmFName` | Full path to grammar model | |
| `-dec:wfst:outSymsFName` | Full path to the output symbols (lexicon) filename | |
| `-dec:wfst:tokenBufferSize` | Token pool size expressed in number of DWORDs | *150000* |
| `-dec:wfstotf:traceBackLogSize` | Number of entries in traceback expressed as log2(N) | *19* |
| `-dec:wfstotf:minStableFrames` | The time expressed in frames, after which the winning hypothesis is recognized as stable and the final result can be printed | *45* |
| `-dec:wfst:maxCumulativeTokenSize` | Maximum fill rate of token buffer before token beam is adjusted to keep token buffer fill constant. Expressed as factor of buffer size (0.0, 1.0) | *0.2f* |
| `-dec:wfst:maxTokenBufferFill` | Active token count number triggering beam tightening expressed as factor of buffer size | *0.6f* |
| `-dec:wfst:maxAvgTokenBufferFill` | Average active token count number for utterance, which triggers beam tightening when exceeded. Expressed as factor of buffer size | *1.0f* |
| `-dec:wfst:tokenBufferMinFill` | Minimum fill rate of token buffer | *0.1f* |
| `-dec:wfst:pruningTighteningDelta` | Beam tightening value when token pool usage reaches the pool capacity | *1.0f* |
| `-dec:wfst:pruningRelaxationDelta` | Beam relaxation value when token pool is not meeting minimum fill ratio criterion | *0.5f* |
| `-dec:wfst:useScoreTrendForEndpointing` | Extend end pointing with acoustic feedback | *1* |
| `-dec:wfstotf:cacheLogSize` | Number of entries in LM cache expressed as log2(N) | *16* |
| `-eng:output:format` | Format of the speech recognition output | *text* |
| `-inference:contextLeft` | IE: Additional stacking option, independent from feature extraction | *0* |
| `-inference:contextRight` | IE: Additional stacking option, independent from feature extraction | *0* |
| `-inference:device`  | IE: Device used for neural computations | CPU |
| `-inference:numThreads` | IE: Number of threads used by GNA in SW mode | *1* |
| `-inference:scaleFactor` | IE: Scale factor used for static quantization | *3000.0* |
| `-inference:quantizationBits` | IE: Quantization resolution in bits | *16* or *8* |


## Demo Output

The resulting transcription for the sample audio file:

```sh
[ INFO ] Using feature transformation
[ INFO ] InferenceEngine API
[ INFO ] Device info:
[ INFO ]        CPU: MKLDNNPlugin
[ INFO ] Batch size: 1
[ INFO ] Model loading time: 61.01 ms
Recognition result:
HOW ARE YOU DOING
```