# Automatic Speech Recognition C++ Sample

This topic shows how to run the speech sample application, which
demonstrates acoustic model inference based on Kaldi\* neural networks
and speech feature vectors.

## How It Works

Upon the start-up, the application reads command line parameters
and loads a Kaldi-trained neural network along with Kaldi ARK speech
feature vector file to the Inference Engine plugin. It then performs
inference on all speech utterances stored in the input ARK
file. Context-windowed speech frames are processed in batches of 1-8
frames according to the `-bs` parameter.  Batching across utterances is
not supported by this sample.  When inference is done, the application
creates an output ARK file.  If the `-r` option is given, error
statistics are provided for each speech utterance as shown above.

### GNA-specific details

#### Quantization

If the GNA device is selected (for example, using the `-d` GNA flag),
the GNA Inference Engine plugin quantizes the model and input feature
vector sequence to integer representation before performing inference.
Several parameters control neural network quantization.  The `-q` flag
determines the quantization mode.  Three modes are supported: static,
dynamic, and user-defined.  In static quantization mode, the first
utterance in the input ARK file is scanned for dynamic range.  The
scale factor (floating point scalar multiplier) required to scale the
maximum input value of the first utterance to 16384 (15 bits) is used
for all subsequent inputs.  The neural network is quantized to
accomodate the scaled input dynamic range.  In user-defined
quantization mode, the user may specify a scale factor via the `-sf`
flag that will be used for static quantization.  In dynamic
quantization mode, the scale factor for each input batch is computed
just before inference on that batch.  The input and network are
(re)quantized on-the-fly using an efficient procedure.

The `-qb` flag provides a hint to the GNA plugin regarding the preferred
target weight resolution for all layers.  For example, when `-qb 8` is
specified, the plugin will use 8-bit weights wherever possible in the
network.  Note that it is not always possible to use 8-bit weights due
to GNA hardware limitations.  For example, convolutional layers always
use 16-bit weights (GNA harware verison 1 and 2).  This limitation
will be removed in GNA hardware version 3 and higher.

#### Execution Modes

Several execution modes are supported via the `-d` flag.  If the device
is set to `CPU` mode, then all calculation will be performed  on CPU device
using CPU Plugin.  If the device is set to `GNA_AUTO`, then the GNA hardware is
used if available and the driver is installed.  Otherwise, the GNA device is 
emulated in fast-but-not-bit-exact mode.  If the device is set to `GNA_HW`,
then the GNA hardware is used if available and the driver is installed.
Otherwise, an error will occur.  If the device is set to `GNA_SW`, the
GNA device is emulated in fast-but-not-bit-exact mode.  Finally, if
the device is set to `GNA_SW_EXACT`, the GNA device is emulated in
bit-exact mode.

#### Loading and Saving Models

The GNA plugin supports loading and saving of the GNA-optimized model
(non-IR) via the `-rg` and `-wg` flags.  Thereby, it is possible to avoid
the cost of full model quantization at run time. The GNA plugin also
supports export of firmware-compatible embedded model images for the
Intel® Speech Enabling Developer Kit and Amazon Alexa* Premium
Far-Field Voice Development Kit via the `-we` flag (save only).

In addition to performing inference directly from a GNA model file, these options make it possible to:
- Convert from IR format to GNA format model file (`-m`, `-wg`)
- Convert from IR format to embedded format model file (`-m`, `-we`)
- Convert from GNA format to embedded format model file (`-rg`, `-we`)


## Running

Running the application with the `-h` option yields the following
usage message:

```sh
$ ./speech_sample -h
InferenceEngine:
    API version ............ <version>
    Build .................. <number>

speech_sample [OPTION]
Options:

    -h                      Print a usage message.
    -i "<path>"             Required. Paths to an .ark files. Example of usage: <file1.ark,file2.ark> or <file.ark>.
    -m "<path>"             Required. Path to an .xml file with a trained model (required if -rg is missing).
    -o "<path>"             Optional. Output file name (default name is "scores.ark").
    -l "<absolute_path>"    Required for CPU custom layers. Absolute path to a shared library with the kernel implementations.
    -d "<device>"           Optional. Specify a target device to infer on. CPU, GPU, GNA_AUTO, GNA_HW, GNA_SW, GNA_SW_EXACT and HETERO with combination of GNA
     as the primary device and CPU as a secondary (e.g. HETERO:GNA,CPU) are supported. The list of available devices is shown below. The sample will look for a suitable plugin for device specified.
    -p                      Optional. Plugin name. For example, GPU. If this parameter is set, the sample will look for this plugin only
    -pc                     Optional. Enables performance report
    -q "<mode>"             Optional. Input quantization mode:  "static" (default), "dynamic", or "user" (use with -sf).
    -qb "<integer>"         Optional. Weight bits for quantization:  8 or 16 (default)
    -sf "<double>"          Optional. Input scale factor for quantization (use with -q user).
    -bs "<integer>"         Optional. Batch size 1-8 (default 1)
    -r "<path>"             Optional. Read reference score .ark file and compare scores.
    -rg "<path>"            Optional. Read GNA model from file using path/filename provided (required if -m is missing).
    -wg "<path>"            Optional. Write GNA model to file using path/filename provided.
    -we "<path>"            Optional. Write GNA embedded model to file using path/filename provided.
    -nthreads "<integer>"   Optional. Number of threads to use for concurrent async inference requests on the GNA.
    -cw_l "<integer>"       Optional. Number of frames for left context windows (default is 0). Works only with context window networks.
                            If you use the cw_l or cw_r flag, then batch size and nthreads arguments are ignored.
    -cw_r "<integer>"       Optional. Number of frames for right context windows (default is 0). Works only with context window networks.
                            If you use the cw_r or cw_l flag, then batch size and nthreads arguments are ignored.

```

Running the application with the empty list of options yields the
usage message given above and an error message.

### Model Preparation

You can use the following model optimizer command to convert a Kaldi
nnet1 or nnet2 neural network to Intel IR format:

```sh
$ python3 mo.py --framework kaldi --input_model wsj_dnn5b_smbr.nnet --counts wsj_dnn5b_smbr.counts --remove_output_softmax
```

Assuming that the model optimizer (`mo.py`), Kaldi-trained neural
network, `wsj_dnn5b_smbr.nnet`, and Kaldi class counts file,
`wsj_dnn5b_smbr.counts`, are in the working directory this produces
the Intel IR network consisting of `wsj_dnn5b_smbr.xml` and
`wsj_dnn5b_smbr.bin`.

The following pre-trained models are available:

* wsj\_dnn5b\_smbr
* rm\_lstm4f
* rm\_cnn4a\_smbr

All of them can be downloaded from [https://download.01.org/openvinotoolkit/models_contrib/speech/kaldi](https://download.01.org/openvinotoolkit/models_contrib/speech/kaldi) or using the OpenVINO [Model Downloader](https://github.com/opencv/open_model_zoo/tree/2018/model_downloader) .


### Speech Inference

Once the IR is created, you can use the following command to do
inference on Intel^&reg; Processors with the GNA co-processor (or
emulation library):

```sh
$ ./speech_sample -d GNA_AUTO -bs 2 -i wsj_dnn5b_smbr_dev93_10.ark -m wsj_dnn5b_smbr_fp32.xml -o scores.ark -r wsj_dnn5b_smbr_dev93_scores_10.ark
```

Here, the floating point Kaldi-generated reference neural network
scores (`wsj_dnn5b_smbr_dev93_scores_10.ark`) corresponding to the input
feature file (`wsj_dnn5b_smbr_dev93_10.ark`) are assumed to be available
for comparison.

> **NOTE**: Before running the sample with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](./docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).

## Sample Output

The acoustic log likelihood sequences for all utterances are stored in
the Kaldi ARK file, `scores.ark`.  If the `-r` option is used, a report on
the statistical score error is generated for each utterance such as
the following:

``` sh
Utterance 0: 4k0c0301
   Average inference time per frame: 6.26867 ms
         max error: 0.0667191
         avg error: 0.00473641
     avg rms error: 0.00602212
       stdev error: 0.00393488
```

## Use of Sample in Kaldi* Speech Recognition Pipeline

The Wall Street Journal DNN model used in this example was prepared
using the Kaldi s5 recipe and the Kaldi Nnet (nnet1) framework.  It is
possible to recognize speech by substituting the `speech_sample` for
Kaldi's nnet-forward command.  Since the speech_sample does not yet
use pipes, it is necessary to use temporary files for speaker-
transformed feature vectors and scores when running the Kaldi speech
recognition pipeline.  The following operations assume that feature
extraction was already performed according to the `s5` recipe and that
the working directory within the Kaldi source tree is `egs/wsj/s5`.
1. Prepare a speaker-transformed feature set given the feature transform specified
  in `final.feature_transform` and the feature files specified in `feats.scp`:
```
nnet-forward --use-gpu=no final.feature_transform "ark,s,cs:copy-feats scp:feats.scp ark:- |" ark:feat.ark
```
2. Score the feature set using the `speech_sample`:
```
./speech_sample -d GNA_AUTO -bs 8 -i feat.ark -m wsj_dnn5b_smbr_fp32.xml -o scores.ark
```
3. Run the Kaldi decoder to produce n-best text hypotheses and select most likely text given the WFST (`HCLG.fst`), vocabulary (`words.txt`), and TID/PID mapping (`final.mdl`):
```
latgen-faster-mapped --max-active=7000 --max-mem=50000000 --beam=13.0 --lattice-beam=6.0 --acoustic-scale=0.0833 --allow-partial=true --word-symbol-table=words.txt final.mdl HCLG.fst ark:scores.ark ark:-| lattice-scale --inv-acoustic-scale=13 ark:- ark:- | lattice-best-path --word-symbol-table=words.txt ark:- ark,t:-  > out.txt &
```
4. Run the word error rate tool to check accuracy given the vocabulary (`words.txt`) and reference transcript (`test_filt.txt`):
```
cat out.txt | utils/int2sym.pl -f 2- words.txt | sed s:\<UNK\>::g | compute-wer --text --mode=present ark:test_filt.txt ark,p:-
```

## See Also
* [Using Inference Engine Samples](./docs/IE_DG/Samples_Overview.md)
* [Model Optimizer](./docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
* [Model Downloader](https://github.com/opencv/open_model_zoo/tree/2018/model_downloader)
