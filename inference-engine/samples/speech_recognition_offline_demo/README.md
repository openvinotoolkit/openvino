# Offline Automatic Speech Recognition C++ Demo

This topic shows how to run speech recognition, demonstrates acoustic model inference and Weighted Finite State Transducer (WFST) language model decoding based on Kaldi\* acoustic neural models, Intel&reg; Rockhopper Trail language models, and speech feature vectors.

## How It Works

The workflow is as follows:
1. The application reads command-line parameters
and loads a Kaldi-trained neural network along with a Kaldi `.ark` speech feature vector file to the Inference Engine plugin.
2. The application performs inference and passes acoustic scores vectors to decoding stage, and
Intel&reg; Rockhopper Trail decoder translates them into a text transcription.
3. The application prints recognized text on a screen.

### Acoustic and Language Model Setup

Pretrained models are available at [Intel&reg; Open Source Technology Center](https://download.01.org/openvinotoolkit/models_contrib/speech/kaldi) and [Intel&reg; OpenVINO&trade; Model Downloader](https://github.com/opencv/open_model_zoo/tree/2018/model_downloader). For this sample, we use models from `librispeech\s5_ext` folder.

To train models from scratch, refer to a shell-script Kaldi training recipe `lspeech_s5_ext_run.sh` and corresponding documentation `lspeech_s5_ext.md`.

To convert a Kaldi acoustic model into an Intermediate Representation (IR) format acceptable by this sample, use the following Model Optimizer command:

```sh
$ python3 mo.py --framework kaldi --input_model lspeech_s5_ext.nnet --counts lspeech_s5_ext.counts --remove_output_softmax
```

The command produces an IR network consisting of `lspeech_s5_ext.xml` and
`lspeech_s5_ext.bin`.

> **NOTE**: Model Optimizer (`mo.py`), Kaldi-trained neural network (`lspeech_s5_ext.nnet`)
and Kaldi class counts file (`lspeech_s5_ext.counts`) must be in your working directory.

### Speech Recognition

Once the IR is created or downloaded, you can use the following command for
speech recognition on Intel&reg; processors with a GNA coprocessor (or
emulation library) and Rockhopper Trail decoder library:

```sh
$ ./speech_recognition_offline_demo -d GNA_AUTO -bs 1 -i test_feat_1_10.ark -m lspeech_s5_ext.xml -hmm rht_language_model/rh.hmm -cl rht_language_model/cl.fst -g rht_language_model/g.fst -labels rht_language_model/labels.bin -amsf 0.08
```

## Sample Output

```
[ INFO ] InferenceEngine:
        API version ............ 1.6
        Build .................. R3
        Description ....... API
[ INFO ] Parsing input parameters
[ INFO ] No extensions provided
[ INFO ] Loading Inference Engine
[ INFO ] Device info:
        GNA
        GNAPlugin version ......... 1.6
        Build ........... GNAPlugin

[ INFO ] Loading network files
[ INFO ] Batch size is 1
[ INFO ] Using scale factor of 4079.14 calculated from first utterance.
[ INFO ] Loading model to the device
[ INFO ] Model loading time 301.864 ms
Utterance 0:
1272-128104-0012        ONLY UNFORTUNATELY HIS OWN WORK NEVER DOES GET GOOD

Total time in Infer (HW and SW):        1522.28 ms
Frames in utterance:                    536 frames
Average Infer time per frame:           2.84008 ms
End of Utterance 0

Utterance 1:
174-84280-0011  BUT NOW IT DOESN'T SEEM TO MATTER VERY MUCH

Total time in Infer (HW and SW):        957.779 ms
Frames in utterance:                    334 frames
Average Infer time per frame:           2.8676 ms
End of Utterance 1

Utterance 2:
1988-147956-0010        I REMEMBERED WHAT THE CONDUCTOR HAD SAID ABOUT HER EYES

Total time in Infer (HW and SW):        1082.91 ms
Frames in utterance:                    384 frames
Average Infer time per frame:           2.82008 ms
End of Utterance 2

Utterance 3:
1988-147956-0026        WE WERE SO DEEP IN THE GRASS THAT WE COULD SEE NOTHING BUT THE BLUE SKY OVER US AND THE GOLD TREE IN FRONT OF US

Total time in Infer (HW and SW):        1963.4 ms
Frames in utterance:                    690 frames
Average Infer time per frame:           2.84551 ms
End of Utterance 3

Utterance 4:
2086-149220-0045        FEWER WORDS THAN BEFORE BUT WITH THE SAME MYSTERIOUS MUSIC IN

Total time in Infer (HW and SW):        1283.32 ms
Frames in utterance:                    453 frames
Average Infer time per frame:           2.83293 ms
End of Utterance 4

Utterance 5:
2277-149874-0011        HE SEEMED TO BE THINKING OF SOMETHING ELSE

Total time in Infer (HW and SW):        690.602 ms
Frames in utterance:                    245 frames
Average Infer time per frame:           2.81878 ms
End of Utterance 5

Utterance 6:
2277-149896-0034        HE RANG AGAIN THIS TIME HARDER STILL NO ANSWER

Total time in Infer (HW and SW):        1128.91 ms
Frames in utterance:                    399 frames
Average Infer time per frame:           2.82934 ms
End of Utterance 6

Utterance 7:
2277-149897-0015        IN ABOUT AN HOUR AND THREE QUARTERS THE BOY RETURNED

Total time in Infer (HW and SW):        857.916 ms
Frames in utterance:                    302 frames
Average Infer time per frame:           2.84078 ms
End of Utterance 7

Utterance 8:
2412-153948-0005        I WAS DELIGHTED WITH THE COUNTRY AND THE MANNER OF LIFE

Total time in Infer (HW and SW):        897.309 ms
Frames in utterance:                    312 frames
Average Infer time per frame:           2.87599 ms
End of Utterance 8

Utterance 9:
3081-166546-0044        HE WAS THE PLAIN FACE DETECTIVE WHO HAD SPOKEN TO GEORGE

Total time in Infer (HW and SW):        1280.3 ms
Frames in utterance:                    448 frames
Average Infer time per frame:           2.8578 ms
End of Utterance 9

[ INFO ] Execution successful
```

## Input Preparation

Speech Recognition Offline Demo application accepts Kaldi binary `.ark` files holding stacked feature frames.
To prepare such files, please follow steps described in `lspeech_s5_ext.md` from folder `librispeech\s5_ext` of Model Zoo.

## See Also
* [Using Inference Engine Samples](./docs/IE_DG/Samples_Overview.md)
* [Model Optimizer](./docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
* [Model Downloader](https://github.com/opencv/open_model_zoo/tree/2018/model_downloader)
