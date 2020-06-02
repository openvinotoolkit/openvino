=====================================================
Demo Scripts for Model Optimizer and Inference Engine
=====================================================

The demo scripts illustrate Intel(R) Deep Learning Deployment Toolkit usage to convert and optimize pre-trained models and perform inference.

Setting Up Demos
================
If you are behind a proxy, set the following environment variables in the console session:

On Linux* and Mac OS:
export http_proxy=http://<proxyHost>:<proxyPort>
export https_proxy=https://<proxyHost>:<proxyPort>

On Windows* OS:
set http_proxy=http://<proxyHost>:<proxyPort>
set https_proxy=https://<proxyHost>:<proxyPort>

Running Demos
=============

The "demo" folder contains three scripts:

1. Classification demo using public SqueezeNet topology (demo_squeezenet_download_convert_run.sh|bat)

2. Security barrier camera demo that showcases three models coming with the product (demo_squeezenet_download_convert_run.sh|bat)

3. Benchmark demo using public SqueezeNet topology (demo_benchmark_app.sh|bat) 

4. Speech recognition demo utilizing models trained on open LibriSpeech dataset

To run the demos, run demo_squeezenet_download_convert_run.sh or demo_security_barrier_camera.sh or demo_benchmark_app.sh or demo_speech_recognition.sh (*.bat on Windows) scripts from the console without parameters, for example:

./demo_squeezenet_download_convert_run.sh

The script allows to specify the target device to infer on using -d <CPU|GPU|MYRIAD|FPGA> option.

Classification Demo Using SqueezeNet
====================================

The demo illustrates the general workflow of using the Intel(R) Deep Learning Deployment Toolkit and performs the following:

  - Downloads a public SqueezeNet model using the Model Downloader (open_model_zoo\tools\downloader\downloader.py)
  - Installs all prerequisites required for running the Model Optimizer using the scripts from the "model_optimizer\install_prerequisites" folder
  - Converts SqueezeNet to an IR using the Model Optimizer (model_optimizer\mo.py) via the Model Converter (open_model_zoo\tools\downloader\converter.py)
  - Builds the Inference Engine classification_sample (inference_engine\samples\classification_sample)
  - Runs the sample with the car.png picture located in the demo folder

The sample application prints top-10 inference results for the picture.
 
For more information about the Inference Engine classification sample, refer to the documentation available in the sample folder.


Security Barrier Camera Demo
============================

The demo illustrates using the Inference Engine with pre-trained models to perform vehicle detection, vehicle attributes and license-plate recognition tasks. 
As the sample produces visual output, it should be run in GUI mode.  

The demo script does the following:

- Builds the Inference Engine security barrier camera sample (inference_engine\samples\security_barrier_camera_sample)
- Runs the sample with the car_1.bmp located in the demo folder

The sample application displays the resulting frame with detections rendered as bounding boxes and text.

For more information about the Inference Engine security barrier camera sample, refer to the documentation available in the sample folder.


Benchmark Demo Using SqueezeNet
===============================

The demo illustrates how to use the Benchmark Application to estimate deep learning inference performance on supported devices.

The demo script does the following:

  - Downloads a public SqueezeNet model using the Model Downloader (open_model_zoo\tools\downloader\downloader.py)
  - Installs all prerequisites required for running the Model Optimizer using the scripts from the "model_optimizer\install_prerequisites" folder
  - Converts SqueezeNet to an IR using the Model Optimizer (model_optimizer\mo.py) via the Model Converter (open_model_zoo\tools\downloader\converter.py)
  - Builds the Inference Engine benchmark tool (inference_engine\samples\demo_benchmark_app)
  - Runs the tool with the car.png picture located in the demo folder

The benchmark app prints performance counters, resulting latency, and throughput values.
 
For more information about the Inference Engine benchmark app, refer to the documentation available in the sample folder.

Speech Recognition Demo Using LibriSpeech models
================================================

The demo illustrates live speech recognition - transcribing speech from microphone or offline (from wave file).
The demo is also capable of live close captioning of an audio clip or movie, where signal is intercepted from the speaker. 

The demo script does the following:

  - Downloads US English models trained on LibriSpeech dataset prepared for direct usage by the Inference Engine
  - Installs the required components
  - Runs the command line offline demo
  - As a final step, runs live speech recognition application with graphical interface

The GUI application prints the speech transcribed from input signal in window. Up to two channels can be transcribed in parallel: microphone & speakers streams.
