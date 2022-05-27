# Get Started with Sample and Demo Applications {#openvino_docs_get_started_get_started_demos}

## Introduction

This section guides you through a simplified workflow for the Intel® Distribution of OpenVINO™ toolkit using code samples and demo applications.
You will perform the following steps:

1. <a href="#download-models">Use the Model Downloader to download suitable models.</a>
2. <a href="#convert-models-to-intermediate-representation">Convert the models with the Model Optimizer.</a> 
3. <a href="#download-media">Download media files to run inference on.</a>
4. <a href="#run-image-classification">Run inference on the sample and see the results:</a>
    - <a href="#run-image-classification">Image Classification Code Sample</a>

This guide assumes you completed all installation and configuration steps. If you have not yet installed and configured the toolkit, follow the, suitable for you, installation guide:

@sphinxdirective
.. tab:: Linux

   :doc:`Install Intel® Distribution of OpenVINO™ toolkit for Linux <openvino_docs_install_guides_installing_openvino_linux>`

.. tab:: Windows

   :doc:`Install Intel® Distribution of OpenVINO™ toolkit for Windows <openvino_docs_install_guides_installing_openvino_windows>`

.. tab:: macOS

   :doc:`Install Intel® Distribution of OpenVINO™ toolkit for macOS <openvino_docs_install_guides_installing_openvino_macos>`
  
@endsphinxdirective

## Installing OpenVINO Development Tools

To install OpenVINO Development Tools for use with Caffe models, run the following command: 

``` sh
   pip install openvino-dev[caffe]
```

## Building Samples and Demos

If you have already built the demos and samples, you can skip this section.

To build OpenVINO samples:

@sphinxdirective
.. tab:: Linux

   Open the :doc:`OpenVINO Samples page <openvino_docs_IE_DG_Samples_Overview>` and follow the "Build the Sample Applications on Linux" section.

.. tab:: Windows

   Open the :doc:`OpenVINO Samples page <openvino_docs_IE_DG_Samples_Overview>` and follow the "Build the Sample Applications on Microsoft Windows* OS" section.

.. tab:: macOS

   Open the :doc:`OpenVINO Samples page <openvino_docs_IE_DG_Samples_Overview>` and follow the "Build the Sample Applications on macOS" section. 

@endsphinxdirective

To build OpenVINO demos:
@sphinxdirective
.. tab:: Linux

   Open the :doc:`Open Model Zoo Demos page <omz_demos>` and follow the "Build the Demo Applications on Linux" section.

.. tab:: Windows

   Open the :doc:`Open Model Zoo Demos page <omz_demos>` and follow the "Build the Demo Applications on Microsoft Windows OS" section.

.. tab:: macOS

   Open the :doc:`Open Model Zoo Demos page <omz_demos>` and follow the "Build the Demo Applications on Linux" section. Use the requirements from "To build OpenVINO samples" above and adapt the Linux building steps for macOS.

@endsphinxdirective

## <a name="download-models"></a> Step 1: Download the Models

For inference task you need to have a specific model. Below are presented model type examples:

- Classification (AlexNet, GoogleNet, SqueezeNet, etc.) - Detects one type of element in an image.
- Object Detection (SSD, YOLO) -- Draws bounding boxes around multiple types of objects in an image.
- Custom - Often based on SSD.

Suitable model for the OpenVINO™ toolkit can be find by:

- Download public or Intel pre-trained models from the [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo) using the [Model Downloader tool](@ref omz_tools_downloader).
- Download from GitHub, Caffe Zoo, TensorFlow Zoo, etc.
- Train your own model with machine learning tools.
  
Run one of the commands below to find a model to use:

* List the models available in the downloader:

``` sh
   omz_info_dumper --print_all
```

* Use `grep` to list models that have a specific name pattern

``` sh
   omz_info_dumper --print_all | grep <model_name>
```

* Use Model Downloader to download models.

 This guide uses `<models_dir>` and `<models_name>` as placeholders for the model directory and model name:

``` sh
   omz_downloader --name <model_name> --output_dir <models_dir>
```

* Download the following models to run the Image Classification Sample:

|Model Name                                     | Code Sample or Demo App                  |
|-----------------------------------------------|------------------------------------------|
|`googlenet-v1`                                 | Image Classification Sample              |

@sphinxdirective
.. raw:: html

   <div class="collapsible-section" data-title="Click here for an example of downloading the GoogleNet v1 Caffe model to the `models` folder">

@endsphinxdirective


@sphinxdirective

.. tab:: Linux

   .. code-block:: sh

      omz_downloader --name googlenet-v1 --output_dir ~/models

.. tab:: Windows

   .. code-block:: bat

      omz_downloader --name googlenet-v1 --output_dir %USERPROFILE%\Documents\models

.. tab:: macOS

   .. code-block:: sh

      omz_downloader --name googlenet-v1 --output_dir ~/models

@endsphinxdirective

After download your screen will look similar to the one below and show the paths of downloaded files:

@sphinxdirective
.. tab:: Linux

   .. code-block:: sh

      ###############|| Downloading models ||###############

      ========= Downloading /home/username/models/public/googlenet-v1/googlenet-v1.prototxt

      ========= Downloading /home/username/models/public/googlenet-v1/googlenet-v1.caffemodel
      ... 100%, 4834 KB, 3157 KB/s, 1 seconds passed

      ###############|| Post processing ||###############

      ========= Replacing text in /home/username/models/public/googlenet-v1/googlenet-v1.prototxt =========

.. tab:: Windows

   .. code-block:: bat

      ################|| Downloading models ||################

      ========== Downloading C:\Users\username\Documents\models\public\googlenet-v1\googlenet-v1.prototxt
      ... 100%, 9 KB, ? KB/s, 0 seconds passed

      ========== Downloading C:\Users\username\Documents\models\public\googlenet-v1\googlenet-v1.caffemodel
      ... 100%, 4834 KB, 571 KB/s, 8 seconds passed

      ################|| Post-processing ||################

      ========== Replacing text in C:\Users\username\Documents\models\public\googlenet-v1\googlenet-v1.prototxt

.. tab:: macOS

   .. code-block:: sh

      ###############|| Downloading models ||###############

      ========= Downloading /Users/username/models/public/googlenet-v1/googlenet-v1.prototxt
      ... 100%, 9 KB, 44058 KB/s, 0 seconds passed

      ========= Downloading /Users/username/models/public/googlenet-v1/googlenet-v1.caffemodel
      ... 100%, 4834 KB, 4877 KB/s, 0 seconds passed

      ###############|| Post processing ||###############

      ========= Replacing text in /Users/username/models/public/googlenet-v1/googlenet-v1.prototxt =========

@endsphinxdirective

@sphinxdirective
.. raw:: html

   </div>

@endsphinxdirective

For the rest of this guide, OpenVINO™ Model Downloader was used to download pre-trained models. 

## <a name="convert-models-to-intermediate-representation"></a>Step 2: Convert the Model with Model Optimizer

To use your model in the OpenVINO Runtime, it's required to be converted into OpenVINO IR (Intermediate Representation) format. To do this it is required to run trained model through the Model Optimizer.

This tutorial uses the public GoogleNet v1 Caffe model to run the Image Classification Sample. See the example in the <a href="#download-models">Download Models</a> section of this article to learn how to download this model.

The googlenet-v1 model is downloaded in the Caffe format. Use the Model Optimizer to convert it to OpenVINO IR.

Create an `<ir_dir>` directory to contain the model's OpenVINO IR:

@sphinxdirective
.. tab:: Linux

   .. code-block:: sh

      mkdir ~/ir

.. tab:: Windows

   .. code-block:: bat

      mkdir %USERPROFILE%\Documents\ir

.. tab:: macOS

   .. code-block:: sh

      mkdir ~/ir

@endsphinxdirective

The OpenVINO Runtime can infer models where floating-point weights are [compressed to FP16](../MO_DG/prepare_model/FP16_Compression.md). To generate an IR with a specific precision, run the Model Optimizer with the appropriate `--data_type` option.

Generic Model Optimizer script:

``` sh
   mo --input_model <model_dir>/<model_file> --data_type <model_precision> --output_dir <ir_dir>
```

The command with most placeholders filled in and FP16 precision:

@sphinxdirective
.. tab:: Linux

   .. code-block:: sh

      mo --input_model ~/models/public/googlenet-v1/googlenet-v1.caffemodel --data_type FP16 --output_dir ~/ir

.. tab:: Windows

   .. code-block:: bat

      mo --input_model %USERPROFILE%\Documents\models\public\googlenet-v1\googlenet-v1.caffemodel --data_type FP16 --output_dir %USERPROFILE%\Documents\ir

.. tab:: macOS

   .. code-block:: sh

      mo --input_model ~/models/public/googlenet-v1/googlenet-v1.caffemodel --data_type FP16 --output_dir ~/ir

@endsphinxdirective

After the conversion IR files produced by the script are written to the <ir_dir> directory. Models in the IR format always include an `.xml` and `.bin` file. In some cases they may also include other files such as `.json` or `.mapping`.
**Make sure these files are together in a single directory so the OpenVINO Runtime can find them!**

**REQUIRED:** `model_name.xml`, `model_name.bin`
**OPTIONAL:** `model_name.json`, `model_name.mapping`, etc.

## <a name="download-media"></a> Step 3: Download a Video or Still Photo as Media

You can download video media to use as the code samples and demo applications from many available sources, like:

- [Pexels](https://pexels.com)
- [Google Images](https://images.google.com)

Alternatively, the Intel® Distribution of OpenVINO™ toolkit includes several sample images and videos that can be used for running code samples and demo applications:

   - [Sample images and video](https://storage.openvinotoolkit.org/data/test_data/)
   - [Sample videos](https://github.com/intel-iot-devkit/sample-videos)

## <a name="run-image-classification"></a>Step 4: Run Inference on the Sample

<!-- ### Run the Image Classification Code Sample -->

To run the **Image Classification** code sample with an input image using the IR model:

1. Set up the OpenVINO environment variables:
@sphinxdirective
.. tab:: Linux

   .. code-block:: sh

      source  <INSTALL_DIR>/setupvars.sh

.. tab:: Windows

   .. code-block:: bat

      <INSTALL_DIR>\setupvars.bat

.. tab:: macOS

   .. code-block:: sh

      source <INSTALL_DIR>/setupvars.sh

@endsphinxdirective

2. Open directory of the code samples release created when you built the samples earlier:
@sphinxdirective
.. tab:: Linux

   .. code-block:: sh

      cd ~/inference_engine_cpp_samples_build/intel64/Release

.. tab:: Windows

   .. code-block:: bat

      cd  %USERPROFILE%\Documents\Intel\OpenVINO\inference_engine_samples_build\intel64\Release

.. tab:: macOS

   .. code-block:: sh

      cd ~/inference_engine_cpp_samples_build/intel64/Release

@endsphinxdirective

3. Run the executable below with specified input media file, the OpenVINO IR for your model, and a target device for performing inference:

@sphinxdirective
.. tab:: Linux

   .. code-block:: sh

      classification_sample_async -i <path_to_media> -m <path_to_model> -d <target_device>

.. tab:: Windows

   .. code-block:: bat

      classification_sample_async.exe -i <path_to_media> -m <path_to_model> -d <target_device>

.. tab:: macOS

   .. code-block:: sh

      classification_sample_async -i <path_to_media> -m <path_to_model> -d <target_device>

@endsphinxdirective

@sphinxdirective
.. raw:: html

   <div class="collapsible-section" data-title="Click for examples of running the Image Classification code sample on different devices">

@endsphinxdirective

Command below runs the Image Classification Code Sample using the [dog.bmp](https://storage.openvinotoolkit.org/data/test_data/images/224x224/dog.bmp) file as an input image, the model in OpenVINO IR format from the `ir` directory, and on different hardware devices:

   **CPU:**  
@sphinxdirective
.. tab:: Linux

   .. code-block:: sh

      ./classification_sample_async -i ~/Downloads/dog.bmp -m ~/ir/googlenet-v1.xml -d CPU

.. tab:: Windows

   .. code-block:: bat

      .\classification_sample_async.exe -i %USERPROFILE%\Downloads\dog.bmp -m %USERPROFILE%\Documents\ir\googlenet-v1.xml -d CPU

.. tab:: macOS

   .. code-block:: sh

      ./classification_sample_async -i ~/Downloads/dog.bmp -m ~/ir/googlenet-v1.xml -d CPU

@endsphinxdirective

   **GPU:**
   > **NOTE**: Running inference on Intel® Processor Graphics (GPU) requires [additional hardware configuration steps](../install_guides/configurations-for-intel-gpu.md), as described earlier in this article. Running on GPU is not compatible with macOS.

@sphinxdirective
.. tab:: Linux

   .. code-block:: sh

      ./classification_sample_async -i ~/Downloads/dog.bmp -m ~/ir/googlenet-v1.xml -d GPU

.. tab:: Windows

   .. code-block:: bat

      .\classification_sample_async.exe -i %USERPROFILE%\Downloads\dog.bmp -m %USERPROFILE%\Documents\ir\googlenet-v1.xml -d GPU

@endsphinxdirective

   **MYRIAD:**
   > **NOTE**: Running inference on VPU devices (Intel® Movidius™ Neural Compute Stick or Intel® Neural Compute Stick 2) with the MYRIAD plugin requires [additional hardware configuration steps](../install_guides/configurations-for-ncs2.md), as described earlier in this article.

@sphinxdirective
.. tab:: Linux

   .. code-block:: sh

      ./classification_sample_async -i ~/Downloads/dog.bmp -m ~/ir/googlenet-v1.xml -d MYRIAD

.. tab:: Windows

   .. code-block:: bat

      .\classification_sample_async.exe -i %USERPROFILE%\Downloads\dog.bmp -m %USERPROFILE%\Documents\ir\googlenet-v1.xml -d MYRIAD

.. tab:: macOS

   .. code-block:: sh

      ./classification_sample_async -i ~/Downloads/dog.bmp -m ~/ir/googlenet-v1.xml -d MYRIAD

@endsphinxdirective

Once the sample application is complete, the label and confidence for the top 10 categories on the display will prompt. Present below is a sample output with inference results on CPU:

@sphinxdirective

   .. code-block:: sh

   Top 10 results:

   Image dog.bmp

      classid probability label
      ------- ----------- -----
      156     0.6875963   Blenheim spaniel
      215     0.0868125   Brittany spaniel
      218     0.0784114   Welsh springer spaniel
      212     0.0597296   English setter
      217     0.0212105   English springer, English springer spaniel
      219     0.0194193   cocker spaniel, English cocker spaniel, cocker
      247     0.0086272   Saint Bernard, St Bernard
      157     0.0058511   papillon
      216     0.0057589   clumber, clumber spaniel
      154     0.0052615   Pekinese, Pekingese, Peke

@endsphinxdirective

@sphinxdirective
.. raw:: html

   </div>

@endsphinxdirective

## Other Demos/Samples

For more samples and demos, visit pages below. You can review samples and demos by complexity or by usage, run the relevant application, and adapt the code for your use.

[Samples](../OV_Runtime_UG/Samples_Overview.md)

[Demos](@ref omz_demos)
