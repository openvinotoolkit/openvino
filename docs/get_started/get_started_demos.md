# Get Started with C++ Samples {#openvino_docs_get_started_get_started_demos}

@sphinxdirective

.. meta::
   :description: Learn the details on the workflow of Intel® Distribution of OpenVINO™
                 toolkit, and how to run inference, using provided code samples.


The guide presents a basic workflow for building and running C++ code samples in OpenVINO. Note that these steps will not work with the Python samples.

To get started, you must first install OpenVINO Runtime, install OpenVINO Development tools, and build the sample applications. See the :ref:`Prerequisites <prerequisites-samples>` section for instructions.

Once the prerequisites have been installed, perform the following steps:

1. :ref:`Use Model Downloader to download a suitable model <download-models>`.
2. :ref:`Convert the model with mo <convert-models-to-intermediate-representation>`.
3. :ref:`Download media files to run inference <download-media>`.
4. :ref:`Run inference with the Image Classification sample application and see the results <run-image-classification>`.

.. _prerequisites-samples:

Prerequisites
#############

Install OpenVINO Runtime
++++++++++++++++++++++++

To use sample applications, install OpenVINO Runtime via one of the following distribution channels (other distributions do not include sample files):

* Archive files (recommended) - :doc:`Linux <openvino_docs_install_guides_installing_openvino_from_archive_linux>` | :doc:`Windows <openvino_docs_install_guides_installing_openvino_from_archive_windows>` | :doc:`macOS <openvino_docs_install_guides_installing_openvino_from_archive_macos>`
* :doc:`APT <openvino_docs_install_guides_installing_openvino_apt>` or :doc:`YUM <openvino_docs_install_guides_installing_openvino_yum>` for Linux
* :doc:`Docker image <openvino_docs_install_guides_installing_openvino_docker>`
* `Build from source <https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build.md>`__

Make sure that you also `install OpenCV <https://github.com/opencv/opencv/wiki/BuildOpenCV4OpenVINO>`__ , as it's required for running sample applications.



Install OpenVINO Development Tools
++++++++++++++++++++++++++++++++++

.. note::

   Note that OpenVINO support for Apache MXNet, Caffe, and Kaldi is currently being deprecated and will be removed entirely in the future.

To install OpenVINO Development Tools, follow the :doc:`instructions for C++ developers on the Install OpenVINO Development Tools page <openvino_docs_install_guides_install_dev_tools>`. This guide uses the ``googlenet-v1`` model from the Caffe framework, therefore, when you get to Step 4 of the installation, run the following command to install OpenVINO with the Caffe requirements:

.. code-block:: sh

   pip install openvino-dev[caffe]






Build Samples
+++++++++++++

To build OpenVINO samples, follow the build instructions for your operating system on the :doc:`OpenVINO Samples <openvino_docs_OV_UG_Samples_Overview>` page. The build will take about 1-2 minutes, depending on your system.

.. _download-models:

Step 1: Download the Models
###########################

You must have a model that is specific for your inference task. Example model types are:

- Classification (AlexNet, GoogleNet, SqueezeNet, others): Detects one type of element in an image
- Object Detection (SSD, YOLO): Draws bounding boxes around multiple types of objects in an image
- Custom: Often based on SSD

You can use one of the following options to find a model suitable for OpenVINO:

- Download public or Intel pre-trained models from :doc:`Open Model Zoo <model_zoo>` using :doc:`Model Downloader tool <omz_tools_downloader>`
- Download from GitHub, Caffe Zoo, TensorFlow Zoo, etc.
- Train your own model with machine learning tools

This guide uses OpenVINO Model Downloader to get pre-trained models. You can use one of the following commands to find a model with this method:

* List the models available in the downloader.

  .. code-block:: sh

     omz_info_dumper --print_all

* Use ``grep`` to list models that have a specific name pattern (e.g. ``ssd-mobilenet``, ``yolo``). Replace ``<model_name>`` with the name of the model.

  .. code-block:: sh

     omz_info_dumper --print_all | grep <model_name>

* Use Model Downloader to download models. Replace ``<models_dir>`` with the directory to download the model to and ``<model_name>`` with the name of the model.

  .. code-block:: sh

     omz_downloader --name <model_name> --output_dir <models_dir>

This guide used the following model to run the Image Classification Sample:

+------------------+-----------------------------+
| Model Name       | Code Sample or Demo App     |
+==================+=============================+
| ``googlenet-v1`` | Image Classification Sample |
+------------------+-----------------------------+

.. dropdown:: Click to view how to download the GoogleNet v1 Caffe model

   To download the GoogleNet v1 Caffe model to the `models` folder:

   .. tab-set::

      .. tab-item:: Windows
         :sync: windows

         .. code-block:: bat

            omz_downloader --name googlenet-v1 --output_dir %USERPROFILE%\Documents\models

      .. tab-item:: Linux
         :sync: linux

         .. code-block:: sh

            omz_downloader --name googlenet-v1 --output_dir ~/models

      .. tab-item:: macOS
         :sync: macos

         .. code-block:: sh

            omz_downloader --name googlenet-v1 --output_dir ~/models


   Your screen will look similar to this after the download and show the paths of downloaded files:

   .. tab-set::

      .. tab-item:: Windows
         :sync: windows

         .. code-block:: bat

            ################|| Downloading models ||################

            ========== Downloading C:\Users\username\Documents\models\public\googlenet-v1\googlenet-v1.prototxt
            ... 100%, 9 KB, ? KB/s, 0 seconds passed

            ========== Downloading C:\Users\username\Documents\models\public\googlenet-v1\googlenet-v1.caffemodel
            ... 100%, 4834 KB, 571 KB/s, 8 seconds passed

            ################|| Post-processing ||################

            ========== Replacing text in C:\Users\username\Documents\models\public\googlenet-v1\googlenet-v1.prototxt

      .. tab-item:: Linux
         :sync: linux

         .. code-block:: sh

            ###############|| Downloading models ||###############

            ========= Downloading /home/username/models/public/googlenet-v1/googlenet-v1.prototxt

            ========= Downloading /home/username/models/public/googlenet-v1/googlenet-v1.caffemodel
            ... 100%, 4834 KB, 3157 KB/s, 1 seconds passed

            ###############|| Post processing ||###############

            ========= Replacing text in /home/username/models/public/googlenet-v1/googlenet-v1.prototxt =========

      .. tab-item:: macOS
         :sync: macos

         .. code-block:: sh

            ###############|| Downloading models ||###############

            ========= Downloading /Users/username/models/public/googlenet-v1/googlenet-v1.prototxt
            ... 100%, 9 KB, 44058 KB/s, 0 seconds passed

            ========= Downloading /Users/username/models/public/googlenet-v1/googlenet-v1.caffemodel
            ... 100%, 4834 KB, 4877 KB/s, 0 seconds passed

            ###############|| Post processing ||###############

            ========= Replacing text in /Users/username/models/public/googlenet-v1/googlenet-v1.prototxt =========

.. _convert-models-to-intermediate-representation:

Step 2: Convert the Model with ``mo``
#####################################

In this step, your trained models are ready for conversion with ``mo`` to the OpenVINO IR (Intermediate Representation) format. For most model types, this is required before using OpenVINO Runtime with the model.

Models in the IR format always include an ``.xml`` and ``.bin`` file and may also include other files such as ``.json`` or ``.mapping``. Make sure you have these files together in a single directory so OpenVINO Runtime can find them.

REQUIRED: ``model_name.xml``
REQUIRED: ``model_name.bin``
OPTIONAL: ``model_name.json``, ``model_name.mapping``, etc.

This tutorial uses the public GoogleNet v1 Caffe model to run the Image Classification Sample. See the example in the Download Models section of this page to learn how to download this model.

The googlenet-v1 model is downloaded in the Caffe format. You must use ``mo`` to convert the model to IR.

Create an ``<ir_dir>`` directory to contain the model's Intermediate Representation (IR).

.. tab-set::

   .. tab-item:: Windows
      :sync: windows

      .. code-block:: bat

         mkdir %USERPROFILE%\Documents\ir

   .. tab-item:: Linux
      :sync: linux

      .. code-block:: sh

         mkdir ~/ir

   .. tab-item:: macOS
      :sync: macos

      .. code-block:: sh

         mkdir ~/ir

To save disk space for your IR files, OpenVINO stores weights in FP16 format by default.

Generic model conversion script:

.. code-block:: sh

   mo --input_model <model_dir>/<model_file>


The IR files produced by the script are written to the ``<ir_dir>`` directory.

The command with most placeholders filled in and FP16 precision:

.. tab-set::

   .. tab-item:: Windows
      :sync: windows

      .. code-block:: bat

         mo --input_model %USERPROFILE%\Documents\models\public\googlenet-v1\googlenet-v1.caffemodel --compress_to_fp16 --output_dir %USERPROFILE%\Documents\ir

   .. tab-item:: Linux
      :sync: linux

      .. code-block:: sh

         mo --input_model ~/models/public/googlenet-v1/googlenet-v1.caffemodel --compress_to_fp16 --output_dir ~/ir

   .. tab-item:: macOS
      :sync: macos

      .. code-block:: sh

         mo --input_model ~/models/public/googlenet-v1/googlenet-v1.caffemodel --compress_to_fp16 --output_dir ~/ir

.. _download-media:

Step 3: Download a Video or a Photo as Media
############################################

Most of the samples require you to provide an image or a video as the input to run the model on. You can get them from sites like `Pexels <https://pexels.com>`__ or `Google Images <https://images.google.com>`__ .

As an alternative, OpenVINO also provides several sample images and videos for you to run code samples and demo applications:

- `Sample images and video <https://storage.openvinotoolkit.org/data/test_data/>`__
- `Sample videos <https://github.com/intel-iot-devkit/sample-videos>`__

.. _run-image-classification:

Step 4: Run Inference on a Sample
##################################

To run the **Image Classification** code sample with an input image using the IR model:

1. Set up the OpenVINO environment variables:

   .. tab-set::

      .. tab-item:: Windows
         :sync: windows

         .. code-block:: bat

            <INSTALL_DIR>\setupvars.bat

      .. tab-item:: Linux
         :sync: linux

         .. code-block:: sh

            source  <INSTALL_DIR>/setupvars.sh

      .. tab-item:: macOS
         :sync: macos

         .. code-block:: sh

            source <INSTALL_DIR>/setupvars.sh

2. Go to the code samples release directory created when you built the samples earlier:

   .. tab-set::

      .. tab-item:: Windows
         :sync: windows

         .. code-block:: bat

            cd  %USERPROFILE%\Documents\Intel\OpenVINO\openvino_samples_build\intel64\Release

      .. tab-item:: Linux
         :sync: linux

         .. code-block:: sh

            cd ~/openvino_cpp_samples_build/intel64/Release

      .. tab-item:: macOS
         :sync: macos

         .. code-block:: sh

            cd ~/openvino_cpp_samples_build/intel64/Release

3. Run the code sample executable, specifying the input media file, the IR for your model, and a target device for performing inference:

   .. tab-set::

      .. tab-item:: Windows
         :sync: windows

         .. code-block:: bat

            classification_sample_async.exe -i <path_to_media> -m <path_to_model> -d <target_device>

      .. tab-item:: Linux
         :sync: linux

         .. code-block:: sh

            classification_sample_async -i <path_to_media> -m <path_to_model> -d <target_device>

      .. tab-item:: macOS
         :sync: macos

         .. code-block:: sh

            classification_sample_async -i <path_to_media> -m <path_to_model> -d <target_device>

Examples
++++++++

Running Inference on CPU
------------------------

The following command shows how to run the Image Classification Code Sample using the `dog.bmp <https://storage.openvinotoolkit.org/data/test_data/images/224x224/dog.bmp>`__ file as an input image, the model in IR format from the ``ir`` directory, and the CPU as the target hardware:

.. tab-set::

   .. tab-item:: Windows
      :sync: windows

      .. code-block:: bat

         .\classification_sample_async.exe -i %USERPROFILE%\Downloads\dog.bmp -m %USERPROFILE%\Documents\ir\googlenet-v1.xml -d CPU

   .. tab-item:: Linux
      :sync: linux

      .. code-block:: sh

         ./classification_sample_async -i ~/Downloads/dog.bmp -m ~/ir/googlenet-v1.xml -d CPU

   .. tab-item:: macOS
      :sync: macos

      .. code-block:: sh

         ./classification_sample_async -i ~/Downloads/dog.bmp -m ~/ir/googlenet-v1.xml -d CPU

When the sample application is complete, you are given the label and confidence for the top 10 categories. The input image and sample output of the inference results is shown below:

.. image:: _static/images/dog.png

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

The following example shows how to run the same sample using GPU as the target device.

Running Inference on GPU
------------------------

.. note::

   Running inference on Intel® Processor Graphics (GPU) requires :doc:`additional hardware configuration steps <openvino_docs_install_guides_configurations_for_intel_gpu>`, as described earlier on this page. Running on GPU is not compatible with macOS.

.. tab-set::

   .. tab-item:: Windows
      :sync: windows

      .. code-block:: bat

         .\classification_sample_async.exe -i %USERPROFILE%\Downloads\dog.bmp -m %USERPROFILE%\Documents\ir\googlenet-v1.xml -d GPU

   .. tab-item:: Linux
      :sync: linux

      .. code-block:: sh

         ./classification_sample_async -i ~/Downloads/dog.bmp -m ~/ir/googlenet-v1.xml -d GPU


Other Demos and Samples
#######################

See the :doc:`Samples <openvino_docs_OV_UG_Samples_Overview>` page for more sample applications. Each sample page explains how the application works and shows how to run it. Use the samples as a starting point that can be adapted for your own application.

OpenVINO also provides demo applications for using off-the-shelf models from :doc:`Open Model Zoo <model_zoo>`. Visit :doc:`Open Model Zoo Demos <omz_demos>` if you'd like to see even more examples of how to run model inference with the OpenVINO API.

@endsphinxdirective

