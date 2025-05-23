===============================================================================================
OpenVINO™ Test Drive
===============================================================================================


.. meta::
   :description: See how to test your models with OpenVINO, using a simple graphic interface of
                 Test Drive.



OpenVINO™ Test Drive is a cross-platform **graphic user interface** application for running and
testing AI models, both generative and vision based.
It can run directly on your computer or on edge devices using
`OpenVINO™ Runtime <https://github.com/openvinotoolkit/openvino>`__.

OpenVINO™ Test Drive is developed under the `openvino_testdrive repository <https://github.com/openvinotoolkit/openvino_testdrive>`__.

Use OpenVINO™ Test Drive to:

* **Chat with LLMs** and evaluate model performance on your computer or edge device;
* **Experiment with different text prompts** to generate images, using Stable
  Diffusion and Stable DiffusionXL models;
* **Transcribe speech from video**, using Whisper models, including generation
  of timestamps;
* **Run inference of models** trained by Intel® Geti™ and **visualize the results**.


Installation (Windows)
###############################################################################################

.. important::

   For Intel® NPU, use the latest available version of
   `Intel® NPU Driver <https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html>`__.

1. Download the latest archive from the
   `release repository <https://storage.openvinotoolkit.org/repositories/openvino_testdrive/packages>`__.
   To verify the integrity of the downloaded package, use the SHA-256 file attached.

2. Extract the zip file and run the *MSIX* installation package. Click the `Install` button to
   proceed.

3. Launch OpenVINO™ Test Drive, clicking the application name in the Windows app list.


Quick start
###############################################################################################

When starting the application, you can import an LLM model from Hugging Face Hub
or upload an Intel® Geti™ model from a local drive.

Text generation and LLM performance evaluation
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

1. Find a model on `Hugging Face <https://huggingface.co/>`__ and import it.

   .. image:: ../../../assets/images/TestDrive_llm_import.gif
      :align: center
      :alt: how to import a model to test drive

2. Chat with LLMs via the `Playground` tab. You can export an LLM by clicking
   the `Export model` button.

   .. image:: ../../../assets/images/TestDrive_llm_model_chat.gif
      :align: center
      :alt: chatting with llm models in test drive

3. Use the `Performance metrics` tab to get model performance metrics on your
   computer or an edge device.

   .. image:: ../../../assets/images/TestDrive_llm_metrics.gif
      :align: center
      :alt: verifying llm performance in test drive


Retrieval-Augmented Generation with LLMs
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

1. Upload files and create knowledge base for RAG (Retrieval-Augmented Generation),
   using `Knowledge base` tab.

   .. image:: ../../../assets/images/TestDrive_rag_base.gif
      :align: center
      :alt: creating a knowledge base for RAG in test drive

   The knowledge base can be used for text generation with LLM models.

   .. image:: ../../../assets/images/TestDrive_rag_1.gif
      :align: center
      :alt: using a knowledge base for text generation with LLMs

2. You can also upload a document directly, using the `Playground`` tab.

   .. image:: ../../../assets/images/TestDrive_rag_2.gif
      :align: center
      :alt: uploading a document to the knowledge base


Image analysis with Visual Language Models (VLMs)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

1. Import a VLM for image analysis.

   .. image:: ../../../assets/images/TestDrive_vlm_1.gif
      :align: center
      :alt: importing a visual language model for image analysis

2. Select the VLM from the `My models` section, upload an image and analyze it.

   .. image:: ../../../assets/images/TestDrive_vlm_2.gif
      :align: center
      :alt: importing a visual language model for image analysis


Video transcription with Whisper models
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

1. Import a Whisper model for video transcription.

   .. image:: ../../../assets/images/TestDrive_st_import.gif
      :align: center
      :alt: importing a Whisper model for video transcription

2. Select the speech-to-text LLM from the `My models` section, and upload a video for transcription.

   .. image:: ../../../assets/images/TestDrive_ts_video.gif
      :align: center
      :alt: importing a visual language model for image analysis

   You can search for words in the transcript or download it.

3. Use the `Performance metrics` tab to get performance metrics of the LLM on your computer
   or an edge device.


Image generation with LLMs
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

1. Import an image generation LLM from a predefined set of popular models or from
   `Hugging Face <https://huggingface.co/>`__, using `Import model` -> `Hugging Face`.

2. Select the LLM from the `My models` section and start the chat to generate an image.
   You can export the model by clicking the `Export model` button.

   .. image:: ../../../assets/images/TestDrive_image_generation.gif
      :align: center
      :alt: image generation with a chosen LLM

   You can download the generated image.

3. Use the `Performance metrics` tab to get performance metrics of the LLM on your computer
   or an edge device.


Inference of models trained with Intel® Geti™
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

1. Download the deployment code for a model in the OpenVINO IR format trained
   by Intel® Geti™ (refer to the `Intel® Geti™ documentation <https://docs.geti.intel.com>`__
   for more details).

   .. image:: ../../../assets/images/TestDrive_geti_download.gif
      :align: center
      :alt: verifying llm performance in test drive

2. Import the deployment code into OpenVINO™ Test Drive, using the *Import model* and then
   *Local disk* buttons.

3. Use the *Live inference* tab to run and visualize results of inference of individual images.

4. For batch inference, use the *Batch inference* tab and provide paths to the folder
   with input images, as well as one for batch inference results. You can do so by filling out
   the *Source folder* and *Destination folder* fields. Click *Start* to start batch inference.


Build the Application
###############################################################################################

1. Make sure you `Install flutter SDK <https://docs.flutter.dev/get-started/install>`__
   and all its platform-specific dependencies.
2. Build the bindings and place them in the **./bindings** folder.

   OpenVINO™ Test Drive uses bindings to `OpenVINO™ GenAI <https://github.com/openvinotoolkit/openvino.genai>`__
   and `OpenVINO™ Model API <https://github.com/openvinotoolkit/model_api>`__,
   which are located in the **./openvino_bindings** folder. Refer to the
   `GitHub page <https://github.com/openvinotoolkit/openvino_testdrive/blob/main/openvino_bindings/>`__
   for more details.

3. Start the application, using the following command:

   .. code-block:: console

      flutter run

Additional Resources
###############################################################################################

- `OpenVINO™ <https://github.com/openvinotoolkit/openvino>`__ - a software toolkit
  for optimizing and deploying deep learning models.
- `GenAI Repository <https://github.com/openvinotoolkit/openvino.genai>`__ and
  `OpenVINO Tokenizers <https://github.com/openvinotoolkit/openvino_tokenizers>`__
  - resources and tools for developing and optimizing Generative AI applications.
- `Intel® Geti™ <https://docs.geti.intel.com/>`__ - software for building computer
  vision models.
- `OpenVINO™ Model API <https://github.com/openvinotoolkit/model_api>`__
  - a set of wrapper classes for particular tasks and model architectures.
  It simplifies routine procedures, preprocessing and postprocessing of data.
