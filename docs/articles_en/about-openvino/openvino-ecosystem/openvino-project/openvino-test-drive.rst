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
  Diffusion and Stable DiffusionXL models (coming soon);
* **Transcribe speech from video**, using Whisper models, including generation
  of timestamps (coming soon);
* **Run inference of models** trained by Intel® Geti™ and **visualize the results**.


Installation (Windows)
###############################################################################################

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

Inference of models from Hugging Face
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

1. Find a model on `Hugging Face <https://huggingface.co/>`__ and import it.

   .. image:: ../../../assets/images/TestDrive_llm_import.gif
      :align: center
      :alt: how to import a model to test drive

2. Chat with LLMs via the `Playground` tab.

   .. image:: ../../../assets/images/TestDrive_llm_model_chat.gif
      :align: center
      :alt: chatting with llm models in test drive

3. Use the `Performance metrics` tab to get model performance metrics on your
   computer or an edge device.

   .. image:: ../../../assets/images/TestDrive_llm_metrics.gif
      :align: center
      :alt: verifying llm performance in test drive

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
