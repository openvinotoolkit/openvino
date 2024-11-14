Install OpenVINO™ GenAI
====================================

OpenVINO GenAI is a new flavor of OpenVINO, aiming to simplify running inference of generative AI models.
It hides the complexity of the generation process and minimizes the amount of code required.
You can now provide a model and input context directly to OpenVINO, which performs tokenization of the
input text, executes the generation loop on the selected device, and returns the generated text.
For a quickstart guide, refer to the :doc:`GenAI API Guide <../../learn-openvino/llm_inference_guide/genai-guide>`.

To see GenAI in action, check the Jupyter notebooks:
`LLM-powered Chatbot <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/llm-chatbot/README.md>`__ and
`LLM Instruction-following pipeline <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/llm-question-answering/README.md>`__.

The OpenVINO GenAI flavor is available for installation via PyPI and Archive distributions.
A `detailed guide <https://github.com/openvinotoolkit/openvino.genai/blob/releases/2024/3/src/docs/BUILD.md>`__
on how to build OpenVINO GenAI is available in the OpenVINO GenAI repository.

PyPI Installation
###############################

To install the GenAI flavor of OpenVINO via PyPI, follow the standard :doc:`installation steps <install-openvino-pip>`,
but use the *openvino-genai* package instead of *openvino*:

.. code-block:: python

   python -m pip install openvino-genai

Archive Installation
###############################

The OpenVINO GenAI archive package includes the OpenVINO™ Runtime and :doc:`Tokenizers <../../learn-openvino/llm_inference_guide/ov-tokenizers>`.
To install the GenAI flavor of OpenVINO from an archive file, follow the standard installation steps for your system
but instead of using the vanilla package file, download the one with OpenVINO GenAI:

Linux
++++++++++++++++++++++++++

.. tab-set::

   .. tab-item:: x86_64
      :sync: x86-64

      .. tab-set::

         .. tab-item:: Ubuntu 24.04
            :sync: ubuntu-24

            .. code-block:: sh

               curl -L https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2024.4/linux/openvino_genai_ubuntu24_2024.4.0.0_x86_64.tar.gz --output openvino_genai_2024.4.0.0.tgz
               tar -xf openvino_genai_2024.4.0.0.tgz

         .. tab-item:: Ubuntu 22.04
            :sync: ubuntu-22

            .. code-block:: sh

               curl -L https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2024.4/linux/openvino_genai_ubuntu22_2024.4.0.0_x86_64.tar.gz --output openvino_genai_2024.4.0.0.tgz
               tar -xf openvino_genai_2024.4.0.0.tgz

         .. tab-item:: Ubuntu 20.04
            :sync: ubuntu-20

            .. code-block:: sh

               curl -L https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2024.4/linux/openvino_genai_ubuntu20_2024.4.0.0_x86_64.tar.gz  --output openvino_genai_2024.4.0.0.tgz
               tar -xf openvino_genai_2024.4.0.0.tgz


   .. tab-item:: ARM 64-bit
      :sync: arm-64

      .. code-block:: sh

         curl -L https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2024.4/linux/openvino_genai_ubuntu20_2024.4.0.0_arm64.tar.gz -O openvino_genai_2024.4.0.0.tgz
         tar -xf openvino_genai_2024.4.0.0.tgz


Windows
++++++++++++++++++++++++++

.. code-block:: sh

   cd <user_home>/Downloads
   curl -L https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2024.4/windows/openvino_genai_windows_2024.4.0.0_x86_64.zip --output openvino_genai_2024.4.0.0.zip

macOS
++++++++++++++++++++++++++

.. tab-set::

   .. tab-item:: x86, 64-bit
      :sync: x86-64

      .. code-block:: sh

         curl -L https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2024.4/macos/openvino_genai_macos_12_6_2024.4.0.0_x86_64.tar.gz --output openvino_genai_2024.4.0.0.tgz
         tar -xf openvino_genai_2024.4.0.0.tgz

   .. tab-item:: ARM, 64-bit
      :sync: arm-64

      .. code-block:: sh

         curl -L https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2024.4/macos/openvino_genai_macos_12_6_2024.4.0.0_arm64.tar.gz --output openvino_genai_2024.4.0.0.tgz
         tar -xf openvino_genai_2024.4.0.0.tgz


Here are the full guides:
:doc:`Linux <install-openvino-archive-linux>`,
:doc:`Windows <install-openvino-archive-windows>`, and
:doc:`macOS <install-openvino-archive-macos>`.



