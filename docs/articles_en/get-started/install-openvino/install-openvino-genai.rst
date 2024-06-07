Install OpenVINO™ GenAI 
====================================

OpenVINO GenAI is a new flavor of OpenVINO, aiming to simplify running inference of generative AI models.
It hides the complexity of the generation process and minimizes the amount of code required.
You can now provide a model and input context directly to OpenVINO, which performs tokenization of the
input text, executes the generation loop on the selected device, and returns the generated text.
For a quickstart guide, refer to the :doc:`GenAI API Guide <../../learn-openvino/llm_inference_guide/genai-guide>`.

To see GenAI in action, check the Jupyter notebooks: 
`LLM-powered Chatbot <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/llm-chatbot/README.md>`__ and 
`LLM Instruction-following pipeline <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/llm-question-answering/README.md>`__

The OpenVINO GenAI flavor is available for installation via Archive and PyPI distributions:

Archive Installation
###############################

To install the GenAI flavor of OpenVINO from an archive file, follow the standard installation steps for your system
but instead of using the vanilla package file, download the one with OpenVINO GenAI:

.. tab-set::

   .. tab-item:: x86_64
      :sync: x86-64

      .. tab-set::

         .. tab-item:: Ubuntu 24.04
            :sync: ubuntu-24

            .. code-block:: sh


               curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.1/linux/l_openvino_toolkit_ubuntu22_2024.1.0.15008.f4afc983258_x86_64.tgz --output openvino_2024.1.0.tgz
               tar -xf openvino_2024.1.0.tgz
               sudo mv l_openvino_toolkit_ubuntu24_2024.1.0.15008.f4afc983258_x86_64 /opt/intel/openvino_2024.1.0

         .. tab-item:: Ubuntu 22.04
            :sync: ubuntu-22

            .. code-block:: sh

               curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.1/linux/l_openvino_toolkit_ubuntu22_2024.1.0.15008.f4afc983258_x86_64.tgz --output openvino_2024.1.0.tgz
               tar -xf openvino_2024.1.0.tgz
               sudo mv l_openvino_toolkit_ubuntu22_2024.1.0.15008.f4afc983258_x86_64 /opt/intel/openvino_2024.1.0

Here are the full guides:
:doc:`Linux <install-openvino-archive-linux>`, 
:doc:`Windows <install-openvino-archive-windows>`, and
:doc:`macOS <install-openvino-archive-macos>`.


PyPI Installation
###############################

To install the GenAI flavor of OpenVINO via PyPI, follow the standard :doc:`installation steps <install-openvino-pip>`,
but use the *openvino-genai* package instead of *openvino*:

.. code-block:: python

   python -m pip install openvino-genai




