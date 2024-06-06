
Install OpenVINO™ GenAI Flavor
====================================

The new OpenVINO GenAI Flavor is an API designed to hide the complexity of the generation process
and significantly minimize the amount of code needed for the application to work.
Developers can now provide a model and input context directly to the OpenVINO GenAI, which performs
tokenization of the input text, executes the generation loop on the selected device, and then returns the generated text.
For a quickstart guide, refer to the :doc:`GenAI API Guide <../../learn-openvino/llm_inference_guide/genai-guide>`.

OpenVINO GenAI Flavor is available for installation via Archive and PyPI distributions.

Archive Installation
###############################

Use this command to download an archive file with OpenVINO GenAI Flavor for your system:

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



For full instructions, refer to the OpenVINO Runtime Archive installation for your system: :doc:`Linux <install-openvino-archive-linux>`, :doc:`Windows <install-openvino-archive-windows>`, and :doc:`macOS <install-openvino-archive-macos>`.

PyPI Installation
###############################

Use this command to install OpenVINO GenAI via PyPI:

.. code-block:: python

   python -m pip install openvino-genai

Refer to the :doc:`OpenVINO Runtime PyPI installation <install-openvino-pip>` for full instructions.







