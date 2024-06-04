
Install OpenVINO™ GenAI Package
====================================

.. meta::
   :description:


The new GenAI API provides a set of LLM-specific interfaces designed to facilitate the integration
of language models into applications. This API hides the complexity of the generation process,
such as tokenization and managing generation loops, and significantly minimizes the amount of code needed for the application to work.
Developers can now provide a model and input context directly to the OpenVINO GenAI, which performs
tokenization of the input text, executes the generation loop on the selected device, and then returns the generated text.
The GenAI API is available through the following distribution channels:

1. PyPI

.. code-block:: sh

   pip install openvino-genai

2. Archives


For a quickstart guide on how to use the OpenVINO GenAI API, read the :doc:`GenAI API Guide <genai-api-guide>`.
Refer to the GenAI API reference page for more detailed information.

