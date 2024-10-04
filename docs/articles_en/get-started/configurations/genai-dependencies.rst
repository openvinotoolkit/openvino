OpenVINO™ GenAI Dependencies
=================================

OpenVINO™ GenAI depends on both `OpenVINO <https://github.com/openvinotoolkit/openvino>`__ and
`OpenVINO Tokenizers <https://github.com/openvinotoolkit/openvino_tokenizers>`__. During OpenVINO™
GenAI installation from PyPi, the same versions of OpenVINO and OpenVINO Tokenizers
are used (e.g. ``openvino==2024.4.0`` and ``openvino-tokenizers==2024.4.0.0`` are installed for
``openvino-genai==2024.4.0``).

Trying to update any of the dependency packages might result in a version incompatiblibty
due to different Application Binary Interfaces (ABIs), which will result in errors while running
OpenVINO GenAI. Having package version in the ``<MAJOR>.<MINOR>.<PATCH>.<REVISION>`` format, allows
changing the ``<REVISION>`` portion of the full version to ensure ABI compatibility. Changing
``<MAJOR>``, ``<MINOR>`` or ``<PATCH>`` part of the version may break ABI.

GenAI, Tokenizers, and OpenVINO wheels for Linux on PyPI are compiled with ``_GLIBCXX_USE_CXX11_ABI=0``
to cover a wider range of platforms. In the C++ archive distributions for Ubuntu, ``_GLIBCXX_USE_CXX11_ABI=1``
is used instead. Mixing different ABIs is not possible as doing so will result in a link error.

To try OpenVINO GenAI with different dependencies versions (which are **not** prebuilt packages
as archives or python wheels), build OpenVINO GenAI library from
`Source <https://github.com/openvinotoolkit/openvino.genai/blob/releases/2024/3/src/docs/BUILD.md#build-openvino-openvino-tokenizers-and-openvino-genai-from-source>`__.

Additional Resources
#######################

* :doc:`OpenVINO GenAI Installation Guide <../install-openvino/install-openvino-genai>`
* `OpenVINO GenAI repository <https://github.com/openvinotoolkit/openvino.genai>`__
* :doc:`OpenVINO Installation Guide <../install-openvino>`
* :doc:`OpenVINO Tokenizers <../../learn-openvino/llm_inference_guide/ov-tokenizers>`

