Install OpenVINO™ 2024.4
==========================


.. meta::
   :description: install OpenVINO Runtime package, using the distribution channel
                 of your choice.


.. toctree::
   :maxdepth: 3
   :hidden:

   OpenVINO Runtime on Linux <install-openvino/install-openvino-linux>
   OpenVINO Runtime on Windows <install-openvino/install-openvino-windows>
   OpenVINO Runtime on macOS <install-openvino/install-openvino-macos>
   Create an OpenVINO Yocto Image <install-openvino/install-openvino-yocto>
   OpenVINO GenAI Flavor <install-openvino/install-openvino-genai>

.. raw:: html

   <script type="module" crossorigin src="../_static/selector-tool/assets/index-f34d1fad.js"></script>
   <meta name="viewport" content="width=device-width, initial-scale=1.0" />
   <iframe id="selector" src="../_static/selector-tool/selector-8d4cf1d.html" style="width: 100%; border: none" title="Download Intel® Distribution of OpenVINO™ Toolkit"></iframe>

OpenVINO 2024.4, described here, is not a Long-Term-Support version!
All currently supported versions are:

* 2024.4 (development)
* 2023.3 (LTS)
* 2022.3 (LTS)

.. dropdown:: Distributions and Device Support

   Different OpenVINO distributions may support slightly different sets of features.
   Read installation guides for particular distributions for more details.
   Refer to the :doc:`OpenVINO Release Policy <../../../about-openvino/release-notes-openvino/release-policy>`
   to learn more about the release types.

   .. dropdown:: Distribution Comparison for OpenVINO 2024.4

      ===============  ==========  ======  ===============  ========  ============ ========== ========== ==========
      Device           Archives    PyPI    APT/YUM/ZYPPER    Conda     Homebrew     vcpkg      Conan       npm
      ===============  ==========  ======  ===============  ========  ============ ========== ========== ==========
      CPU              V           V       V                V         V            V          V          V
      GPU              V           V       V                V         V            V          V          V
      NPU              V\*         V\*     V\ *             n/a       n/a          n/a        n/a        V\*
      ===============  ==========  ======  ===============  ========  ============ ========== ========== ==========

      | \* **Of the Linux systems, versions 22.04 and 24.04 include drivers for NPU.**
      |  **For Windows, CPU inference on ARM64 is not supported.**

.. dropdown:: Effortless GenAI integration with OpenVINO GenAI Flavor

   A new OpenVINO GenAI Flavor streamlines application development by providing
   LLM-specific interfaces for easy integration of language models, handling tokenization and
   text generation. For installation and usage instructions, proceed to
   :doc:`Install OpenVINO GenAI Flavor <../learn-openvino/llm_inference_guide/genai-guide>` and
   :doc:`Run LLMs with OpenVINO GenAI Flavor <../learn-openvino/llm_inference_guide/genai-guide>`.

.. dropdown:: Deprecation of OpenVINO™ Development Tools Package

   The OpenVINO™ Development Tools package has been deprecated and removed from the default
   installation options. For new projects, the OpenVINO runtime package now includes
   all necessary components.

   The OpenVINO Development Tools is still available for older versions of OpenVINO,
   as well as the current one, from the GitHub repository and PyPI. :doc:`Learn more <../documentation/legacy-features/install-dev-tools>`.

.. dropdown:: Building OpenVINO from Source

   OpenVINO Toolkit source files are available on GitHub as open source. If you want to build your own version of OpenVINO for your platform,
   follow the `OpenVINO Build Instructions <https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build.md>`__.




