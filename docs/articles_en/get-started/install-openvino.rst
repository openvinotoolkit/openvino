Install OpenVINO™ 2024.2
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


.. raw:: html

   <script type="module" crossorigin src="../_static/selector-tool/assets/index-f34d1fad.js"></script>
   <meta name="viewport" content="width=device-width, initial-scale=1.0" />
   <iframe id="selector" src="../_static/selector-tool/selector-1c16038.html" style="width: 100%; border: none" title="Download Intel® Distribution of OpenVINO™ Toolkit"></iframe>

.. tip::

   The new OpenVINO GenAI API package provides a set of LLM-specific interfaces to facilitate the integration
   of language models into applications. This API hides the complexity of the generation process
   and significantly minimizes the amount of code needed for the application to work.
   Developers can now provide a model and input context directly to the OpenVINO GenAI, which performs
   tokenization of the input text, executes the generation loop on the selected device, and then returns the generated text.
   For a quickstart guide, refer to the :doc:`GenAI API Guide <../learn-openvino/llm_inference_guide/genai-guide>` or the GenAI API reference page.

.. warning::

   The OpenVINO™ Development Tools package has been deprecated and removed from the default
   installation options. For new projects, the OpenVINO runtime package now includes
   all necessary components.

   The OpenVINO Development Tools is still available for older versions of OpenVINO,
   as well as the current one, from the GitHub repository and PyPI. :doc:`Learn more <../documentation/legacy-features/install-dev-tools>`.


.. tip::

   OpenVINO 2024.2, described here, is not a Long-Term-Support version!
   All currently supported versions are:

   * 2024.2 (development)
   * 2023.3 (LTS)
   * 2022.3 (LTS)

   Moreover, different OpenVINO distributions may support slightly different sets of features.
   Read installation guides for particular distributions for more details.

   .. dropdown:: Distribution Comparison for OpenVINO 2024.2

      ===============  ==========  ======  ===============  ========  ============ ========== ========== ==========
       Device           Archives    PyPI    APT/YUM/ZYPPER    Conda     Homebrew     vcpkg      Conan       npm
      ===============  ==========  ======  ===============  ========  ============ ========== ========== ==========
       CPU              V           V       V                V         V            V          V          V
       GPU              V           V       V                V         V            V          V          V
       NPU              V\*         V\*     V\ *             n/a       n/a          n/a        n/a        V\*
      ===============  ==========  ======  ===============  ========  ============ ========== ========== ==========

      | \* **Of the Linux systems, versions 22.04 and 24.04 include drivers for NPU.**
      |  **For Windows, CPU inference on ARM64 is not supported.**

| **Build OpenVINO from source**
|    OpenVINO Toolkit source files are available on GitHub as open source. If you want to build your own version of OpenVINO for your platform,
     follow the `OpenVINO Build Instructions <https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build.md>`__.




