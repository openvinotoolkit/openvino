# Install OpenVINO™ 2023.2 {#openvino_docs_install_guides_overview}

@sphinxdirective

.. meta::
   :description: install OpenVINO Runtime package, using the distribution channel 
                 of your choice.


.. toctree::
   :maxdepth: 3
   :hidden:

   OpenVINO Runtime on Linux <openvino_docs_install_guides_installing_openvino_linux_header>
   OpenVINO Runtime on Windows <openvino_docs_install_guides_installing_openvino_windows_header>
   OpenVINO Runtime on macOS <openvino_docs_install_guides_installing_openvino_macos_header>  
   Create a Yocto Image <openvino_docs_install_guides_installing_openvino_yocto>


.. raw:: html

   <script type="module" crossorigin src="_static/selector-tool/assets/index-f34d1fad.js"></script>
   <meta name="viewport" content="width=device-width, initial-scale=1.0" />
   <iframe id="selector" src="_static/selector-tool/selector-c1c409a.html" style="width: 100%; border: none" title="Download Intel® Distribution of OpenVINO™ Toolkit"></iframe>

.. warning::
   
   The OpenVINO Development Tools package has been deprecated and removed from the default
   installation options. For new projects, the OpenVINO runtime package now includes
   all necessary components.

   The OpenVINO Development Tools is still available for older versions of OpenVINO,
   as well as the current one, from the GitHub repository and PyPI. :doc:`Learn more <openvino_docs_install_guides_install_dev_tools>`.


.. tip::
   
   OpenVINO 2023.2, described here, is not a Long-Term-Support version!
   All currently supported versions are:

   * 2023.2 (development)
   * 2022.3 (LTS)
   * 2021.4 (LTS) 

   Moreover, different OpenVINO distributions may support slightly different sets of features.
   Read installation guides for particular distributions for more details. 

   .. dropdown:: Distribution Comparison for OpenVINO 2023.2
   
      ===============  ==========  ======  =========  ========  ============ ========== ========== 
       Device           Archives    PyPI    APT/YUM    Conda     Homebrew     vcpkg      Conan     
      ===============  ==========  ======  =========  ========  ============ ========== ========== 
       CPU              V           V        V         V          V           V          V         
       GPU              V           V        V         V          V           V          V         
       GNA              V          n/a      n/a       n/a        n/a         n/a        n/a        
       NPU              V          n/a      n/a       n/a        n/a         n/a        n/a        
      ===============  ==========  ======  =========  ========  ============ ========== ========== 

| **Build OpenVINO from source**
|    OpenVINO Toolkit source files are available on GitHub as open source. If you want to build your own version of OpenVINO for your platform, 
     follow the `OpenVINO Build Instructions <https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build.md>`__.



@endsphinxdirective

