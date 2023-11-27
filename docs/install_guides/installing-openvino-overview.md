# Install OpenVINO™ 2023.1 {#openvino_docs_install_guides_overview}

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



.. card:: Installing this version is not recommended
    :link: https://docs.openvino.ai/install

    This is not the most recent release of OpenVINO. Since it is not a Long-term-support
    version either, the selector tool for it is no longer available. 
    
    Click here to visit the current Selector Tool version and install the current
    stable version of OpenVINO...




.. warning::
   
   The OpenVINO Development Tools package has been deprecated and removed from the default
   installation options. For new projects, the OpenVINO runtime package now includes
   all necessary components.

   The OpenVINO Development Tools is still available for older versions of OpenVINO,
   as well as the current one, from the GitHub repository and PyPI. :doc:`Learn more <openvino_docs_install_guides_install_dev_tools>`.


.. tip::
   
   OpenVINO 2023.1, described here, is not a Long-Term-Support version!
   All currently supported versions are:

   * 2023.1 (development)
   * 2022.3 (LTS)
   * 2021.4 (LTS) 

   Moreover, different OpenVINO distributions may support slightly different sets of features.
   Read installation guides for particular distributions for more details. 

   .. dropdown:: Distribution Comparison for OpenVINO 2023.1
   
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

