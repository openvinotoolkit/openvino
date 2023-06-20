# Uninstalling the Intel® Distribution of OpenVINO™ Toolkit {#openvino_docs_install_guides_uninstalling_openvino}

@sphinxdirective

.. note::

   Uninstallation procedures remove all Intel® Distribution of OpenVINO™ Toolkit component files but don't affect user files in the installation directory.

Uninstall Using the Original Installation Package
#################################################

If you have installed OpenVINO Runtime from archive files, you can uninstall it by deleting the archive files and the extracted folders.

.. tab-set::

   .. tab-item:: Windows
      :sync: windows
   
      If you have created the symbolic link, remove the link first.
    
      Use either of the following methods to delete the files:
    
      * Use Windows Explorer to remove the files.
      * Open a Command Prompt and run:
    
        .. code-block:: sh
    
          rmdir /s <extracted_folder>
          del <path_to_archive>
   
   
   .. tab-item:: Linux & macOS
      :sync: linmac
   
      If you have created the symbolic link, remove the link first:
    
      .. code-block:: sh
    
        sudo rm /opt/intel/openvino_2023
    
      To delete the files:
    
      .. code-block:: sh
    
        rm -r <extracted_folder> && rm <path_to_archive>


@endsphinxdirective

