# Uninstalling the Intel® Distribution of OpenVINO™ Toolkit {#openvino_docs_install_guides_uninstalling_openvino}

> **NOTE**: Uninstallation procedures remove all Intel® Distribution of OpenVINO™ Toolkit component files but don't affect user files in the installation directory.

## Uninstall Using the Original Installation Package

@sphinxdirective
.. tab:: Windows

  Use initial bootstrapper file ``w_openvino_toolkit_p_<version>.exe`` to select product for uninstallation. Follow the wizard instructions. Select **Remove** option when presented. If you have more product versions installed, you can select one from a drop-down menu in GUI.

  .. image:: _static/images/openvino-uninstall-dropdown-win.png
    :width: 500px
    :align: center
    
.. tab:: Linux

  If you want to use graphical user interface (GUI) installation wizard, run the script without any parameters:
  
  .. code-block:: sh
  
    ./l_openvino_toolkit_p_<version>.sh

  Follow the wizard instructions.

  Otherwise, you can add parameters `-a` for additional arguments and `--cli` to run installation in command line (CLI):
  
  .. code-block:: sh
    
    ./l_openvino_toolkit_p_<version>.sh -a --cli

  Follow the wizard. Select **Remove** option when presented. If you have more product versions installed, you can select one from a drop-down menu in GUI and from a list in CLI.

  .. image:: _static/images/openvino-uninstall-dropdown-linux.png
    :width: 500px
    :align: center

.. tab:: macOS

  Use initial bootstrapper file ``m_openvino_toolkit_p_<version>.dmg`` to select product for uninstallation. Mount the file and double-click ``bootstrapper.app``. Follow the wizard instructions. Select **Remove** option when presented. If you have more product versions installed, you can select one from a drop-down menu in GUI.

  .. image:: _static/images/openvino-uninstall-dropdown-macos.png
    :width: 500px
    :align: center

@endsphinxdirective


