# Uninstalling the Intel® Distribution of OpenVINO™ Toolkit {#openvino_docs_install_guides_uninstalling_openvino}

> **NOTE**: Uninstallation procedures remove all Intel® Distribution of OpenVINO™ Toolkit component files and don't affect user files in the installation directory.

## Uninstall Using the Original Installation Package

@sphinxdirective
.. tab:: Windows

  1. Open the initial bootstrapper file ``w_openvino_toolkit_p_<version>.exe``. 
  2. Follow the wizard instructions on Your screen. 
  3. Select **Remove** option once it presents. 
  
  > **NOTE**: If you have more product versions installed, you can select the one to uninstall from a drop-down menu in GUI.

  .. image:: _static/images/openvino-uninstall-dropdown-win.png
    :width: 500px
    :align: center
    
.. tab:: Linux

  To use graphical user interface (GUI) installation wizard, run in terminal:
  
  .. code-block:: sh
  
    ./l_openvino_toolkit_p_<version>.sh

  To use command line (CLI), run in terminal:
  
  .. code-block:: sh
    
    ./l_openvino_toolkit_p_<version>.sh -a --cli

  After using uninstallation method:
  1. Follow the wizard. 
  2. Select **Remove** option when presented. 
  
  > **NOTE**: If you have more product versions installed, you can select one to uninstall from a drop-down menu in GUI and from a list in CLI.

  .. image:: _static/images/openvino-uninstall-dropdown-linux.png
    :width: 500px
    :align: center

.. tab:: macOS

  1. Open initial bootstrapper file ``m_openvino_toolkit_p_<version>.dmg``.
  2. Mount the file and double-click ``bootstrapper.app``. 
  3. Follow the wizard instructions. 
  4. Select **Remove** option once it presents. 
  
  > **NOTE**: If you have more product versions installed, you can select one to uninstall from a drop-down menu in GUI.

  .. image:: _static/images/openvino-uninstall-dropdown-macos.png
    :width: 500px
    :align: center

@endsphinxdirective

## Uninstall Using the Intel® Software Installer

@sphinxdirective
.. tab:: Windows

  **Option 1:**
  1. Open the **Apps & Features** option from the Windows Settings app.
  2. From the list of installed applications, select the Intel® Distribution of OpenVINO™ Toolkit and click **Uninstall**.
  3. Follow the uninstallation wizard instructions.

  **Option 2:**
  1. Go to installation directory on Your PC.
  2. In ``OpenVINO`` directory find ``Installer`` folder and open it.
  3. Double-click on ``installer.exe`` and you will be presented with dialog box shown below.

.. tab:: Linux

  1. Run the installer file from the user mode installation directory:
   
  .. code-block:: sh
  
    /home/<user>/intel/openvino_installer/installer

  In a case of administrative installation:

  .. code-block:: sh

    /opt/intel/openvino_installer/installer

  2. Follow the uninstallation wizard instructions.
  
.. tab:: macOS

  1. Open the installer file from the installation directory:
   
  .. code-block:: sh
  
    open /opt/intel/openvino_installer/installer.app

  2. Follow the uninstallation wizard instructions.


Complete the procedure with pressing on **Modify** and then selecting **Uninstall** option from drop down menu:

.. tab:: Windows
  
  .. image:: _static/images/openvino-uninstall-win.png
    :width: 500px
    :align: center

.. tab:: Linux
 
  .. image:: _static/images/openvino-uninstall-linux.png
    :width: 500px
    :align: center
    
  if GUI is not available, uninstallation can also be run in CLI:

  .. image:: _static/images/openvino-uninstall-cli.png
     :width: 500px
     :align: center
  
.. tab:: macOS

  .. image:: _static/images/openvino-uninstall-macos.png
    :width: 500px
    :align: center

@endsphinxdirective
