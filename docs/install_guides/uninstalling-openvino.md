# Uninstalling the Intel® Distribution of OpenVINO™ Toolkit {#openvino_docs_install_guides_uninstalling_openvino}

> **NOTE**: Uninstallation procedures remove all Intel® Distribution of OpenVINO™ Toolkit component files but don't affect user files in the installation directory.

##There are 2 ways of uninstalling OpenVINO:

### Bootstrapper file

@sphinxdirective
.. tab:: Windows

  Go to the `Downloads` folder and double-click w_openvino_toolkit_p_<version>.exe. Follow the wizard instructions.

.. tab:: Linux

  If you want to use graphical user interface (GUI) installation wizard, run the script without any parameters:
  ```sh
  ./l_openvino_toolkit_p_<version>.sh
  ```
  Follow the wizard instructions.

  Otherwise, you can add parameters `-a` for additional arguments and `--cli` to run installation in command line (CLI):
  ```sh
  ./l_openvino_toolkit_p_<version>.sh -a --cli
  ```
  Follow the wizard. Select **Remove** option when presented.

.. tab:: macOS

  Go to the `Downloads` folder and double-click m_openvino_toolkit_p_<version>.dmg. Follow the wizard instructions.

@endsphinxdirective

### System specific

.. tab:: Windows

  1. Choose the **Apps & Features** option from the Windows* Settings app.
  2. From the list of installed applications, select the Intel® Distribution of OpenVINO™ Toolkit and click **Uninstall**.
  3. Follow the uninstallation wizard instructions.

.. tab:: Linux & macOS

  1. Run the installer file from the installation directory:
   ```sh
   ./<INSTALL_DIR>/intaller/installer
   ```
  2. Follow the uninstallation wizard instructions.

Finally, you need to complete the procedure with clicking on **Modify** and then selecting **Uninstall** option:

   @sphinxdirective

   .. image:: _static/images/openvino-uninstall.png
      :width: 500px
      :align: center

   @endsphinxdirective
