# Uninstalling the Intel® Distribution of OpenVINO™ Toolkit {#openvino_docs_install_guides_uninstalling_openvino}

> **NOTE**: Uninstallation procedures remove all Intel® Distribution of OpenVINO™ Toolkit component files but don't affect user files in the installation directory.

## <a name="uninstall"></a>Uninstall the Toolkit in Linux
Choose one of the options provided below to uninstall the Intel® Distribution of OpenVINO™ Toolkit from your system.

### Uninstall with GUI
1. Run the uninstallation script from `<INSTALL_DIR>/openvino_toolkit_uninstaller`:
   ```sh
   sudo ./uninstall_GUI.sh
   ```
2. Follow the uninstallation wizard instructions.


### Uninstall with Command Line (Interactive Mode)
1. Run the uninstallation script from `<INSTALL_DIR>/openvino_toolkit_uninstaller`:
   ```sh
   sudo ./uninstall.sh
   ```
2. Follow the instructions on your screen.
4. When uninstallation is complete, press **Enter**.

### Uninstall with Command Line (Silent Mode)
1. Run the following command from `<INSTALL_DIR>/openvino_toolkit_uninstaller`:
   ```sh
   sudo ./uninstall.sh -s
   ```
2. Intel® Distribution of OpenVINO™ Toolkit is now uninstalled from your system.

## <a name="uninstall"></a>Uninstall the Toolkit in Windows

Follow the steps below to uninstall the Intel® Distribution of OpenVINO™ Toolkit from your system:
1. Choose the **Apps & Features** option from the Windows* Settings app.
2. From the list of installed applications, select the Intel® Distribution of OpenVINO™ Toolkit and click **Uninstall**.
3. Follow the uninstallation wizard instructions.
4. When uninstallation is complete, click **Finish**. 

## <a name="uninstall"></a>Uninstall the Toolkit in macOS

Follow the steps below to uninstall the Intel® Distribution of OpenVINO™ Toolkit from your system:

1. Run the uninstall application from the installation directory (by default, `/opt/intel/openvino_2021`):
   ```sh
   open openvino_toolkit_uninstaller.app
   ```
2. Follow the uninstallation wizard instructions.
3. When uninstallation is complete, click **Finish**. 
