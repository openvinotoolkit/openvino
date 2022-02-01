# Intel® Neural Compute Stick 2 (NCS2) Setup Guide for Use with Intel® Distribution of OpenVINO™ toolkit {#openvino_docs_install_guides_ncs2_setup_guide}

@sphinxdirective

.. _ncs guide:

@endsphinxdirective

Once you have your Intel® Distribution of OpenVINO™ toolkit installed, follow the steps to be able to work on NCS2:

1. Go to the install_dependencies directory:
   ```sh
   cd <INSTALL_DIR>/intel/openvino_2022/install_dependencies/
   ```
2. Run the `install_NCS_udev_rules.sh` script:
   ```
   ./install_NCS_udev_rules.sh
   ```
3. You may need to reboot your machine for this to take effect.

You've completed all required configuration steps to perform inference on Intel® Neural Compute Stick 2. 
Proceed to the <a href="#get-started">Start Using the Toolkit</a> section to learn the basic OpenVINO™ toolkit workflow and run code samples and demo applications.
