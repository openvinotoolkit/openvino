# Configurations for Intel® Neural Compute Stick 2 {#openvino_docs_install_guides_configurations_for_ncs2}

@sphinxdirective

.. _ncs guide:

@endsphinxdirective

This page introduces additional configurations for Intel® Neural Compute Stick 2 with Intel® Distribution of OpenVINO™ toolkit on Linux, Raspbian OS and macOS.

## Linux

Once you have your Intel® Distribution of OpenVINO™ toolkit installed, follow the steps to be able to work on NCS2:

1. Go to the install_dependencies directory:
   ```sh
   cd <INSTALL_DIR>/install_dependencies/
   ```
2. Run the `install_NCS_udev_rules.sh` script:
   ```
   ./install_NCS_udev_rules.sh
   ```
3. You may need to reboot your machine for this to take effect.

You've completed all required configuration steps to perform inference on Intel® Neural Compute Stick 2. 
Proceed to the [Get Started Guide](@ref get_started) section to learn the basic OpenVINO™ toolkit workflow and run code samples and demo applications.

@sphinxdirective

.. _ncs guide raspbianos:

@endsphinxdirective

## Raspbian OS

1. Add the current Linux user to the `users` group:
   ```sh
   sudo usermod -a -G users "$(whoami)"
   ```
   Log out and log in for it to take effect.
2. If you didn't modify `.bashrc` to permanently set the environment variables, run `setupvars.sh` again after logging in:
   ```sh
   source /opt/intel/openvino_2022/setupvars.sh
   ```
3. To perform inference on the Intel® Neural Compute Stick 2, install the USB rules running the `install_NCS_udev_rules.sh` script:
   ```sh
   sh /opt/intel/openvino_2022/install_dependencies/install_NCS_udev_rules.sh
   ```
4. Plug in your Intel® Neural Compute Stick 2.

5. (Optional) If you want to compile and run the Image Classification sample to verify the OpenVINO™ toolkit installation follow the next steps.

   a. Navigate to a directory that you have write access to and create a samples build directory. This example uses a directory named `build`:
   ```sh
   mkdir build && cd build
   ```
   b. Build the Hello Classification Sample:
   ```sh
   cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=armv7-a" /opt/intel/openvino_2022/samples/cpp
   ```
   ```sh
   make -j2 hello_classification
   ```
   c. Download the pre-trained squeezenet1.1 image classification model with the Model Downloader or copy it from the host machine:
   ```sh
   git clone --depth 1 https://github.com/openvinotoolkit/open_model_zoo
   cd open_model_zoo/tools/model_tools
   python3 -m pip install --upgrade pip
   python3 -m pip install -r requirements.in
   python3 downloader.py --name squeezenet1.1 
   ```
   d. Run the sample specifying the model, a path to the input image, and the VPU required to run with the Raspbian OS:
   ```sh
   ./armv7l/Release/hello_classification <path_to_model>/squeezenet1.1.xml <path_to_image> MYRIAD
   ```
   The application outputs to console window top 10 classification results.

@sphinxdirective

.. _ncs guide macos:

@endsphinxdirective

## macOS

These steps are required only if you want to perform inference on Intel® Neural Compute Stick 2 powered by the Intel® Movidius™ Myriad™ X VPU.

To perform inference on Intel® Neural Compute Stick 2, the `libusb` library is required. You can build it from the [source code](https://github.com/libusb/libusb) or install using the macOS package manager you prefer: [Homebrew](https://brew.sh/), [MacPorts](https://www.macports.org/) or other.

For example, to install the `libusb` library using Homebrew, use the following command:
```sh
brew install libusb
```

You've completed all required configuration steps to perform inference on your Intel® Neural Compute Stick 2.
Proceed to the <a href="openvino_docs_install_guides_installing_openvino_macos.html#get-started">Start Using the Toolkit</a> section to learn the basic OpenVINO™ toolkit workflow and run code samples and demo applications.
