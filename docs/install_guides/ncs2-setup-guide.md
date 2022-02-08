# Intel® Neural Compute Stick 2 (NCS2) Setup Guide for Use with Intel® Distribution of OpenVINO™ toolkit {#openvino_docs_install_guides_ncs2_setup_guide}

@sphinxdirective

.. _ncs guide:

@endsphinxdirective

## Linux

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
