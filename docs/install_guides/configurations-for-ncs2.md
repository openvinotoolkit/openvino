# Configurations for Intel® Neural Compute Stick 2 {#openvino_docs_install_guides_configurations_for_ncs2}

@sphinxdirective

.. _ncs guide:

@endsphinxdirective


## Linux

Once you have OpenVINO™ Runtime installed, follow these steps to be able to work on NCS2:

1. Add the current Linux user to the `users` group:
   ```sh
   sudo usermod -a -G users "$(whoami)"
   ```
2. Go to the install_dependencies directory:
   ```sh
   cd <INSTALL_DIR>/install_dependencies/
   ```
3. Copy the `97-myriad-usbboot.rules` file to the udev rules directory:
   ```
   sudo cp 97-myriad-usbboot.rules /etc/udev/rules.d/
   ``` 
4. Now reload udev rules with rules that you copied

   ```
   sudo udevadm control --reload-rules
   sudo udevadm trigger
   sudo ldconfig
   ``` 
5. You may need to reboot your machine for this to take effect.

You've completed all required configuration steps to perform inference on Intel® Neural Compute Stick 2.


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

## What’s Next?

Now you are ready to try out OpenVINO™. You can use the following tutorials to write your applications using Python and C++.

Developing in Python:
   * [Start with tensorflow models with OpenVINO™](https://docs.openvino.ai/latest/notebooks/101-tensorflow-to-openvino-with-output.html)
   * [Start with ONNX and PyTorch models with OpenVINO™](https://docs.openvino.ai/latest/notebooks/102-pytorch-onnx-to-openvino-with-output.html)
   * [Start with PaddlePaddle models with OpenVINO™](https://docs.openvino.ai/latest/notebooks/103-paddle-onnx-to-openvino-classification-with-output.html)

Developing in C++:
   * [Image Classification Async C++ Sample](@ref openvino_inference_engine_samples_classification_sample_async_README)
   * [Hello Classification C++ Sample](@ref openvino_inference_engine_samples_hello_classification_README)
   * [Hello Reshape SSD C++ Sample](@ref openvino_inference_engine_samples_hello_reshape_ssd_README)
