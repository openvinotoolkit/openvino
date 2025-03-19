# NPU Compile Tool

This page demonstrates how to use NPU Compile Tool to convert OpenVINO™ Intermediate Representation (IR) of an AI model or a model in ONNX format to a "blob" file that is compiled by NPU NN Compiler and serialized to the format accessible for NPU Driver and NPU Runtime to execute.


## Description

Compile tool is a C++ application that enables you to compile a model for inference on a specific device and export the compiled representation to a binary file.
With this tool, you can compile a model using supported OpenVINO Runtime devices on a machine that does not have the physical device connected, i.e. without NPU driver and Runtime loading, and then transfer a generated file to any machine with the target inference device available.

Using Compile Tool is not a basic approach to end-to-end execution and/or application but mostly suitable for debugging and validation and some specific use cases. If one is looking for the standard way of reducing application startup delays by exporting and reusing the compiled model automatically, refer to [Model Caching article](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html#model-caching)

## Workflow of the Compile tool

First, the application reads command-line parameters and loads a model to the OpenVINO Runtime device. After that, the application exports a blob with the compiled model and writes it to the output file.

## How to build

### Within NPU Plugin build

See [How to build](https://github.com/openvinotoolkit/openvino/wiki#how-to-build). If `ENABLE_INTEL_NPU=ON` is provided, no additional steps are required for Compile Tool. It will be built unconditionally with every NPU Plugin build. It can be found in `bin` folder.

If you need to configure a release package layout and have Compile Tool in it, use `cmake --install <dir> --component npu_internal` from your `build` folder. After installation compile_tool executable can be found in `<install_dir>/tools/compile_tool` folder.

### Standalone build

#### Prerequisites
* [OpenVINO™ Runtime release package](https://docs.openvino.ai/2025/get-started/install-openvino.html)

#### Build instructions
1. Download and install OpenVINO™ Runtime package
2. Build Compile Tool
    ```sh
    mkdir compile_tool_build && cd compile_tool_build
    cmake -DOpenVINO_DIR=<openvino_install_dir>/runtime/cmake <compile_tool_source_dir>
    cmake --build . --config Release
    cmake --install . --prefix <compile_tool_install_dir>
    ```
    > Note 1: command line instruction might differ on different platforms (e.g. Windows cmd)
    > Note 2: this example is based on OpenVINO Archive distribution. If you have chosen another installation method, specifying OpenVINO_DIR and calling `setupvars` script might not be needed. Refer [documentation](https://docs.openvino.ai/2025/get-started/install-openvino.html) for details.
    > Note 3: `<compile_tool_install_dir>` can be any directory on your filesystem that you want to use for installation including `<openvino_install_dir>` if you wish to extend OpenVINO package
3. Verify the installation
    ```sh
    source <openvino_install_dir>/setupvars.sh
    <compile_tool_install_dir>/tools/compile_tool/compile_tool -h
    ```
    > Note 1: command line might differ depending on your platform
    > Note 2: this example is based on OpenVINO Archive distribution. If you have chosen another installation method, calling setupvars might not be needed. Refer [documentation](https://docs.openvino.ai/2025/get-started/install-openvino.html) for details.

    Successful build will show the information about Compile Tool CLI options


## How to run

Running the application with the `-h` option yields the following usage message:
```
OpenVINO Runtime version ......... 202x.y.z
Build ........... 202x.y.z-build-hash
Parsing command-line arguments
compile_tool [OPTIONS]

 Common options:
    -h                                       Optional. Print the usage message.
    -m                           <value>     Required. Path to the XML model.
    -d                           <value>     Required. Specify a target device for which executable network will be compiled.
                                             Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin.
                                             Use "-d MULTI:<comma-separated_devices_list>" format to specify MULTI plugin.
                                             The application looks for a suitable plugin for the specified device.
    -o                           <value>     Optional. Path to the output file. Default value: "<model_xml_file>.blob".
    -c                           <value>     Optional. Path to the configuration file.
    -ip                          <value>     Optional. Specifies precision for all input layers of the network.
    -op                          <value>     Optional. Specifies precision for all output layers of the network.
    -iop                        "<value>"    Optional. Specifies precision for input and output layers by name.
                                             Example: -iop "input:FP16, output:FP16".
                                             Notice that quotes are required.
                                             Overwrites precision from ip and op options for specified layers.
    -il                          <value>     Optional. Specifies layout for all input layers of the network.
    -ol                          <value>     Optional. Specifies layout for all output layers of the network.
    -iol                        "<value>"    Optional. Specifies layout for input and output layers by name.
                                             Example: -iol "input:NCHW, output:NHWC".
                                             Notice that quotes are required.
                                             Overwrites layout from il and ol options for specified layers.
    -iml                         <value>     Optional. Specifies model layout for all input layers of the network.
    -oml                         <value>     Optional. Specifies model layout for all output layers of the network.
    -ioml                       "<value>"    Optional. Specifies model layout for input and output tensors by name.
                                             Example: -ionl "input:NCHW, output:NHWC".
                                             Notice that quotes are required.
                                             Overwrites layout from il and ol options for specified layers.
    -shape                       <value>      Set shape for model input. For example, "input1[1,3,224,224],input2[1,4]" or "[1,3,224,224]" in case of one input size. This parameter affect model input shape and can be dynamic. For dynamic dimensions use symbol `?` or '-1'. Ex. [?,3,?,?]. For bounded dimensions specify range 'min..max'. Ex. [1..10,3,?,?].
```
Running the application with the empty list of options yields an error message.

For example, to compile a blob for inference on Intel® Core™ Ultra NPU, run the command below
```
./compile_tool -m <path_to_model>/model_name.xml -d NPU.3720
```

You can pass a config file via `-c` option which allow you to specify some public or private properties. More details in [Supported Properties](https://github.com/openvinotoolkit/openvino/tree/master/src/plugins/intel_npu#supported-properties) and in [configs](https://github.com/openvinotoolkit/openvino/tree/master/src/plugins/intel_npu/src/al/include/intel_npu/config). For example, to use a custom build of NPU Compiler instaed of the release Compiler distributed within NPU driver, create a config file with the following content:
```
NPU_COMPILER_TYPE MLIR
```
