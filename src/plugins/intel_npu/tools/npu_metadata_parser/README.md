# NPU MetaData Parser Tool

This page demonstrates how to use NPU Blob Parser Tool to separate NPU plugin specific metadata printing its content from the raw compiled model ("blob").


## Description

NPU Metadata Parser tool is a C++ application that extracts a raw compiled model from a generated file from NPU plugin where it appended it's specific metadata at the beginning of that file.

## Workflow of the NPU MetaData Parser Tool

First, the application reads command-line parameters and loads a compiled model from the disk. After that, the application exports a raw blob without metadata to the output file, but gives information about parsed metadata information.

## How to build

### Within NPU Plugin build

See [How to build](https://github.com/openvinotoolkit/openvino/wiki#how-to-build). If `ENABLE_INTEL_NPU=ON` is provided, no additional steps are required for NPU MetaData Parser Tool. It will be built unconditionally with every NPU Plugin build. It can be found in `bin` folder.

If you need to configure a release package layout and have NPU MetaData Parser Tool in it, use `cmake --install <dir> --component npu_internal` from your `build` folder. After installation npu_metadata_parser executable can be found in `<install_dir>/tools/npu_metadata_parser` folder.

### Standalone build

#### Prerequisites
* [OpenVINO™ Runtime release package](https://docs.openvino.ai/2025/get-started/install-openvino.html)

#### Build instructions
1. Download and install OpenVINO™ Runtime package
2. Build NPU MetaData Parser Tool
    ```sh
    mkdir npu_metadata_parser_build && cd npu_metadata_parser_tool_build
    cmake -DOpenVINO_DIR=<openvino_install_dir>/runtime/cmake <npu_metadata_parser_source_dir>
    cmake --build . --config Release
    cmake --install . --prefix <npu_metadata_parser_install_dir>
    ```
    > Note 1: command line instruction might differ on different platforms (e.g. Windows cmd)
    > Note 2: this example is based on OpenVINO Archive distribution. If you have chosen another installation method, specifying OpenVINO_DIR and calling `setupvars` script might not be needed. Refer [documentation](https://docs.openvino.ai/2025/get-started/install-openvino.html) for details.
    > Note 3: `<npu_metadata_parser_install_dir>` can be any directory on your filesystem that you want to use for installation including `<openvino_install_dir>` if you wish to extend OpenVINO package
3. Verify the installation
    ```sh
    source <openvino_install_dir>/setupvars.sh
    <npu_metadata_parser_install_dir>/tools/npu_metadata_parser/npu_metadata_parser -h
    ```
    > Note 1: command line might differ depending on your platform
    > Note 2: this example is based on OpenVINO Archive distribution. If you have chosen another installation method, calling setupvars might not be needed. Refer [documentation](https://docs.openvino.ai/2025/get-started/install-openvino.html) for details.

    Successful build will show the information about NPU MetaData Parser CLI options


## How to run

Running the application with the `-h` option yields the following usage message:
```
OpenVINO Runtime version ......... 202x.y.z
Build ........... 202x.y.z-build-hash
Parsing command-line arguments
npu_metadata_parser [OPTIONS]

 Common options:                             
    -h                                       Optional. Print the usage message.
    -b                           <value>     Required. Path to the NPU generated blob (with metadata).
    -o                           <value>     Optional. Path to the output file. Default value: "raw_<npu_blob_file>.blob".
```
Running the application with the empty list of options yields an error message.
