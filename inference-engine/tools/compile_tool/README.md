# Compile Tool {#openvino_inference_engine_tools_compile_tool_README}

The Compile tool is a C++ application that enables you to dump a loaded executable network blob.
The tool is delivered as an executable file that can be run on both Linux\* and Windows\*.
The tool is located in the `<INSTALLROOT>/deployment_tools/inference_engine/lib/intel64/` directory on Linux
and `<INSTALL_DIR\deployment_tools\inference_engine\bin\intel64\Release>` on Windows.

The workflow of the Compile tool is as follows:

1. Upon the start, the tool application reads command-line parameters and loads a network to the Inference Engine device.
2. The application exports a blob with the compiled network and writes it to the output file.

## Run the Compile Tool

Running the application with the `-h` option yields the following usage message:

```sh
./compile_tool -h
Inference Engine:
        API version ............ 2.1
        Build .................. custom_vv/compile-tool_8b57af00330063c7f302aaac4d41805de21fc54a
        Description ....... API

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
    -ol                          <value>     Optional. Specifies layout for all input layers of the network.
    -iol                        "<value>"    Optional. Specifies layout for input and output layers by name.
                                             Example: -iol "input:NCHW, output:NHWC".
                                             Notice that quotes are required.
                                             Overwrites layout from il and ol options for specified layers.

 MYRIAD-specific options:
      -VPU_NUMBER_OF_SHAVES      <value>     Optional. Specifies number of shaves.
                                             Should be set with "VPU_NUMBER_OF_CMX_SLICES".
                                             Overwrites value from config.

      -VPU_NUMBER_OF_CMX_SLICES  <value>     Optional. Specifies number of CMX slices.
                                             Should be set with "VPU_NUMBER_OF_SHAVES".
                                             Overwrites value from config.
      -VPU_TILING_CMX_LIMIT_KB   <value>     Optional. Specifies CMX limit for data tiling.
                                             Value should be equal or greater than -1.
                                             Overwrites value from config.

 FPGA-specific options:
      -DLA_ARCH_NAME             <value>     Optional. Specify architecture name used to compile executable network for FPGA device.
```

Running the application with the empty list of options yields an error message.

To dump a blob using a trained network, use the command below:

```sh
./compile_tool -m <path_to_model>/model_name.xml
```

## FPGA Option

You can compile executable network without a connected FPGA device with a loaded DLA bitstream.
To do that, specify the architecture name of the DLA bitstream using the parameter `-DLA_ARCH_NAME`.

## Import and Export Functionality

### Export

To save a blob file from your application, call the `InferenceEngine::ExecutableNetwork::Export()`
method:

```cpp
InferenceEngine::ExecutableNetwork executableNetwork = core.LoadNetwork(network, "MYRIAD", {});
std::ofstream file{"model_name.blob"}
executableNetwork.Export(file);
```

### Import

To import a blob with the network into your application, call the
`InferenceEngine::Core::ImportNetwork` method:

Example:

```cpp
InferenceEngine::Core ie;
std::ifstream file{"model_name.blob"};
InferenceEngine::ExecutableNetwork = ie.ImportNetwork(file, "MYRIAD", {});
```

> **NOTE**: Prior to the import, models must be converted to the Inference Engine format
> (\*.xml + \*.bin) using the [Model Optimizer tool](https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer).
