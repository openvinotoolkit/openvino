# Compile Tool {#openvino_inference_engine_tools_compile_tool_README}


The Compile tool is a C++ application that enables you to dump a loaded 
executable network blob. The tool is delivered as an executable file that can be
run on both Linux\* and Windows\*. The tool is located in the `<INSTALLROOT>/deployment_tools/inference_engine/lib/intel64/` directory on 
Linux and `<INSTALL_DIR\deployment_tools\inference_engine\bin\intel64\Release>`
on Windows. 

The workflow of the Compile tool is as follows:
1. Upon the start, the tool application reads command-line parameters and loads a network to the 
Inference Engine device.
2. The application exports a blob with the compiled network and writes it to the output file.

## Run the Compile Tool

Running the application with the `-h` option yields the following usage message:

```sh
./compile_tool -h
Inference Engine:
        API version ............ <version>
        Build .................. <build>

compile_tool [OPTIONS]
[OPTIONS]:
    -h                                       Optional. Print the usage message.
    -m                           <value>     Required. Path to the XML model.
    -d                           <value>     Required. Target device name.
    -o                           <value>     Optional. Path to the output file. Default value: "<model_xml_file>.blob".
    -c                           <value>     Optional. Path to the configuration file. Default value: "config".
    -ip                          <value>     Optional. Specifies precision for all input layers of the network. Supported values: FP32, FP16, U8. Default value: FP16.
    -op                          <value>     Optional. Specifies precision for all output layers of the network. Supported values: FP32, FP16, U8. Default value: FP16.
    -iop                        "<value>"    Optional. Specifies precision for input and output layers by name.
                                             By default, all inputs and outputs have the FP16 precision.
                                             Available precisions: FP32, FP16, U8.
                                             Example: -iop "input:FP16, output:FP16".
                                             Notice that quotes are required.
                                             Overwrites precision from ip and op options for specified layers.

    VPU options:
        -VPU_MYRIAD_PLATFORM      <value>     Optional. Specifies Movidius platform. Supported values: VPU_MYRIAD_2450, VPU_MYRIAD_2480. Overwrites value from config.
                                                 This option must be used in order to compile blob without a connected Myriad device.
        -VPU_NUMBER_OF_SHAVES     <value>     Optional. Specifies number of shaves. Should be set with "VPU_NUMBER_OF_CMX_SLICES". Overwrites value from config.
        -VPU_NUMBER_OF_CMX_SLICES <value>     Optional. Specifies number of CMX slices. Should be set with "VPU_NUMBER_OF_SHAVES". Overwrites value from config.

    DLA options:
        -DLA_ARCH_NAME            <value>     Optional. Specify architecture name used to compile executable network for FPGA device.
```

Running the application with the empty list of options yields an error message.

To dump a blob using a trained Faster R-CNN network, use the command below:

```sh
./compile_tool -m <path_to_model>/model_name.xml
```

## MYRIAD Platform Option

You can dump a blob without a connected MYRIAD device.
To do that, specify the type of an Intel® Movidius™ platform using the `-VPU_MYRIAD_PLATFORM` parameter.

Supported values: `VPU_MYRIAD_2450`, `VPU_MYRIAD_2480`.

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
