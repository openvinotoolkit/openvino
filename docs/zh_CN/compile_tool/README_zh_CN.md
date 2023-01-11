# 编译工具 {#openvino_inference_engine_tools_compile_tool_README_zh_CN}

编译工具是一个 C++ 应用，使您能够编译模型以在特定设备上进行推理，并将编译后的表示导出到二进制文件。
您可以使用此工具在未连接物理设备的机器上使用支持的 OpenVINO™ 运行时设备编译模型，然后将生成的文件传输到具有可用目标推理设备的任何机器上。如需了解哪些设备支持导入/导出功能，请参阅[功能支持表](../OV_Runtime_UG/supported_plugins/Device_Plugins_zh_CN.md)。

该工具使用相应的 OpenVINO™ 运行时插件为以下目标设备编译网络：英特尔® 神经电脑棒 2（MYRIAD 插件）。

该工具以可执行文件的形式提供，可在 Linux 和 Windows 上运行。它位于 `<INSTALL_DIR>/tools/compile_tool` 目录中。

## 编译工具的工作流程

首先，应用读取命令行参数，将模型加载到 OpenVINO 运行时设备。然后，应用使用已编译模型导出一个 Blob，并将其写入输出文件。

编译工具还支持以下功能：
- 嵌入[布局](../../OV_Runtime_UG/layout_overview.md)和精度转换（有关更多详情，请参阅[优化预处理](../../OV_Runtime_UG/preprocessing_overview.md)）。要使用高级预处理功能编译模型，请参阅[用例 - 将预处理步骤集成并保存到 OpenVINO™ IR 中](../../OV_Runtime_UG/preprocessing_usecase_save.md)。其中说明了如何在已编译 Blob 中进行所有预处理。
- 默认情况下会为 OpenVINO™ 运行时 API 2.0 编译 Blob，或者为带有显式选项 `-ov_api_1_0` 的推理引擎 API 编译 Blob。
- 接受用于自定义编译流程的设备特定选项。

## 运行编译工具

使用 `-h` 选项运行应用会生成以下使用消息：

```sh
./compile_tool -h
OpenVINO Runtime version ......... 2022.1.0
Build ........... custom_changed_compile_tool_183a1adfcd7a001974fe1c5cfa21ec859b70ca2c

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
    -ov_api_1_0                              Optional. Compile model to legacy format for usage in Inference Engine API,
                                             by default compiles to OV 2.0 API

 MYRIAD-specific options:
    -VPU_NUMBER_OF_SHAVES        <value>     Optional. Specifies number of shaves.
                                             Should be set with "VPU_NUMBER_OF_CMX_SLICES".
                                             Overwrites value from config.

    -VPU_NUMBER_OF_CMX_SLICES    <value>     Optional. Specifies number of CMX slices.
                                             Should be set with "VPU_NUMBER_OF_SHAVES".
                                             Overwrites value from config.
    -VPU_TILING_CMX_LIMIT_KB     <value>     Optional. Specifies CMX limit for data tiling.
                                             Value should be equal or greater than -1.
                                             Overwrites value from config.
```

使用空选项列表运行应用会产生一个错误消息。

例如，要从经过训练的网络编译用于在英特尔® 神经电脑棒 2 上推理的 Blob，请运行以下命令：

```sh
./compile_tool -m <path_to_model>/model_name.xml -d MYRIAD
```

### 将已编译的 Blob 文件导入应用

要从生成的文件中将带有网络的 Blob 导入应用中，请使用
`ov::Core::import_model` 方法：

```cpp
ov::Core ie;
std::ifstream file{"model_name.blob"};
ov::CompiledModel compiled_model = ie.import_model(file, "MYRIAD");
```
