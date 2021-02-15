Cross Check Tool {#openvino_docs_IE_DG_Cross_Check_Tool}
================

Cross Check Tool is a console application that enables comparing accuracy and performance metrics for two successive
model inferences that are performed
on two different supported Intel&reg; devices or with different precisions.
The Cross Check Tool can compare metrics per layer or all over the model.

On Linux* OS, before running the Cross Check Tool binary, make sure your application can find the
Deep Learning Inference Engine libraries.
Navigate to the `<INSTALL_DIR>/deployment_tools/inference_engine/bin` folder and run the `setvars.sh` script to
set all necessary environment variables:

```sh
source setvars.sh
```

## Running the Cross Check Tool

Cross Check Tool is distributed as a binary file and there is no need to build it. To run the Cross Check Tool,
execute the tool's binary file with necessary parameters. Please note that the Inference Engine assumes that weights
are in the same folder as the _.xml_ file.

You can get the list of all available options using the -h option:

```sh
$./cross_check_tool -h
InferenceEngine:
  API version ............ 1.0
  Build .................. ###
[ INFO ] Parsing input parameters

./cross_check_tool [OPTION]
Options:

    -h                     Prints a usage message.
    -i "<path>"            Optional. Path to an input image file or multi-input file to infer. Generates input(s) from normal distribution if empty
    -m "<path>"            Required. Path to an .xml file that represents the first IR of the trained model to infer.
      -l "<absolute_path>" Required for MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the kernels implementation.
          Or
      -c "<absolute_path>" Required for clDNN (GPU)-targeted custom kernels. Absolute path to the xml file with the kernels description.
    -conf "<path>"         Optional. Path to config file for -d device plugin
    -ref_conf "<path>"     Optional. Path to config file for -ref_d device plugin
    -pp "<path>"           Optional. Path to a plugin folder.
    -d "<device>"          Required. The first target device to infer the model specified with the -m option. CPU, GPU, HDDL or MYRIAD is acceptable.
    -ref_m "<path>"        Optional. Path to an .xml file that represents the second IR in different precision to compare the metrics.
    -ref_d "<device>"      Required. The second target device to infer the model and compare the metrics. CPU, GPU, HDDL or MYRIAD is acceptable.
    -layers "<options>"    Defines layers to check. Options: all, None - for output layers check, list of comma-separated layer names to check. Default value is None.
    -eps "<float>"         Optional. Threshold for filtering out those blob statistics that do not statify the condition: max_abs_diff < eps.
    -dump                  Enables blobs statistics dumping
    -load "<path>"         Path to a file to load blobs from
```
### Examples

1. To check per-layer accuracy and performance of inference in FP32 precision on the CPU against the GPU, run:
```sh
./cross_check_tool -i <path_to_input_image_or_multi_input_file> \
                   -m <path_to_FP32_xml>    \
                   -d CPU                   \
                   -ref_d GPU               \
                   -layers all
```
The output looks as follows:
```
InferenceEngine:
  API version ............ 1.0
  Build .................. ###
[ INFO ] Parsing input parameters
    The same IR on both devices: <path_to_IR>

[ INFO ] No extensions provided

  API version ............ 1.0
  Build .................. lnx_20180510
  Description ....... MKLDNNPlugin

  API version ............ 0.1
  Build .................. ci-main-03659
  Description ....... clDNNPlugin
[ INFO ] Inputs detected: Placeholder
[ INFO ] Statistics will be dumped for X layers: <layer_1_name>, <layer_2_name>, ... , <layer_X_name>
[ INFO ] Layer <layer_1_name> statistics
    Max absolute difference: 1.52588e-05
    Min absolute difference: 0
    Max relative difference: 0.000288028%
    Min relative difference: 0%
                  Blob size: 1000

                    Devices:            CPU_FP32            GPU_FP32
                     Status:            EXECUTED            EXECUTED
                 Layer type:             Reshape             Reshape
        Real time, microsec:                  20                 154
             Execution type:             unknown                 GPU
              Number of NAN:                   0                   0
              Number of INF:                   0                   0
             Number of ZERO:                   0                   0
...
<list_of_layer_statistics>
...

[ INFO ] Overall max absolute difference 2.81334e-05 was reached by <layer_name> layer
[ INFO ] Overall min absolute difference 0 was reached by <layer_name> layer
[ INFO ] Overall max relative difference 0.744893% was reached by <layer_name> layer
[ INFO ] Overall min relative difference -2.47948% was reached by <layer_name> layer
[ INFO ] Execution successful
```

2. To check the overall accuracy and performance of inference on the CPU in FP32 precision against the
Intel&reg; Movidius&trade; Myriad&trade; device in FP16 precision, run:
```sh
./cross_check_tool -i <path_to_input_image_or_multi_input_file> \
                   -m <path_to_FP16_xml>    \
                   -ref_d CPU               \
                   -ref_m <path_to_FP32_xml>\
                   -d MYRIAD                \
```
The output looks as follows:
```
InferenceEngine:
  API version ............ 1.0
  Build .................. ###

[ INFO ] Parsing input parameters
[ INFO ] MYRIAD vs CPU
    IR for MYRIAD : <path_to_FP16_xml>
    IR for CPU : <path_to_FP32_xml>

[ INFO ] No extensions provided
[ INFO ] Loading plugins

  API version ............ 0.1
  Build .................. ###
  Description ....... myriadPlugin


  API version ............ 1.0
  Build .................. ###
  Description ....... MKLDNNPlugin

[ INFO ] Inputs detected: <list_of_input_layers>
[ INFO ] Statistics will be dumped for 1 layers: <output_layer_name(s)>
[ INFO ] Layer <output_layer_name> statistics
    Max absolute difference: 0.003889
    Min absolute difference: 2.49778e-12
    Max relative difference: 290.98%
    Min relative difference: 0.0327804%
                    Devices:         MYRIAD_FP16            CPU_FP32
        Real time, microsec:        69213.978946         4149.904940
[ INFO ] Execution successful
```

3. To dump layer statistics from specific list of layers, run:
```sh
./cross_check_tool -i <path_to_input_image_or_multi_input_file> \
                   -m <path_to_FP16_xml>                        \
                   -d MYRIAD                                    \
                   -dump                                        \
                   -layers <comma_separated_list_of_layers>
```
The output looks as follows:
```
InferenceEngine:
  API version ............ 1.0
  Build .................. ###
[ INFO ] Blob and statistics dumping enabled
[ INFO ] No extensions provided

  API version ............ 0.1
  Build .................. custom_releases/cvsdk-2018-r2_e28ec0278fb749d6b999c688a8e90a8a25c0f2b5
  Description ....... myriadPlugin

[ INFO ] Inputs detected: <list_of_input_layers>
[ INFO ] Statistics will be dumped for X layers: <comma_separated_list_of_layers>
[ INFO ] Dump path: <path_where_dump_will_be_saved>
[ INFO ] <layer_1_name> layer processing
...
[ INFO ] <layer_X_name> layer processing
[ INFO ] Execution successful
```
If you do not provide the `-i` key, the Cross Check Tool generates an input from normal distributed noise and saves
it in a multi-input file format with the filename `<path_to_xml>_input_layers_dump.txt` in the same folder as the IR.
4. To check the overall accuracy and performance of inference on the CPU in FP32 precision against dumped results, run:
```sh
./cross_check_tool -i <path_to_input_image_or_multi_input_file> \
                   -m <path_to_FP32_xml>                        \
                   -d CPU                                       \
                   -load <path_to_dump>                         \
                   -layers all
```
The output looks as follows:
```
InferenceEngine:
  API version ............ 1.0
  Build .................. ###
[ INFO ] Blob and statistics loading enabled. File /localdisk/models/FP16/icv_squeezenet_v1.0_MYRIAD_FP16_dump.txt
    The same IR on both devices: <path_to_FP32_xml>   

[ INFO ] No extensions provided

  API version ............ 0.1
  Build .................. ###
  Description ....... myriadPlugin

[ INFO ] Inputs detected: <list_of_input_layers>
[ INFO ] Statistics will be dumped for X layers: <layer_1_name>, <layer_2_name>, ... , <layer_X_name>
[ INFO ] <layer_1_name> layer processing
[ INFO ] Layer <layer_1_name> statistics
    Max absolute difference: 0
    Min absolute difference: 0
    Max relative difference: 0%
    Min relative difference: 0%
                  Blob size: 1000

                    Devices:         MYRIAD_FP16  MYRIAD_FP16_loaded
                     Status:            EXECUTED            EXECUTED
                 Layer type:             SoftMax             SoftMax
        Real time, microsec:                  43                  43
             Execution type:             SoftMax             SoftMax
              Number of NAN:                   0                   0
              Number of INF:                   0                   0
             Number of ZERO:                   0                   0
...
<list_of_layer_statistics>
...
[ INFO ] Overall max absolute difference 0
[ INFO ] Overall min absolute difference 0 was reached by <layer_1_name> layer
[ INFO ] Overall max relative difference 0%
[ INFO ] Overall min relative difference 0% was reached by <layer_1_name> layer
[ INFO ] Execution successful
```

### Multi-input and dump file experimental format

Text file contains description of each layer in structure like this:
* 1<sup>st</sup> line is layer name (required)
* 2<sup>nd</sup> line is shape like "(1,224,224,3)" (required)
* 3<sup>rd</sup> line is a device and precision information like "CPU_FP32" (optional for multi-input file)
* 4<sup>th</sup> line is execution status Options are: EXECUTED, OPTIMIZED_OUT (optional for multi-input file)
* 5<sup>th</sup> line is type of layer (optional for multi-input file)
* 6<sup>th</sup> line is execution time in microseconds (optional for multi-input file)
* 7<sup>th</sup> line is type of execution (optional for multi-input file)
* 8<sup>th</sup> line is word "CONTENT" which means that the next line or lines are consisted of blob elements   
* Next line or lines are for blob elements. They may be separated with one or several spaces, tabs and new lines.


#### Multi-input file example

```
Input_1
(1,10)
CONTENT
0 0.000628471375 0.00185108185
0.000580787659
0.00137138367
0.000561237335 0.0040473938 0 0 0
Input_2
(1,8)
CONTENT
0 0 0.00194549561 0.0017490387 7.73072243e-05 0.000135779381 0.000186920166 0 7.52806664e-05
```

#### Dump file example

```
Softmax
(1,10)
MYRIAD_FP16
EXECUTED
SoftMax
43
SoftMax
CONTENT
7.44462013e-05
0
0.000810623169
0.000361680984
0
9.14335251e-05
0
0
8.15987587e-05
0
```


### Configuration file

There is an option to pass configuration file to plugin by providing
`-conf` and/or `--ref_conf` keys.

Configuration file is a text file with content of pairs of keys and values.

Structure of configuration file:

```sh
KEY VALUE
ANOTHER_KEY ANOTHER_VALUE,VALUE_1
```
