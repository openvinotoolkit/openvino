# Cross Check Tool {#openvino_inference_engine_tools_cross_check_tool_README}

Cross Check Tool is a console application that enables comparing accuracy and performance metrics for two successive 
model inferences that are performed on two different supported Intel&reg; devices or with different precisions.
The Cross Check Tool can compare the metrics per layer or all over the model.

## Running the Cross Check Tool

Cross Check Tool is distributed as a Python module and there is no need to build it. To run the Cross Check Tool, 
execute the `cross_check_tool.py` file with necessary parameters. Please note that the Inference Engine assumes that weights 
are in the same folder as the `.xml` file.

You can get the list of all available options using the `-h` option:

```sh
$python3 cross_check_tool.py -h

Cross Check Tool is a console application that enables comparing accuracy and
provides performance metrics

optional arguments:
  -h, --help            show this help message and exit

Model specific arguments:
  --input INPUT, -i INPUT
                        Path to an input image file or multi-input file to
                        infer. Generates input(s) from normal distribution if
                        empty
  --batch BATCH, -b BATCH
                        Overrides batch size. Default is inherited from model
  --model MODEL, -m MODEL
                        Path to an .xml file that represents the first IR of
                        the trained model to infer.
  --reference_model REFERENCE_MODEL, -ref_m REFERENCE_MODEL
                        Path to an .xml file that represents the second IR in
                        different precision to compare the metrics.
  --layers LAYERS, -layers LAYERS
                        Defines layers to check. Options: all, None - for
                        output layers check, list of comma-separated layer
                        names to check. Default value is None.
  --mapping MAPPING, -map MAPPING
                        Model Optimizer provided mapping for --model/-m
  --reference_mapping REFERENCE_MAPPING, -ref_map REFERENCE_MAPPING
                        Model Optimizer provided mapping for
                        --reference_model/-ref_model
  --num_of_iterations NUM_OF_ITERATIONS, -ni NUM_OF_ITERATIONS
                        Number of iterations to collect all over the net
                        performance

Plugin specific arguments:
  --plugin_path PLUGIN_PATH, -pp PLUGIN_PATH
                        Path to a plugin folder.
  --device DEVICE, -d DEVICE
                        The first target device to infer the model specified
                        with the -m or --model option. CPU, GPU, HDDL or
                        MYRIAD are acceptable.
  --config CONFIG, -conf CONFIG
                        Path to config file for -d or -device device plugin
  --reference_device REFERENCE_DEVICE, -ref_d REFERENCE_DEVICE
                        The second target device to infer the model and
                        compare the metrics. CPU, GPU, HDDL or MYRIAD are
                        acceptable.
  --reference_config REFERENCE_CONFIG, -ref_conf REFERENCE_CONFIG
                        Path to config file for -ref_d or -reference_device
                        device plugin
  -l L                  Required for MKLDNN (CPU)-targeted custom layers.
                        Comma separated paths to a shared libraries with the
                        kernels implementation.

CCT mode arguments:
  --dump                Enables blobs statistics dumping
  --load LOAD           Path to a file to load blobs from

```
### Examples

1. To check per-layer accuracy and performance of inference in FP32 precision on the CPU against the GPU, run:
   ```sh
   $python3 cross_check_tool.py -i <path_to_input_image_or_multi_input_file> \
                 -m <path_to_FP32_xml>                            \
                 -d GPU                                           \
                 -ref_d CPU                                       \
                 --layers all
   ```
   
   The output looks as follows:
   ```sh
   [ INFO ] Cross check with one IR was enabled
   [ INFO ] GPU:FP32 vs CPU:FP32
   [ INFO ] The same IR on both devices: <path_to_IR> 
   [ INFO ] Statistics will be dumped for X layers: <layer_1_name>, <layer_2_name>, ... , <layer_X_name>
   [ INFO ] Layer <layer_1_name> statistics 
        Max absolute difference : 1.15204E-03
        Min absolute difference : 0.0
        Max relative difference : 1.15204E+17
        Min relative difference : 0.0
        Min reference value : -1.69513E+03
        Min absolute reference value : 2.71080E-06
        Max reference value : 1.17132E+03
        Max absolute reference value : 1.69513E+03
        Min actual value : -1.69513E+03
        Min absolute actual value : 8.66465E-05
        Max actual value : 1.17132E+03
        Max absolute actual value : 1.69513E+03
          Device:           -d GPU       -ref_d CPU
          Status:    OPTIMIZED_OUT    OPTIMIZED_OUT
          Layer type:      Convolution      Convolution
        Real time, microsec:     0              120
          Number of NAN:         0                0
          Number of INF:         0                0
          Number of ZERO:        0                0
    ...
   <list_of_layer_statistics>
   ...
   
   [ INFO ] Overall max absolute difference = 0.00115203857421875
   [ INFO ] Overall min absolute difference = 0.0
   [ INFO ] Overall max relative difference = 1.1520386483093504e+17
   [ INFO ] Overall min relative difference = 0.0
   [ INFO ] Execution successful
   ```

2. To check the overall accuracy and performance of inference on the CPU in FP32 precision against the 
   Intel&reg; Movidius&trade; Myriad&trade; device in FP16 precision, run:
   ```sh
   $python3 cross_check_tool.py    -i <path_to_input_image_or_multi_input_file> \
                   -m <path_to_FP16_xml>                        \
                   -d MYRIAD                                    \
                   -ref_m <path_to_FP32_xml>                    \
                   -ref_d CPU                                   
   ```
   
   The output looks as follows:
   ```sh
   [ INFO ] Cross check with two IRs was enabled
   [ INFO ] GPU:FP16 vs CPU:FP32
   [ INFO ] IR for MYRIAD : <path_to_FP16_xml>
   [ INFO ] IR for CPU : <path_to_FP32_xml>
   [ INFO ] Statistics will be dumped for 1 layer: <output_layer_name(s)>
   [ INFO ] Layer <output_layer_name> statistics 
        Max absolute difference : 2.32944E-02
        Min absolute difference : 3.63002E-13
        Max relative difference : 6.41717E+10
        Min relative difference : 1.0
        Min reference value : 3.63002E-13
        Min absolute reference value : 3.63002E-13
        Max reference value : 7.38138E-01
        Max absolute reference value : 7.38138E-01
        Min actual value : 0.0
        Min absolute actual value : 0.0
        Max actual value : 7.14844E-01
        Max absolute actual value : 7.14844E-01
          Device:        -d MYRIAD       -ref_d CPU
          Status:    OPTIMIZED_OUT    OPTIMIZED_OUT
          Layer type:          Reshape          Reshape
        Real time, microsec:      0                0
          Number of NAN:          0                0
          Number of INF:          0                0
          Number of ZERO:         0                0
   ----------------------------------------------------------------------
     Overall performance, microseconds:      2.79943E+05      6.24670E+04
   ----------------------------------------------------------------------
   [ INFO ] Overall max absolute difference = 0.023294448852539062
   [ INFO ] Overall min absolute difference = 3.630019191052519e-13
   [ INFO ] Overall max relative difference = 64171696128.0
   [ INFO ] Overall min relative difference = 1.0
   [ INFO ] Execution successful
   ```

3. To dump layer statistics from a specific list of layers, run:
   ```sh
   $python3 cross_check_tool.py    -i <path_to_input_image_or_multi_input_file> \
                   -m <path_to_FP16_xml>                        \
                   -d MYRIAD                                    \
                   --dump                                       \
                   --layers <comma_separated_list_of_layers>
   ```
   
   The output looks as follows:
   ```sh
   [ INFO ] Dump mode was enabled
   [ INFO ] <layer_1_name> layer processing
   ...
   [ INFO ] <layer_X_name> layer processing
   [ INFO ] Dump file path: <path_where_dump_will_be_saved>
   [ INFO ] Execution successful
   ```
   
   If you do not provide the `-i` key, the Cross Check Tool generates an input from normal distributed noise and saves 
   it in a multi-input file format with the filename `<path_to_xml>_input_layers_dump.txt` in the same folder as the Intermediate Representation (IR).

4. To check the overall accuracy and performance of inference on the CPU in FP32 precision against dumped results, run:
   ```sh
   $python3 cross_check_tool.py    -i <path_to_input_image_or_multi_input_file> \
                   -m <path_to_FP32_xml>                        \
                   -d CPU                                       \
                   --load <path_to_dump>                        \
                   --layers all
   ```
   
   The output looks as follows:
   ```sh
   [ INFO ] Load mode was enabled
   [ INFO ] IR for CPU : <path_to_FP32_xml>
   [ INFO ] Loading blob from /localdisk/models/FP16/icv_squeezenet_v1.0.xml_GPU_dump.npz
   [ INFO ] Statistics will be dumped for X layers:  <layer_1_name>, <layer_2_name>, ... , <layer_X_name>
   [ INFO ] Layer <layer_1_name> statistics
        Max absolute difference : 0.0
        Min absolute difference : 0.0
        Max relative difference : 0.0
        Min relative difference : 0.0
        Min reference value : 0.0
        Min absolute reference value : 0.0
        Max reference value : 7.14844E-01
        Max absolute reference value : 7.14844E-01
        Min actual value : 0.0
        Min absolute actual value : 0.0
        Max actual value : 7.14844E-01
        Max absolute actual value : 7.14844E-01
          Device:           -d CPU        -load GPU
          Status:    OPTIMIZED_OUT    OPTIMIZED_OUT
          Layer type:          Reshape          Reshape
        Real time, microsec:      0                0
          Number of NAN:          0                0
          Number of INF:          0                0
          Number of ZERO:        609              699
   
   ...
   <list_of_layer_statistics>
   ...
               
   [ INFO ] Overall max absolute difference = 0.0
   [ INFO ] Overall min absolute difference = 0.0
   [ INFO ] Overall max relative difference = 0.0
   [ INFO ] Overall min relative difference = 0.0
   [ INFO ] Execution successful
   ```
   
### Multi-input and dump file format

Multi-input and dump file is a numpy compressed `.npz` file with hierarchy:

```sh
{
  ‘layer_name’: {
    ‘blob’: np.array([…])
    ‘pc’: {
      ‘device’: ‘device_name’,
      ‘real_time’: int_real_time_in_microseconds_from_plugin,
      ‘exec_type’: ‘exec_type_from_plugin’,
      ‘layer_type’: ‘layer_type_from_plugin’,
      ‘status’: ‘status_from_plugin’
    }
  },
  ‘another_layer_name’: {
    ‘blob’: np.array([…])
    ‘pc’: {
      ‘device’: ‘device_name’,
      ‘real_time’: int_real_time_in_microseconds_from_plugin,
      ‘exec_type’: ‘exec_type_from_plugin’,
      ‘layer_type’: ‘layer_type_from_plugin’,
      ‘status’: ‘status_from_plugin’
    }
  },
  ...
}
```

### Configuration file

There is an option to pass configuration file to plugin by providing 
`--config` and/or `--reference_config` keys.

Configuration file is a text file with content of pairs of keys and values.

Structure of configuration file:

```sh
KEY VALUE
ANOTHER_KEY ANOTHER_VALUE,VALUE_1
```
