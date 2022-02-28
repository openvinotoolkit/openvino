# Benchmark C++ Tool

This topic demonstrates how to use the Benchmark C++ Tool to estimate deep learning inference performance on supported devices. Performance can be measured for two inference modes: synchronous (latency-oriented) and asynchronous (throughput-oriented).

> **NOTE**: This topic describes usage of C++ implementation of the Benchmark Tool. For the Python* implementation, refer to [Benchmark Python* Tool](../../benchmark_tool/README.md).


## How It Works

Upon start-up, the application reads command-line parameters and loads a network and images/binary files to the Inference Engine plugin, which is chosen depending on a specified device. The number of infer requests and execution approach depend on the mode defined with the `-api` command-line parameter.

> **NOTE**: By default, Inference Engine samples, tools and demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the sample or demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model to Intermediate Representation (IR)](../../../docs/MO_DG/prepare_model/convert_model/Converting_Model.md).

If you run the application in the synchronous mode, it creates one infer request and executes the `Infer` method.
If you run the application in the asynchronous mode, it creates as many infer requests as specified in the `-nireq` command-line parameter and executes the `StartAsync` method for each of them. If `-nireq` is not set, the application will use the default value for specified device.

A number of execution steps is defined by one of the following parameters:
* Number of iterations specified with the `-niter` command-line argument
* Time duration specified with the `-t` command-line argument
* Both of them (execution will continue until both conditions are met)
* Predefined duration if `-niter` and `-t` are not specified. Predefined duration value depends on a device.

During the execution, the application collects latency for each executed infer request.

Reported latency value is calculated as a median value of all collected latencies. Reported throughput value is reported
in frames per second (FPS) and calculated as a derivative from:
* Reported latency in the Sync mode
* The total execution time in the Async mode

Throughput value also depends on batch size.

The application also collects per-layer Performance Measurement (PM) counters for each executed infer request if you
enable statistics dumping by setting the `-report_type` parameter to one of the possible values:
* `no_counters` report includes configuration options specified, resulting FPS and latency.
* `average_counters` report extends the `no_counters` report and additionally includes average PM counters values for each layer from the network.
* `detailed_counters` report extends the `average_counters` report and additionally includes per-layer PM counters and latency for each executed infer request.

Depending on the type, the report is stored to `benchmark_no_counters_report.csv`, `benchmark_average_counters_report.csv`,
or `benchmark_detailed_counters_report.csv` file located in the path specified in `-report_folder`.

The application also saves executable graph information serialized to an XML file if you specify a path to it with the
`-exec_graph_path` parameter.

## Building
This tools can be built as part of OpenVINO, during standard building process. More information about building OpenVINO can be found here[Build OpenVINO Inference Engine](https://github.com/openvinotoolkit/openvino/wiki/BuildingCode)

## Run the Tool

Note that the benchmark_app usually produces optimal performance for any device out of the box.

**So in most cases you don't need to play the app options explicitly and the plain device name is enough**, for example, for CPU:
```sh
./benchmark_app -m <model> -i <input> -d CPU
```

But it is still may be sub-optimal for some cases, especially for very small networks. More details can read in [Performance Optimization Guide](../../../docs/optimization_guide/dldt_optimization_guide.md).

As explained in the  [Performance Optimization Guide](../../../docs/optimization_guide/dldt_optimization_guide.md) section, for all devices, including new [MULTI device](../../../docs/OV_Runtime_UG/supported_plugins/MULTI.md) it is preferable to use the FP16 IR for the model.
Also if latency of the CPU inference on the multi-socket machines is of concern, please refer to the same
[Performance Optimization Guide](../../../docs/optimization_guide/dldt_optimization_guide.md).

Running the application with the `-h` option yields the following usage message:
```
./benchmark_app -h
InferenceEngine:
        API version ............ <version>
        Build .................. <number>
[ INFO ] Parsing input parameters

benchmark_app [OPTION]
Options:

    -h, --help                  Print a usage message
    -m "<path>"                 Required. Path to an .xml/.onnx/.prototxt file with a trained model or to a .blob files with a trained compiled model.
    -i "<path>"                 Optional. Path to a folder with images and/or binaries or to specific image or binary file.
    -d "<device>"               Optional. Specify a target device to infer on (the list of available devices is shown below). Default value is CPU.
                                Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin.
                                Use "-d MULTI:<comma-separated_devices_list>" format to specify MULTI plugin.
    The application looks for a suitable plugin for the specified device.
    -l "<absolute_path>"        Required for CPU custom layers. Absolute path to a shared library with the kernels implementations.
          Or
    -c "<absolute_path>"        Required for GPU custom kernels. Absolute path to an .xml file with the kernels description.
    -api "<sync/async>"         Optional. Enable Sync/Async API. Default value is "async".
    -niter "<integer>"          Optional. Number of iterations. If not specified, the number of iterations is calculated depending on a device.
    -nireq "<integer>"          Optional. Number of infer requests. Default value is determined automatically for a device.
    -b "<integer>"              Optional. Batch size value. If not specified, the batch size value is determined from Intermediate Representation.
    -stream_output              Optional. Print progress as a plain text. When specified, an interactive progress bar is replaced with a multiline output.
    -t                          Optional. Time, in seconds, to execute topology.
    -progress                   Optional. Show progress bar (can affect performance measurement). Default values is "false".
    -shape                      Optional. Set shape for input. For example, "input1[1,3,224,224],input2[1,4]" or "[1,3,224,224]" in case of one input size.
    -layout                     Optional. Prompts how network layouts should be treated by application. For example, "input1[NCHW],input2[NC]" or "[NCHW]" in case of one input size.
    -cache_dir "<path>"         Optional. Enables caching of loaded models to specified directory.
    -load_from_file             Optional. Loads model from file directly without ReadNetwork.

  CPU-specific performance options:
    -nstreams "<integer>"       Optional. Number of streams to use for inference on the CPU, GPU or MYRIAD devices
                                (for HETERO and MULTI device cases use format <device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>).
                                Default value is determined automatically for a device.
                                Please note that although the automatic selection usually provides a reasonable performance,
                                it still may be non-optimal for some cases, especially for very small networks.
                                Also, using nstreams>1 is inherently throughput-oriented option, while for the best-latency
                                estimations the number of streams should be set to 1.
    -nthreads "<integer>"       Optional. Number of threads to use for inference on the CPU (including HETERO and MULTI cases).
    -enforcebf16="<true/false>" Optional. By default floating point operations execution in bfloat16 precision are enforced if supported by platform.
    -pin "YES"/"HYBRID_AWARE"/"NUMA"/"NO"
                                Optional. Explicit inference threads binding options (leave empty to let the OpenVINO to make a choice):
					            enabling threads->cores pinning ("YES", which is already default for a conventional CPU),  
			                    letting the runtime to decide on the threads->different core types ("HYBRID_AWARE", which is default on the hybrid CPUs)
			                    threads->(NUMA)nodes ("NUMA") or 
			      	            completely disable ("NO") CPU inference threads pinning.
    -ip "U8"/"FP16"/"FP32"      Optional. Specifies precision for all input layers of the network.
    -op "U8"/"FP16"/"FP32"      Optional. Specifies precision for all output layers of the network.
    -iop                        Optional. Specifies precision for input and output layers by name. Example: -iop "input:FP16, output:FP16". Notice that quotes are required. Overwrites precision from ip and op options for specified layers.

  Statistics dumping options:
    -report_type "<type>"       Optional. Enable collecting statistics report. "no_counters" report contains configuration options specified, resulting FPS and latency. "average_counters" report extends "no_counters" report and additionally includes average PM counters values for each layer from the network. "detailed_counters" report extends "average_counters" report and additionally includes per-layer PM counters and latency for each executed infer request.
    -report_folder              Optional. Path to a folder where statistics report is stored.
    -exec_graph_path            Optional. Path to a file where to store executable graph information serialized.
    -pc                         Optional. Report performance counters.
    -dump_config                Optional. Path to XML/YAML/JSON file to dump IE parameters, which were set by application.
    -load_config                Optional. Path to XML/YAML/JSON file to load custom IE parameters. Please note, command line parameters have higher priority then parameters from configuration file.
```

Running the application with the empty list of options yields the usage message given above and an error message.

Application supports topologies with one or more inputs. If a topology is not data-sensitive, you can skip the input parameter. In this case, inputs are filled with random values.
If a model has only image input(s), please provide a folder with images or a path to an image as input.
If a model has some specific input(s) (not images), please prepare a binary file(s) that is filled with data of appropriate precision and provide a path to them as input.
If a model has mixed input types, input folder should contain all required files. Image inputs are filled with image files one by one. Binary inputs are filled with binary inputs one by one.

To run the tool, you can use [public](@ref omz_models_group_public) or [Intel's](@ref omz_models_group_intel) pre-trained models from the Open Model Zoo. The models can be downloaded using the [Model Downloader](@ref omz_tools_downloader).

> **NOTE**: Before running the tool with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
>
> The sample accepts models in ONNX format (.onnx) that do not require preprocessing.

## Examples of Running the Tool

This section provides step-by-step instructions on how to run the Benchmark Tool with the `googlenet-v1` public model on CPU or GPU devices. As an input, the `car.png` file from the `<INSTALL_DIR>/samples/scripts/` directory is used.

> **NOTE**: The Internet access is required to execute the following steps successfully. If you have access to the Internet through the proxy server only, please make sure that it is configured in your OS environment.

1. Download the model. Install openvino-dev package into Python virtual environment from PyPi and run omz_downloader:
   ```sh
   omz_downloader --name googlenet-v1 -o <models_dir>
2. Convert the model to the Inference Engine IR format. Run the Model Optimizer using the `mo` command with the path to the model, model format (which must be FP32 for CPU and FPG) and output directory to generate the IR files:
   ```sh
   mo --input_model <models_dir>/public/googlenet-v1/googlenet-v1.caffemodel --data_type FP32 --output_dir <ir_dir>
   ```
3. Run the tool with specifying the `<INSTALL_DIR>/samples/scripts/car.png` file as an input image, the IR of the `googlenet-v1` model and a device to perform inference on. The following commands demonstrate running the Benchmark Tool in the asynchronous mode on CPU and GPU devices:

   * On CPU:
   ```sh
   ./benchmark_app -m <ir_dir>/googlenet-v1.xml -i <INSTALL_DIR>/samples/scripts/car.png  -d CPU -api async --progress true
   ```
   * On GPU:
   ```sh
   ./benchmark_app -m <ir_dir>/googlenet-v1.xml -i <INSTALL_DIR>/samples/scripts/car.png -d GPU -api async --progress true
   ```

The application outputs the number of executed iterations, total duration of execution, latency, and throughput.
Additionally, if you set the `-report_type` parameter, the application outputs statistics report. If you set the `-pc` parameter, the application outputs performance counters. If you set `-exec_graph_path`, the application reports executable graph information serialized. All measurements including per-layer PM counters are reported in milliseconds.

Below are fragments of sample output:

   ```
   [Step 10/11] Measuring performance (Start inference asynchronously, 4 inference requests using 4 streams for CPU, limits: 60000 ms duration)
   [ INFO ] BENCHMARK IS IN INFERENCE ONLY MODE.
   [ INFO ] Input blobs will be filled once before performance measurements.
   [ INFO ] First inference took 26.26 ms
   Progress: [................... ]  99% done

   [Step 11/11] Dumping statistics report
   [ INFO ] Count:      6640 iterations
   [ INFO ] Duration:   60039.70 ms
   [ INFO ] Latency:
   [ INFO ]        Median:  35.36 ms
   [ INFO ]        Avg:    36.12 ms
   [ INFO ]        Min:    18.55 ms
   [ INFO ]        Max:    88.96 ms
   [ INFO ] Throughput: 110.59 FPS
   ```

## See Also
* [Model Optimizer](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
* [Model Downloader](@ref omz_tools_downloader)
