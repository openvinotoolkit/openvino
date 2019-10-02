# vpu_profile tool

This topic demonstrates how to run the `vpu_profile` tool application, which intended to get per layer or per stage
performance statistics for vpu plugins of Inference Engine by configuration options.

## How It Works

Upon the start-up, the tool application reads command line parameters and loads a network to the Inference Engine plugin, 
performs inference and prints performance statistics, according to provided by plugin performance counts to standard output.
Only Bitmap(.bmp) images are supported as inputs, need to convert from other formats if your inputs are not Bitmap.

## Running

Running the application with the "-h" option yields the following usage message:

```sh
$./vpu_profile -h
Inference Engine:
        API version ............ <version>
        Build .................. <build>
vpu_profile [OPTIONS]
[OPTIONS]:
	-h          	         	Print a help(this) message.
	-model      	 <value> 	Path to xml model.
	-inputs_dir 	 <value> 	Path to folder with images, only bitmap(.bmp) supported. Default: ".".
	-config     	 <value> 	Path to the configuration file. Default value: "config".
	-iterations 	 <value> 	Specifies number of iterations. Default value: 16.
	-plugin     	 <value> 	Specifies plugin. Supported values: myriad, hddl.
	            	         	Default value: "myriad".
	-report     	 <value> 	Specifies report type. Supported values: per_layer, per_stage.
	            	         	Overrides value in configuration file if provided. Default value: "per_layer"
```

Running the application with the empty list of options yields an error message.

You can use the following command to simply execute network:

```sh
$./vpu_profile -model <path_to_model>/model_name.xml
```
> **NOTE**: Models should be first converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer).

## Plugin Option

You have to select between Myriad and HDDL plugin manually, by default vpu_profile will try to use myriad plugin
If you need to run HDDL, need to set it explicitly

```sh
$./vpu_profile -model <path_to_model>/model_name.xml -plugin hddl
```

## Iterations Option

Sets amount of Infer requests to be executed, will affect overall inference time, performance counts will be reported for last iteration

```sh
$./vpu_profile -model <path_to_model>/model_name.xml -iterations 30
```

## Configuration file

Set configuration keys for plugin, file contents converted to a config map used for LoadNetwork call
Format of file - each line represent config key and it's value like:
"<CONFIG_KEY> <CONFIG_VALUE>"
Below example shows how to set performance reporting

```sh
$echo "VPU_PERF_REPORT_MODE VPU_PER_STAGE" > config.txt
```
```sh
$./vpu_profile -model <path_to_model>/model_name.xml -config ./congig.txt
```

## Report Option

By default performance counts are reported per layer in Inference Engine.
vpu_profile sets "per_layer" profiling by default - means performance report will be provided for each layer.
To switch report you can use configuration file or report option when changed to "per_stage" - 
statistics will be provided with finer granularity - for each executed stage.
If wrongly specified - switch back to default mode.

```sh
$./vpu_profile -model <path_to_model>/model_name.xml -report "per_stage"
```

Performace counts will be provided for executed only stages/layers, in next format:
```sh
Index           Name           Type                Time (ms)
<stage index>   <stage name>   <stage exec_type>   <time in millisecond>
```
Where 
  * stage index - correspons to execution order of a stage, in case of per_layer output this corresponds to the first stage order 
  * stage name - corresponds to the name of a stage or layer in case of per_layer output, 
  * stage exec_type - corresponds to stage execution type, e.g. MyriadXHwOp mean that stage was executed at HW fixed function, otherwise - shaves.
In case of per_layer output exec_type can be not accurate - if the first stage of a layer performed on shaves - whole layer won't be shown as HW operation
  * time in millisecond - time in millisecond took corresponding stage or layer to execute.

At the end of report total accumulated execution time printed. `Important`: this timing doesn't represent throughput, but latency of execution on device.
Throughput will depend on asyncrounous operation depth and device capacity, so for example by default MyriadX usually running 2 infer requests asynchrounously
which exected in parallel on a device, so below number can be close to 1000/(total_time*2).
```sh
                Total inference time:              <total_time>
```

