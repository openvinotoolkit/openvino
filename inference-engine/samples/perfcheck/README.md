# Perfcheck Sample

This topic demonstrates how to build and run the Perfcheck sample application, which estimates performance by calculating minimum, average, and maximum FPS.

## How It Works

Upon the start-up, the sample application reads command line parameters and loads a network and its inputs from given directory to the Inference Engine plugin.
Then application starts infer requests in asynchronous mode till specified number of iterations is finished.
After inference stage, Perfcheck sample computes total time of execution, divides execution time in 10 intervals and evaluates minimum, average and maximum FPS among these intervals.

## Running

Running the application with the <code>-h</code> option yields the following usage message:

```sh
./perfcheck -h
[ INFO ] Inference Engine:
        API version ............ <version>
        Build .................. <number>

perfcheck [OPTIONS]
[OPTIONS]:
        -m                       <value>        Required. Path to an .xml file with a trained model.
        -h                                      Optional. Print a usage message.
        -d                       <value>        Optional. Specify the target device to infer on. Sample will look for a suitable plugin for device specified. Default value: CPU.
        -pp                      <value>        Optional. Path to a plugin folder.
        -l                       <value>        Optional. Required for CPU custom layers. Absolute path to a shared library with the kernels implementation.
        -c                       <value>        Optional. Required for GPU custom kernels. Absolute path to an .xml file with the kernels description.
        -inputs_dir              <value>        Optional. Path to a folder with images and binaries for inputs. Default value: ".".
        -config                  <value>        Optional. Path to a configuration file.
        -num_iterations          <value>        Optional. Specify number of iterations. Default value: 1000. Must be greater than or equal to 1000.
        -batch                   <value>        Optional. Specify batch. Default value: 1.
        -num_networks            <value>        Optional. Specify number of networks. Default value: 1. Must be less than or equal to 16.
        -num_requests            <value>        Optional. Specify number of infer requests. Default value depends on specified device.
        -num_fpga_devices        <value>        Optional. Specify number of FPGA devices. Default value: 1.
```

Running the application with the empty list of options yields an error message.

You can use the following command to do inference on Intel® Processors on images from a folder using a trained Faster R-CNN network:

```sh
./perfcheck -m <path_to_model>/faster_rcnn.xml -inputs_dir <path_to_inputs> -d CPU
```

> **NOTE**: Public models should be first converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer).

## Sample Output

The application outputs a performance statistics that shows: total execution time (in milliseconds), number of iterations, batch size, minimum, average and maximum FPS.
Example of sample output:

```sh
[ INFO ] Inference Engine:
	API version ............ <version>
	Build .................. <number>
[ INFO ] Loading network files:
[ INFO ] 	<path_to_model_xml_file>
[ INFO ] 	<path_to_model_bin_file>
[ INFO ] Loading network 0
[ INFO ] All networks are loaded

Total time:     8954.61 ms
Num iterations: 1000
Batch:          1
Min fps:        110.558
Avg fps:        111.674
Max fps:        112.791
```

## See Also

* [Using Inference Engine Samples](./docs/IE_DG/Samples_Overview.md)
