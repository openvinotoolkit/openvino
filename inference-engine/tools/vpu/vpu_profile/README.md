# vpu_profile tool

This topic demonstrates how to run the `vpu_profile` tool application, which estimates performance by calculating average time of each stage in model.

## How It Works

Upon the start-up, the sample application reads command line parameters and loads a network and its inputs from given directory to the Inference Engine plugin.
Then application starts infer requests in asynchronous mode till specified number of iterations is finished.
After inference stage, profile tool computes average time that each stage took.

## Running

Running the application with the <code>-h</code> option yields the following usage message:

```sh
Inference Engine:
	API version ............ <version>
	Build .................. <number>

vpu_profile [OPTIONS]
[OPTIONS]:
	-help       	         	Optional. Print a usage message.
	-model      	 <value> 	Required. Path to xml model.
	-inputs_dir 	 <value> 	Required. Path to folder with images. Default: ".".
	-plugin_path	 <value> 	Optional. Path to a plugin folder.
	-config     	 <value> 	Optional. Path to the configuration file. Default value: "config".
	-platform   	 <value> 	Optional. Specifies movidius platform.
	-iterations 	 <value> 	Optional. Specifies number of iterations. Default value: 16.
	-plugin     	 <value> 	Optional. Specifies plugin. Supported values: myriad.
	            	         	Default value: "myriad".
```

Running the application with the empty list of options yields an error.

To run the sample, you can use public or pre-trained models. To download the pre-trained models, use the OpenVINO [Model Downloader](https://github.com/opencv/open_model_zoo/tree/2018/model_downloader) or go to [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **note**: before running the sample with a trained model, make sure the model is converted to the inference engine format (\*.xml + \*.bin) using the [model optimizer tool](./docs/mo_dg/deep_learning_model_optimizer_devguide.md).

You can use the following command to do inference on images from a folder using a trained Faster R-CNN network:

```sh
./perfcheck -model <path_to_model>/faster_rcnn.xml -inputs_dir <path_to_inputs>
```
