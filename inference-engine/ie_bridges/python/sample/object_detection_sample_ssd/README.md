# Object Detection Python* Sample SSD

This sample demonstrates how to run the Object Detection sample application.

The sample demonstrates how to use the new Infer Request API of Inference Engine in applications.
Refer to [Integrate the Inference Engine New Request API with Your Application](./docs/IE_DG/Integrate_with_customer_application_new_API.md) for details.
The sample demonstrates how to build and execute an inference request on example of object detection networks.

Due to properties of SSD networks, this sample works correctly only on a batch of the size 1. For a greater number of images in a batch, network reshape is required.

## How It Works

Upon the start-up, the sample application reads command line parameters and loads specified network and input images (or a
folder with images) to the Inference Engine plugin.

Then, the sample creates an inference request object and executes inference on it.

When inference is done, the application outputs data to the standard output stream and creates an output image with bounding boxes drawn atop the initial image.

> **NOTE**: By default, Inference Engine samples and demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the sample or demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](./docs/MO_DG/prepare_model/convert_model/Converting_Model_General.md).

## Running

Running the application with the <code>-h</code> option yields the following usage message:
```
python3 object_detection_sample_ssd.py -h
```
The command yields the following usage message:
```
usage: object_detection_sample_ssd.py [-h] -m MODEL -i INPUT [INPUT ...]
                                      [-l CPU_EXTENSION]
                                      [-d DEVICE] [--labels LABELS]
                                      [-nt NUMBER_TOP]

Options:
  -h, --help            Show this help message and exit
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model
  -i INPUT [INPUT ...], --input INPUT [INPUT ...]
                        Required. Path to a folder with images or path to an
                        image files
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        Optional. Required for CPU custom layers. Absolute
                        path to a shared library with the kernels
                        implementations
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU,
                        GPU, FPGA, HDDL or MYRIAD is acceptable. The sample
                        will look for a suitable plugin for device specified
                        Default value is CPU
  --labels LABELS       Optional. Labels mapping file
  -nt NUMBER_TOP, --number_top NUMBER_TOP
                        Optional. Number of top results
```

Running the application with the empty list of options yields the usage message given above and an error message.

To run the sample, you can use RMNet_SSD or other object-detection models. You can download the pre-trained models with the OpenVINO [Model Downloader](https://github.com/opencv/open_model_zoo/tree/2018/model_downloader) or from [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the sample with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](./docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).


You can do inference of an image using a trained RMNet_SSD network on FPGA with fallback to CPU using the following command:
```
    python3 object_detection_sample_ssd.py -i <path_to_image>/cat.bmp -m <path_to_model>/alexnet_fp32.xml -nt 5 -d HETERO:FPGA,CPU
```

## Sample Output

By default, the application outputs all inference results and draws bounding boxes for inference results with an over 50% confidence.

## See Also
* [Using Inference Engine Samples](./docs/IE_DG/Samples_Overview.md)
