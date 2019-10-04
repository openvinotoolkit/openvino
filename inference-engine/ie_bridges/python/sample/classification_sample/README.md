# Image Classification Python* Sample

This topic demonstrates how to run the Image Classification sample application, which performs
inference using image classification networks such as AlexNet and GoogLeNet.

## How It Works

Upon the start-up, the sample application reads command line parameters and loads a network and an image to the Inference
Engine plugin. When inference is done, the application creates an
output image and outputs data to the standard output stream.

> **NOTE**: By default, Inference Engine samples and demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the sample or demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](./docs/MO_DG/prepare_model/convert_model/Converting_Model_General.md).

## Running

Run the application with the `-h` option yields the usage message:
```
python3 classification_sample.py -h
```
The command yields the following usage message:
```
usage: classification_sample.py [-h] -m MODEL -i INPUT [INPUT ...]
                                [-l CPU_EXTENSION]
                                [-d DEVICE] [--labels LABELS] [-nt NUMBER_TOP]

Options:
  -h, --help            Show this help message and exit.
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model.
  -i INPUT [INPUT ...], --input INPUT [INPUT ...]
                        Required. Path to a folder with images or path to an
                        image files
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        Optional. Required for CPU custom layers. MKLDNN (CPU)-targeted custom layers.
                        Absolute path to a shared library with the kernels
                        implementations.
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU,
                        GPU, FPGA, HDDL or MYRIAD is acceptable. The sample
                        will look for a suitable plugin for device specified.
                        Default value is CPU
  --labels LABELS       Optional. Path to a labels mapping file
  -nt NUMBER_TOP, --number_top NUMBER_TOP
                        Optional. Number of top results
```

Running the application with the empty list of options yields the usage message given above.

To run the sample, you can use AlexNet and GoogLeNet or other image classification models. You can download the pre-trained models with the OpenVINO [Model Downloader](https://github.com/opencv/open_model_zoo/tree/2018/model_downloader) or from [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the sample with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](./docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).

For example, to perform inference of an AlexNet model (previously converted to the Inference Engine format) on CPU, use the following command:

```
    python3 classification_sample.py -i <path_to_image>/cat.bmp -m <path_to_model>/alexnet_fp32.xml
```

## Sample Output

By default the application outputs top-10 inference results.
Add the `-nt` option to the previous command to modify the number of top output results.
For example, to get the top-5 results on GPU, run the following command:
```
    python3 classification_sample.py -i <path_to_image>/cat.bmp -m <path_to_model>/alexnet_fp32.xml -nt 5 -d GPU
```

## See Also
* [Using Inference Engine Samples](./docs/IE_DG/Samples_Overview.md)
* [Model Optimizer tool](./docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
* [Model Downloader](https://github.com/opencv/open_model_zoo/tree/2018/model_downloader)
