# Neural Style Transfer Python* Sample

This topic demonstrates how to run the Neural Style Transfer sample application, which performs 
inference of style transfer models.

> **NOTE**: The OpenVINO™ toolkit does not include a pre-trained model to run the Neural Style Transfer sample. A public model from the [Zhaw's Neural Style Transfer repository](https://github.com/zhaw/neural_style) can be used. Read the [Converting a Style Transfer Model from MXNet*](./docs/MO_DG/prepare_model/convert_model/mxnet_specific/Convert_Style_Transfer_From_MXNet.md) topic from the [Model Optimizer Developer Guide](./docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md) to learn about how to get the trained model and how to convert it to the Inference Engine format (\*.xml + \*.bin).

## How It Works

> **NOTE**: By default, Inference Engine samples and demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the sample or demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](./docs/MO_DG/prepare_model/convert_model/Converting_Model_General.md).

## Running

Running the application with the <code>-h</code> option yields the following usage message:
```
python3 style_transfer_sample.py --help
```
The command yields the following usage message:
```
usage: style_transfer_sample.py [-h] -m MODEL -i INPUT [INPUT ...]
                                [-l CPU_EXTENSION] [-d DEVICE] 
                                [-nt NUMBER_TOP]
                                [--mean_val_r MEAN_VAL_R]
                                [--mean_val_g MEAN_VAL_G]
                                [--mean_val_b MEAN_VAL_B]

Options:
  -h, --help            Show this help message and exit.
  -m MODEL, --model MODEL
                        Path to an .xml file with a trained model.
  -i INPUT [INPUT ...], --input INPUT [INPUT ...]
                        Path to a folder with images or path to an image files
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        Optional. Required for CPU custom layers. Absolute
                        MKLDNN (CPU)-targeted custom layers. Absolute path to
                        a shared library with the kernels implementations
  -d DEVICE, --device DEVICE
                        Specify the target device to infer on; CPU, GPU, FPGA,
                        HDDL or MYRIAD is acceptable. Sample will look for a
                        suitable plugin for device specified. Default value is CPU
  -nt NUMBER_TOP, --number_top NUMBER_TOP
                        Number of top results
  --mean_val_r MEAN_VAL_R, -mean_val_r MEAN_VAL_R
                        Mean value of red chanel for mean value subtraction in
                        postprocessing
  --mean_val_g MEAN_VAL_G, -mean_val_g MEAN_VAL_G
                        Mean value of green chanel for mean value subtraction
                        in postprocessing
  --mean_val_b MEAN_VAL_B, -mean_val_b MEAN_VAL_B
                        Mean value of blue chanel for mean value subtraction
                        in postprocessing
```

Running the application with the empty list of options yields the usage message given above and an error message.

To perform inference of an image using a trained model of NST network on Intel® CPUs, use the following command:
```
    python3 style_transfer_sample.py -i <path_to_image>/cat.bmp -m <path_to_model>/1_decoder_FP32.xml
```

### Demo Output

The application outputs an image (`out1.bmp`) or a sequence of images (`out1.bmp`, ..., `out<N>.bmp`) which are redrawn in style of the style transfer model used for sample. 

## See Also 
* [Using Inference Engine Samples](./docs/IE_DG/Samples_Overview.md)


