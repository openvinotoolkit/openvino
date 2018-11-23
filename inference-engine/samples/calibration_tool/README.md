# Calibration Tool

Inference Engine Calibration Tool calibrates a given FP32 model so that is can be run in low-precision 8-bit integer
mode while keeping the input data of this model in the original precision.

## Calibration Tool Options

The core command-line options for the Calibration Tool are the same as for
[Validation Application](./samples/validation_app/README.md). However, the Calibration Tool has the following specific options: `-t`, `-subset`, `-output`, and `-threshold`.

Running the Calibration Tool with the `-h` option yields the following usage message with all CLI options listed:
```sh  
Usage: calibration_tool [OPTION]

Available options:

    -h                        Print a help message
    -t <type>                 Type of an inferred network ("C" by default)
      -t "C" to calibrate Classification network and write the calibrated network to IR
      -t "OD" to calibrate Object Detection network and write the calibrated network to IR
      -t "RawC" to collect only statistics for Classification network and write statistics to IR. With this option, a model is not calibrated. For calibration and statisctics collection, use "-t C" instead.
      -t "RawOD" to collect only statistics for Object Detection network and write statistics to IR. With this option, a model is not calibrated. For calibration and statisctics collection, use "-t OD" instead
    -i <path>                 Required. Path to a directory with validation images. For Classification models, the directory must contain folders named as labels with images inside or a .txt file with a list of images. For Object Detection models, the dataset must be in VOC format.
    -m <path>                 Required. Path to an .xml file with a trained model, including model name and extension.
    -l <absolute_path>        Required for CPU custom layers. Absolute path to a shared library with the kernel implementations.
    -c <absolute_path>        Required for GPU custom kernels. Absolute path to an .xml file with the kernel descriptions.
    -d <device>               Target device to infer on: CPU (default), GPU, FPGA, or MYRIAD. The application looks for a suitable plugin for the specified device.
    -b N                      Batch size value. If not specified, the batch size value is taken from IR
    -ppType <type>            Preprocessing type. Options: "None", "Resize", "ResizeCrop"
    -ppSize N                 Preprocessing size (used with ppType="ResizeCrop")
    -ppWidth W                Preprocessing width (overrides -ppSize, used with ppType="ResizeCrop")
    -ppHeight H               Preprocessing height (overrides -ppSize, used with ppType="ResizeCrop")
    --dump                    Dump file names and inference results to a .csv file
    -subset                  Number of pictures from the whole validation set tocreate the calibration dataset. Default value is 0, which stands forthe whole provided dataset
    -output <output_IR>      Output name for calibrated model. Default is <original_model_name>_i8.xml|bin
    -threshold               Threshold for a maximum accuracy drop of quantized model. Must be an integer number (percents) without a percent sign. Default value is 1, which stands for accepted accuracy drop in 1%

    Classification-specific options:
      -Czb true               "Zero is a background" flag. Some networks are trained with a modified dataset where the class IDs  are enumerated from 1, but 0 is an undefined "background" class (which is never detected)

    Object detection-specific options:
      -ODkind <kind>          Type of an Object Detection model. Options: SSD
      -ODa <path>             Required for Object Detection models. Path to a directory containing an .xml file with annotations for images.
      -ODc <file>             Required for Object Detection models. Path to a file with a list of classes
      -ODsubdir <name>        Directory between the path to images (specified with -i) and image name (specified in the .xml file). For VOC2007 dataset, use JPEGImages.
```

The tool options are divided into two categories:
1. **Common options** named with a single letter or a word, such as <code>-b</code> or <code>--dump</code>.
   These options are the same in all calibration tool modes.
2. **Network type-specific options** named as an acronym of the network type (<code>C</code> or <code>OD</code>)
   followed by a letter or a word.


## Calibrate a Classification Model

To calibrate a classification convolutional neural network (CNN)
on a subset of images (first 2000 images) from the given dataset (specified with the `-i` option), run the following command:

```bash
./calibration_tool -t C -i <path_to_images_directory_or_txt_file> -m <path_to_classification_model>/<model_name>.xml -d <CPU|GPU> -subset 2000
```

The dataset must have the correct format. Classification models support two formats: folders
named as labels that contain all images of this class and ImageNet*-like format, with the
`.txt` file containing list of images and IDs of classes.

For more information on the structure of the datasets, refer to the **Prepare a Dataset** section of the
[Validation Application document](./samples/validation_app/README.md).

If you decide to use the subset of the given dataset, use the ImageNet-like format
instead of "folder as classes" format. This brings a more accurate calibration as you are likely to get images
representing different classes.

For example, to calibrate the pretrained TensorFlow\* `inception_v4_tf.xml` classification model,
run the following command:

```bash
./calibration_tool -t C -m inception_v4_tf.xml -i ILSVRC2012_val.txt -Czb false -ppType "ResizeCrop" -ppSize 342 -b 1 -d CPU -subset 2000
```

## Calibrate Object Detection Model

This topic demonstrates how to run the Calibration Tool on the Object Detection CNN on a set of images. Please
review the list of Object Detection models used for validation of the Calibration Tool
in the [8-bit Inference Introduction](./docs/Inference_Engine_Developer_Guide/Int8Inference.md).
Any network that can be inferred with the Inference Engine and has the same input and output
format as the SSD CNN should be supported as well.

### Run SSD Network on the VOC dataset

Before you start calibrating the model, make sure your dataset is in the correct format. For more information,
refer to the **Prepare a Dataset** section of the
[Validation Application document](./samples/validation_app/README.md).

Once you have prepared the dataset, you can calibrate the model on it by running the following command:
```bash
./calibration_tool -d CPU -t OD -ODa "<path_to_image_annotations>/VOCdevkit/VOC2007/Annotations" -i "<path_to_image_directory>/VOCdevkit" -m "<path_to_model>/vgg_voc0712_ssd_300x300.xml" -ODc "<path_to_classes_list>/VOC_SSD_Classes.txt" -ODsubdir JPEGImages -subset 500
```

## See Also

* [Using Inference Engine Samples](./docs/Inference_Engine_Developer_Guide/Samples_Overview.md)
