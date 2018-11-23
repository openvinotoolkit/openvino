# Validation Application

Inference Engine Validation Application is a tool that allows to infer deep learning models with
standard inputs and outputs configuration and to collect simple
validation metrics for topologies. It supports **top-1** and **top-5** metric for Classification networks and
11-points **mAP** metric for Object Detection networks.

Possible use cases of the tool:
* Check if the Inference Engine infers the public topologies well (the engineering team uses the Validation Application for
  regular testing)
* Verify if a custom model is compatible with the default input/output configuration and compare its
  accuracy with the public models
* Use Validation Application as another sample: although the code is much more complex than in classification and object
  detection samples, the source code is open and can be re-used.

## Validation Application Options

The Validation Application provides the following command-line interface (CLI):
```sh
Usage: validation_app [OPTION]

Available options:

    -h                        Print a help message
    -t <type>                 Type of an inferred network ("C" by default)
      -t "C" for classification
      -t "OD" for object detection
    -i <path>                 Required. Folder with validation images. Path to a directory with validation images. For Classification models, the directory must contain folders named as labels with images inside or a .txt file with a list of images. For Object Detection models, the dataset must be in VOC format.
    -m <path>                 Required. Path to an .xml file with a trained model
    -l <absolute_path>        Required for CPU custom layers. Absolute path to a shared library with the kernel implementations
    -c <absolute_path>        Required for GPU custom kernels.Absolute path to an .xml file with the kernel descriptions.
    -d <device>               Target device to infer on: CPU (default), GPU, FPGA, or MYRIAD. The application looks for a suitable plugin for the specified device.
    -b N                      Batch size value. If not specified, the batch size value is taken from IR
    -ppType <type>            Preprocessing type. Options: "None", "Resize", "ResizeCrop"
    -ppSize N                 Preprocessing size (used with ppType="ResizeCrop")
    -ppWidth W                Preprocessing width (overrides -ppSize, used with ppType="ResizeCrop")
    -ppHeight H               Preprocessing height (overrides -ppSize, used with ppType="ResizeCrop")
    --dump                    Dump file names and inference results to a .csv file

    Classification-specific options:
      -Czb true               "Zero is a background" flag. Some networks are trained with a modified dataset where the class IDs  are enumerated from 1, but 0 is an undefined "background" class (which is never detected)

    Object detection-specific options:
      -ODkind <kind>          Type of an Object Detection model. Options: SSD
      -ODa <path>             Required for Object Detection models. Path to a directory containing an .xml file with annotations for images.
      -ODc <file>             Required for Object Detection models. Path to a file containing a list of classes
      -ODsubdir <name>        Directory between the path to images (specified with -i) and image name (specified in the .xml file). For VOC2007 dataset, use JPEGImages.
```
The tool options are divided into two categories:
1. **Common options** named with a single letter or a word, such as `-b` or `--dump`.
   These options are the same in all Validation Application modes.
2. **Network type-specific options** named as an acronym of the network type (`C` or `OD`)
   followed by a letter or a word.

## General Workflow

When executed, the Validation Application perform the following steps:

1. Loads a model to an Inference Engine plugin
2. Reads validation set (specified with the `-i` option):
    - if you specified a directory, the application tries to load labels first. To do this, it searches for the file
      with the same name as a model, but with `.labels` extension (instead of `.xml`).
      Then it searches for the specified folder, detects its sub-folders named as known labels, and adds all images from these sub-folders to the validation set. When there are no such sub-folders, validation set is considered empty.

    - if you specified a `.txt` file, the application reads this file expecting every line to be in the correct format.
      For more information about the format, refer to the <a href="#preparing">Preparing the Dataset</a> section below.

3. Reads the batch size value specified with the `-b` option and loads this number of images to the plugin
   **Note**: Images loading time is not a part of inference time reported by the application.

4. The plugin infers the model, and the Validation Application collects the statistics.

You can also retrieve infer result by specifying the `--dump` option, however it generates a report only
for Classification models. This CLI option enables creation (if possible) of an inference report in
the `.csv` format.

The structure of the report is a set of lines, each of them contains semicolon-separated values:
* image path
* a flag representing correctness of prediction
* ID of Top-1 class
* probability that the image belongs to Top-1 class in per cents
* ID of Top-2 class
* probability that the image belongs to Top-2 class in per cents
*

This is an example line from such report:
```bash
"ILSVRC2012_val_00002138.bmp";1;1;8.5;392;6.875;123;5.875;2;5.5;396;5;
```
It means that the given image was predicted correctly. The most probable prediction is that this image
represents class *1* with the probability *0.085*.

## <a name="preparing"></a>Prepare a Dataset

You must prepare the dataset before running the Validation Application. The format of dataset depends on
a type of the model you are going to validate. Make sure that the dataset is format is applicable
for the chosen model type.

### Dataset Format for Classification: Folders as Classes

In this case, a dataset has the following structure:
```sh
|-- <path>/dataset
    |-- apron
        |-- apron1.bmp
        |-- apron2.bmp
    |-- collie
        |-- a_big_dog.jpg
    |-- coral reef
        |-- reef.bmp
    |-- Siamese
        |-- cat3.jpg
```

This structure means that each folder in dataset directory must have the name of one of the classes and contain all images of this class. In the given example, there are two images that represent the class `apron`, while three other classes have only one image
each.

**NOTE:** A dataset can contain images of both `.bmp` and `.jpg` formats.

The correct way to use such dataset is to specify the path as `-i <path>/dataset`.

### Dataset Format for Classification: List of Images (ImageNet-like)

If you want to use this dataset format, create a single file with a list of images. In this case, the correct set of files must be similar to the following:
```bash
|-- <path>/dataset
    |-- apron1.bmp
    |-- apron2.bmp
    |-- a_big_dog.jpg
    |-- reef.bmp
    |-- cat3.jpg
    |-- labels.txt
```

Where `labels.txt` looks like:
```bash
apron1.bmp 411
apron2.bmp 411
cat3.jpg 284
reef.bmp 973
a_big_dog.jpg 231
```

Each line of the file must contain the name of the image and the ID of the class
that it represents in the format `<image_name> tabulation <class_id>`. For example, `apron1.bmp` represents the class with ID `411`.

**NOTE:** A dataset can contain images of both `.bmp` and `.jpg` formats.

The correct way to use such dataset is to specify the path as `-i <path>/dataset/labels.txt`.

### Dataset Format for Object Detection (VOC-like)

Object Detection SSD models can be inferred on the original dataset that was used as a testing dataset during the model training.
To prepare the VOC dataset, follow the steps below :

1. Download the pre-trained SSD-300 model from the SSD GitHub* repository at
   [https://github.com/weiliu89/caffe/tree/ssd](https://github.com/weiliu89/caffe/tree/ssd).

2. Download VOC2007 testing dataset:
  ```bash
  $wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
  tar -xvf VOCtest_06-Nov-2007.tar
  ```
3. Convert the model with the [Model Optimizer](docs/Model_Optimizer_Developer_Guide/prepare_trained_model/convert_model/Convert_Model_From_Caffe.md).

4. Create a proper `.txt` class file from the original `labelmap_voc.prototxt`. The new file must be in
the following format:
```sh
	none_of_the_above 0
	aeroplane 1
	bicycle 2
	bird 3
	boat 4
	bottle 5
	bus 6
	car 7
	cat 8
	chair 9
	cow 10
	diningtable 11
	dog 12
	horse 13
	motorbike 14
	person 15
	pottedplant 16
	sheep 17
	sofa 18
	train 19
	tvmonitor 20
```
Save this file as `VOC_SSD_Classes.txt`.

## Validate Classification Models

Once you have prepared the dataset (refer to the <a href="#preparing">Preparing the Dataset</a> section above),
run the following command to infer a classification model on the selected dataset:
```bash
./validation_app -t C -i <path_to_images_directory_or_txt_file> -m <path_to_classification_model>/<model_name>.xml -d <CPU|GPU>
```

## Validate Object Detection Models

**Note**: Validation Application was validated with SSD CNN. Any network that can be inferred by the Inference Engine
and has the same input and output format as one of these should be supported as well.

Once you have prepared the dataset (refer to the <a href="#preparing">Preparing the Dataset</a> section above),
run the following command to infer an Object Detection model on the selected dataset:
```bash
./validation_app -d CPU -t OD -ODa "<path_to_VOC_dataset>/VOCdevkit/VOC2007/Annotations" -i "<path_to_VOC_dataset>/VOCdevkit" -m "<path_to_model>/vgg_voc0712_ssd_300x300.xml" -ODc "<path_to_classes_file>/VOC_SSD_Classes.txt" -ODsubdir JPEGImages
```

## Understand Validation Application Output

During the validation process, you can see the interactive progress bar that represents the current validation stage. When it is
full, the validation process is over, and you can analyze the output.

Key data from the output:
* **Network loading time** - time spent on topology loading in ms
* **Model** - path to a chosen model
* **Model Precision** - precision of the chosen model
* **Batch size** - specified batch size
* **Validation dataset** - path to a validation set
* **Validation approach** - type of the model: Classification or Object Detection
* **Device** - device type

Below you can find the example output for Classification models, which reports average infer time and
**Top-1** and **Top-5** metric values:
```bash
Average infer time (ms): 588.977 (16.98 images per second with batch size = 10)

Top1 accuracy: 70.00% (7 of 10 images were detected correctly, top class is correct)
Top5 accuracy: 80.00% (8 of 10 images were detected correctly, top five classes contain required class)
```

Below you can find the example output for Object Detection models:

```bash
Progress: [....................] 100.00% done    
[ INFO ] Processing output blobs
Network load time: 27.70ms
Model: /home/user/models/ssd/withmean/vgg_voc0712_ssd_300x300/vgg_voc0712_ssd_300x300.xml
Model Precision: FP32
Batch size: 1
Validation dataset: /home/user/Data/SSD-data/testonly/VOCdevkit
Validation approach: Object detection network

Average infer time (ms): 166.49 (6.01 images per second with batch size = 1)
Average precision per class table:

Class	AP
1	0.796
2	0.839
3	0.759
4	0.695
5	0.508
6	0.867
7	0.861
8	0.886
9	0.602
10	0.822
11	0.768
12	0.861
13	0.874
14	0.842
15	0.797
16	0.526
17	0.792
18	0.795
19	0.873
20	0.773

Mean Average Precision (mAP): 0.7767
```

This output shows the resulting `mAP` metric value for the SSD300 model used to prepare the
dataset. This value repeats the result stated in the
[SSD GitHub* repository](https://github.com/weiliu89/caffe/tree/ssd) and in the
[original arXiv paper](http://arxiv.org/abs/1512.02325).



## See Also

* [Using Inference Engine Samples](./docs/Inference_Engine_Developer_Guide/Samples_Overview.md)
