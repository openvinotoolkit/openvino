# Quantizing Semantic Segmentation Model {#pot_example_segmentation_README}

This example demonstrates the use of the [Post-training Optimization Tool API](@ref pot_compression_api_README) for the task of quantizing a segmentation model.
The [DeepLabV3](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/deeplabv3/deeplabv3.md) model from TensorFlow* is used for this purpose.
A custom `DataLoader` is created to load the [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) dataset for semantic segmentation task 
and the implementation of Mean Intersection Over Union metric is used for the model evaluation. The code of the example is available on [GitHub](https://github.com/openvinotoolkit/openvino/tree/master/tools/pot/openvino/tools/pot/api/samples/segmentation).

## How to prepare the data

To run this example, you will need to download the validation part of the Pascal VOC 2012 image database http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#data.
Images are placed in the `JPEGImages` folder, ImageSet file with the list of image names for the segmentation task can be found at `ImageSets/Segmentation/val.txt` 
and segmentation masks are kept in the `SegmentationClass` directory.


## How to Run the example

1. Launch [Model Downloader](@ref omz_tools_downloader) tool to download `deeplabv3` model from the Open Model Zoo repository.
   ```sh
   omz_downloader --name deeplabv3
   ```
2. Launch [Model Converter](@ref omz_tools_downloader) tool to generate Intermediate Representation (IR) files for the model:
   ```sh
   omz_converter --name deeplabv3 --mo <PATH_TO_MODEL_OPTIMIZER>/mo.py
   ```
3. Launch the example script from the example directory:
   ```sh
   python3 ./segmentation_example.py -m <PATH_TO_IR_XML> -d <VOCdevkit/VOC2012/JPEGImages> --imageset-file <VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt> --mask-dir <VOCdevkit/VOC2012/SegmentationClass>
   ```
   Optional: you can specify .bin file of IR directly using the `-w`, `--weights` options.
