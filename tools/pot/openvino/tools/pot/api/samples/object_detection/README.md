# Quantizatiing Object Detection Model with Accuracy Control {#pot_example_object_detection_README}

This example demonstrates the use of the [Post-training Optimization Toolkit API](@ref pot_compression_api_README) to
 quantize an object detection model in the [accuracy-aware mode](@ref pot_compression_algorithms_quantization_accuracy_aware_README).
The [MobileNetV1 FPN](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/ssd_mobilenet_v1_fpn_coco/ssd_mobilenet_v1_fpn_coco.md) model from TensorFlow* for object detection task is used for this purpose.
A custom `DataLoader` is created to load the [COCO](https://cocodataset.org/) dataset for object detection task 
and the implementation of mAP COCO is used for the model evaluation.

## How to prepare the data

To run this example, you will need to download the validation part of the [COCO](https://cocodataset.org/). The images should be placed in a separate folder, which will be later referred as `<IMAGES_DIR>` and annotation file `instances_val2017.json` later referred as `<ANNOTATION_FILE>`.  
## How to Run the example

1. Launch [Model Downloader](@ref omz_tools_downloader) tool to download `ssd_mobilenet_v1_fpn_coco` model from the Open Model Zoo repository.
   ```sh
   python3 ./downloader.py --name ssd_mobilenet_v1_fpn_coco
2. Launch [Model Converter](@ref omz_tools_downloader) tool to generate Intermediate Representation (IR) files for the model:
   ```sh
   python3 ./converter.py --name ssd_mobilenet_v1_fpn_coco --mo <PATH_TO_MODEL_OPTIMIZER>/mo.py
   ```
3. Launch the example script:
   ```sh
   python <POT_DIR>/api/examples/object_detection/object_detection_example.py -m <PATH_TO_IR_XML> -d <IMAGES_DIR> --annotation-path <ANNOTATION_FILE>
   ```
   
*  Optional: you can specify .bin file of IR directly using the `-w`, `--weights` options.
