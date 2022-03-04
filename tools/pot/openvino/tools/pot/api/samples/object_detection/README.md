# API usage sample for object_detection {#pot_sample_object_detection_README}

This sample demonstrates the use of the [Post-training Optimization Toolkit API](@ref pot_compression_api_README) to
 quantize an object detection model in the [accuracy-aware mode](@ref pot_compression_algorithms_quantization_accuracy_aware_README).
The [MobileNetV1 FPN](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/ssd_mobilenet_v1_fpn_coco/ssd_mobilenet_v1_fpn_coco.md) model from TensorFlow* for object detection task is used for this purpose.
A custom `DataLoader` is created to load the [COCO](https://cocodataset.org/) dataset for object detection task 
and the implementation of mAP COCO is used for the model evaluation.

## How to prepare the data

To run this sample, you will need to download the validation part of the [COCO](https://cocodataset.org/). The images should be placed in a separate folder, which will be later referred as `<IMAGES_DIR>` and annotation file `instances_val2017.json` later referred as `<ANNOTATION_FILE>`.  
## How to Run the Sample
In the instructions below, the Post-Training Optimization Tool directory `<POT_DIR>` is referred to:
- `<ENV>/lib/python<version>/site-packages/` in the case of PyPI installation, where `<ENV>` is a Python* 
  environment where OpenVINO is installed and `<version>` is a Python* version, e.g. `3.6`.
- `<INSTALL_DIR>/deployment_tools/tools/post_training_optimization_toolkit` in the case of OpenVINO distribution package. 
  `<INSTALL_DIR>` is the directory where Intel&reg; Distribution of OpenVINO&trade; toolkit is installed.

1. To get started, follow the [Installation Guide](@ref pot_InstallationGuide).
2. Launch [Model Downloader](@ref omz_tools_downloader) tool to download `ssd_mobilenet_v1_fpn_coco` model from the Open Model Zoo repository.
   ```sh
   python3 ./downloader.py --name ssd_mobilenet_v1_fpn_coco
3. Launch [Model Converter](@ref omz_tools_downloader) tool to generate Intermediate Representation (IR) files for the model:
   ```sh
   python3 ./converter.py --name ssd_mobilenet_v1_fpn_coco --mo <PATH_TO_MODEL_OPTIMIZER>/mo.py
   ```
4. Launch the sample script:
   ```sh
   python <POT_DIR>/api/samples/object_detection/object_detection_sample.py -m <PATH_TO_IR_XML> -d <IMAGES_DIR> --annotation-path <ANNOTATION_FILE>
   ```
   
*  Optional: you can specify .bin file of IR directly using the `-w`, `--weights` options.
