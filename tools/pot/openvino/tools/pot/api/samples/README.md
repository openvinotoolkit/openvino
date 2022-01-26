# Post-training Optimization Tool API samples {#pot_sample_README}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:
   
   Image Classification Quantization Sample <pot_sample_classification_README>
   Accuracy-Aware Quantization Sample <pot_sample_object_detection_README>
   Cascaded Model Quantization Sample <pot_sample_face_detection_README>
   Semantic segmentation quantization sample <pot_sample_segmentation_README>
   3D Segmentation quantization sample <pot_sample_3d_segmentation_README>
   GNA speech sample <pot_sample_speech_README>

@endsphinxdirective

The Post-training Optimization Tool contains multiple samples that demonstrate how to use its [Software API](@ref pot_compression_api_README) 
to optimize DL models which require special inference pipeline, data loading or metric calculation that 
are not supported through the `AccuracyCheker` or `Simplified` engines (see [Best Practices](../../../../../docs/BestPractices.md) for more details).

All available samples can be found in `<POT_DIR>/api/samples` folder, where `<POT_DIR>` is a directory where the Post-Training Optimization Tool is installed.
> **NOTE**: - `<POT_DIR>` is referred to `<ENV>/lib/python<version>/site-packages/` in the case of PyPI installation, where `<ENV>` is a Python* 
> environment where OpenVINO is installed and `<version>` is a Python* version, e.g. `3.6` or to `<INSTALL_DIR>/deployment_tools/tools/post_training_optimization_toolkit` in the case of OpenVINO distribution package. 
> `<INSTALL_DIR>` is the directory where Intel&reg; Distribution of OpenVINO&trade; toolkit is installed.

There are currently the following samples that demonstrate the implementation of `Engine`, `Metric` and `DataLoader` interfaces 
for classification, detection and segmentation tasks:

1. [Classification sample](./classification/README.md)
    - Uses single `MobilenetV2` model from TensorFlow*
    - Implements `DataLoader` to load .JPEG images and annotations of Imagenet database
    - Implements `Metric` interface to calculate Accuracy at top-1 metric
    - Uses DefaultQuantization algorithm for quantization model

2. [Object Detection sample](./object_detection/README.md)
    - Uses single `MobileNetV1 FPN` model from TensorFlow*
    - Implements `Dataloader` to load images of COCO database
    - Implements `Metric` interface to calculate mAP@[.5:.95] metric
    - Uses `AccuracyAwareQuantization` algorithm for quantization model

3. [Segmentation sample](./segmentation/README.md)
    - Uses single `DeepLabV3` model from TensorFlow*
    - Implements `DataLoader` to load .JPEG images and annotations of Pascal VOC 2012 database
    - Implements `Metric` interface to calculate Mean Intersection Over Union metric
    - Uses DefaultQuantization algorithm for quantization model

4. [3D Segmentation sample](./3d_segmentation/README.md)
    - Uses single `Brain Tumor Segmentation` model from PyTorch*
    - Implements `DataLoader` to load images in NIfTI format from Medical Segmentation Decathlon BRATS 2017 database
    - Implements `Metric` interface to calculate Dice Index metric
    - Demonstrates how to use image metadata obtained during data loading to post-process the raw model output
    - Uses DefaultQuantization algorithm for quantization model

5. [Face Detection sample](./face_detection/README.md)
    - Uses cascaded (composite) `MTCNN` model from Caffe* that consists of three separate models in an OpenVino&trade; Intermediate Representation (IR)
    - Implements `Dataloader` to load .jpg images of WIDER FACE database
    - Implements `Metric` interface to calculate Recall metric
    - Implements `Engine` class that is inherited from `IEEngine` to create a complex staged pipeline to sequentially execute 
    each of the three stages of the MTCNN model, represented by multiple models in IR. It uses engine helpers to set model in 
    OpenVino&trade; Inference Engine and process raw model output for the correct statistics collection
    - Uses DefaultQuantization algorithm for quantization model

6. [GNA speech sample](./speech/README.md)
    - Uses models from Kaldi*
    - Implements `DataLoader` to data in .ark format
    - Uses DefaultQuantization algorithm for quantization model
   
After execution of each sample above the quantized model is placed into the folder `optimized`. The accuracy validation of the quantized model is performed right after the quantization. 
