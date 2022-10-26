# Post-training Optimization Tool API Examples {#pot_example_README}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:
   
   Quantizatiing Image Classification Model <pot_example_classification_README>
   Quantizatiing Object Detection Model with Accuracy Control  <pot_example_object_detection_README>
   Quantizatiing Cascaded Model <pot_example_face_detection_README>
   Quantizatiing Semantic Segmentation Model <pot_example_segmentation_README>
   Quantizatiing 3D Segmentation Model <pot_example_3d_segmentation_README>
   Quantizatiing for GNA Device <pot_example_speech_README>

@endsphinxdirective

The Post-training Optimization Tool contains multiple examples that demonstrate how to use its [API](@ref pot_compression_api_README) 
to optimize DL models. All available examples can be found on [GitHub](https://github.com/openvinotoolkit/openvino/tree/master/tools/pot/openvino/tools/pot/api/samples).

The following examples demonstrate the implementation of `Engine`, `Metric`, and `DataLoader` interfaces for various use cases:

1. [Quantizing Image Classification model](./classification/README.md)
    - Uses single `MobilenetV2` model from TensorFlow
    - Implements `DataLoader` to load .JPEG images and annotations of Imagenet database
    - Implements `Metric` interface to calculate Accuracy at top-1 metric
    - Uses DefaultQuantization algorithm for quantization model

2. [Quantizing Object Detection Model with Accuracy Control](./object_detection/README.md)
    - Uses single `MobileNetV1 FPN` model from TensorFlow
    - Implements `Dataloader` to load images of COCO database
    - Implements `Metric` interface to calculate mAP@[.5:.95] metric
    - Uses `AccuracyAwareQuantization` algorithm for quantization model

3. [Quantizing Semantic Segmentation Model](./segmentation/README.md)
    - Uses single `DeepLabV3` model from TensorFlow
    - Implements `DataLoader` to load .JPEG images and annotations of Pascal VOC 2012 database
    - Implements `Metric` interface to calculate Mean Intersection Over Union metric
    - Uses DefaultQuantization algorithm for quantization model

4. [Quantizing 3D Segmentation Model](./3d_segmentation/README.md)
    - Uses single `Brain Tumor Segmentation` model from PyTorch
    - Implements `DataLoader` to load images in NIfTI format from Medical Segmentation Decathlon BRATS 2017 database
    - Implements `Metric` interface to calculate Dice Index metric
    - Demonstrates how to use image metadata obtained during data loading to post-process the raw model output
    - Uses DefaultQuantization algorithm for quantization model

5. [Quantizing Cascaded model](./face_detection/README.md)
    - Uses cascaded (composite) `MTCNN` model from Caffe that consists of three separate models in an OpenVino&trade; Intermediate Representation (IR)
    - Implements `Dataloader` to load .jpg images of WIDER FACE database
    - Implements `Metric` interface to calculate Recall metric
    - Implements `Engine` class that is inherited from `IEEngine` to create a complex staged pipeline to sequentially execute 
    each of the three stages of the MTCNN model, represented by multiple models in IR. It uses engine helpers to set model in 
    OpenVino&trade; Inference Engine and process raw model output for the correct statistics collection
    - Uses DefaultQuantization algorithm for quantization model

6. [Quantizing for GNA Device](./speech/README.md)
    - Uses models from Kaldi
    - Implements `DataLoader` to data in .ark format
    - Uses DefaultQuantization algorithm for quantization model
   
After execution of each example above the quantized model is placed into the folder `optimized`. The accuracy validation of the quantized model is performed right after the quantization. 

## See the tutorials
* [Quantization of Image Classification model](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/301-tensorflow-training-openvino)
* [Quantization of Object Detection model from Model Zoo](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/111-detection-quantization)
* [Quantization of Segmentation model for medical data](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/110-ct-segmentation-quantize)
* [Quantization of BERT for Text Classification](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/105-language-quantize-bert)
