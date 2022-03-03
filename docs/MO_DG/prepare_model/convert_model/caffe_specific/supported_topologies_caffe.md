# Supported Caffe Topologies {#openvino_docs_MO_DG_prepare_model_convert_model_caffe_specific_supported_topologies}

* **Classification models:**
	* AlexNet
	* VGG-16, VGG-19
	* SqueezeNet v1.0, SqueezeNet v1.1
	* ResNet-50, ResNet-101, Res-Net-152
	* Inception v1, Inception v2, Inception v3, Inception v4
	* CaffeNet
	* MobileNet
	* Squeeze-and-Excitation Networks: SE-BN-Inception, SE-Resnet-101, SE-ResNet-152, SE-ResNet-50, SE-ResNeXt-101, SE-ResNeXt-50
	* ShuffleNet v2

* **Object detection models:**
	* SSD300-VGG16, SSD500-VGG16
	* Faster-RCNN
	* RefineDet (MYRIAD plugin only)

* **Face detection models:**
	* VGG Face
    * SSH: Single Stage Headless Face Detector

* **Semantic segmentation models:**
	* FCN8

> **NOTE**: It is necessary to specify mean and scale values for most of the Caffe\* models to convert them with the Model Optimizer. The exact values should be determined separately for each model. For example, for Caffe\* models trained on ImageNet, the mean values usually are `123.68`, `116.779`, `103.939` for blue, green and red channels respectively. The scale value is usually `127.5`. Refer to the **General Conversion Parameters** section in [Converting a Model to Intermediate Representation (IR)](../Converting_Model.md) for the information on how to specify mean and scale values.
