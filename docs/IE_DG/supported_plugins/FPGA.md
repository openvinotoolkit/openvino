FPGA Plugin {#openvino_docs_IE_DG_supported_plugins_FPGA}
===========

## Introducing FPGA Plugin

The FPGA plugin provides an opportunity for high performance scoring of neural networks on Intel&reg; FPGA devices.

> **NOTE**: Before using the FPGA plugin, ensure that you have installed and configured either the Intel® Vision Accelerator Design with an Intel&reg; Arria&reg; 10 FPGA (Speed Grade 2) or the Intel&reg; Programmable Acceleration Card with Intel&reg; Arria&reg; 10 GX FPGA. For installation and configuration details, see [FPGA installation](Supported_Devices.md).

## Heterogeneous Execution

When your topology contains layers that are not supported by the Intel&reg; FPGA plugin, use [Heterogeneous plugin](HETERO.md) with dedicated fallback device.

If a network has layers that are not supported in the Intel&reg; FPGA plugin or in a fallback plugin, you can implement a custom layer on the CPU/GPU and use the [Extensibility mechanism](../Extensibility_DG/Intro.md).
In addition to adding custom kernels, you must still point to the CPU plugin or the GPU plugin as fallback devices for heterogeneous plugin.

## Supported Networks

The following network topologies are supported in heterogeneous mode, running on FPGA with fallback to CPU or GPU devices.

> **IMPORTANT**: Use only bitstreams from the current version of the OpenVINO toolkit. Bitstreams from older versions of the OpenVINO toolkit are incompatible with later versions of the OpenVINO toolkit. For example, you cannot use the `1-0-1_A10DK_FP16_Generic` bitstream, when the OpenVINO toolkit supports the `2019R2_PL2_FP16_InceptionV1_SqueezeNet_VGG_YoloV3.aocx` bitstream.


| Network                              | Bitstreams (Intel&reg; Vision Accelerator Design with an Intel&reg; Arria&reg; 10 FPGA (Speed Grade 2)) | Bitstreams (Intel&reg; Programmable Acceleration Card with Intel&reg; Arria&reg; 10 GX FPGA) |
|:-------------------------------------|:-------------------------------------------------------------------|:---------------------------------------------------------------------------------------------|
| AlexNet                              | 2020-4_PL2_FP16_AlexNet_GoogleNet_SSD300_Generic, 2020-4_PL2_FP11_AlexNet_GoogleNet_Generic | 2020-4_RC_FP16_AlexNet_GoogleNet_Generic, 2020-4_RC_FP11_AlexNet_GoogleNet_Generic |
| GoogleNet v1                         | 2020-4_PL2_FP16_AlexNet_GoogleNet_SSD300_Generic, 2020-4_PL2_FP11_AlexNet_GoogleNet_Generic | 2020-4_RC_FP16_AlexNet_GoogleNet_Generic, 2020-4_RC_FP11_AlexNet_GoogleNet_Generic |
| VGG-16                               | 2020-4_PL2_FP16_SqueezeNet_TinyYolo_VGG, 2020-4_PL2_FP11_InceptionV1_ResNet_VGG | 2020-4_RC_FP16_InceptionV1_SqueezeNet_TinyYolo_VGG, 2020-4_RC_FP16_ResNet_TinyYolo_VGG |
| VGG-19                               | 2020-4_PL2_FP16_SqueezeNet_TinyYolo_VGG, 2020-4_PL2_FP11_InceptionV1_ResNet_VGG | 2020-4_RC_FP16_InceptionV1_SqueezeNet_TinyYolo_VGG, 2020-4_RC_FP16_ResNet_TinyYolo_VGG |
| SqueezeNet v 1.0                     | 2020-4_PL2_FP16_SqueezeNet_TinyYolo_VGG, 2020-4_PL2_FP11_SqueezeNet | 2020-4_RC_FP16_InceptionV1_SqueezeNet_YoloV3, 2020-4_RC_FP16_InceptionV1_SqueezeNet_YoloV3 |
| SqueezeNet v 1.1                     | 2020-4_PL2_FP16_SqueezeNet_TinyYolo_VGG, 2020-4_PL2_FP11_SqueezeNet | 2020-4_RC_FP16_InceptionV1_SqueezeNet_YoloV3, 2020-4_RC_FP16_InceptionV1_SqueezeNet_YoloV3 |
| ResNet-18                            | 2020-4_PL2_FP16_ResNet_YoloV3, 2020-4_PL2_FP11_InceptionV1_ResNet_VGG | 2020-4_RC_FP16_ResNet_YoloV3, 2020-4_RC_FP16_ResNet_TinyYolo_VGG |
| ResNet-50                            | 2020-4_PL2_FP16_ResNet_YoloV3, 2020-4_PL2_FP11_InceptionV1_ResNet_VGG | 2020-4_RC_FP16_ResNet_YoloV3, 2020-4_RC_FP16_ResNet_TinyYolo_VGG |
| ResNet-101                           | 2020-4_PL2_FP16_ResNet_YoloV3, 2020-4_PL2_FP11_InceptionV1_ResNet_VGG | 2020-4_RC_FP16_ResNet_YoloV3, 2020-4_RC_FP16_ResNet_TinyYolo_VGG |
| ResNet-152                           | 2020-4_PL2_FP16_ResNet_YoloV3, 2020-4_PL2_FP11_InceptionV1_ResNet_VGG | 2020-4_RC_FP16_ResNet_YoloV3, 2020-4_RC_FP16_ResNet_TinyYolo_VGG |
| MobileNet (Caffe)                    | 2020-4_PL2_FP16_MobileNet_Clamp, 2020-4_PL2_FP11_MobileNet_Clamp | 2020-4_RC_FP16_MobileNet_Clamp, 2020-4_RC_FP11_MobileNet_Clamp |
| MobileNet (TensorFlow)               | 2020-4_PL2_FP16_MobileNet_Clamp, 2020-4_PL2_FP11_MobileNet_Clamp | 2020-4_RC_FP16_MobileNet_Clamp, 2020-4_RC_FP11_MobileNet_Clamp|
| SqueezeNet-based variant of the SSD* | 2020-4_PL2_FP16_SqueezeNet_TinyYolo_VGG, 2020-4_PL2_FP11_SqueezeNet | 2020-4_RC_FP16_InceptionV1_SqueezeNet_TinyYolo_VGG, 2020-4_RC_FP16_InceptionV1_SqueezeNet_YoloV3 |
| ResNet-based variant of SSD          | 2020-4_PL2_FP16_ResNet_YoloV3, 2020-4_PL2_FP11_InceptionV1_ResNet_VGG | 2020-4_RC_FP16_ResNet_YoloV3, 2020-4_RC_FP16_ResNet_TinyYolo_VGG |
| RMNet                                | 2020-4_PL2_FP16_RMNet, 2020-4_PL2_FP11_RMNet | 2020-4_RC_FP16_RMNet, 2020-4_RC_FP11_RMNet |
| Yolo v3                              | 2020-4_PL2_FP16_ResNet_YoloV3, 2020-4_PL2_FP11_YoloV3_ELU | 2020-4_RC_FP16_ResNet_YoloV3, 2020-4_RC_FP16_InceptionV1_SqueezeNet_YoloV3 |


In addition to the list above, arbitrary topologies having big continues subgraphs consisting of layers supported by FPGA plugin are recommended to be executed on FPGA plugin.

## Bitstreams that are Optimal to Use with the Intel's Pre-Trained Models

The table below provides you with a list of Intel&reg; Vision Accelerator Design with an Intel&reg; Arria&reg; 10 FPGA (Speed Grade 2) bitstreams that are optimal to use for the Intel's pre-trained models.  

<details>
  <summary><strong>Click to expand/collapse the table</strong></summary>

| Model Name | FP11 Bitstreams | FP16 Bitstreams |
| :---       | :---            | :---            |
| action-recognition-0001-decoder | 2020-3_PL2_FP11_AlexNet_GoogleNet_Generic.aocx | 2020-3_PL2_FP16_SwishExcitation.aocx |
| action-recognition-0001-encoder | 2020-3_PL2_FP11_InceptionV1_ResNet_VGG.aocx | 2020-3_PL2_FP16_ResNet_YoloV3.aocx |
| age-gender-recognition-retail-0013 | 2020-3_PL2_FP11_AlexNet_GoogleNet_Generic.aocx | 2020-3_PL2_FP16_MobileNet_Clamp.aocx |
| asl-recognition-0004 | 2020-3_PL2_FP11_AlexNet_GoogleNet_Generic.aocx | 2020-3_PL2_FP16_AlexNet_GoogleNet_SSD300_Generic.aocx |
| driver-action-recognition-adas-0002-decoder | 2020-3_PL2_FP11_AlexNet_GoogleNet_Generic.aocx | 2020-3_PL2_FP16_MobileNet_Clamp.aocx |
| driver-action-recognition-adas-0002-encoder | 2020-3_PL2_FP11_YoloV3_ELU.aocx | 2020-3_PL2_FP16_SwishExcitation.aocx |
| emotions-recognition-retail-0003 | 2020-3_PL2_FP11_AlexNet_GoogleNet_Generic.aocx | 2020-3_PL2_FP16_SqueezeNet_TinyYolo_VGG.aocx |
| face-detection-0100 | 2020-3_PL2_FP11_MobileNet_Clamp.aocx | 2020-3_PL2_FP16_MobileNet_Clamp.aocx |
| face-detection-0102 | 2020-3_PL2_FP11_MobileNet_Clamp.aocx | 2020-3_PL2_FP16_MobileNet_Clamp.aocx |
| face-detection-0104 | 2020-3_PL2_FP11_MobileNet_Clamp.aocx | 2020-3_PL2_FP16_MobileNet_Clamp.aocx |
| face-detection-0105 | 2020-3_PL2_FP11_MobileNet_Clamp.aocx | 2020-3_PL2_FP16_MobileNet_Clamp.aocx |
| face-detection-0106 | 2020-3_PL2_FP11_InceptionV1_ResNet_VGG.aocx | 2020-3_PL2_FP16_ResNet_YoloV3.aocx |
| face-detection-adas-0001 | 2020-3_PL2_FP11_YoloV3_ELU.aocx | 2020-3_PL2_FP16_SwishExcitation.aocx |
| face-detection-adas-binary-0001 | 2020-3_PL2_FP11_AlexNet_GoogleNet_Generic.aocx | 2020-3_PL2_FP16_AlexNet_GoogleNet_SSD300_Generic.aocx |
| face-detection-retail-0004 | 2020-3_PL2_FP11_TinyYolo_SSD300.aocx | 2020-3_PL2_FP16_AlexNet_GoogleNet_SSD300_Generic.aocx |
| face-detection-retail-0005 | 2020-3_PL2_FP11_YoloV3_ELU.aocx | 2020-3_PL2_FP16_MobileNet_Clamp.aocx |
| face-reidentification-retail-0095 | 2020-3_PL2_FP11_SqueezeNet.aocx | 2020-3_PL2_FP16_MobileNet_Clamp.aocx |
| facial-landmarks-35-adas-0002 | 2020-3_PL2_FP11_AlexNet_GoogleNet_Generic.aocx | 2020-3_PL2_FP16_MobileNet_Clamp.aocx |
| faster-rcnn-resnet101-coco-sparse-60-0001 | 2020-3_PL2_FP11_InceptionV1_ResNet_VGG.aocx | 2020-3_PL2_FP16_SwishExcitation.aocx |
| gaze-estimation-adas-0002 | 2020-3_PL2_FP11_SqueezeNet.aocx | 2020-3_PL2_FP16_AlexNet_GoogleNet_SSD300_Generic.aocx |
| handwritten-japanese-recognition-0001 | 2020-3_PL2_FP11_InceptionV1_ResNet_VGG.aocx | 2020-3_PL2_FP16_ResNet_YoloV3.aocx |
| handwritten-score-recognition-0003 | 2020-3_PL2_FP11_InceptionV1_ResNet_VGG.aocx | 2020-3_PL2_FP16_SqueezeNet_TinyYolo_VGG.aocx |
| head-pose-estimation-adas-0001 | 2020-3_PL2_FP11_SqueezeNet.aocx | 2020-3_PL2_FP16_AlexNet_GoogleNet_SSD300_Generic.aocx |
| human-pose-estimation-0001 | 2020-3_PL2_FP11_YoloV3_ELU.aocx | 2020-3_PL2_FP16_SwishExcitation.aocx |
| icnet-camvid-ava-0001 | 2020-3_PL2_FP11_AlexNet_GoogleNet_Generic.aocx | 2020-3_PL2_FP16_AlexNet_GoogleNet_SSD300_Generic.aocx |
| icnet-camvid-ava-sparse-30-0001 | 2020-3_PL2_FP11_SqueezeNet.aocx | 2020-3_PL2_FP16_SwishExcitation.aocx |
| icnet-camvid-ava-sparse-60-0001 | 2020-3_PL2_FP11_SqueezeNet.aocx | 2020-3_PL2_FP16_SwishExcitation.aocx |
| image-retrieval-0001 | 2020-3_PL2_FP11_MobileNet_Clamp.aocx | 2020-3_PL2_FP16_MobileNet_Clamp.aocx |
| instance-segmentation-security-0010 | 2020-3_PL2_FP11_InceptionV1_ResNet_VGG.aocx | 2020-3_PL2_FP16_SqueezeNet_TinyYolo_VGG.aocx |
| instance-segmentation-security-0050 | 2020-3_PL2_FP11_InceptionV1_ResNet_VGG.aocx | 2020-3_PL2_FP16_ResNet_YoloV3.aocx |
| instance-segmentation-security-0083 | 2020-3_PL2_FP11_AlexNet_GoogleNet_Generic.aocx | 2020-3_PL2_FP16_AlexNet_GoogleNet_SSD300_Generic.aocx |
| instance-segmentation-security-1025 | 2020-3_PL2_FP11_InceptionV1_ResNet_VGG.aocx | 2020-3_PL2_FP16_AlexNet_GoogleNet_SSD300_Generic.aocx |
| landmarks-regression-retail-0009 | 2020-3_PL2_FP11_SqueezeNet.aocx | 2020-3_PL2_FP16_AlexNet_GoogleNet_SSD300_Generic.aocx |
| license-plate-recognition-barrier-0001 | 2020-3_PL2_FP11_SqueezeNet.aocx | 2020-3_PL2_FP16_SqueezeNet_TinyYolo_VGG.aocx |
| pedestrian-and-vehicle-detector-adas-0001 | 2020-3_PL2_FP11_YoloV3_ELU.aocx | 2020-3_PL2_FP16_SwishExcitation.aocx |
| pedestrian-detection-adas-0002 | 2020-3_PL2_FP11_YoloV3_ELU.aocx | 2020-3_PL2_FP16_SwishExcitation.aocx |
| pedestrian-detection-adas-binary-0001 | 2020-3_PL2_FP11_AlexNet_GoogleNet_Generic.aocx | 2020-3_PL2_FP16_AlexNet_GoogleNet_SSD300_Generic.aocx |
| person-attributes-recognition-crossroad-0230 | 2020-3_PL2_FP11_SqueezeNet.aocx | 2020-3_PL2_FP16_MobileNet_Clamp.aocx |
| person-detection-action-recognition-0005 | 2020-3_PL2_FP11_YoloV3_ELU.aocx | 2020-3_PL2_FP16_MobileNet_Clamp.aocx |
| person-detection-action-recognition-0006 | 2020-3_PL2_FP11_YoloV3_ELU.aocx | 2020-3_PL2_FP16_MobileNet_Clamp.aocx |
| person-detection-action-recognition-teacher-0002 | 2020-3_PL2_FP11_YoloV3_ELU.aocx | 2020-3_PL2_FP16_MobileNet_Clamp.aocx |
| person-detection-asl-0001 | 2020-3_PL2_FP11_AlexNet_GoogleNet_Generic.aocx | 2020-3_PL2_FP16_AlexNet_GoogleNet_SSD300_Generic.aocx |
| person-detection-raisinghand-recognition-0001 | 2020-3_PL2_FP11_YoloV3_ELU.aocx | 2020-3_PL2_FP16_MobileNet_Clamp.aocx |
| person-detection-retail-0002 | 2020-3_PL2_FP11_InceptionV1_ResNet_VGG.aocx | 2020-3_PL2_FP16_MobileNet_Clamp.aocx |
| person-detection-retail-0013 | 2020-3_PL2_FP11_MobileNet_Clamp.aocx | 2020-3_PL2_FP16_MobileNet_Clamp.aocx |
| person-reidentification-retail-0031 | 2020-3_PL2_FP11_YoloV3_ELU.aocx | 2020-3_PL2_FP16_ELU.aocx |
| person-reidentification-retail-0248 | 2020-3_PL2_FP11_MobileNet_Clamp.aocx | 2020-3_PL2_FP16_MobileNet_Clamp.aocx |
| person-reidentification-retail-0249 | 2020-3_PL2_FP11_MobileNet_Clamp.aocx | 2020-3_PL2_FP16_MobileNet_Clamp.aocx |
| person-reidentification-retail-0300 | 2020-3_PL2_FP11_InceptionV1_ResNet_VGG.aocx | 2020-3_PL2_FP16_SwishExcitation.aocx |
| person-vehicle-bike-detection-crossroad-0078 | 2020-3_PL2_FP11_YoloV3_ELU.aocx | 2020-3_PL2_FP16_ELU.aocx |
| person-vehicle-bike-detection-crossroad-1016 | 2020-3_PL2_FP11_MobileNet_Clamp.aocx | 2020-3_PL2_FP16_SwishExcitation.aocx |
| product-detection-0001 | 2020-3_PL2_FP11_MobileNet_Clamp.aocx | 2020-3_PL2_FP16_MobileNet_Clamp.aocx |
| resnet18-xnor-binary-onnx-0001 | 2020-3_PL2_FP11_AlexNet_GoogleNet_Generic.aocx | 2020-3_PL2_FP16_RMNet.aocx |
| resnet50-binary-0001 | 2020-3_PL2_FP11_AlexNet_GoogleNet_Generic.aocx | 2020-3_PL2_FP16_AlexNet_GoogleNet_SSD300_Generic.aocx |
| road-segmentation-adas-0001 | 2020-3_PL2_FP11_MobileNet_Clamp.aocx | 2020-3_PL2_FP16_MobileNet_Clamp.aocx |
| semantic-segmentation-adas-0001 | 2020-3_PL2_FP11_MobileNet_Clamp.aocx | 2020-3_PL2_FP16_AlexNet_GoogleNet_SSD300_Generic.aocx |
| single-image-super-resolution-1032 | 2020-3_PL2_FP11_MobileNet_Clamp.aocx | 2020-3_PL2_FP16_RMNet.aocx |
| single-image-super-resolution-1033 | 2020-3_PL2_FP11_MobileNet_Clamp.aocx | 2020-3_PL2_FP16_RMNet.aocx |
| text-detection-0003 | 2020-3_PL2_FP11_MobileNet_Clamp.aocx | 2020-3_PL2_FP16_MobileNet_Clamp.aocx |
| text-detection-0004 | 2020-3_PL2_FP11_MobileNet_Clamp.aocx | 2020-3_PL2_FP16_MobileNet_Clamp.aocx |
| text-image-super-resolution-0001 | 2020-3_PL2_FP11_MobileNet_Clamp.aocx | 2020-3_PL2_FP16_RMNet.aocx |
| text-recognition-0012 | 2020-3_PL2_FP11_AlexNet_GoogleNet_Generic.aocx | 2020-3_PL2_FP16_AlexNet_GoogleNet_SSD300_Generic.aocx |
| text-spotting-0002-detector | 2020-3_PL2_FP11_InceptionV1_ResNet_VGG.aocx | 2020-3_PL2_FP16_ResNet_YoloV3.aocx |
| text-spotting-0002-recognizer-decoder | 2020-3_PL2_FP11_AlexNet_GoogleNet_Generic.aocx | 2020-3_PL2_FP16_AlexNet_GoogleNet_SSD300_Generic.aocx |
| text-spotting-0002-recognizer-encoder | 2020-3_PL2_FP11_InceptionV1_ResNet_VGG.aocx | 2020-3_PL2_FP16_SqueezeNet_TinyYolo_VGG.aocx |
| unet-camvid-onnx-0001 | 2020-3_PL2_FP11_InceptionV1_ResNet_VGG.aocx | 2020-3_PL2_FP16_SwishExcitation.aocx |
| vehicle-attributes-recognition-barrier-0039 | 2020-3_PL2_FP11_SqueezeNet.aocx | 2020-3_PL2_FP16_SqueezeNet_TinyYolo_VGG.aocx |
| vehicle-detection-adas-0002 | 2020-3_PL2_FP11_YoloV3_ELU.aocx | 2020-3_PL2_FP16_SwishExcitation.aocx |
| vehicle-detection-adas-binary-0001 | 2020-3_PL2_FP11_AlexNet_GoogleNet_Generic.aocx | 2020-3_PL2_FP16_AlexNet_GoogleNet_SSD300_Generic.aocx |
| vehicle-license-plate-detection-barrier-0106 | 2020-3_PL2_FP11_MobileNet_Clamp.aocx | 2020-3_PL2_FP16_MobileNet_Clamp.aocx |
| yolo-v2-ava-0001 | 2020-3_PL2_FP11_SqueezeNet.aocx | 2020-3_PL2_FP16_SqueezeNet_TinyYolo_VGG.aocx |
| yolo-v2-ava-sparse-35-0001 | 2020-3_PL2_FP11_SqueezeNet.aocx | 2020-3_PL2_FP16_SqueezeNet_TinyYolo_VGG.aocx |
| yolo-v2-ava-sparse-70-0001 | 2020-3_PL2_FP11_SqueezeNet.aocx | 2020-3_PL2_FP16_SqueezeNet_TinyYolo_VGG.aocx |
| yolo-v2-tiny-ava-0001 | 2020-3_PL2_FP11_SqueezeNet.aocx | 2020-3_PL2_FP16_ResNet_YoloV3.aocx |
| yolo-v2-tiny-ava-sparse-30-0001 | 2020-3_PL2_FP11_SqueezeNet.aocx | 2020-3_PL2_FP16_ResNet_YoloV3.aocx |
| yolo-v2-tiny-ava-sparse-60-0001 | 2020-3_PL2_FP11_SqueezeNet.aocx | 2020-3_PL2_FP16_ResNet_YoloV3.aocx |

</details>

## <a name="TranslatingArchtoBitstream"></a>Translate from Architecture to FPGA Bitstream Files

Various FPGA bitstreams that support CNN are available in the OpenVINO&trade; toolkit package for FPGA.

To select the correct bitstream (`.aocx`) file for an architecture, select a network (for example, Resnet-18) from the table above for either the Intel® Vision Accelerator Design with an Intel® Arria 10 FPGA (Speed Grade 1), Intel® Vision Accelerator Design with an Intel® Arria 10 FPGA (Speed Grade 2) or the Intel&reg; Programmable Acceleration Card (PAC) with Intel&reg; Arria&reg; 10 GX FPGA and note the corresponding architecture.

The following table describes several parameters that might help you to select the proper bitstream for your needs:

| Name                                           | Board                                                                           | Precision | LRN Support | Leaky ReLU Support | PReLU Support | Clamp Support | ELU Support |
|:------------------------------------------|:--------------------------------------------------------------------------------|:----------|:------------|:-------------------|:--------------|:--------------|:------------|
| 2020-4_PL2_FP11_AlexNet_GoogleNet_Generic | Intel&reg; Vision Accelerator Design with an Intel&reg; Arria&reg; 10 FPGA (Speed Grade 2) | FP11 | true | true | true | false | false |
| 2020-4_PL2_FP11_SqueezeNet | Intel&reg; Vision Accelerator Design with an Intel&reg; Arria&reg; 10 FPGA (Speed Grade 2) | FP11 | false | true | true | false | false |
| 2020-4_PL2_FP11_MobileNet_Clamp | Intel&reg; Vision Accelerator Design with an Intel&reg; Arria&reg; 10 FPGA (Speed Grade 2) | FP11 | false | true | true | true | false |
| 2020-4_PL2_FP11_InceptionV1_ResNet_VGG | Intel&reg; Vision Accelerator Design with an Intel&reg; Arria&reg; 10 FPGA (Speed Grade 2) | FP11 | false | false | false | false | false |
| 2020-4_PL2_FP11_RMNet | Intel&reg; Vision Accelerator Design with an Intel&reg; Arria&reg; 10 FPGA (Speed Grade 2) | FP11 | false | true | true | false | true |
| 2020-4_PL2_FP11_TinyYolo_SSD300 | Intel&reg; Vision Accelerator Design with an Intel&reg; Arria&reg; 10 FPGA (Speed Grade 2) | FP11 | true | true | true | false | false |
| 2020-4_PL2_FP11_YoloV3_ELU | Intel&reg; Vision Accelerator Design with an Intel&reg; Arria&reg; 10 FPGA (Speed Grade 2) | FP11 | false | true | true | false | true |
| 2020-4_PL2_FP11_Streaming_InternalUseOnly | Intel&reg; Vision Accelerator Design with an Intel&reg; Arria&reg; 10 FPGA (Speed Grade 2) | FP11 | false | false | false | false | false |
| 2020-4_PL2_FP11_Streaming_Slicing_InternalUseOnly | Intel&reg; Vision Accelerator Design with an Intel&reg; Arria&reg; 10 FPGA (Speed Grade 2) | FP11 | false | false | false | false | false |
| 2020-4_PL2_FP11_SwishExcitation | Intel&reg; Vision Accelerator Design with an Intel&reg; Arria&reg; 10 FPGA (Speed Grade 2) | FP11 | false | false | false | false | false |
| 2020-4_PL2_FP16_AlexNet_GoogleNet_SSD300_Generic | Intel&reg; Vision Accelerator Design with an Intel&reg; Arria&reg; 10 FPGA (Speed Grade 2) | FP16 | true | true | true | false | false |
| 2020-4_PL2_FP16_ELU | Intel&reg; Vision Accelerator Design with an Intel&reg; Arria&reg; 10 FPGA (Speed Grade 2) | FP16 | false | true | true | false | true |
| 2020-4_PL2_FP16_MobileNet_Clamp | Intel&reg; Vision Accelerator Design with an Intel&reg; Arria&reg; 10 FPGA (Speed Grade 2) | FP16 | false | true | true | true | false |
| 2020-4_PL2_FP16_ResNet_YoloV3 | Intel&reg; Vision Accelerator Design with an Intel&reg; Arria&reg; 10 FPGA (Speed Grade 2) | FP16 | false | true | true | false | false |
| 2020-4_PL2_FP16_RMNet | Intel&reg; Vision Accelerator Design with an Intel&reg; Arria&reg; 10 FPGA (Speed Grade 2) | FP16 | false | true | true | false | true |
| 2020-4_PL2_FP16_SqueezeNet_TinyYolo_VGG | Intel&reg; Vision Accelerator Design with an Intel&reg; Arria&reg; 10 FPGA (Speed Grade 2) | FP16 | false | true | true | false | false |
| 2020-4_PL2_FP16_SqueezeNet_TinyYolo_VGG | Intel&reg; Vision Accelerator Design with an Intel&reg; Arria&reg; 10 FPGA (Speed Grade 2) | FP16 | false | false | false | false | false |
| 2020-4_RC_FP11_AlexNet_GoogleNet_Generic | Intel&reg; Programmable Acceleration Card with Intel&reg; Arria&reg; 10 GX FPGA | FP11 | true | true | true | false | false |
| 2020-4_RC_FP11_RMNet | Intel&reg; Programmable Acceleration Card with Intel&reg; Arria&reg; 10 GX FPGA | FP11 | false | true | true | false | true |
| 2020-4_RC_FP11_Streaming_InternalUseOnly | Intel&reg; Programmable Acceleration Card with Intel&reg; Arria&reg; 10 GX FPGA | FP11 | true | false | false | false | false |
| 2020-4_RC_FP11_Streaming_Slicing_InternalUseOnly | Intel&reg; Programmable Acceleration Card with Intel&reg; Arria&reg; 10 GX FPGA | FP11 | true | false | false | false | false |
| 2020-4_RC_FP11_ELU | Intel&reg; Programmable Acceleration Card with Intel&reg; Arria&reg; 10 GX FPGA | FP11 | false | true | true | false | true |
| 2020-4_RC_FP11_SwishExcitation | Intel&reg; Programmable Acceleration Card with Intel&reg; Arria&reg; 10 GX FPGA | FP11 | false | false | false | false | false |
| 2020-4_RC_FP11_InceptionV1_ResNet_SqueezeNet_TinyYolo_YoloV3 | Intel&reg; Programmable Acceleration Card with Intel&reg; Arria&reg; 10 GX FPGA | FP11 | false | true | true | false | false |
| 2020-4_RC_FP11_MobileNet_Clamp | Intel&reg; Programmable Acceleration Card with Intel&reg; Arria&reg; 10 GX FPGA | FP11 | false | true | true | true | false |
| 2020-4_RC_FP16_AlexNet_GoogleNet_Generic | Intel&reg; Programmable Acceleration Card with Intel&reg; Arria&reg; 10 GX FPGA | FP16 | true | true | true | false | false |
| 2020-4_RC_FP16_InceptionV1_SqueezeNet_TinyYolo_VGG | Intel&reg; Programmable Acceleration Card with Intel&reg; Arria&reg; 10 GX FPGA | FP16 | false | true | true | false | false |
| 2020-4_RC_FP16_RMNet | Intel&reg; Programmable Acceleration Card with Intel&reg; Arria&reg; 10 GX FPGA | FP16 | false | true | true | false | true |
| 2020-4_RC_FP16_SwishExcitation | Intel&reg; Programmable Acceleration Card with Intel&reg; Arria&reg; 10 GX FPGA | FP16 | false | false | false | false | false |
| 2020-4_RC_FP16_MobileNet_Clamp | Intel&reg; Programmable Acceleration Card with Intel&reg; Arria&reg; 10 GX FPGA | FP16 | false | true | true | true | false |
| 2020-4_RC_FP16_ResNet_YoloV3 | Intel&reg; Programmable Acceleration Card with Intel&reg; Arria&reg; 10 GX FPGA | FP16 | false | true | true | false | false |
| 2020-4_RC_FP16_InceptionV1_SqueezeNet_YoloV3 | Intel&reg; Programmable Acceleration Card with Intel&reg; Arria&reg; 10 GX FPGA | FP16 | false | true | true | false | false |

## Set Environment for Running the FPGA Plugin

To make the FPGA plugin run directly or through the heterogeneous plugin, set up the environment:
1. Set up environment to access Intel&reg; FPGA RTE for OpenCL:
```
source /opt/altera/aocl-pro-rte/aclrte-linux64/init_opencl.sh
```
2. Set the following environment variable and program the board with a DLA bitstream. Programming of the board is not supported during runtime and must be done before running an application.

  | Variable                           | Setting                                                                                                                                                                                                                                                                                                                                                                                                                                       |
  | :----------------------------------| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | ACL_PCIE_USE_JTAG_PROGRAMMING      | Set this variable to a value of 1 to force FPGA reprogramming using JTAG                                                                                                                                                                                                                                                                                                                                                                      |

## Analyzing Heterogeneous Execution

Besides generation of .dot files, you can use the error listening mechanism:

```cpp
class FPGA_ErrorListener : public InferenceEngine::IErrorListener
{
public:
    virtual void onError(const char *msg) noexcept override {
        std::cout << msg;
    }
};
...
FPGA_ErrorListener err_listener;
core.SetLogCallback(err_listener); // will be used for FPGA device as well
```
If during network loading some layers are decided to be executed on a fallback plugin, the following message is printed:

```cpp
Layer (Name: detection_out, Type: DetectionOutput) is not supported:
	custom or unknown.
	Has (3) sets of inputs, must be 1, or 2.
	Input dimensions (2) should be 4.
```

## Multiple FPGA Devices Support

The Inference Engine FPGA plugin provides an ability to load different networks on multiple FPGA devices. For example, to load two networks AlexNet and MobileNet v2 on two different FPGA devices, follow the steps below:

1. Program each FGPA device with a corresponding bitstream:
```bash
aocl program acl0 2019R3_PV_PL1_FP16_AlexNet_GoogleNet_InceptionV1_SSD300_Generic.aocx
```
```bash
aocl program acl1 2019R3_PV_PL1_FP16_MobileNet_Clamp.aocx
```
For more information about bitstream programming instructions, refer to [Installation Guide for Linux* with Support for FPGA](Supported_Devices.md)
2. All FPGA devices are enumerated with unique ID starting from `0`. By default, all networks are loaded to the default
device with ID `0`. If you want to load a network on a particular non-default device, specify the `KEY_DEVICE_ID`
parameter for C++ and `DEVICE_ID` parameter for Python\*.
The following code snippets demonstrates how to load the AlexNet network on the FPGA device with ID `0` and the
MobileNet v2 network on the device with ID `1`:
    * With C++:
```cpp
InferenceEngine::Core core;

// Load AlexNet network on the first FPGA device programmed with bitstream supporting AlexNet
auto alexnetNetwork = core.ReadNetwork("alexnet.xml");
auto exeNetwork1 = core.LoadNetwork(alexnetNetwork, "FPGA.0");

// Load MobileNet network on the second FPGA device programmed with MobileNet bitstream
auto mobilenetNetwork = core.ReadNetwork("mobilenet_v2.xml");
auto exeNetwork2 = core.LoadNetwork(mobilenetNetwork, "FPGA", { { KEY_DEVICE_ID, "1" } });
```
    * With Python:
```python
# Load AlexNet network on the first FPGA device programmed with bitstream supporting AlexNet
net1 = IENetwork(model="alexnet.xml", weights="alexnet.bin")
plugin.load(network=net1, config={"DEVICE_ID": "0"})

# Load MobileNet network on the second FPGA device programmed with MobileNet bitstream
net2 = IENetwork(model="mobilenet_v2.xml", weights="mobilenet_v2.bin")
plugin.load(network=net2, config={"DEVICE_ID": "1"})
```
Note that you have to use asynchronous infer requests to utilize several FPGA devices, otherwise the execution on devices is performed sequentially.

## Import and Export Network Flow

Since the 2019 R4 release, FPGA and HETERO plugins support the export and import flow, which allows to export a compiled network from a plugin to a binary blob by running the command below:

```bash
$ ./compile_tool -m resnet.xml -DLA_ARCH_NAME 4x2x16x32_fp16_sb9408_fcd1024_actk4_poolk4_normk1_owk2_image300x300x8192_mbfr -d HETERO:FPGA,CPU
Inference Engine: 
	API version ............ 2.1
	Build .................. 6db44e09a795cb277a63275ea1395bfcb88e46ac
	Description ....... API
Done
```

Once the command is executed, the binary blob named `resnet.blob` is created at the working directory. Refer to the [Compile tool](../../../inference-engine/tools/compile_tool/README.md) documentation for more details.

A compiled binary blob can be later imported via `InferenceEngine::Core::Import`:

```cpp
InferenceEngine::Core core;
std::ifstream strm("resnet.blob");
auto execNetwork = core.Import(strm);
```

## How to Interpret Performance Counters

As a result of collecting performance counters using <code>InferenceEngine::InferRequest::GetPerformanceCounts</code> you can find out performance data about execution on FPGA, pre-processing and post-processing data and data transferring from/to FPGA card.

If network is sliced to two parts that are executed on CPU, you can find performance data about Intel&reg; MKL-DNN kernels, their types, and other useful information.

## Limitations of the FPGA Support for CNN

The Inference Engine FPGA plugin has limitations on network topologies, kernel parameters, and batch size.

* Depending on the bitstream loaded on the target device, the FPGA performs calculations with precision rates ranging from FP11 to FP16. This might have accuracy implications. Use the [Accuracy Checker](@ref omz_tools_accuracy_checker_README) to verify the network accuracy on the validation data set.
* Networks that have many CNN layers that are not supported on FPGA stayed in topologies between supported layers might lead to dividing of graph to many subgraphs that might lead to `CL_OUT_OF_HOST_MEMORY` error. These topologies are not FPGA friendly for this release.
* When you use the heterogeneous plugin, the affinity and distribution of nodes by devices depends on the FPGA bitstream that you use. Some layers might not be supported by a bitstream or parameters of the layer are not supported by the bitstream.

## See Also
* [Supported Devices](Supported_Devices.md)
