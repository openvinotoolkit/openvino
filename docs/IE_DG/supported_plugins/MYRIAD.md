# MYRIAD Plugin {#openvino_docs_IE_DG_supported_plugins_MYRIAD}

## Introducing MYRIAD Plugin

The Inference Engine MYRIAD plugin is developed for inference of neural networks on Intel&reg; Neural Compute Stick 2.

## Installation on Linux* OS

For installation instructions, refer to the [Installation Guide for Linux*](../../install_guides/installing-openvino-linux.md).


## Installation on Windows* OS

For installation instructions, refer to the [Installation Guide for Windows*](../../install_guides/installing-openvino-windows.md).

## Supported networks

The Inference Engine MYRIAD plugin supports the following networks:

**Caffe\***:
* AlexNet
* CaffeNet
* GoogleNet (Inception) v1, v2, v4
* VGG family (VGG16, VGG19)
* SqueezeNet v1.0, v1.1
* ResNet v1 family (18\*\*\*, 50, 101, 152)
* MobileNet (mobilenet-v1-1.0-224, mobilenet-v2)
* Inception ResNet v2
* DenseNet family (121,161,169,201)
* SSD-300, SSD-512, SSD-MobileNet, SSD-GoogleNet, SSD-SqueezeNet

**TensorFlow\***:
* AlexNet
* Inception v1, v2, v3, v4
* Inception ResNet v2
* MobileNet v1, v2
* ResNet v1 family (50, 101, 152)
* ResNet v2 family (50, 101, 152)
* SqueezeNet v1.0, v1.1
* VGG family (VGG16, VGG19)
* Yolo family (yolo-v2, yolo-v3, tiny-yolo-v1, tiny-yolo-v2, tiny-yolo-v3)
* faster_rcnn_inception_v2, faster_rcnn_resnet101
* ssd_mobilenet_v1
* DeepLab-v3+

**MXNet\***:
* AlexNet and CaffeNet
* DenseNet family (121,161,169,201)
* SqueezeNet v1.1
* MobileNet v1, v2
* NiN
* ResNet v1 (101, 152)
* ResNet v2 (101)
* SqueezeNet v1.1
* VGG family (VGG16, VGG19)
* SSD-Inception-v3, SSD-MobileNet, SSD-ResNet-50, SSD-300

\*\*\* Network is tested on Intel&reg; Neural Compute Stick 2 with BatchNormalization fusion optimization disabled during Model Optimizer import

## Supported Configuration Parameters

See VPU common configuration parameters for the [VPU Plugins](VPU.md).
When specifying key values as raw strings (that is, when using Python API), omit the `KEY_` prefix.

In addition to common parameters, the MYRIAD plugin accepts the following options:

| Parameter Name        | Parameter Values | Default    | Description                                                                        |
| :---                  | :---             | :---       | :---                                                                               |
| `KEY_VPU_MYRIAD_PLATFORM`    | empty string/`VPU_MYRIAD_2450`/`VPU_MYRIAD_2480` | empty string | If set, the plugin will use a device with specific platform to allocate a network. |
| `KEY_VPU_MYRIAD_PROTOCOL`    | empty string/`VPU_MYRIAD_USB`/`VPU_MYRIAD_PCIE` | empty string | If set, the plugin will use a device with specific protocol to allocate a network. |
| `KEY_VPU_MYRIAD_FORCE_RESET` | `YES`/`NO`                             | `NO`        | Enables force reset of all booted devices when new ExecutableNetwork is created.<br />This is a plugin scope option and must be used with the plugin's SetConfig method only.<br />See <a href="#MYRIAD_DEVICE_ALLOC">Device allocation</a> section for details. |
| `KEY_VPU_PLATFORM`           | empty string/`VPU_2450`/`VPU_2480`     | empty string | **Deprecated** Use `KEY_VPU_MYRIAD_PLATFORM` instead. <br />If set, the plugin will use a device with specific platform to allocate a network. |
| `KEY_VPU_FORCE_RESET`        | `YES`/`NO`                             | `NO`         | **Deprecated** Use `KEY_VPU_MYRIAD_FORCE_RESET` instead. <br />Enables force reset of all booted devices when new ExecutableNetwork is created.<br />This is a plugin scope option and must be used with the plugin's SetConfig method only.<br />See <a href="#MYRIAD_DEVICE_ALLOC">Device allocation</a> section for details. |

## Device allocation <a name="MYRIAD_DEVICE_ALLOC">&nbsp;</a>

Each `IExecutableNetwork` instance tries to allocate new device on `InferenceEngine::Core::LoadNetwork`, but if all available devices are already allocated it will use the one with the minimal number of uploaded networks.
The maximum number of networks single device can handle depends on device memory capacity and the size of the networks.

If `KEY_VPU_MYRIAD_FORCE_RESET` option is set to `YES` the plugin will reset all VPU devices in the system.

Single device cannot be shared across multiple processes.

## See Also

* [Supported Devices](Supported_Devices.md)
* [VPU Plugins](VPU.md)
* [Intel&reg; Neural Compute Stick 2 Get Started](https://software.intel.com/en-us/neural-compute-stick/get-started)
