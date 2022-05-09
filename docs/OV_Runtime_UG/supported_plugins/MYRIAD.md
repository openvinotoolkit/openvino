# MYRIAD Device {#openvino_docs_OV_UG_supported_plugins_MYRIAD}

## Introducing MYRIAD Plugin

The OpenVINO Runtime MYRIAD plugin has been developed for inference of neural networks on Intel&reg; Neural Compute Stick 2.

## Configuring the MYRIAD Plugin

To configure your Intel® Vision Accelerator Design With Intel® Movidius™ on supported operating systems, refer to the installation [guide](../../install_guides/installing-openvino-config-ivad-vpu).

> **NOTE**: The HDDL and MYRIAD plugins may cause conflicts when used at the same time.
> To ensure proper operation in such a case, the number of booted devices needs to be limited in the *`hddl_autoboot.config`* file.
> Otherwise, the HDDL plugin will boot all available Intel® Movidius™ Myriad™ X devices.

## Supported Configuration Parameters

See VPU common configuration parameters for the [VPU Plugins](VPU.md).
When specifying key values as raw strings (when using the Python API), omit the *`KEY_`* prefix.

In addition to common parameters, the MYRIAD plugin accepts the following options:

| Parameter Name        | Parameter Values | Default    | Description                                                                        |
| :---                  | :---             | :---       | :---                                                                               |
| *`KEY_VPU_MYRIAD_PROTOCOL`*    | empty string/*`VPU_MYRIAD_USB`*/*`VPU_MYRIAD_PCIE`* | empty string | If set, the plugin will use a device with specific protocol to allocate a network. |
| *`KEY_VPU_MYRIAD_FORCE_RESET`* | *`YES`*/*`NO`*                             | *`NO`*        | Enables force reset of all booted devices when new ExecutableNetwork is created.<br />This is a plugin scope option and must be used with the plugin's SetConfig method only.<br />See <a href="#MYRIAD_DEVICE_ALLOC">Device allocation</a> section for details. |
| *`KEY_VPU_FORCE_RESET`*        | *`YES`*/*`NO`*                             | *`NO`*         | **Deprecated** Use *`KEY_VPU_MYRIAD_FORCE_RESET`* instead. <br />Enables force reset of all booted devices when new ExecutableNetwork is created.<br />This is a plugin scope option and must be used with the plugin's SetConfig method only.<br />See <a href="#MYRIAD_DEVICE_ALLOC">Device allocation</a> section for details. |

## Device allocation <a name="MYRIAD_DEVICE_ALLOC">&nbsp;</a>

Each *`IExecutableNetwork`* instance tries to allocate new device on `InferenceEngine::Core::LoadNetwork`, but if all available devices are already allocated it will use the one with the minimal number of uploaded networks.
The maximum number of networks a single device can handle depends on device memory capacity and the size of the networks.

If the *`KEY_VPU_MYRIAD_FORCE_RESET`* option is set to *`YES`*, the plugin will reset all VPU devices in the system.

Single device cannot be shared across multiple processes.

## See Also

* [Supported Devices](Supported_Devices.md)
* [VPU Plugins](VPU.md)
* [Intel&reg; Neural Compute Stick 2 Get Started](https://software.intel.com/en-us/neural-compute-stick/get-started)
