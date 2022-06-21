# MYRIAD device {#openvino_docs_OV_UG_supported_plugins_MYRIAD}

## Introducing MYRIAD Plugin

The OpenVINO Runtime MYRIAD plugin has been developed for inference of neural networks on Intel&reg; Neural Compute Stick 2.

## Configuring the MYRIAD Plugin

To configure your Intel® Vision Accelerator Design With Intel® Movidius™ on supported operating systemss, refer to the Steps for Intel® Vision Accelerator Design with Intel® Movidius™ VPUs section in the installation guides for [Linux](../../install_guides/installing-openvino-linux.md) or [Windows](../../install_guides/installing-openvino-windows.md).

 > **NOTE**: The HDDL and MYRIAD plugins may cause conflicts when used at the same time.
> To ensure proper operation in such a case, the number of booted devices needs to be limited in the 'hddl_autoboot.config' file.
> Otherwise, the HDDL plugin will boot all available Intel® Movidius™ Myriad™ X devices.

## Supported Configuration Parameters

See VPU common configuration parameters for the [VPU Plugins](VPU.md).
When specifying key values as raw strings (that is, when using the Python API), omit the `KEY_` prefix.

In addition to common parameters, the MYRIAD plugin accepts the following options:

| Parameter Name        | Parameter Values | Default    | Description                                                                        |
| :---                  | :---             | :---       | :---                                                                               |
| `KEY_VPU_MYRIAD_PROTOCOL`    | empty string/`VPU_MYRIAD_USB`/`VPU_MYRIAD_PCIE` | empty string | If set, the plugin will use a device with specific protocol to allocate a network. |
| `KEY_VPU_MYRIAD_FORCE_RESET` | `YES`/`NO`                             | `NO`        | Enables force reset of all booted devices when new ExecutableNetwork is created.<br />This is a plugin scope option and must be used with the plugin's SetConfig method only.<br />See <a href="#MYRIAD_DEVICE_ALLOC">Device allocation</a> section for details. |
| `KEY_VPU_FORCE_RESET`        | `YES`/`NO`                             | `NO`         | **Deprecated** Use `KEY_VPU_MYRIAD_FORCE_RESET` instead. <br />Enables force reset of all booted devices when new ExecutableNetwork is created.<br />This is a plugin scope option and must be used with the plugin's SetConfig method only.<br />See <a href="#MYRIAD_DEVICE_ALLOC">Device allocation</a> section for details. |

## Device allocation <a name="MYRIAD_DEVICE_ALLOC">&nbsp;</a>

Each `IExecutableNetwork` instance tries to allocate new device on `InferenceEngine::Core::LoadNetwork`, but if all available devices are already allocated it will use the one with the minimal number of uploaded networks.
The maximum number of networks a single device can handle depends on device memory capacity and the size of the networks.

If the `KEY_VPU_MYRIAD_FORCE_RESET` option is set to `YES`, the plugin will reset all VPU devices in the system.

Single device cannot be shared across multiple processes.

## Intel® Movidius™ Myriad X firmwares from MDK R18

By default OpenVINO Runtime MYRIAD plugin is linked to the Intel® Movidius™ Myriad X firmwares generated from MDK R17 release which are automatically downloaded in the install process. But it is also compatible with the Intel® Movidius™ Myriad X firmwares generated from MDK R18 release (latest firmwares version) which are available on [Intel RDC](https://cdrdv2.intel.com/v1/dl/getContent/730726?explicitVersion=true).

The user have the possibility to download the Intel® Movidius™ Myriad X firmwares generated from MDK R18 relese from the source mentioned above and to manually replace the default firmwares in the directory where the OpenVINO Runtime was installed. The Intel® Movidius™ Myriad X firmwares consist in the following three files:

| FileName              | Setups           |
| :---                  | :---             |
| `usb-ma2x8x.mvcmd`    | Used to boot the Intel Movidius Myriad X VPU devices over USB connection for all supported operating systems.                   |
| `pcie-ma2x8x.mvcmd`   | Used to boot the Intel Movidius Myriad X VPU devices over PCIE connection for all supported operating systems excepting Windows. |
| `pcie-ma2x8x.elf`     | Used to boot the Intel Movidius Myriad X VPU devices over PCIE connection for Windows operating systems.                     |

Depending on the source used to install OpenVINO Runtime the paths to Intel® Movidius™ Myriad X firmwares are different.

For OpenVINO installed from Intel® Distribution of OpenVINO™ Toolkit these files are located to the following paths:

```
<install-dir>/<openvino-version>/runtime/3rdparty/hddl/lib/mvnc/usb-ma2x8x.mvcmd
<install-dir>/<openvino-version>/runtime/lib/intel64/usb-ma2x8x.mvcmd
<install-dir>/<openvino-version>/runtime/lib/intel64/pcie-ma2x8x.mvcmd
<install-dir>/<openvino-version>/runtime/lib/intel64/pcie-ma2x8x.elf
```

For OpenVINO Runtime installed from source code, the paths to Intel Movidius Myriad X OpenVINO firmwares are:

```
openvino/bin/intel64/<Release|Debug|RelWithDebInfo>/lib/usb-ma2x8x.mvcmd
openvino/bin/intel64/<Release|Debug|RelWithDebInfo>/lib/pcie-ma2x8x.mvcmd
openvino/bin/intel64/<Release|Debug|RelWithDebInfo>/lib/pcie-ma2x8x.elf
```

## See Also

* [Supported Devices](Supported_Devices.md)
* [VPU Plugins](VPU.md)
* [Intel&reg; Neural Compute Stick 2 Get Started](https://software.intel.com/en-us/neural-compute-stick/get-started)
