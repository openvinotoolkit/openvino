# Configuration Guide for the Intel® Distribution of OpenVINO™ toolkit and the Intel® Vision Accelerator Design with Intel® Movidius™ VPUs on Linux* {#openvino_docs_install_guides_installing_openvino_linux_ivad_vpu}

> **NOTES**: 
> - These steps are only required if you want to perform inference on Intel® Vision Accelerator Design with Intel® Movidius™ VPUs.
> - If you installed the Intel® Distribution of OpenVINO™ to the non-default install directory, replace `/opt/intel` with the directory in which you installed the software.


## Configuration Steps

For Intel® Vision Accelerator Design with Intel® Movidius™ VPUs, the following additional installation steps are required.

1. Set the environment variables:
```sh
source /opt/intel/openvino/bin/setupvars.sh
```
> **NOTE**: The `HDDL_INSTALL_DIR` variable is set to `<openvino_install_dir>/deployment_tools/inference_engine/external/hddl`. If you installed the Intel® Distribution of OpenVINO™ to the default install directory, the `HDDL_INSTALL_DIR` was set to `/opt/intel/openvino//deployment_tools/inference_engine/external/hddl`.

2. Install dependencies:
```sh
${HDDL_INSTALL_DIR}/install_IVAD_VPU_dependencies.sh
```
Note, if the Linux kernel is updated after the installation, it is required to install drivers again: 
```sh
cd ${HDDL_INSTALL_DIR}/drivers
```
```sh
sudo ./setup.sh install
```
Now the dependencies are installed and you are ready to use the Intel® Vision Accelerator Design with Intel® Movidius™ with the Intel® Distribution of OpenVINO™ toolkit.

## Optional Steps

* For advanced configuration steps for your **IEI Mustang-V100-MX8-R10** accelerator, see [Intel® Movidius™ VPUs Setup Guide for Use with Intel® Distribution of OpenVINO™ toolkit](movidius-setup-guide.md). **IEI Mustang-V100-MX8-R11** accelerator doesn't require any additional steps. 

* After you've configured your Intel® Vision Accelerator Design with Intel® Movidius™ VPUs, see [Intel® Movidius™ VPUs Programming Guide for Use with Intel® Distribution of OpenVINO™ toolkit](movidius-programming-guide.md) to learn how to distribute a model across all 8 VPUs to maximize performance.

## Troubleshooting

### Unable to run inference with the MYRIAD Plugin after running with the HDDL Plugin

Running inference with the MYRIAD Plugin after running with the HDDL Plugin is failed with the following error generated:

```sh
E: [ncAPI] [    965618] [MainThread] ncDeviceOpen:677   Failed to find a device, rc: X_LINK_ERROR
```

**Possible solutions (use one of the following):**

* Reboot the host system and run with the MYRIAD Plugin

* Kill the HDDL Plugin backend service (`hddldaemon`) and reset all Intel® Movidius™ VPUs before running an  application that uses the MYRIAD Plugin:
```sh
kill -9 $(pidof hddldaemon autoboot)
pidof hddldaemon autoboot # Make sure none of them is alive
source /opt/intel/openvino/bin/setupvars.sh
${HDDL_INSTALL_DIR}/bin/bsl_reset
```

---
### Get the "No space left on device" error while loading a network
When the application runs inference of a network with a big size(>4MB) of input/output or if the system is running out of the DMA buffer, 
the HDDL Plugin will fall back to use shared memory. 
In this case, if the application exits abnormally, the shared memory is not released automatically. 
To release it manually, remove files with the `hddl_` prefix from the `/dev/shm` directory:
```sh
sudo rm -f /dev/shm/hddl_*
```

---
### How to solve the permission issue?

Make sure that the following udev rules exist:
   - `/etc/udev/rules.d/97-myriad-usbboot.rules`
   - `/etc/udev/rules.d/98-hddlbsl.rules`
   - `/etc/udev/rules.d/99-hddl-ion.rules`
   - `/etc/udev/rules.d/99-myriad-vsc.rules`

Also make sure that the current user is included in the users groups. If not, run the command below to include:
```sh
sudo usermod -a -G users "$(whoami)"
```

---
### `setup.sh` doesn't install the driver binaries to `/lib/modules` on CentOS systems

As a temporary workaround, run the commands below to install the drivers. This issue will be fixed in future releases.

```sh
sudo mkdir -p /lib/modules/$(uname -r)/kernel/drivers/myd/
```
```sh
sudo cp drv_vsc/myd_vsc.ko /lib/modules/$(uname -r)/kernel/drivers/myd/
```
```sh
sudo cp drv_ion/myd_ion.ko /lib/modules/$(uname -r)/kernel/drivers/myd/
```
```sh
sudo touch /etc/modules-load.d/intel_vision_accelerator.conf
```
```sh
sudo echo "myd_vsc" >> /etc/modules-load.d/intel_vision_accelerator.conf
```
```sh
sudo echo "myd_ion" >> /etc/modules-load.d/intel_vision_accelerator.conf
```
```sh
sudo depmod
```
```sh
sudo modprobe myd_vsc
```
```sh
sudo modprobe myd_ion 
```

---
### Host machine reboots after running an inference application with the HDDL plugin

**Symptom:** Boot up the host machine, run the inference application with the HDDL plugin. System reboots in a uncertain time.

**Root Cause:** The I2C address of the reset device of the Intel® Vision Accelerator Design with Intel® Movidius™ VPUs conflicts with another device I2C address in 0x20-0x27 range. If the target Intel® Vision Accelerator Design with Intel® Movidius™ VPUs device needs to be reset (for example, in case of device errors), the `libbsl` library, which is responsible for reset, expects that the target reset device I2C address is in the 0x20-0x27 range on SMBUS. If there is another device on SMBUS in this address range, `libbsl` treats this device as the target reset device and writes an unexpected value to this address. This causes system reboot.

**Solution:** Detect if there is any I2C device on SMBUS with address in 0x20-0x27 range. If yes, do the following:

1. Change the DIP switch on the target PCIE card
2. Disable autoscan for the reset device by setting `"autoscan": false` in `${HDDL_INSTALL_DIR}/config/bsl.json`
3. Set the correct address of the I2C reset device (for example, `0x21`) in `${HDDL_INSTALL_DIR}/config/bsl.json`

```sh
{
  "autoscan": false,
  "ioexpander": {
    "enabled": true,
    "i2c_addr": [ 33 ]
  }
}
```

---
###Cannot reset VPU device and cannot find any 0x20-0x27 (Raw data card with HW version Fab-B and before) I2C addresses on SMBUS (using i2c-tools)

Please contact your motherboard vendor to verify SMBUS pins are connected to the PCIe slot.

---
### Get "Error: ipc_connection_linux_UDS : bind() failed" in hddldaemon log.

You may have run hddldaemon under another user. Run the command below and try again:
```sh
sudo rm -rf /var/tmp/hddl_*
```

---
### Get "I2C bus: SMBus I801 adapter at not found!" in hddldaemon log

Run the following command to check if a SMBUS I801 adapter can be found:
```sh
i2cdetect -l
```
Then run:
```sh
sudo modprobe i2c-i801
```
---
### Get "open /dev/ion failed!" in hddldaemon log

Check if `myd_ion` kernel module is installed by running the following command:
```sh
lsmod | grep myd_ion
```
If you do not see any output from the command, reinstall the `myd_ion` module.

---
### Constantly get "\_name\_mapping open failed err=2,No such file or directory" in hddldaemon log

Check if myd_vsc kernel module is installed by running the following command:
```sh
lsmod | grep myd_vsc
```
If you do not see any output from the command reinstall the `myd_vsc` module.

---
### Get "Required key not available" when trying to install the `myd_ion` or `myd_vsc` modules

Run the following commands:
```sh
sudo apt install mokutil
```
```sh
sudo mokutil --disable-validation
```
