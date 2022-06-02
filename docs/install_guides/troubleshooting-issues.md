# Issues & Solutions for OpenVINO™ Installation & Configuration {#openvino_docs_get_started_guide_troubleshooting_issues}

This page lists issues that you may encounter during the installation and configuration of OpenVINO™, as well as their possible solutions.

<!-- this part was from Docker installation -->

## <a name="install-for-prc"></a>Errors with Installing via PIP for PRC Users

PRC users might encounter errors while downloading sources via PIP during OpenVINO™ installation. To resolve the issues, try one of the following options:
   
* Add the download source using the ``-i`` parameter with the Python ``pip`` command. For example: 

   ``` sh
   pip install openvino-dev -i https://mirrors.aliyun.com/pypi/simple/
   ```
   Use the ``--trusted-host`` parameter if the URL above is ``http`` instead of ``https``.
   You can also run the following command to install specific framework. For example:
   
   ```
   pip install openvino-dev[tensorflow2] -i https://mirrors.aliyun.com/pypi/simple/
   ```
   
* If you run into incompatibility issues between components after installing OpenVINO, try running ``requirements.txt`` with the following command:

   ``` sh
   pip install -r <INSTALL_DIR>/tools/requirements.txt
   ```

## Issues with Installing OpenVINO on Linux from Docker

### <a name="proxy-issues"></a>Proxy Issues

If you met proxy issues during the installation with Docker, please set up proxy settings for Docker. See the Proxy section in the [Install the DL Workbench from DockerHub](https://docs.openvino.ai/latest/workbench_docs_Workbench_DG_Prerequisites.html#set-proxy) topic.

### Permission Errors for /dev/shm

If you encounter a permission error for files in `/dev/shm` (see `hddldaemon.log`). A possible cause is that the uid and gid of the container user are different from the uid and gid of the user who created `hddldaemon` service on the host.

Try one of these solutions:

* Create the user in the Docker container with the same uid and gid as the HDDL daemon user.
* Set the container user umask to 0000: `umask 000`.
* (NOT RECOMMENDED) Start HDDL daemon on the host as root and start the container as root with the `-u root:root` option.

## Issues with Configurations for Intel® Vision Accelerator Design with Intel® Movidius™ VPUs 

<!-- this part was taken from original configurations-for-ivad-vpu.md -->

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
source /opt/intel/openvino_2022/setupvars.sh
${HDDL_INSTALL_DIR}/bin/bsl_reset
```

---
### "No space left on device" error while loading a network
When the application runs inference of a network with a big size(>4MB) of input/output or if the system is running out of the DMA buffer, 
the HDDL Plugin will fall back to use shared memory. 
In this case, if the application exits abnormally, the shared memory is not released automatically. 
To release it manually, remove files with the `hddl_` prefix from the `/dev/shm` directory:
```sh
sudo rm -f /dev/shm/hddl_*
```

---
### Solutions to the permission issue

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
### setup.sh doesn't install the driver binaries to /lib/modules on CentOS systems

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
### "Error: ipc_connection_linux_UDS : bind() failed" in hddldaemon log

You may have run hddldaemon under another user. Run the command below and try again:
```sh
sudo rm -rf /var/tmp/hddl_*
```

---
### "I2C bus: SMBus I801 adapter at not found!" in hddldaemon log

Run the following command to check if a SMBUS I801 adapter can be found:
```sh
i2cdetect -l
```
Then run:
```sh
sudo modprobe i2c-i801
```
---
### "open /dev/ion failed!" in hddldaemon log

Check if `myd_ion` kernel module is installed by running the following command:
```sh
lsmod | grep myd_ion
```
If you do not see any output from the command, reinstall the `myd_ion` module.

---
### Constantly getting "\_name\_mapping open failed err=2,No such file or directory" in hddldaemon log

Check if myd_vsc kernel module is installed by running the following command:
```sh
lsmod | grep myd_vsc
```
If you do not see any output from the command reinstall the `myd_vsc` module.

---
### "Required key not available" appears when trying to install the myd_ion or myd_vsc modules

Run the following commands:
```sh
sudo apt install mokutil
```
```sh
sudo mokutil --disable-validation
```
