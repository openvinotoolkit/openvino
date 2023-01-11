# OpenVINO™ 安全附加组件 {#ovsa_get_started_zh_CN}

本指南为使用 OpenVINO™ 安全附加组件创建、分发和使用通过 OpenVINO™ 工具套件创建的模型的人员提供了说明：

* **模型开发人员**：模型开发人员与独立软件开发商进行交互，以控制用户对模型的访问。本文档介绍如何设置硬件和虚拟机以使用 OpenVINO™ 安全附加组件定义对 OpenVINO™ 模型的访问控制，然后向用户提供访问控制模型。
* **独立软件开发商**：使用本指南获取有关使用 OpenVINO™ 安全附加组件验证向您的客户（用户）提供的访问控制模型的许可的说明。
* **用户**：本文档包含面向需要通过 OpenVINO™ 安全附加组件访问和运行访问控制模型的最终用户的说明。

在此版本中，一个人既可以是模型开发人员，也可以是独立软件开发商。因此，本文档提供有关为这两个角色配置一个系统和为用户角色配置一个系统的说明。本文档还提供一种方法，让同一个人扮演模型开发人员、独立软件开发商和用户的角色，以便让您从用户的角度了解 OpenVINO™ 安全附加组件的工作原理。


## 概述

OpenVINO™ 安全附加组件可与英特尔® 架构上的 [OpenVINO™ 模型服务器](@ref ovms_what_is_openvino_model_server_zh_CN)配合使用。OpenVINO™ 安全附加组件和 OpenVINO™ 模型服务器一起为模型开发人员和独立软件开发商提供了一种方法，可以使用安全封装和安全模型执行来实现对 OpenVINO™ 模型的访问控制，并让模型用户在指定的范围内运行推理。

OpenVINO™ 安全附加组件包括三个组件，这些组件在基于内核的虚拟机 (KVM) 中运行。这些组件提供了一种在隔离环境中执行安全敏感操作的方法。这三个组件的简要说明如下所示。单击每条带三角形的线条以获取有关它的更多信息。


@sphinxdirective

.. raw:: html

   <div class="collapsible-section" data-title="<strong>OpenVINO™ Security Add-on Tool</strong>: As a Model Developer or Independent Software Vendor, you use the OpenVINO™ Security Add-on Tool(`ovsatool`) to generate a access controlled model and master license.">

@endsphinxdirective



- 模型开发人员从 OpenVINO™ 工具套件输出生成访问控制模型。访问控制模型使用模型的中间表示 (IR) 文件创建分发给模型用户的访问控制输出文件存档。开发人员还可以将存档文件放在长期存储中或将其备份，而无需考虑额外的安全性。

- 模型开发人员使用 OpenVINO™ 安全附加组件工具 (<code>ovsatool</code>) 为访问控制模型生成和管理加密密钥和相关附属内容。加密材料仅在虚拟机 (VM) 环境中可用。OpenVINO™ 安全附加组件密钥管理系统可以让模型开发人员获取外部证书颁发机构生成的证书，并将其添加到密钥存储中。

- 模型开发人员以 JSON 格式文件为访问控制模型生成用户特定的许可。模型开发人员可以定义全局或用户特定的许可，并将许可策略附加到许可上。例如，模型开发人员可以为模型添加时间限制或限制用户可以运行模型的次数。


@sphinxdirective

.. raw:: html

   </div>

@endsphinxdirective




@sphinxdirective

.. raw:: html

   <div class="collapsible-section" data-title="<strong>OpenVINO™ Security Add-on License Service</strong>: Use the OpenVINO™ Security Add-on License Service to verify user parameters.">

@endsphinxdirective



- 独立软件开发商托管 OpenVINO™ 安全附加组件许可服务，当用户尝试在模型服务器中加载访问控制模型时，该服务会响应许可验证请求。许可通过 OpenVINO™ 安全附加组件许可服务注册。

- 当用户加载模型时，OpenVINO™ 安全附加组件运行时与许可服务联系，以确保许可是有效的，并且在模型开发人员用 OpenVINO™ 安全附加组件工具 (<code>ovsatool</code>) 定义的参数范围内。用户必须能够通过互联网访问独立软件开发商的许可服务。


@sphinxdirective

.. raw:: html

   </div>

@endsphinxdirective




@sphinxdirective

.. raw:: html

   <div class="collapsible-section" data-title="<strong>OpenVINO™ Security Add-on Runtime</strong>: Users install and use the OpenVINO™ Security Add-on Runtime on a virtual machine. ">

@endsphinxdirective



用户在虚拟机中托管 OpenVINO™ 安全附加组件运行时组件。

用户从 OpenVINO™ 安全附加组件的外部将访问控制模型添加到 OpenVINO™ 模型服务器配置文件中。OpenVINO™ 模型服务器尝试在内存中加载模型。此时，OpenVINO™ 安全附加组件运行时组件会根据独立软件开发商提供的许可服务中存储的信息验证访问控制模型的用户许可。

成功验证许可后，OpenVINO™ 模型服务器加载模型并为推理请求提供服务。


@sphinxdirective

.. raw:: html

   </div>

@endsphinxdirective

 

<br>
**OpenVINO™ 安全附加组件适合模型开发和部署的情况**

![安全附加组件图表](../../ovsa/ovsa_diagram.png)

[本文档](https://github.com/openvinotoolkit/security_addon/blob/master/docs/fingerprint-changes.md)介绍了 SWTPM（访客 VM 中使用的 vTPM）和 HW TPM（主机上的 TPM）之间的关系

## 关于安装
模型开发人员、独立软件开发商和用户必须各自准备一台物理硬件设备和一个基于内核的虚拟机 (KVM)。此外，每个人都必须为其扮演的每个角色准备一台访客虚拟机 （访客 VM）。

例如：
* 如果一个人既是模型开发人员又是独立软件开发商，则该人员必须准备两台访客 VM。两台访客 VM 可以位于同一物理硬件（主机）上，并且在该主机上的同一 KVM 下。
* 如果一个人扮演所有这三个角色，则该人员必须准备三台访客 VM。所有三个访客 VM 都可以位于同一主机上，并且在该主机上的同一 KVM 下。

**每台机器的用途**

| 机器 | 用途 |
| ----------- | ----------- |
| 主机 | 设置 KVM 和访客 VM 共享的物理硬件。 |
| 基于内核的虚拟机 (KVM) | OpenVINO™ 安全附加组件在此虚拟机中运行，因为它为安全敏感操作提供了一个隔离环境。 |
| 访客 VM | 模型开发人员使用访客 VM 来启用对已完成模型的访问控制。<br>独立软件开发商使用访客 VM 托管许可服务。<br>用户使用访客 VM 与许可服务联系并运行访问控制模型。 |


## 必备条件<a name="prerequisites"></a>

**硬件**
* 英特尔® 酷睿™ 或至强® 处理器<br>

**操作系统、固件和软件**
* 主机上使用 Ubuntu* Linux* 18.04。<br>
* TPM 版本 2.0 与独立可信平台模块 (dTPM) 或固件可信平台模块 (fTPM) 兼容
* 支持安全启动。<br>

**其他**
* 独立软件开发商必须有权访问执行在线证书状态协议 (OCSP) 的证书颁发机构 (CA)，从而支持椭圆曲线加密 (ECC) 证书以进行部署。
* 本文档中的示例使用自签名证书。

## 如何准备主机<a name="setup-host"></a>

本部分适用于模型开发人员和独立软件开发商的组合角色和单独的用户角色。

### 步骤 1：在主机上设置程序包<a name="setup-packages"></a>

在满足 <a href="#prerequisites">必备条件</a>的英特尔® 酷睿™ 或至强® 处理器设备上开始执行此步骤。

> **NOTE**: 作为手动执行步骤 1-11 的替代方法，您可以在 OpenVINO™ 安全附加组件存储库下的 `Scripts/reference directory` 中运行脚本 `install_host_deps.sh`。如果脚本发现任何问题，它会停止并显示错误消息。如果脚本由于错误而停止，请更正导致错误的问题并重新启动脚本。脚本运行几分钟，并提供进度信息。

1. 可信平台模块 (TPM) 支持测试：
   ```sh
   dmesg | grep -i TPM 
   ```	
   输出表明 TPM 在内核启动日志中可用。查找是否存在以下设备以表明是否支持 TPM：
   * `/dev/tpm0`
   * `/dev/tpmrm0`
   
   如果您没有看到此信息，则说明您的系统不满足 使用 OpenVINO™ 安全附加组件的<a href="#prerequisites">必备条件</a>。
2. 确保在 BIOS 中启用了硬件虚拟化支持：
   ```sh
   kvm-ok 
   ```
   输出应显示：<br>
   `INFO: /dev/kvm exists` <br>
   `KVM acceleration can be used`
	
   如果输出不同，请修改 BIOS 设置以启用硬件虚拟化。
   
   如果 `kvm-ok` 命令不存在，请安装该命令：
   ```sh
   sudo apt install -y cpu-checker
   ```
3. 安装基于内核的虚拟机 (KVM) 和 QEMU 程序包。
	```sh	
	sudo apt install qemu qemu-kvm libvirt-bin  bridge-utils  virt-manager 
	```	
4. 检查 QEMU 版本：
   ```sh	
   qemu-system-x86_64 --version 
   ```	
   如果响应消息显示 QEMU 版本低于 2.12.0，则从 [QEMU 网站](https://www.qemu.org/download)下载、编译并安装最新的 QEMU 版本。
5. 构建和安装 [libtpm 程序包](https://github.com/stefanberger/libtpms/)。
6. 构建和安装 [swtpm 程序包](https://github.com/stefanberger/swtpm/)。
7. 将 `swtpm` 程序包添加到 `$PATH` 环境变量中。
8. 安装软件工具 [tpm2-tss](https://github.com/tpm2-software/tpm2-tss/releases/download/2.4.4/tpm2-tss-2.4.4.tar.gz)。有关安装信息，请参阅[此处](https://github.com/tpm2-software/tpm2-tss/blob/master/INSTALL.md)
9. 安装软件工具 [tpm2-abmrd](https://github.com/tpm2-software/tpm2-abrmd/releases/download/2.3.3/tpm2-abrmd-2.3.3.tar.gz)。有关安装信息，请参阅[此处](https://github.com/tpm2-software/tpm2-abrmd/blob/master/INSTALL.md)
10. 安装 [tpm2-tools](https://github.com/tpm2-software/tpm2-tools/releases/download/4.3.0/tpm2-tools-4.3.0.tar.gz)。有关安装信息，请参阅[此处](https://github.com/tpm2-software/tpm2-tools/blob/master/docs/INSTALL.md)
11. 安装 [Docker 程序包](https://docs.docker.com/engine/install/ubuntu/)。	
   > **NOTE**:  无论您是否使用 `install_host_deps.sh` 脚本，都需要完成步骤 12，才能在主机上完成程序包的设置。
12. 如果您在代理后运行，请[为 Docker 设置代理](https://docs.docker.com/config/daemon/systemd/)。

以下组件已安装并且随时可用：
* 基于内核的虚拟机 (KVM)
* QEMU
* SW-TPM
* HW-TPM 支持
* Docker<br>
	
您已经准备好为网络配置主机。

### 步骤 2：在主机上设置网络<a name="setup-networking"></a>

此步骤适用于模型开发人员和独立软件开发商的组合角色。如果模型用户 VM 在不同的物理主机上运行，则还要在该主机上重复以下步骤。

在此步骤中，准备两个网桥：
* KVM 可以通过互联网访问的全局 IP 地址。用户设备上的 OpenVINO™ 安全附加组件运行时软件使用该地址验证他们是否具有有效许可。
* 用于在访客 VM 和 QEMU 主机操作系统之间提供通信的仅用于主机的本地地址。

本步骤中的此示例使用以下名称。您的配置可能使用不同的名称：
* `50-cloud-init.yaml` 作为示例配置文件名称。
* `eno1` 作为示例网络接口名称。
* `br0` 作为示例桥接名称。
* `virbr0` 作为示例桥接名称。

1. 打开网络配置文件以进行编辑。此文件位于 `/etc/netplan` 中，其名称类似于 `50-cloud-init.yaml`
2. 在文件中查找这些行：
   ```sh	
   network:
     ethernets:
        eno1:
          dhcp4: true
          dhcp-identifier: mac
     version: 2
   ```
3. 更改现有行，并添加 `br0` 网桥。这些更改支持外部网络访问：
   ```sh	
   network:
     ethernets:
        eno1:
          dhcp4: false
     bridges:
        br0:
          interfaces: [eno1]
          dhcp4: yes
		  dhcp-identifier: mac
     version: 2
   ```
4. 保存并关闭网络配置文件。
5. 运行两个命令以激活更新的网络配置文件。如果使用 ssh，在发出这些命令时可能会断开网络连接。如果是这样，请重新连接网络。
```sh
sudo netplan generate
```
```sh
sudo netplan apply
```	
创建桥接，并将 IP 地址指定到新桥接。
6.验证新桥接：
   ```sh
   ip a | grep br0
   ```	
输出看起来与此类似，并显示有效的 IP 地址：
   ```sh	
   4: br0:<br><BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc noqueue state UP group default qlen 1000<br>inet 123.123.123.123/<mask> brd 321.321.321.321 scope global dynamic br0
   ```	
7. 创建名为 `br0-qemu-ifup` 的脚本以启动 `br0` 接口。添加以下脚本内容：
   ```sh
   #!/bin/sh
   nic=$1
   if [ -f /etc/default/qemu-kvm ]; then
   	. /etc/default/qemu-kvm
   fi
   switch=br0
   ifconfig $nic 0.0.0.0 up
   brctl addif ${switch} $nic
   ```
8. 创建名为 `br0-qemu-ifdown` 的脚本，以关闭 `br0` 接口。添加以下脚本内容：
   ```sh
   #!/bin/sh
   nic=$1
   if [ -f /etc/default/qemu-kvm ]; then
   	. /etc/default/qemu-kvm
   fi
   switch=br0
   brctl delif $switch $nic
   ifconfig $nic 0.0.0.0 down
   ```
9. 创建名为 `virbr0-qemu-ifup` 的脚本以启动 `virbr0` 接口。添加以下脚本内容：
   ```sh
   #!/bin/sh
   nic=$1
   if [ -f /etc/default/qemu-kvm ]; then
   	. /etc/default/qemu-kvm
   fi
   switch=virbr0
   ifconfig $nic 0.0.0.0 up
   brctl addif ${switch} $nic
   ```
10. 创建名为 `virbr0-qemu-ifdown` 的脚本，以关闭 `virbr0` 接口。添加以下脚本内容：
   ```sh
   #!/bin/sh
   nic=$1
   if [ -f /etc/default/qemu-kvm ]; then
   . /etc/default/qemu-kvm
   fi
   switch=virbr0
   brctl delif $switch $nic
   ifconfig $nic 0.0.0.0 down
   ```

有关 QEMU 网络配置的更多信息，请参见 QEMU 文档。

在主机上设置网络。继续执行步骤 3，为模型开发人员和独立软件开发商的组合角色准备一个访客 VM。

### 步骤 3：克隆 OpenVINO™ 安全附加组件

下载 [OpenVINO™ 安全附加组件](https://github.com/openvinotoolkit/security_addon)。


### 步骤 4：为模型开发人员和独立软件开发商的组合角色设置一个访客 VM<a name="dev-isv-vm"></a>

您必须为您扮演的每个单独角色准备一个名为访客 VM 的虚拟机。因为在本版本中，模型开发人员和独立软件开发商角色组合在一起，这些指令可以引导您设置名为 `ovsa_isv` 的访客 VM。

在主机上开始这些步骤。

或者，您可以使用 `virsh` 和虚拟机管理器创建和启动访客 VM。如果您想执行此操作，请参见 `libvirtd` 文档以获取说明。

1. 下载 ** 服务器，为支持 [Ubuntu 18.04](https://releases.ubuntu.com/18.04/) 的 64 位 (AMD64) 电脑**安装 ISO 映像

2. 创建一个空的虚拟磁盘映像，作为模型开发人员和独立软件开发商组合角色的访客 VM：
   ```sh
   sudo qemu-img create -f qcow2 <path>/ovsa_isv_dev_vm_disk.qcow2 20G
   ```
3. 在访客 VM 上安装 Ubuntu 18.04。将访客 VM 命名为 `ovsa_isv`：
   ```sh
   sudo qemu-system-x86_64 -m 8192 -enable-kvm \
   -cpu host \
   -drive if=virtio,file=<path-to-disk-image>/ovsa_isv_dev_vm_disk.qcow2,cache=none \
   -cdrom <path-to-iso-image>/ubuntu-18.04.5-live-server-amd64.iso \
   -device e1000,netdev=hostnet1,mac=52:54:00:d1:66:5f \
   -netdev tap,id=hostnet1,script=<path-to-scripts>/virbr0-qemu-ifup,downscript=<path-to-scripts>/virbr0-qemu-ifdown \
   -vnc :1
   ```
4. 通过 `<host-ip-address>:1` 连接 VNC 客户端
5. 按照屏幕上的提示完成访客 VM 的安装。将 VM 命名为 `ovsa_isv_dev`
6. 关闭访客 VM。
7. 删除 cdrom 映像的选项后，重新启动访客 VM：
   ```sh
   sudo qemu-system-x86_64 -m 8192 -enable-kvm \
   -cpu host \
   -drive if=virtio,file=<path-to-disk-image>/ovsa_isv_dev_vm_disk.qcow2,cache=none \
   -device e1000,netdev=hostnet1,mac=52:54:00:d1:66:5f \
   -netdev tap,id=hostnet1,script=<path-to-scripts>/virbr0-qemu-ifup,downscript=<path-to-scripts>/virbr0-qemu-ifdown \
   -vnc :1
   ```
8. 选择其中一个选项以安装其他所需的软件：
   * **选项 1**：使用脚本安装其他软件
      1. 将脚本 `install_guest_deps.sh` 从 OVSA 存储库的 `Scripts/reference directory` 复制到访客 VM
      2. 运行脚本。
      3. 关闭访客 VM。<br>
   * **选项 2**：手动安装其他软件
      1. 安装软件工具 [tpm2-tss](https://github.com/tpm2-software/tpm2-tss/releases/download/2.4.4/tpm2-tss-2.4.4.tar.gz)。
         有关安装信息，请参阅[此处](https://github.com/tpm2-software/tpm2-tss/blob/master/INSTALL.md)
      2. 安装软件工具 [tpm2-abmrd](https://github.com/tpm2-software/tpm2-abrmd/releases/download/2.3.3/tpm2-abrmd-2.3.3.tar.gz)。
         有关安装信息，请参阅[此处](https://github.com/tpm2-software/tpm2-abrmd/blob/master/INSTALL.md)
      3. 安装 [tpm2-tools](https://github.com/tpm2-software/tpm2-tools/releases/download/4.3.0/tpm2-tools-4.3.0.tar.gz)。
         有关安装信息，请参阅[此处](https://github.com/tpm2-software/tpm2-tools/blob/master/docs/INSTALL.md)
      4. 安装 [Docker 程序包](https://docs.docker.com/engine/install/ubuntu/)
      5. 关闭访客 VM。<br>
9. 在主机上创建一个目录，以支持虚拟 TPM 设备并提供其证书。仅 `root` 应具有此目录的读/写权限：
   ```sh
   sudo mkdir -p /var/OVSA/
   sudo mkdir /var/OVSA/vtpm
   sudo mkdir /var/OVSA/vtpm/vtpm_isv_dev
   
   export XDG_CONFIG_HOME=~/.config
   /usr/share/swtpm/swtpm-create-user-config-files
   swtpm_setup --tpmstate /var/OVSA/vtpm/vtpm_isv_dev --create-ek-cert --create-platform-cert --overwrite --tpm2 --pcr-banks -
   ```
   > **NOTE**:  对于步骤 10 和 11，您可以复制和编辑 OpenVINO™ 安全附加组件存储库中的 `Scripts/reference` 目录中名为 `start_ovsa_isv_dev_vm.sh` 的脚本，而不是手动运行命令。如果使用脚本，无论您是模型开发人员还是独立软件开发商，都请选择文件名中带有 `isv` 的脚本。编辑脚本以指向正确的目录位置，并为每台访客 VM 递增 `vnc`。
10. 在主机上启动 vTPM，将 HW TPM 数据写入其 NVRAM 中，然后重新启动适用于 QEMU 的 vTPM：
   ```sh
    sudo swtpm socket --tpm2 --server port=8280 \
                      --ctrl type=tcp,port=8281 \
                      --flags not-need-init --tpmstate dir=/var/OVSA/vtpm/vtpm_isv_dev &

    sudo tpm2_startup --clear -T swtpm:port=8280
    sudo tpm2_startup -T swtpm:port=8280
    python3 <path to Security-Addon source>/Scripts/host/OVSA_write_hwquote_swtpm_nvram.py 8280
    sudo pkill -f vtpm_isv_dev
     
   swtpm socket --tpmstate dir=/var/OVSA/vtpm/vtpm_isv_dev \
    --tpm2 \
    --ctrl type=unixio,path=/var/OVSA/vtpm/vtpm_isv_dev/swtpm-sock \
    --log level=20
   ```
	
11. 启动访客 VM：
   ```sh
   sudo qemu-system-x86_64 \
    -cpu host \
    -enable-kvm \
    -m 8192 \
    -smp 8,sockets=1,cores=8,threads=1 \
    -device e1000,netdev=hostnet0,mac=52:54:00:d1:66:6f \
    -netdev tap,id=hostnet0,script=<path-to-scripts>/br0-qemu-ifup,downscript=<path-to-scripts>/br0-qemu-ifdown \
    -device e1000,netdev=hostnet1,mac=52:54:00:d1:66:5f \
    -netdev tap,id=hostnet1,script=<path-to-scripts>/virbr0-qemu-ifup,downscript=<path-to-scripts>/virbr0-qemu-ifdown \
    -drive if=virtio,file=<path-to-disk-image>/ovsa_isv_dev_vm_disk.qcow2,cache=none \
    -chardev socket,id=chrtpm,path=/var/OVSA/vtpm/vtpm_isv_dev/swtpm-sock \
    -tpmdev emulator,id=tpm0,chardev=chrtpm \
    -device tpm-tis,tpmdev=tpm0 \
    -vnc :1
   ```
使用命令中的 QEMU 运行时选项更改分配给此访客 VM 的内存量或 CPU。
   
12. 使用 VNC 客户端登录到地址为 `<host-ip-address>:1` 的访客 VM

### 步骤 5：为用户角色设置访客 VM

1. 选择其中**一个**选项以创建适用于用户角色的访客 VM：<br>
   **选项 1：复制和重命名 ovsa_isv_dev_vm_disk.qcow2 磁盘映像**
   1. 将 `ovsa_isv_dev_vm_disk.qcow2` 磁盘映像复制到名为 `ovsa_runtime_vm_disk.qcow2` 的新映像。您已经创建了 `ovsa_isv_dev_vm_disk.qcow2` 磁盘映像 <a  href="#dev-isv-vm">步骤 4</a>。
   2. 启动新映像。
   3. 将主机名从 `ovsa_isv_dev` 更改为 `ovsa_runtime`。
   ```sh 
   sudo hostnamectl set-hostname ovsa_runtime
   ```
   4. 在新映像中将 `ovsa_isv_dev` 的所有实例替换为 `ovsa_runtime`。
   ```sh 	
   sudo nano /etc/hosts
   ```
   5. 更改 `/etc/machine-id`：
   ```sh
   sudo rm /etc/machine-id
   systemd-machine-id-setup
   ```
   6. 关闭访客 VM。<br><br>
       
   **选项 2：手动创建访客 VM**
   1. 创建一个空的虚拟磁盘映像：
   ```sh
   sudo qemu-img create -f qcow2 <path>/ovsa_ovsa_runtime_vm_disk.qcow2 20G
   ```
   2. 在访客 VM 上安装 Ubuntu 18.04。将访客 VM 命名为 `ovsa_runtime`：
   ```sh
   sudo qemu-system-x86_64 -m 8192 -enable-kvm \
   -cpu host \
   -drive if=virtio,file=<path-to-disk-image>/ovsa_ovsa_runtime_vm_disk.qcow2,cache=none \
   -cdrom <path-to-iso-image>/ubuntu-18.04.5-live-server-amd64.iso \
   -device e1000,netdev=hostnet1,mac=52:54:00:d1:66:5f \
   -netdev tap,id=hostnet1,script=<path-to-scripts>/virbr0-qemu-ifup,   downscript=<path-to-scripts>/virbr0-qemu-ifdown \
   -vnc :2
   ```
   3. 通过 `<host-ip-address>:2` 连接 VNC 客户端。
   4. 按照屏幕上的提示完成访客 VM 的安装。将访客 VM 命名为 `ovsa_runtime`。
   5. 关闭访客 VM。
   6. 重新启动访客 VM：
   ```sh
   sudo qemu-system-x86_64 -m 8192 -enable-kvm \
   -cpu host \
   -drive if=virtio,file=<path-to-disk-image>/ovsa_ovsa_runtime_vm_disk.qcow2,cache=none \
   -device e1000,netdev=hostnet1,mac=52:54:00:d1:66:5f \
   -netdev tap,id=hostnet1,script=<path-to-scripts>/virbr0-qemu-ifup,   downscript=<path-to-scripts>/virbr0-qemu-ifdown \
   -vnc :2
   ```
   7. 选择其中**一个**选项以安装其他所需的软件：
      
      **选项 1：使用脚本安装其他软件**
      1. 将脚本 `install_guest_deps.sh` 从 OVSA 存储库的 `Scripts/reference` 目录复制到访客 VM
      2. 运行脚本。
      3. 关闭访客 VM。<br><br>
	        
      **选项 2：手动安装其他软件**
      1. 安装软件工具 [tpm2-tss](https://github.com/tpm2-software/tpm2-tss/releases/download/2.4.4/tpm2-tss-2.4.4.tar.gz)。有关安装信息，请参阅[此处](https://github.com/tpm2-software/tpm2-tss/blob/master/INSTALL.md)
      2. 安装软件工具 [tpm2-abmrd](https://github.com/tpm2-software/tpm2-abrmd/releases/download/2.3.3/tpm2-abrmd-2.3.3.tar.gz)。
         有关安装信息，请参阅[此处](https://github.com/tpm2-software/tpm2-abrmd/blob/master/INSTALL.md)
      3. 安装 [tpm2-tools](https://github.com/tpm2-software/tpm2-tools/releases/download/4.3.0/tpm2-tools-4.3.0.tar.gz)。
         有关安装信息，请参阅[此处](https://github.com/tpm2-software/tpm2-tools/blob/master/docs/INSTALL.md)
      4. 安装 [Docker 程序包](https://docs.docker.com/engine/install/ubuntu/)
      5. 关闭访客 VM。

2. 创建一个目录以支持虚拟 TPM 设备并提供其证书。仅 `root` 应具有此目录的读/写权限：
   ```sh
   sudo mkdir /var/OVSA/vtpm/vtpm_runtime
    
   export XDG_CONFIG_HOME=~/.config
   /usr/share/swtpm/swtpm-create-user-config-files
   swtpm_setup --tpmstate /var/OVSA/vtpm/vtpm_runtime --create-ek-cert --create-platform-cert --overwrite --tpm2 --pcr-banks -
   ```
   > **NOTE**: 对于步骤 3 和 4，您可以复制和编辑 OpenVINO™ 安全附加组件存储库中的 `Scripts/reference` 目录中名为 `start_ovsa_runtime_vm.sh` 的脚本，而不是手动运行命令。编辑脚本以指向正确的目录位置，并为每台访客 VM 递增 `vnc`。这意味着，如果您在同一主机上创建第三方访客 VM，则需要将 `-vnc :2` 更改为 `-vnc :3`
3. 启动 vTPM，将 HW TPM 数据写入其 NVRAM 中，然后重新启动适用于 QEMU 的 vTPM：
   ```sh
   sudo swtpm socket --tpm2 --server port=8380 \
                     --ctrl type=tcp,port=8381 \
                     --flags not-need-init --tpmstate dir=/var/OVSA/vtpm/vtpm_runtime &

   sudo tpm2_startup --clear -T swtpm:port=8380
   sudo tpm2_startup -T swtpm:port=8380
   python3 <path to Security-Addon source>/Scripts/host/OVSA_write_hwquote_swtpm_nvram.py 8380
   sudo pkill -f vtpm_runtime
	
   swtpm socket --tpmstate dir=/var/OVSA/vtpm/vtpm_runtime \
   --tpm2 \
   --ctrl type=unixio,path=/var/OVSA/vtpm/vtpm_runtime/swtpm-sock \
   --log level=20
   ```
4. 在新的终端启动访客 VM：
   ```sh
   sudo qemu-system-x86_64 \
    -cpu host \
    -enable-kvm \
    -m 8192 \
    -smp 8,sockets=1,cores=8,threads=1 \
    -device e1000,netdev=hostnet2,mac=52:54:00:d1:67:6f \
    -netdev tap,id=hostnet2,script=<path-to-scripts>/br0-qemu-ifup,downscript=<path-to-scripts>/br0-qemu-ifdown \
    -device e1000,netdev=hostnet3,mac=52:54:00:d1:67:5f \
    -netdev tap,id=hostnet3,script=<path-to-scripts>/virbr0-qemu-ifup,downscript=<path-to-scripts>/virbr0-qemu-ifdown \
    -drive if=virtio,file=<path-to-disk-image>/ovsa_runtime_vm_disk.qcow2,cache=none \
    -chardev socket,id=chrtpm,path=/var/OVSA/vtpm/vtpm_runtime/swtpm-sock \
    -tpmdev emulator,id=tpm0,chardev=chrtpm \
    -device tpm-tis,tpmdev=tpm0 \
    -vnc :2
   ```
   使用命令中的 QEMU 运行时选项更改分配给此访客 VM 的内存量或 CPU。
5. 使用 VNC 客户端登录到地址为 `<host-ip-address>:<x>` 的访客 VM。其中 `<x>` 对应于 `start_ovsa_isv_vm.sh` 中或步骤 8 中的 vnc 编号。

## 如何构建和安装 OpenVINO™ 安全附加组件软件<a name="install-ovsa"></a>

按照以下步骤在主机和不同的 VM 上构建和安装 OpenVINO™ 安全附加组件。

### 步骤 1：构建 OpenVINO™ 模型服务器映像
根据 OpenVINO™ 模型服务器 docker 容器构建 OpenVINO™ 安全附加组件。首先在主机上下载并构建 OpenVINO™ 模型服务器。

1. 下载 [OpenVINO™ 模型服务器软件](https://github.com/openvinotoolkit/model_server)

2. 构建 [OpenVINO™ 模型服务器 Docker 映像](https://github.com/openvinotoolkit/model_server/blob/main/docs/docker_container.md)
   ```sh
   git clone https://github.com/openvinotoolkit/model_server.git
   cd model_server
   make docker_build
   ```

### 步骤 2：构建全部角色所需的软件

本步骤适用于模型开发人员和独立软件开发商的组合角色以及用户

1. 转到之前克隆的顶级 OpenVINO™ 安全附加组件源代码目录。
   ```sh
   cd security_addon
   ```
2. 构建 OpenVINO™ 安全附加组件：
   ```sh
   make clean all
   sudo -s make package
   ```
   以下程序包是在 `release_files` 目录下创建的：
   - `ovsa-kvm-host.tar.gz`：主机文件
   - `ovsa-developer.tar.gz`：面向模型开发人员和独立软件开发商
   - `ovsa-model-hosting.tar.gz`：面向用户

### 步骤 3：安装主机软件
本步骤适用于模型开发人员和独立软件开发商的组合角色以及用户。

1. 转到 `release_files` 目录：
   ```sh
   cd release_files
   ```
2. 设置路径：
   ```sh
   export OVSA_RELEASE_PATH=$PWD
   ```
3. 在主机上安装 OpenVINO™ 安全附加组件软件：
   ```sh
   cd $OVSA_RELEASE_PATH
   tar xvfz ovsa-kvm-host.tar.gz
   cd ovsa-kvm-host
   ./install.sh
   ```

如果您使用多个主机，请在每个主机上重复步骤 3。

### 步骤 4：安装 OpenVINO™ 安全附加组件模型开发人员/ISV 组件
本步骤适用于模型开发人员和独立软件开发商的组合角色。对访客 VM 的参考指向 `ovsa_isv_dev`。
 
1. 以 `<user>` 身份登录到访客 VM。
2. 在主目录中创建 OpenVINO™ 安全附加组件目录
   ```sh
   mkdir -p ~/OVSA
   ```
3. 转到访客 VM 之外的主机。
4. 将 `ovsa-developer.tar.gz` 从 `release_files`复 制到访客 VM：
   ```sh
   cd $OVSA_RELEASE_PATH
   scp ovsa-developer.tar.gz username@<isv-developer-vm-ip-address>:/<username-home-directory>/OVSA
   ```
5. 转到访客 VM。
6. 创建 `ovsa` 用户
   ``sh
   sudo useradd -m ovsa
   sudo passwd ovsa
   ```

7. 将软件安装到访客 VM：
   ```sh
   cd ~/OVSA
   tar xvfz ovsa-developer.tar.gz
   cd ovsa-developer
   sudo ./install.sh
   ```
8. 以 `ovsa` 用户身份在单独终端上启动许可服务器。
   ```sh
   source /opt/ovsa/scripts/setupvars.sh
   cd /opt/ovsa/bin
   ./license_server
   ```
   > **NOTE**: 如果您受到防火墙的保护，请检查并设置代理设置以确保许可服务器能够验证证书。

### 步骤 5：安装 OpenVINO™ 安全附加组件模型托管组件

此步骤适用于用户。对访客 VM 的参考指向 `ovsa_runtime`。

模型托管组件基于 OpenVINO™ 模型服务器 NGINX Docker 安装 OpenVINO™ 安全附加组件运行时 Docker 容器，以托管访问控制模型。

1. 以 `<user>` 身份登录到访客 VM。
2. 在主目录中创建 OpenVINO™ 安全附加组件目录
    ```sh
    mkdir -p ~/OVSA
    ```
3. 在主机上，将 ovsa-model-hosting.tar.gz 从 release_files 复制到访客 VM 时：
   ```sh
   cd $OVSA_RELEASE_PATH
   scp ovsa-model-hosting.tar.gz username@<runtime-vm-ip-address>:/<username-home-directory>/OVSA
   ```
4. 转到访客 VM。
5. 创建 `ovsa` 用户
   ```sh
   sudo useradd -m ovsa
   sudo passwd ovsa
   sudo usermod -aG docker ovsa
   ``` 
6. 将软件安装到访客 VM：
   ```sh
   cd ~/OVSA
   tar xvfz ovsa-model-hosting.tar.gz
   cd ovsa-model-hosting
   sudo ./install.sh
   ```

## 如何使用 OpenVINO™ 安全附加组件

本节需要模型开发人员/独立软件开发商和用户之间进行交互。所有角色必须完成所有适用的 <a href="#setup-host">设置步骤</a>和 <a href="#ovsa-install">安装步骤</a>之后才能开始学习本节。

本文档以 [face-detection-retail-0004](@ref omz_models_model_face_detection_retail_0044) 模型为例。

下图描述了模型开发人员、独立软件开发商和用户之间的交互。

> **TIP**: 模型开发人员/独立软件开发商和用户角色与虚拟机的使用相关，一个人可以填补多个角色所需完成的任务。在本文档中，模型开发人员和独立软件开发商的任务进行组合，并使用名为 `ovsa_isv` 的访客 VM。在同一台主机上可以设置所有角色。

![OpenVINO™ 安全附加组件示例图表](../../ovsa/ovsa_example.png)

### 模型开发人员指令

模型开发人员创建模型，定义访问控制并创建用户许可。创建模型、启用访问控制并准备好许可后，模型开发人员先向独立软件开发商提供许可详细信息，然后才能共享给模型用户。

对访客 VM 的参考指向 `ovsa_isv_dev`。以 `ovsa` 用户身份登录到访客 VM。

#### 步骤 1：设置工件目录

创建名为工件的目录。此目录将包含创建许可所需的工件：
```sh
mkdir -p ~/OVSA/artefacts
cd ~/OVSA/artefacts
export OVSA_DEV_ARTEFACTS=$PWD
source /opt/ovsa/scripts/setupvars.sh
```
#### 步骤 2：创建密钥存储，并为其添加证书
1. 创建文件以请求证书：
   此示例使用自签名的证书进行演示。在生产环境中，使用 CSR 文件请求 CA 签名的证书。
   ```sh
   cd $OVSA_DEV_ARTEFACTS
   /opt/ovsa/bin/ovsatool keygen -storekey -t ECDSA -n Intel -k isv_keystore -r  isv_keystore.csr -e "/C=IN/CN=localhost"
   ```
   以下两个文件与密钥存储文件一起创建：
   - `isv_keystore.csr` - 证书签名请求 (CSR)
   - `isv_keystore.csr.crt` - 自签名证书
   
   在生产环境中将 `isv_keystore.csr` 发送到 CA 以请求 CA 签名的证书。
	
3. 将证书添加到密钥存储
   ```sh
   /opt/ovsa/bin/ovsatool keygen -storecert -c isv_keystore.csr.crt -k isv_keystore
   ```	
	
#### 步骤 3：创建模型

此示例使用 `curl` 从 OpenVINO™ Model Zoo 下载 `face-detection-retail-004` 模型。如果您受到防火墙的保护，请检查并设置代理设置。

从 Model Zoo 下载模型：
```sh
curl --create-dirs https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/1/face-detection-retail-0004/FP32/face-detection-retail-0004.xml https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/1/face-detection-retail-0004/FP32/face-detection-retail-0004.bin -o model/face-detection-retail-0004.xml -o model/face-detection-retail-0004.bin
```
将模型下载到 `OVSA_DEV_ARTEFACTS/model` 目录

#### 步骤 4：为模型定义访问控制，并为其创建原版使用许可

定义和启用模型访问控制和原版使用许可：
```sh	
uuid=$(uuidgen)
/opt/ovsa/bin/ovsatool controlAccess -i model/face-detection-retail-0004.xml model/face-detection-retail-0004.bin -n "face detection" -d "face detection retail" -v 0004 -p face_detection_model.dat -m face_detection_model.masterlic -k isv_keystore -g $uuid
```
`face-detection-retail-0004` 模型的中间表示文件加密为 `face_detection_model.dat`，原版使用许可生成为 `face_detection_model.masterlic`。

#### 步骤 5：创建运行时参考 TCB

使用运行时参考 TCB 为访问控制模型和特定运行时创建客户许可。

生成运行时参考 TCB
```sh
/opt/ovsa/bin/ovsaruntime gen-tcb-signature -n "Face Detect @ Runtime VM" -v "1.0" -f face_detect_runtime_vm.tcb -k isv_keystore
```
	
#### 步骤 6：发布访问控制模型和运行时参考 TCB
访问控制模型已准备好与用户共享，参考 TCB 已准备好执行许可检查。

#### 步骤 7：接收用户请求
1. 从需要访问访问控制模型的用户处获取工件：
   * 来自客户密钥存储的客户证书。
   * 适用于您的许可做法的其他信息，例如用户需要访问模型的时间长度

2. 创建客户许可配置
   ```sh
   cd $OVSA_DEV_ARTEFACTS
   /opt/ovsa/bin/ovsatool licgen -t TimeLimit -l30 -n "Time Limit License Config" -v 1.0 -u "<isv-developer-vm-ip-address>:<license_server-port>" /opt/ovsa/certs/server.crt  -k isv_keystore -o 30daylicense.config
   ```
   > **NOTE**: 参数 /opt/ovsa/certs/server.crt 包含许可服务器所用的证书。将服务器证书添加到客户许可中，并在使用期间进行验证。请参阅 [OpenVINO™ 安全附加组件许可服务器证书锁定](https://github.com/openvinotoolkit/security_addon/blob/releases/2022/2/docs/ovsa_license_server_cert_pinning.md)
3. 创建客户许可
   ```sh
   cd $OVSA_DEV_ARTEFACTS
   /opt/ovsa/bin/ovsatool sale -m face_detection_model.masterlic -k isv_keystore -l 30daylicense.config -t face_detect_runtime_vm.tcb -p custkeystore.csr.crt -c face_detection_model.lic
   ```
	
4. 使用许可更新许可服务器数据库。
   ```sh
   cd /opt/ovsa/DB
   python3 ovsa_store_customer_lic_cert_db.py ovsa.db $OVSA_DEV_ARTEFACTS/face_detection_model.lic $OVSA_DEV_ARTEFACTS/custkeystore.csr.crt
   ```

5. 向用户提供这些文件：
	* `face_detection_model.dat`
	* `face_detection_model.lic`

### 模型用户指令
对访客 VM 的参考指向 `ovsa_rumtime`。以 `ovsa` 用户身份登录到访客 VM。

#### 步骤 1：设置工件目录

1. 创建名为工件的目录。此目录将包含创建许可所需的工件：
   ```sh
   mkdir -p ~/OVSA/artefacts
   cd ~/OVSA/artefacts
   export OVSA_RUNTIME_ARTEFACTS=$PWD
   source /opt/ovsa/scripts/setupvars.sh
   ```

#### 步骤 2：将 CA 签名的证书添加到密钥存储

1. 生成客户密钥存储文件：
   ```sh
   cd $OVSA_RUNTIME_ARTEFACTS
   /opt/ovsa/bin/ovsatool keygen -storekey -t ECDSA -n Intel -k custkeystore -r  custkeystore.csr -e "/C=IN/CN=localhost"
   ```
   以下两个文件与密钥存储文件一起创建：
   * `custkeystore.csr` - 证书签名请求 (CSR)
   * `custkeystore.csr.crt` - 自签名证书

3. 将 `custkeystore.csr` 发送到 CA 以请求 CA 签名的证书。

4. 将证书添加到密钥存储：
   ```sh
   /opt/ovsa/bin/ovsatool keygen -storecert -c custkeystore.csr.crt -k custkeystore
   ```

#### 步骤 3：请求模型开发人员提供访问控制模型
此示例使用 scp 在同一主机上的 ovsa_runtime 和 ovsa_dev 访客 VM 之间共享数据。

1. 将您对模型的需求传达给模型开发人员。开发人员将要求您提供密钥存储中的证书和其他信息。此示例使用模型所需的可用的时间长度。
2. 模型用户的证书需要提供给开发人员：
   ```sh
   cd $OVSA_RUNTIME_ARTEFACTS
   scp custkeystore.csr.crt username@<developer-vm-ip-address>:/<username-home-directory>/OVSA/artefacts
   ```
#### 步骤 4：接收访问控制模型并将其载入 OpenVINO™ 模型服务器中
1. 接收以下文件名的模型：
   * face_detection_model.dat
   * face_detection_model.lic
   ```sh
   cd $OVSA_RUNTIME_ARTEFACTS
   scp username@<developer-vm-ip-address>:/<username-home-directory>/OVSA/artefacts/face_detection_model.dat .
   scp username@<developer-vm-ip-address>:/<username-home-directory>/OVSA/artefacts/face_detection_model.lic .
   ```

2. 准备环境：
   ```sh
   cd $OVSA_RUNTIME_ARTEFACTS/..
   cp /opt/ovsa/example_runtime ovms -r
   cd ovms
   mkdir -vp model/fd/1
   ```
   `$OVSA_RUNTIME_ARTEFACTS/../ovms` 目录包含用于启动模型服务器的脚本和样本配置 JSON 文件。
3. 复制模型开发人员提供的工件：
   ```sh
   cd $OVSA_RUNTIME_ARTEFACTS/../ovms
   cp $OVSA_RUNTIME_ARTEFACTS/face_detection_model.dat model/fd/1/.
   cp $OVSA_RUNTIME_ARTEFACTS/face_detection_model.lic model/fd/1/.
   cp $OVSA_RUNTIME_ARTEFACTS/custkeystore model/fd/1/.
   ```
4. 重命名并编辑 `sample.json` 以包含模型开发人员提供的访问控制模型工件的名称。该文件如下所示：
   ```sh
   {
   "custom_loader_config_list":[
   	{
   		"config":{
   				"loader_name":"ovsa",
   				"library_path": "/ovsa-runtime/lib/libovsaruntime.so"
   		}
   	}
   ],
   "model_config_list":[
   	{
   	"config":{
		"name":"controlled-access-model",
   		"base_path":"/sampleloader/model/fd",
		"custom_loader_options": {"loader_name":  "ovsa", "keystore":  "custkeystore", "controlled_access_file": "face_detection_model"}
   	}
   	}
   ]
   }
   ```

#### 步骤 5：启动 NGINX 模型服务器
NGINX 模型服务器发布访问控制模型。
   ```sh
   ./start_secure_ovsa_model_server.sh
   ```
有关 NGINX 接口的信息，请参阅[此处](https://github.com/openvinotoolkit/model_server/blob/main/extras/nginx-mtls-auth/README.md)。

#### 步骤 6：准备运行推理

1. 从另一个终端登录到访客 VM。

2. 为您的设置安装 Python 依赖项。例如：
   ```sh
   sudo apt install pip3
   pip3 install cmake
   pip3 install scikit-build
   pip3 install opencv-python
   pip3 install futures==3.1.1
   pip3 install tensorflow-serving-api==1.14.0
   ```
3. 从 `/opt/ovsa/example_client` 中的 example_client 复制`face_detection.py`
   ```sh
   cd ~/OVSA/ovms
   cp /opt/ovsa/example_client/* .
   ```
4. 复制样本映像以进行推理。创建的映像目录包括用于推理的样本映像。
   ```sh
   curl --create-dirs https://raw.githubusercontent.com/openvinotoolkit/model_server/master/example_client/images/people/people1.jpeg -o images/people1.jpeg
   ```

#### 步骤 7：运行推理

运行 `face_detection.py` 脚本：
```sh
python3 face_detection.py --grpc_port 3335 --batch_size 1 --width 300 --height 300 --input_images_dir images --output_dir results --tls --server_cert /var/OVSA/Modelserver/server.pem --client_cert /var/OVSA/Modelserver/client.pem --client_key /var/OVSA/Modelserver/client.key --model_name controlled-access-model
```	

## 摘要
您已经完成这些任务：
- 设置一台或多台电脑（主机），每台电脑一台 KVM，并在主机上设置一台或多台虚拟机（访客 VM）
- 安装 OpenVINO™ 安全附加组件
- 结合使用 OpenVINO™ 模型服务器与 OpenVINO™ 安全附加组件
- 作为模型开发人员或独立软件开发商，您可以对模型启用访问控制并为其准备许可。
- 作为模型开发人员或独立软件开发商，您准备和运行许可服务器，并使用许可服务器验证用户是否具有使用访问控制模型的有效许可。
- 作为用户，您向模型开发人员或独立软件开发商提供用于获取访问控制模型和模型许可的信息。
- 作为用户，您设置并启动一个主机服务器，在该主机服务器上，您可以运行许可和访问控制模型。
- 作为用户，您加载访问控制模型，验证模型许可并使用模型运行推理。

## 参考资料
使用这些链接，以便获取更多信息：
- [OpenVINO&trade; 工具套件](https://software.intel.com/en-us/openvino-toolkit)
- [OpenVINO™ 模型服务器快速入门指南](https://github.com/openvinotoolkit/model_server/blob/main/docs/ovms_quickstart.md)
- [模型存储库](https://github.com/openvinotoolkit/model_server/blob/main/docs/models_repository.md)
