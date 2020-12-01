# OpenVINO™ Security Add-on {#ovsa_get_started}

This guide shows you how to protect your OpenVINO models and then provide the models to your users. This guide also includes instructions to provide to your users. To protect and distribute your models, you use a Host and Virtual Machine that runs software to which your users connect. This software confirms the user is entitled to use your protected model.

## Overview

The OpenVINO™ Security Add-on (OVSA) works with the [OpenVINO™ Model Server (OVMS)](@ref openvino_docs_ovms) on Intel® architecture. Together, the OVSA and OVMS help you protect and control your OpenVINO™ models through secure packaging and secure model execution. Your users install a component on their systems to allow them to use the models within the limits that you assign.

The OVSA consists of three components that run in a Kernel-based Virtual Machine (KVM) to run security-sensitive operations in an isolated environment. A brief description of the three OVSA components are as follows. Click each triangled line for full descriptions. 
 
<details>
    <summary><strong>OVSA Protect Tool</strong>: As a model developer, you use the OVSA Protect Tool to generate protected models. You provide instructions to your users use this tool.</summary>

The Protect Tool:

* Generates and manages cryptographic keys and related collateral for protected models. Cryptographic material is only available in the virtual machine (VM) environment. The OVSA key management system lets you use of external Certificate Authorities to generate certificates that you add a key-store. 

* Generates protected models from the OpenVINO™ toolkit output. The protected models use the model’s IR files to create a protected archive output file that you can distribute to your users. You can also put the archive file in long-term storage or back it up without additional security. 

* Generates user-specific licenses in a JSON format file for the protected model. You can define licenses for specific users and attach licensing policies. For example, you can limit the time for which a model will work or the number of times the user can use the model. 

</details>

<details>
    <summary><strong>OVSA License Service</strong>: Use the OVSA License Service to define user parameters.</summary>
    
You host the OVSA License Service, which responds to license validation requests when a user attempts to run a protected model is executed by a user. When you generate a user license for a model, the license is registered with the OVSA License Service.

When a user loads the model, the OVSE Run Time contacts the License Service to make sure the license is valid and within the parameters you defined with the Protect Tool. The user must be able to reach the License Service over the Internet. 

</details>

<details>
    <summary><strong>OVSA Run Time</strong>: Your users install and use use the OVSA Run Time on a virtual machine. </summary>

When a user attempts to run your protected model, OVMS loads the model in memory and uses OVSA Run Time to check for license validity. Users must host the OVSA Run Time and related components in a VM in their environment to use your protected model. 

</details> 


**Where the OpenVINO™ Security Add-on Fits into Model Development and Deployment**

![Security Add-on Diagram](ovsa_diagram.png)
 
## Prerequisites 

**Hardware**
* Intel® Core™ or Xeon® processor<br>

**Operating system, firmware, and software**
* Ubuntu* Linux* 18.04 or 20.04 (preferred) on the Host machine.<br>
* Support for version 2.0-conformant Discrete Trusted Platform Module (dTPM) or Firmware Trusted Platform Module (fTPM)
* Secure boot is enabled.<br>
* The GCC compiler and libraries are installed in the KVM.<br>
* The latest version of the OpenVINO™ toolkit is installed.<br>
* The latest version of the OpenVINO™ Model Server (OVSM) is installed.<br>
* NGINX open-source web server.

**Other**
* You must have access to a Certificate Authority (CA) that implements the Online Certificate Status Protocol (OCSP), supporting Elliptic Curve Cryptography (ECC) certificates. 


## How to Prepare the Host Machine

In this section, you prepare the physical machine that will serve as the host.

### Step 1: Set up Packages on the Host Machine

Use this step on the Intel® Core™ or Xeon® processor machine that meets the prerequisites. 

1. Test the Trusted Platform Module (TPM) support:
   ```sh	
   dmesg | grep -i TPM 
   ```	
   The output indicates TPM availability in the kernel boot logs. Look for these values to indicate support is available:
   * `/dev/tpm0`
   * `/dev/tpmrm0`
   
   If you do not see this information, your system does not meet the prerequisites required to use the OVSA.
2. Make sure hardware virtualization support is enabled in the BIOS:
   ```sh	
   kvm-ok 
   ```
   The output should be `INFO: /dev/kvm exists`. If your output is different, modify your BIOS settings to enable hardware virtualization.
3. Install the Kernel-based Virtual Machine (KVM) and QEMU packages. 
   * Ubuntu 20.04 (preferred):
```sh	
sudo apt install qemu qemu-utils qemu-kvm virt-manager libvirt-daemon-system libvirt-clients bridge-utils 
```	
   * Ubuntu 18.04:
```sh	
sudo apt install qemu qemu-kvm libvirt-bin  bridge-utils  virt-manager 
```	
4. Check the QEMU version:
   ```sh	
   qemu-system-x86_64 --version 
   ```	
   If the response indicates a QEMU version lower than 2.12.0 download and compile QEMU version 2.12.0 from [https://www.   qemu.org/download](https://www.qemu.org/download).
5. Build and install the [`libtpm` package](https://github.com/stefanberger/libtpms/) 
6. Build and install [`swtpm` package](https://github.com/stefanberger/swtpm/)
7. Add the `swtpm` package to the `$PATH` environment variable.
8. Install the hardware tool [`tpm2-tss`]( 	https://github.com/tpm2-software/tpm2-tss/releases/download/2.4.4/tpm2-tss-2.4.4.tar.gz)<br>Information is at https://github.com/tpm2-software/tpm2-tss/blob/master/INSTALL.md
9. Install the hardware tool [tpm2-abmrd](https://github.com/tpm2-software/tpm2-abrmd/releases/download/2.3.3/tpm2-abrmd-2.3.3.tar.gz)<br> Information is at 
	https://github.com/tpm2-software/tpm2-abrmd/blob/master/INSTALL.md
10. Install [tpm2-tools](	https://github.com/tpm2-software/tpm2-tools/releases/download/4.3.0/tpm2-tools-4.3.0.tar.gz)<br>Information is at https://github.com/tpm2-software/tpm2-tools/blob/master/INSTALL.md
11. Install [Docker packages](https://docs.docker.com/engine/install/ubuntu/)
	
The following are installed and ready to use:
* Kernel-based Virtual Machine (KVM)
* QEMU
* SW-TPM
* HW-TPM support
* Docker
	
You are ready to configure the Host Machine for networking. 

### Step 2: Configure the Host Machine for Networking

In this step you prepare two IP addresses:
* A global IP address that is visible outside of the Host Machine. This IP address lets the virtual machine access the public Internet. This is the address that the OVSA Run Time on your user's machine will contact to verify they have a valid license.
* A host-only local address for communication between a Guest VM and the QEMU host operating system.

This step uses the following names. Your configuration might use different names:
* `50-cloud-init.yaml` as an example configuration file name.
* `eno1` as an example network interface name. 
* `br0` as an example bridge name.
* `virbr0` as an example bridge name.

1. Open the network configuration file for editing. This file is in `/etc/netplan` with a name such as `50-cloud-init.yaml`
2. Look for these lines:
   ```sh	
   network:
     ethernets:
        eno1:
          dhcp4: true
          dhcp-identifier: mac
     version: 2
   ```
3. Change these lines to add the `br0` network bridge for external network access:
   ```sh	
   network:
     ethernets:
        eno1:
          dhcp4: false
     bridges:
        br0:
          interfaces: [eno1]
          dhcp4: yes
     version: 2
   ```
4. Save and close the network configuration file.
5. Run two commands to activate the new network configuration file. If you use ssh, you might lose network connectivity when issuing these commands. If so, reconnect to the network.
```sh
sudo netplan generate
```
   ```sh
   sudo netplan apply
   ```	
   A bridge is created and an IP address is assigned to the new bridge.
6. Verify the the new bridge:
   ```sh
   ip a | grep br0
   ```	
   The output looks similar to this and shows valid IP addresses:
   ```sh	
   4: br0:<br><BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc noqueue state UP group default qlen 1000<br>inet 123.123.   123.123/<mask> brd 321.321.321.321 scope global dynamic br0
   ```	
7. Create a script named `br0-qemu-ifup` to bring up the `br0` interface. The script contents are:
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
8. Create a script named `br0-qemu-ifdown` to bring down the `br0` interface. The script contents are:
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
9. Create a script named `virbr0-qemu-ifup` to bring up the `virbr0` interface. The script contents are:
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
10. Create a script named `virbr0-qemu-ifdown ` to bring down the `virbr0` interface. The script contents are:
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

See the QEMU documentation for more information about the QEMU network configuration.

Networking is set up on the Host Machine. Continue to the next section to prepare the Guest VM.
	
## How to Prepare the Guest VM
In this section, you prepare a virtual machine that will serve as the guest. This is the Guest VM. Begin these steps on the Host Machine. 

As an option, you can use `virsh` and the virtual machine manager to create the virtual machine and complete the bring-up. See the `libvirtd` documentation for instructions if you'd like to do this.

1. Download the [Ubuntu 18.04 server ISO image](https://releases.ubuntu.com/18.04/ubuntu-18.04.5-live-server-amd64.iso)
2. Create an empty virtual disk image:
   ```sh
   sudo qemu-img create -f qcow2 <path>/ovsa_vm_disk.qcow2 20G
   ```
   This is the Guest VM.
3. Install Ubuntu 18.04 on the Guest VM:
   ```sh
   sudo qemu-system-x86_64 -m 8192 -enable-kvm \
    -drive if=virtio,file=<path-to-disk-image>/ovsa_vm_disk.qcow2,cache=none \
    -cdrom <path-to-iso-image>/ubuntu-18.04.5-live-server-amd64.iso \
    -vnc :1
   ```
4. Connect a VNC client with `<host-ip-address>:1`.
5. Follow the prompts on the screen to finish installing the Guest VM.
6. Shut down the Guest VM. You are working directly on the Host Machine again.
7. Create a directory to support the virtual TPM device. Only `root` should have read/write permission to this directory:
   ```sh
   sudo mkdir /var/ovsa/vtpm
   ```
8. Start the vTPM on the Host Machine:
   ```sh
   swtpm socket --tpmstate dir=<path_to_tpmdata>/vtpm1 \
     --tpm2 \
     --ctrl type=unixio,path=<path_to_tpmdata>/vtpm1/swtpm-sock \
     --log level=20
   ```
9. Start the Guest VM:
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
     -nographic  \
     -drive if=virtio,file=<path-to-disk-image>/ovsa_vm_disk.qcow2,cache=none \
     -chardev socket,id=chrtpm,path=<path_to_tpmdata>/vtpm1/swtpm-sock \
     -tpmdev emulator,id=tpm0,chardev=chrtpm \
     -device tpm-tis,tpmdev=tpm0 \
     -vnc :1
   ```
10. Use a VNC client to log on to the Guest VM at `<host-ip-address>:1`
12. Install the [Docker package](https://docs.docker.com/engine/install/ubuntu/) in the Guest VM.

## How to Build Images

### Step 1: Build the OpenVINO Model Server (OVSM) Images

Complete these steps on the Host Machine.

1. Configure the NGINX server: https://github.com/openvinotoolkit/model_server/tree/main/extras/nginx-mtls-auth1. 
2. Download the OVMS from https://github.com/openvinotoolkit/model_server
3. Build the OVMS Docker images: https://github.com/openvinotoolkit/model_server/blob/main/docs/docker_container.md

The OVMS images are available.

### Step 2: Build the OVSA Docker Image

Begin these steps on the Host Machine.

1. Download the [OpenVINO Security Add-on (OVSA)[(https://github.com/openvinotoolkit/OVSA)
2. Go to the top-level OVSA source directory.
3. Build the OVSA Docker image:
   ```sh
   make docker_images
   make all
   sudo ./host_install.sh
   ```
   A file named `ovsa.tar.gz` is created.
4. Copy `ovsa.tar.gz` to the Guest VM:
   ```sh
   scp ovsa.tar.gz username@<guest-vm-ip-address>:/<username-home-directory>/OVSA
   ```
   The file is copied to `/<username-home-directory>/OVSA`.
5. Log on to the Guest VM as `<user>`.
6. Unpack `ovsa.tar.gz` in `/<username-home-directory>/OVSA`:
   ```sh
   cd OVSA
   tar -xvfz ovsa.tar.gz
   ```
7. Install OVSA and its related components:
   ```sh
   sudo ./install.sh
   ```

## How to Use the OVSA

### Step 1: Create a Key Store and Add a Certificate to it
<strong>Is this on the Host Machine or on the Guest VM?</strong>

In this step you use the Protect tool to create a certificate and put the certificate in a key store file. 

1. Generate the key store file:
   ```sh	
   ./bin/ovsatool keygen -storekey -t ECDSA -n Intel -k keystore/isv_keystore -r artefacts/isv_keystore.csr -e "/C=IN/   ST=KA/O=Intel, Inc./CN=intel.com/L=Bangalore/mail=xyz@intel.com"
   ```
   A certificate signing request file is created named `isv_keystore.csr`
	
2. Contact your Certificate Authority (CA) and use `isv_keystore.csr` to get a CA-signed certificate in PEM format. Instructions to get the certificate are outside the scope of this document.

3. Add the certificate to the key store: 
   ```sh	
   ./bin/ovsatool keygen -storecert -c artefacts/isv_keystore.csr.crt -k keystore/isv_keystore
   ```

### Step 2: Use a Sample Model to Learn to Protect Models	
1. Use the Model Downloader to download the sample model named `Person_Detection_Retail_0013_Model` from the Model Zoo.

2. Protect the model:
   ```sh	
   ./bin/ovsatool protect -i model/Person_Detection_Retail_0013_Model.bin model/Person_Detection_Retail_0013_Model.xml -n    "person detection" -d "person detection retail" -v 0002 -p artefacts/person_detection_model.dat -m artefacts/   person_detection_model_master.lic -k keystore/isvkeystore -g fcb6c944-7c69-4827-a47c-4c61a5731a03
   ```

### Step 3: Create a Runtime File
In this step you create a reference quote that you will provide to users who use your protected model.
1. Go to the `Ovsa_runtime` directory:
   ```sh
   cd ../Ovsa_runtime
   ```
2. Generate the reference quote for the runtime:
   ```sh
   ./bin/ovsaruntime gen-tcb-signature -n "Person Detect" -v "1.0" -f artefacts/person_detect.tcb -k keystore/isvkeystore
   ```

### Step 4: Generate a User License

This example creates a configuration file that is valid for 180 days. 

1. Go to the `Ovsa_tool` directory:
   ```sh
   cd ../Ovsa_tool
   ```
2. Create a license configuration file that is valid for 180 days:
   ```sh
   ./bin/ovsatool -t TimeLimit -l180 -n "Time Bound license" -v 1234.0.1 -u "14650@192.166.248.2" "http://mylicen.com"    "http:192.168.3.2" "localhost:4420" "localhost:4450" -k keystore/isvkeystore -o artefacts/license_time.conf
   ```	
3. In a real-life scenario, get a file named `custkeystore.csr.crt` from the user. For this example, generate your own certificate file:
   ```sh
   ./bin/ovsatool sale -m artefacts/person_detection_model_master.lic -k keystore/isvkeystore -l artefacts/license_inst.   conf -t artefacts/person_detect.tcb -p artefacts/custkeystore.csr.crt -c artefacts/person_detection_model.lic
   ```
4. Go to the `/DB` directory:
   ```sh
   cd ../DB
   ```
5. Generate the customer license:
   ```sh
   ./bin/ovsatool -t TimeLimit -l180 -n "Time Bound license" -v 1234.0.1 -u "14650@192.166.248.2" "http://mylicen.com"    "http:192.168.3.2" "localhost:4420" "localhost:4450" -k keystore/isvkeystore -o artefacts/license_time.conf
   ```	

Step 4: Start the Attestation Service and Host Server
1. Go to the Guest VM.
2. Start the attestation service. This must be run in the background:
   ```sh
   ./bin/attestation_server
   ```
3. Go to the Host Machine.
4. Go to the Host Server directory:
   ```sh
   cd <KVM Host Server path>
   ```
5. Start the Host Server:
   ```sh
   ./bin/ovsa_host_server
   ```

### Step 5. Test your Protected Model in a User Environment
Provide these items to the user:
* `person_detection_model.dat`
* `person_detection_model.lic`
* The User Instructions at the end of this document

### Step 6. Next Steps
You have successfully:
* Protected a model
* Applied a certificate file
* Defined license parameters
* Started a Licensing service
* Tested the model distribution 

You are ready to return to How to Use the OVSA to protect your own model. 
To do so:
1. Adapt the license configuration file to fit your own needs and those of your user. 
2. Update the User Instructions to refer to your model instead of the sample model.
3. Provide the items listed above to the user.

## User Instructions

### Step 1: Install the OpenVINO™ Security Add-on (OVSA) Run Time Software
<strong>The Word source file refers to Step 7, but the numbers are missing from the Word file. Need to copy/paste the correct steps here</strong>

### Step 2: Create a Key Store
1. Go to the `Ovsa_tool` directory.
   ```sh
   cd openvino-security/Ovsa_tool	
   ```
2. Create the key-store:
   ```sh
   ./bin/ovsatool keygen -storekey -t ECDSA -n Intel -k keystore/customer_one_keystore -r artefacts/customer_one_keystore.   csr -e "/C=IN/ST=KA/O=Intel, Inc./CN=intel.com/L=Bangalore/mail=xyz@intel.com"
   ```	
3. Contact your Certificate Authority (CA) and use `customer_one_keystore.csr` to get a CA-signed certificate in PEM format. Instructions to get the certificate are outside the scope of this document.

4. Add the certificate to the key store: 
   ```sh	
   ./bin/ovsatool keygen -storecert -c artefacts/customer_one_keystore.csr.crt -k keystore/customer_one_keystore
   ```
### Step 3: Start the Host Server
3. Go to the Host Machine.
4. Go to the Host Server directory:
   ```sh
   cd <KVM Host Server path>
   ```
5. Start the Host Server:
   ```sh
   ./bin/ovsa_host_server
   ```
### Step 4: Create an OpenVINO Model Server Configuration File	
1. Create a JSON that includes the following: **location and file name pending**:
   ```sh
   {
     "custom_loader_config_list":[
       {
           "config":{
           "loader_name":"ovsa-loader",
           "library_path": "/ovsa/openvino-security/Ovsa_runtime/lib/libovsaruntime.so"
           }
       }
       ],
       "model_config_list":[
       {
           "config":{
           "name":"sampleloader-model",
           "base_path":"/sampleloader/model/fdsample",
                      "custom_loader_options": {"loader_name":  "ovsa-loader", "keystore":  "custkeystore", "protected_file": "person_detection_model", "cert_path": "test-ca-sha384.crt"}
               }
       }
       ]
   }
   ```
### Step 5: Run the Model
1. Copy the artefacts to a folder that can be mounted to Docker. The artefacts include the key-store:
   ```sh
   docker run -d -v <path where protected artifacts are stored>:/sampleloader -p 9000:9000 openvino/model_server:latest    --config_path /sampleloader/sampleloader.json --port 9000  --log_level DEBUG
   ```
2. Perform inferencing with OVMS client on a separate terminal:
   ```sh
   python3 face_detection.py --batch_size 1 --width 300 --height 300 --input_images_dir images --output_dir results    --model_name sampleloader-model
   ```