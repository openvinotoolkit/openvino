# Configurations for Intel® Processor Graphics (GPU) with OpenVINO™ {#openvino_docs_install_guides_configurations_for_intel_gpu}


@sphinxdirective

.. _gpu guide:


To use the OpenVINO™ GPU plugin and offload inference to Intel® Processor Graphics (GPU), Intel® Graphics Driver must be properly configured on your system.


Linux
#####

In oder to use GPU device for inference with OpenVINO you need to meet the following prerequisites:

- use supported Linux kernel according to documentatin from https://dgpu-docs.intel.com/driver/kernel-driver-types.html
- install GPU runtime drivers: The Intel(R) Graphics Compute Runtime for oneAPI Level Zero and OpenCL(TM) Driver
- install Intel Graphics Memory Management Library and Intel® Graphics Compiler for OpenCL™
- install OpenCl ICD loader package

Depending on OS, there might be different methods for installing the packages above

Ubuntu22:
Download and install the `deb` packages published on https://github.com/intel/compute-runtime/releases/latest
and install apt package `ocl-icd-libopencl1` with OpenCl ICD loader.

Altenrnatively add apt repository like described on https://dgpu-docs.intel.com/driver/installation.html
For example for ARC device use command:
```
apt-get update && apt-get install -y gpg gpg-agent curl
curl https://repositories.intel.com/graphics/intel-graphics.key | \
  gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg
echo 'deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu jammy arc' | \
  tee  /etc/apt/sources.list.d/intel.gpu.jammy.list
```
Install the apt packages: `ocl-icd-libopencl1`, `intel-opencl-icd`, `intel-level-zero-gpu`, `level-zero`.
```
apt-get install -y ocl-icd-libopencl1 intel-opencl-icd intel-level-zero-gpu level-zero
```

Ubuntu20:
Ubuntu20 is not uppdated with the latest driver versions. You can install the versions up to version 22.43 from apt. 
```
apt-get update && apt-get install -y --no-install-recommends curl gpg gpg-agent && \
curl https://repositories.intel.com/graphics/intel-graphics.key | gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg && \
echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu focal-legacy main' | tee  /etc/apt/sources.list.d/intel.gpu.focal.list && \
apt-get update
apt-get update && apt-get install -y --no-install-recommends intel-opencl-icd intel-level-zero-gpu level-zero
```
Alternatively download older `deb` version from https://github.com/intel/compute-runtime/releases
Note that older driver version might not include some of the bug fixes and might be not supported on some latest platforms. Check the supported HW for the versions you are installing.

RH8:
Follow the instruction from https://dgpu-docs.intel.com/driver/installation.html#red-hat-enterprise-linux-8-6 to add yum repository.
Install packages: `intel-opencl`,`level-zero`,`intel-level-zero-gpu` ,`intel-igc-core`, `intel-igc-cm`, `intel-gmmlib`, `intel-ocloc`.
Install OpenCL ICD Loader via `rpm -ivh http://mirror.centos.org/centos/8-stream/AppStream/x86_64/os/Packages/ocl-icd-2.2.12-1.el8.x86_64.rpm`


.. _gpu guide windows:


Windows
#######

On 

To install the Intel Graphics Driver for Windows on your system, follow the `driver installation guide <https://www.intel.com/content/www/us/en/support/articles/000005629/graphics.html>`_.


To check if you have this driver installed:

1. Type **device manager** in your **Search Windows** box and press Enter. The **Device Manager** opens.
2. Click the drop-down arrow to view the **Display adapters**. You can see the adapter that is installed in your computer:  

   .. image:: _static/images/DeviceManager.PNG
      :width: 400

3. Right-click the adapter name and select **Properties**.
4. Click the **Driver** tab to see the driver version. 

   .. image:: _static/images/DeviceDriverVersion.PNG
   current version is 31.

   Note: this version is old and not supported on latest GPU

      :width: 400

You are done updating your device driver and ready to use your GPU.


Windows Subsystem for Linux (WSL)
######
WSL lets the developers to run a GNU/Linux development environment on Windows operating system. Using GPU on WSL is very similar to a native Linux environment.
Here are the steps to make it work with OpenVINO:
- Installed GPU drivers like described above
- Use the latest version of WSL2- in PowerShell run `wsl --update` and `wsl --shutdown`
- While Ubuntu20 or Ubuntu22 is started, install the same drivers like described above in the Linux section

Note: In WSL, GPU device is accessible via a character device `/dev/drx` versus `/dev/dri` on a native Linux OS.



Additional info
###############

For your reference, the following versions of Intel® Graphics Driver were used in the OpenVINO internal validation:

+------------------+-------------------------------------------------------------------------------------------+
| Operation System | Driver version                                                                            |
+==================+===========================================================================================+
| Ubuntu 22.04     | `22.43.24595.30 <https://github.com/intel/compute-runtime/releases/tag/22.43.24595.30>`__ |
+------------------+-------------------------------------------------------------------------------------------+
| Ubuntu 20.04     | `22.35.24055 <https://github.com/intel/compute-runtime/releases/tag/22.35.24055>`__       |
+------------------+-------------------------------------------------------------------------------------------+
| Ubuntu 18.04     | `21.38.21026 <https://github.com/intel/compute-runtime/releases/tag/21.38.21026>`__       |
+------------------+-------------------------------------------------------------------------------------------+
| CentOS 7         | `19.41.14441 <https://github.com/intel/compute-runtime/releases/tag/19.41.14441>`__       |
+------------------+-------------------------------------------------------------------------------------------+
| RHEL 8           | `22.28.23726 <https://github.com/intel/compute-runtime/releases/tag/22.28.23726>`__       |
+------------------+-------------------------------------------------------------------------------------------+


What’s Next?
############

You can try out the toolkit with:


Link to GPU plugin
Link to GPU benchmarking
Link to a demo with GPU
Using containers with GPU

@endsphinxdirective


