# Install Intel® Distribution of OpenVINO™ toolkit for Linux* from a Docker* Image {#openvino_docs_install_guides_installing_openvino_docker_linux}

The Intel® Distribution of OpenVINO™ toolkit quickly deploys applications and solutions that emulate human vision. Based on Convolutional Neural Networks (CNN), the toolkit extends computer vision (CV) workloads across Intel® hardware, maximizing performance. The Intel® Distribution of OpenVINO™ toolkit includes the Intel® Deep Learning Deployment Toolkit.  

This guide provides the steps for creating a Docker* image with Intel® Distribution of OpenVINO™ toolkit for Linux* and further installation.  

## System Requirements

**Target Operating Systems**

- Ubuntu\* 18.04 long-term support (LTS), 64-bit

**Host Operating Systems**

- Linux with installed GPU driver and with Linux kernel supported by GPU driver

## Use Docker* Image for CPU

- Kernel reports the same information for all containers as for native application, for example, CPU, memory information.
- All instructions that are available to host process available for process in container, including, for example, AVX2, AVX512. No restrictions.
- Docker* does not use virtualization or emulation. The process in Docker* is just a regular Linux process, but it is isolated from external world on kernel level. Performance penalty is small.

### <a name="building-for-cpu"></a>Build a Docker* Image for CPU

To build a Docker image, create a `Dockerfile` that contains defined variables and commands required to create an OpenVINO toolkit installation image. 

Create your `Dockerfile` using the following example as a template:

<details>
  <summary>Click to expand/collapse</summary>

```sh
FROM ubuntu:18.04

USER root
WORKDIR /

SHELL ["/bin/bash", "-xo", "pipefail", "-c"]

# Creating user openvino
RUN useradd -ms /bin/bash openvino && \
    chown openvino -R /home/openvino

ARG DEPENDENCIES="autoconf \
                  automake \
                  build-essential \
                  cmake \
                  cpio \
                  curl \
                  gnupg2 \
                  libdrm2 \
                  libglib2.0-0 \
                  lsb-release \
                  libgtk-3-0 \
                  libtool \
                  udev \
                  unzip \
                  dos2unix"

RUN apt-get update && \
    apt-get install -y --no-install-recommends ${DEPENDENCIES} && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /thirdparty
RUN sed -Ei 's/# deb-src /deb-src /' /etc/apt/sources.list && \
    apt-get update && \
    apt-get source ${DEPENDENCIES} && \
    rm -rf /var/lib/apt/lists/*

# setup Python
ENV PYTHON python3.6

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-pip python3-dev lib${PYTHON}=3.6.9-1~18.04 && \
    rm -rf /var/lib/apt/lists/*

ARG package_url=http://registrationcenter-download.intel.com/akdlm/irc_nas/16612/l_openvino_toolkit_p_0000.0.000.tgz
ARG TEMP_DIR=/tmp/openvino_installer

WORKDIR ${TEMP_DIR}
ADD ${package_url} ${TEMP_DIR}

# install product by installation script
ENV INTEL_OPENVINO_DIR /opt/intel/openvino

RUN tar -xzf ${TEMP_DIR}/*.tgz --strip 1
RUN sed -i 's/decline/accept/g' silent.cfg && \
    ${TEMP_DIR}/install.sh -s silent.cfg && \
    ${INTEL_OPENVINO_DIR}/install_dependencies/install_openvino_dependencies.sh

WORKDIR /tmp
RUN rm -rf ${TEMP_DIR}

# installing dependencies for package
WORKDIR /tmp

RUN ${PYTHON} -m pip install --no-cache-dir setuptools && \
    find "${INTEL_OPENVINO_DIR}/" -type f -name "*requirements*.*" -path "*/${PYTHON}/*" -exec ${PYTHON} -m pip install --no-cache-dir -r "{}" \; && \
    find "${INTEL_OPENVINO_DIR}/" -type f -name "*requirements*.*" -not -path "*/post_training_optimization_toolkit/*" -not -name "*windows.txt"  -not -name "*ubuntu16.txt" -not -path "*/python3*/*" -not -path "*/python2*/*" -exec ${PYTHON} -m pip install --no-cache-dir -r "{}" \;

WORKDIR ${INTEL_OPENVINO_DIR}/deployment_tools/open_model_zoo/tools/accuracy_checker
RUN source ${INTEL_OPENVINO_DIR}/bin/setupvars.sh && \
    ${PYTHON} -m pip install --no-cache-dir -r ${INTEL_OPENVINO_DIR}/deployment_tools/open_model_zoo/tools/accuracy_checker/requirements.in && \
    ${PYTHON} ${INTEL_OPENVINO_DIR}/deployment_tools/open_model_zoo/tools/accuracy_checker/setup.py install

WORKDIR ${INTEL_OPENVINO_DIR}/deployment_tools/tools/post_training_optimization_toolkit
RUN if [ -f requirements.txt ]; then \
        ${PYTHON} -m pip install --no-cache-dir -r ${INTEL_OPENVINO_DIR}/deployment_tools/tools/post_training_optimization_toolkit/requirements.txt && \
        ${PYTHON} ${INTEL_OPENVINO_DIR}/deployment_tools/tools/post_training_optimization_toolkit/setup.py install; \
    fi;

# Post-installation cleanup and setting up OpenVINO environment variables
RUN if [ -f "${INTEL_OPENVINO_DIR}"/bin/setupvars.sh ]; then \
        printf "\nsource \${INTEL_OPENVINO_DIR}/bin/setupvars.sh\n" >> /home/openvino/.bashrc; \
        printf "\nsource \${INTEL_OPENVINO_DIR}/bin/setupvars.sh\n" >> /root/.bashrc; \
    fi;
RUN find "${INTEL_OPENVINO_DIR}/" -name "*.*sh" -type f -exec dos2unix {} \;

USER openvino
WORKDIR ${INTEL_OPENVINO_DIR}

CMD ["/bin/bash"]
```

</details>

> **NOTE**: Please replace direct link to the Intel® Distribution of OpenVINO™ toolkit package to the latest version in the `package_url` argument. You can copy the link from the [Intel® Distribution of OpenVINO™ toolkit download page](https://software.seek.intel.com/openvino-toolkit) after registration. Right click on **Offline Installer** button on the download page for Linux in your browser and press **Copy link address**.

You can select which OpenVINO components will be installed by modifying `COMPONENTS` parameter in the `silent.cfg` file. For example to install only CPU runtime for the Inference Engine, set 
`COMPONENTS=intel-openvino-ie-rt-cpu__x86_64` in `silent.cfg`.

To get a full list of available components for installation, run the `./install.sh --list_components` command from the unpacked OpenVINO™ toolkit package.

To build a Docker* image for CPU, run the following command:
```sh
docker build . -t <image_name> \
--build-arg HTTP_PROXY=<http://your_proxy_server.com:port> \
--build-arg HTTPS_PROXY=<https://your_proxy_server.com:port>
```

### Run the Docker* Image for CPU

Run the image with the following command:
```sh
docker run -it <image_name>
```
## Use a Docker* Image for GPU
### Build a Docker* Image for GPU

**Prerequisites:**
- GPU is not available in container by default, you must attach it to the container.
- Kernel driver must be installed on the host.
- Intel® OpenCL™ runtime package must be included into the container.
- In the container, user must be in the `video` group.

Before building a Docker* image on GPU, add the following commands to the `Dockerfile` example for CPU above:

```sh
WORKDIR /tmp/opencl
RUN usermod -aG video openvino
RUN apt-get update && \
    apt-get install -y --no-install-recommends ocl-icd-libopencl1 && \
    rm -rf /var/lib/apt/lists/* && \
    curl -L "https://github.com/intel/compute-runtime/releases/download/19.41.14441/intel-gmmlib_19.3.2_amd64.deb" --output "intel-gmmlib_19.3.2_amd64.deb" && \
    curl -L "https://github.com/intel/compute-runtime/releases/download/19.41.14441/intel-igc-core_1.0.2597_amd64.deb" --output "intel-igc-core_1.0.2597_amd64.deb" && \
    curl -L "https://github.com/intel/compute-runtime/releases/download/19.41.14441/intel-igc-opencl_1.0.2597_amd64.deb" --output "intel-igc-opencl_1.0.2597_amd64.deb" && \
    curl -L "https://github.com/intel/compute-runtime/releases/download/19.41.14441/intel-opencl_19.41.14441_amd64.deb" --output "intel-opencl_19.41.14441_amd64.deb" && \
    curl -L "https://github.com/intel/compute-runtime/releases/download/19.41.14441/intel-ocloc_19.04.12237_amd64.deb" --output "intel-ocloc_19.04.12237_amd64.deb" && \
    dpkg -i /tmp/opencl/*.deb && \
    ldconfig && \
    rm /tmp/opencl
```

To build a Docker* image for GPU, run the following command:
```sh
docker build . -t <image_name> \
--build-arg HTTP_PROXY=<http://your_proxy_server.com:port> \
--build-arg HTTPS_PROXY=<https://your_proxy_server.com:port>
```

### Run the Docker* Image for GPU

To make GPU available in the container, attach the GPU to the container using `--device /dev/dri` option and run the container:
```sh
docker run -it --device /dev/dri <image_name>
```

## Use a Docker* Image for Intel® Neural Compute Stick 2

### Build a Docker* Image for Intel® Neural Compute Stick 2

Build a Docker image using the same steps as for CPU.

### Run the Docker* Image for Intel® Neural Compute Stick 2

**Known limitations:**

- Intel® Neural Compute Stick 2 device changes its VendorID and DeviceID during execution and each time looks for a host system as a brand new device. It means it cannot be mounted as usual.
- UDEV events are not forwarded to the container by default it does not know about device reconnection.
- Only one device per host is supported.

Use one of the following options to run **Possible solutions for Intel® Neural Compute Stick 2:**

- **Solution #1**:
	1. Get rid of UDEV by rebuilding `libusb` without UDEV support in the Docker* image (add the following commands to the `Dockerfile` example for CPU above):<br>
```sh
RUN usermod -aG users openvino
WORKDIR /opt
RUN curl -L https://github.com/libusb/libusb/archive/v1.0.22.zip --output v1.0.22.zip && \
    unzip v1.0.22.zip

WORKDIR /opt/libusb-1.0.22
RUN ./bootstrap.sh && \
    ./configure --disable-udev --enable-shared && \
    make -j4
RUN apt-get update && \
    apt-get install -y --no-install-recommends libusb-1.0-0-dev=2:1.0.21-2 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/libusb-1.0.22/libusb
RUN /bin/mkdir -p '/usr/local/lib' && \
    /bin/bash ../libtool --mode=install /usr/bin/install -c   libusb-1.0.la '/usr/local/lib' && \
    /bin/mkdir -p '/usr/local/include/libusb-1.0' && \
    /usr/bin/install -c -m 644 libusb.h '/usr/local/include/libusb-1.0' && \
    /bin/mkdir -p '/usr/local/lib/pkgconfig'

WORKDIR /opt/libusb-1.0.22/
RUN /usr/bin/install -c -m 644 libusb-1.0.pc '/usr/local/lib/pkgconfig' && \
    ldconfig
```
<br>
	2. Run the Docker* image:<br>
```sh
docker run --device-cgroup-rule='c 189:* rmw' -v /dev/bus/usb:/dev/bus/usb <image_name>
```

- **Solution #2**:
   Run container in privileged mode, enable Docker network configuration as host, and mount all devices to container:<br>
```sh
docker run --privileged -v /dev:/dev --network=host <image_name>
```

> **Notes**:
> - It is not secure
> - Conflicts with Kubernetes* and other tools that use orchestration and private networks

## Use a Docker* Image for Intel® Vision Accelerator Design with Intel® Movidius™ VPUs

### Build Docker* Image for Intel® Vision Accelerator Design with Intel® Movidius™ VPUs
To use the Docker container for inference on Intel® Vision Accelerator Design with Intel® Movidius™ VPUs:

1. Set up the environment on the host machine, that is going to be used for running Docker*. It is required to execute `hddldaemon`, which is responsible for communication between the HDDL plugin and the board. To learn how to set up the environment (the OpenVINO package must be pre-installed), see [Configuration Guide for Intel® Vision Accelerator Design with Intel® Movidius™ VPUs](installing-openvino-linux-ivad-vpu.md).
2. Prepare the Docker* image. As a base image, you can use the image from the section [Building Docker Image for CPU](#building-for-cpu). To use it for inference on Intel® Vision Accelerator Design with Intel® Movidius™ VPUs you need to rebuild the image with adding the following dependencies:
```sh
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libboost-filesystem1.65-dev=1.65.1+dfsg-0ubuntu5 \
        libboost-thread1.65-dev=1.65.1+dfsg-0ubuntu5 \
        libjson-c3=0.12.1-1.3 libxxf86vm-dev=1:1.1.4-1 && \
    rm -rf /var/lib/apt/lists/*
```
3. Run `hddldaemon` on the host in a separate terminal session using the following command:
```sh
$HDDL_INSTALL_DIR/hddldaemon
```

### Run the Docker* Image for Intel® Vision Accelerator Design with Intel® Movidius™ VPUs
To run the built Docker* image for Intel® Vision Accelerator Design with Intel® Movidius™ VPUs, use the following command:
```sh
docker run --device=/dev/ion:/dev/ion -v /var/tmp:/var/tmp -ti <image_name>
```

> **NOTE**:
> - The device `/dev/ion` need to be shared to be able to use ion buffers among the plugin, `hddldaemon` and the kernel.
> - Since separate inference tasks share the same HDDL service communication interface (the service creates mutexes and a socket file in `/var/tmp`), `/var/tmp` needs to be mounted and shared among them.

In some cases, the ion driver is not enabled (for example, due to a newer kernel version or iommu incompatibility). `lsmod | grep myd_ion` returns empty output. To resolve, use the following command:
```sh
docker run --rm --net=host -v /var/tmp:/var/tmp –ipc=host  -ti <image_name>
```
> **NOTE**:
> - When building docker images, create a user in the docker file that has the same UID and GID as the user which runs hddldaemon on the host.
> - Run the application in the docker with this user.
> - Alternatively, you can start hddldaemon with the root user on host, but this approach is not recommended.

## Use a Docker* Image for FPGA
### Build a Docker* Image for FPGA

FPGA card is not available in container by default, but it can be mounted there with the following pre-requisites:
- FPGA device is up and ready to run inference.
- FPGA bitstreams were pushed to the device over PCIe.

To build a Docker* image for FPGA:

1. Set additional environment variables in the `Dockerfile`:<br>
```sh
ENV CL_CONTEXT_COMPILER_MODE_INTELFPGA=3
ENV DLA_AOCX=/opt/intel/openvino/a10_devkit_bitstreams/2-0-1_RC_FP11_Generic.aocx
ENV PATH=/opt/altera/aocl-pro-rte/aclrte-linux64/bin:$PATH
```
2. Install the following UDEV rule:<br>
```sh
cat <<EOF > fpga.rules
KERNEL=="acla10_ref*",GROUP="users",MODE="0660"
EOF
sudo cp fpga.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
sudo ldconfig
```
Make sure that a container user is added to the "users" group with the same GID as on host.

### Run the Docker* container for FPGA 

To run the built Docker* container for FPGA, use the following command:

```sh
docker run --rm -it \
--mount type=bind,source=/opt/intel/intelFPGA_pro,destination=/opt/intel/intelFPGA_pro \
--mount type=bind,source=/opt/altera,destination=/opt/altera \
--mount type=bind,source=/etc/OpenCL/vendors,destination=/etc/OpenCL/vendors \
--mount type=bind,source=/opt/Intel/OpenCL/Boards,destination=/opt/Intel/OpenCL/Boards \
--device /dev/acla10_ref0:/dev/acla10_ref0 \
<image_name>
```

## Examples
* [ubuntu18_runtime dockerfile](https://docs.openvinotoolkit.org/downloads/ubuntu18_runtime.dockerfile) - Can be used to build OpenVINO™ runtime image containing minimal dependencies needed to use OpenVINO™ in production environment.
* [ubuntu18_dev dockerfile](https://docs.openvinotoolkit.org/downloads/ubuntu18_dev.dockerfile) - Can be used to build OpenVINO™ developer image containing full OpenVINO™ package to use in development environment.

## Additional Resources

* Intel® Distribution of OpenVINO™ toolkit home page: [https://software.intel.com/en-us/openvino-toolkit](https://software.intel.com/en-us/openvino-toolkit)  

* OpenVINO™ toolkit documentation: [https://docs.openvinotoolkit.org](https://docs.openvinotoolkit.org)

* Intel® Neural Compute Stick 2 Get Started: [https://software.intel.com/en-us/neural-compute-stick/get-started](https://software.intel.com/en-us/neural-compute-stick/get-started)

* Intel® Distribution of OpenVINO™ toolkit Docker Hub* home page: [https://hub.docker.com/u/openvino](https://hub.docker.com/u/openvino)
