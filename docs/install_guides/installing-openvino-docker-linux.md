# Install Intel® Distribution of OpenVINO™ toolkit for Linux* from a Docker* Image {#openvino_docs_install_guides_installing_openvino_docker_linux}

The Intel® Distribution of OpenVINO™ toolkit quickly deploys applications and solutions that emulate human vision. Based on Convolutional Neural Networks (CNN), the toolkit extends computer vision (CV) workloads across Intel® hardware, maximizing performance. The Intel® Distribution of OpenVINO™ toolkit includes the Intel® Deep Learning Deployment Toolkit.  

This guide provides the steps for creating a Docker* image with Intel® Distribution of OpenVINO™ toolkit for Linux* and further installation.  

## System Requirements

**Target Operating Systems**

- Ubuntu\* 18.04 long-term support (LTS), 64-bit
- Ubuntu\* 20.04 long-term support (LTS), 64-bit
- CentOS\* 7
- RHEL\* 8

**Host Operating Systems**

- Linux with installed GPU driver and with Linux kernel supported by GPU driver

## Prebuilt images

Prebuilt images are available on [Docker Hub](https://hub.docker.com/u/openvino).

## Use Docker* Image for CPU

- Kernel reports the same information for all containers as for native application, for example, CPU, memory information.
- All instructions that are available to host process available for process in container, including, for example, AVX2, AVX512. No restrictions.
- Docker* does not use virtualization or emulation. The process in Docker* is just a regular Linux process, but it is isolated from external world on kernel level. Performance penalty is small.

### <a name="building-for-cpu"></a>Build a Docker* Image for CPU

You can use [available Dockerfiles](https://github.com/openvinotoolkit/docker_ci/tree/master/dockerfiles) or generate a Dockerfile with your setting via [DockerHub CI Framework](https://github.com/openvinotoolkit/docker_ci) for Intel® Distribution of OpenVINO™ toolkit. 
The Framework can generate a Dockerfile, build, test, and deploy an image with the Intel® Distribution of OpenVINO™ toolkit.

### Run the Docker* Image for CPU

Run the image with the following command:
```sh
docker run -it --rm <image_name>
```
## Use a Docker* Image for GPU
### Build a Docker* Image for GPU

**Prerequisites:**
- GPU is not available in container by default, you must attach it to the container.
- Kernel driver must be installed on the host.
- Intel® OpenCL™ runtime package must be included into the container.
- In the container, user must be in the `video` group.

Before building a Docker* image on GPU, add the following commands to a Dockerfile:

**Ubuntu 18.04/20.04**:
```sh
WORKDIR /tmp/opencl
RUN useradd -ms /bin/bash -G video,users openvino && \
    chown openvino -R /home/openvino
RUN apt-get update && \
    apt-get install -y --no-install-recommends ocl-icd-libopencl1 && \
    rm -rf /var/lib/apt/lists/* && \
    curl -L "https://github.com/intel/compute-runtime/releases/download/19.41.14441/intel-gmmlib_19.3.2_amd64.deb" --output "intel-gmmlib_19.3.2_amd64.deb" && \
    curl -L "https://github.com/intel/compute-runtime/releases/download/19.41.14441/intel-igc-core_1.0.2597_amd64.deb" --output "intel-igc-core_1.0.2597_amd64.deb" && \
    curl -L "https://github.com/intel/compute-runtime/releases/download/19.41.14441/intel-igc-opencl_1.0.2597_amd64.deb" --output "intel-igc-opencl_1.0.2597_amd64.deb" && \
    curl -L "https://github.com/intel/compute-runtime/releases/download/19.41.14441/intel-opencl_19.41.14441_amd64.deb" --output "intel-opencl_19.41.14441_amd64.deb" && \
    curl -L "https://github.com/intel/compute-runtime/releases/download/19.41.14441/intel-ocloc_19.41.14441_amd64.deb" --output "intel-ocloc_19.04.12237_amd64.deb" && \
    dpkg -i /tmp/opencl/*.deb && \
    ldconfig && \
    rm /tmp/opencl
```
**CentOS 7/RHEL 8**:
```sh
WORKDIR /tmp/opencl
RUN useradd -ms /bin/bash -G video,users openvino && \
    chown openvino -R /home/openvino
RUN groupmod -g 44 video

RUN yum update -y && yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm && \
    yum update -y && yum install -y ocl-icd ocl-icd-devel && \ 
    yum clean all && rm -rf /var/cache/yum && \
    curl -L https://sourceforge.net/projects/intel-compute-runtime/files/19.41.14441/centos-7/intel-gmmlib-19.3.2-1.el7.x86_64.rpm/download -o intel-gmmlib-19.3.2-1.el7.x86_64.rpm && \
    curl -L https://sourceforge.net/projects/intel-compute-runtime/files/19.41.14441/centos-7/intel-gmmlib-devel-19.3.2-1.el7.x86_64.rpm/download -o intel-gmmlib-devel-19.3.2-1.el7.x86_64.rpm && \
    curl -L https://sourceforge.net/projects/intel-compute-runtime/files/19.41.14441/centos-7/intel-igc-core-1.0.2597-1.el7.x86_64.rpm/download -o intel-igc-core-1.0.2597-1.el7.x86_64.rpm && \
    curl -L https://sourceforge.net/projects/intel-compute-runtime/files/19.41.14441/centos-7/intel-igc-opencl-1.0.2597-1.el7.x86_64.rpm/download -o intel-igc-opencl-1.0.2597-1.el7.x86_64.rpm && \
    curl -L https://sourceforge.net/projects/intel-compute-runtime/files/19.41.14441/centos-7/intel-igc-opencl-devel-1.0.2597-1.el7.x86_64.rpm/download -o  intel-igc-opencl-devel-1.0.2597-1.el7.x86_64.rpm && \
    curl -L https://sourceforge.net/projects/intel-compute-runtime/files/19.41.14441/centos-7/intel-opencl-19.41.14441-1.el7.x86_64.rpm/download -o intel-opencl-19.41.14441-1.el7.x86_64.rpm \
    rpm -ivh ${TEMP_DIR}/*.rpm && \
    ldconfig && \
    rm -rf ${TEMP_DIR} && \
    yum remove -y epel-release
```

### Run the Docker* Image for GPU

To make GPU available in the container, attach the GPU to the container using `--device /dev/dri` option and run the container:
```sh
docker run -it --rm --device /dev/dri <image_name>
```

## Use a Docker* Image for Intel® Neural Compute Stick 2

### Build and Run the Docker* Image for Intel® Neural Compute Stick 2

**Known limitations:**

- Intel® Neural Compute Stick 2 device changes its VendorID and DeviceID during execution and each time looks for a host system as a brand new device. It means it cannot be mounted as usual.
- UDEV events are not forwarded to the container by default it does not know about device reconnection.
- Only one device per host is supported.

Use one of the following options as **Possible solutions for Intel® Neural Compute Stick 2:**

#### Option #1
1. Get rid of UDEV by rebuilding `libusb` without UDEV support in the Docker* image (add the following commands to a `Dockerfile`):
   - **Ubuntu 18.04/20.04**:
```sh
ARG BUILD_DEPENDENCIES="autoconf \
                        automake \
                        build-essential \
                        libtool \
                        unzip \
                        udev"
RUN apt-get update && \
    apt-get install -y --no-install-recommends ${BUILD_DEPENDENCIES} && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN curl -L https://github.com/libusb/libusb/archive/v1.0.22.zip --output v1.0.22.zip && \
    unzip v1.0.22.zip

WORKDIR /opt/libusb-1.0.22
RUN ./bootstrap.sh && \
    ./configure --disable-udev --enable-shared && \
    make -j4

WORKDIR /opt/libusb-1.0.22/libusb
RUN /bin/mkdir -p '/usr/local/lib' && \
    /bin/bash ../libtool --mode=install /usr/bin/install -c   libusb-1.0.la '/usr/local/lib' && \
    /bin/mkdir -p '/usr/local/include/libusb-1.0' && \
    /usr/bin/install -c -m 644 libusb.h '/usr/local/include/libusb-1.0' && \
    /bin/mkdir -p '/usr/local/lib/pkgconfig'

WORKDIR /opt/libusb-1.0.22/
RUN /usr/bin/install -c -m 644 libusb-1.0.pc '/usr/local/lib/pkgconfig' && \
    cp /opt/intel/openvino/deployment_tools/inference_engine/external/97-myriad-usbboot.rules /etc/udev/rules.d/ && \
    ldconfig
```
   - **CentOS 7**:
```sh
ARG BUILD_DEPENDENCIES="autoconf \
                        automake \
                        libtool \
                        unzip \
                        udev"

# hadolint ignore=DL3031, DL3033
RUN yum update -y && yum install -y ${BUILD_DEPENDENCIES} && \
    yum group install -y "Development Tools" && \
    yum clean all && rm -rf /var/cache/yum

WORKDIR /opt
RUN curl -L https://github.com/libusb/libusb/archive/v1.0.22.zip --output v1.0.22.zip && \
    unzip v1.0.22.zip && rm -rf v1.0.22.zip

WORKDIR /opt/libusb-1.0.22
RUN ./bootstrap.sh && \
    ./configure --disable-udev --enable-shared && \
    make -j4

WORKDIR /opt/libusb-1.0.22/libusb
RUN /bin/mkdir -p '/usr/local/lib' && \
    /bin/bash ../libtool   --mode=install /usr/bin/install -c   libusb-1.0.la '/usr/local/lib' && \
    /bin/mkdir -p '/usr/local/include/libusb-1.0' && \
    /usr/bin/install -c -m 644 libusb.h '/usr/local/include/libusb-1.0' && \
    /bin/mkdir -p '/usr/local/lib/pkgconfig' && \
    printf "\nexport LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:/usr/local/lib\n" >> /opt/intel/openvino/bin/setupvars.sh

WORKDIR /opt/libusb-1.0.22/
RUN /usr/bin/install -c -m 644 libusb-1.0.pc '/usr/local/lib/pkgconfig' && \
    cp /opt/intel/openvino/deployment_tools/inference_engine/external/97-myriad-usbboot.rules /etc/udev/rules.d/ && \
    ldconfig
```
2. Run the Docker* image:
```sh
docker run -it --rm --device-cgroup-rule='c 189:* rmw' -v /dev/bus/usb:/dev/bus/usb <image_name>
```

#### Option #2
Run container in the privileged mode, enable the Docker network configuration as host, and mount all devices to the container:
```sh
docker run -it --rm --privileged -v /dev:/dev --network=host <image_name>
```
> **NOTES**:
> - It is not secure.
> - Conflicts with Kubernetes* and other tools that use orchestration and private networks may occur.

## Use a Docker* Image for Intel® Vision Accelerator Design with Intel® Movidius™ VPUs

### Build Docker* Image for Intel® Vision Accelerator Design with Intel® Movidius™ VPUs
To use the Docker container for inference on Intel® Vision Accelerator Design with Intel® Movidius™ VPUs:

1. Set up the environment on the host machine, that is going to be used for running Docker*. 
It is required to execute `hddldaemon`, which is responsible for communication between the HDDL plugin and the board. 
To learn how to set up the environment (the OpenVINO package or HDDL package must be pre-installed), see [Configuration guide for HDDL device](https://github.com/openvinotoolkit/docker_ci/blob/master/install_guide_vpu_hddl.md) or [Configuration Guide for Intel® Vision Accelerator Design with Intel® Movidius™ VPUs](installing-openvino-linux-ivad-vpu.md).
2. Prepare the Docker* image (add the following commands to a Dockerfile).
   - **Ubuntu 18.04**:
```sh
WORKDIR /tmp
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libboost-filesystem1.65-dev \
        libboost-thread1.65-dev \
        libjson-c3 libxxf86vm-dev && \
    rm -rf /var/lib/apt/lists/* && rm -rf /tmp/*
```
   - **Ubuntu 20.04**:
```sh
WORKDIR /tmp
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libboost-filesystem-dev \
        libboost-thread-dev \
        libjson-c4 \
        libxxf86vm-dev && \
    rm -rf /var/lib/apt/lists/* && rm -rf /tmp/*
```
   - **CentOS 7**:
```sh
WORKDIR /tmp
RUN yum update -y && yum install -y \
        boost-filesystem \
        boost-thread \
        boost-program-options \
        boost-system \
        boost-chrono \
        boost-date-time \
        boost-regex \
        boost-atomic \
        json-c \
        libXxf86vm-devel && \
    yum clean all && rm -rf /var/cache/yum
```
3. Run `hddldaemon` on the host in a separate terminal session using the following command:
```sh
$HDDL_INSTALL_DIR/hddldaemon
```

### Run the Docker* Image for Intel® Vision Accelerator Design with Intel® Movidius™ VPUs
To run the built Docker* image for Intel® Vision Accelerator Design with Intel® Movidius™ VPUs, use the following command:
```sh
docker run -it --rm --device=/dev/ion:/dev/ion -v /var/tmp:/var/tmp <image_name>
```

> **NOTES**:
> - The device `/dev/ion` need to be shared to be able to use ion buffers among the plugin, `hddldaemon` and the kernel.
> - Since separate inference tasks share the same HDDL service communication interface (the service creates mutexes and a socket file in `/var/tmp`), `/var/tmp` needs to be mounted and shared among them.

In some cases, the ion driver is not enabled (for example, due to a newer kernel version or iommu incompatibility). `lsmod | grep myd_ion` returns empty output. To resolve, use the following command:
```sh
docker run -it --rm --net=host -v /var/tmp:/var/tmp –ipc=host <image_name>
```
> **NOTES**:
> - When building docker images, create a user in the docker file that has the same UID and GID as the user which runs hddldaemon on the host.
> - Run the application in the docker with this user.
> - Alternatively, you can start hddldaemon with the root user on host, but this approach is not recommended.

### Run Demos in the Docker* Image 

To run the Security Barrier Camera Demo on a specific inference device, run the following commands with the root privileges (additional third-party dependencies will be installed):

**CPU**:
```sh
docker run -itu root:root --rm --device=/dev/ion:/dev/ion -v /var/tmp:/var/tmp --device /dev/dri:/dev/dri --device-cgroup-rule='c 189:* rmw' -v /dev/bus/usb:/dev/bus/usb <image_name>
/bin/bash -c "apt update && apt install sudo && deployment_tools/demo/demo_security_barrier_camera.sh -d CPU -sample-options -no_show"
```

**GPU**:
```sh
docker run -itu root:root --rm --device=/dev/ion:/dev/ion -v /var/tmp:/var/tmp --device /dev/dri:/dev/dri --device-cgroup-rule='c 189:* rmw' -v /dev/bus/usb:/dev/bus/usb <image_name>
/bin/bash -c "apt update && apt install sudo && deployment_tools/demo/demo_security_barrier_camera.sh -d GPU -sample-options -no_show"
```

**MYRIAD**:
```sh
docker run -itu root:root --rm --device=/dev/ion:/dev/ion -v /var/tmp:/var/tmp --device /dev/dri:/dev/dri --device-cgroup-rule='c 189:* rmw' -v /dev/bus/usb:/dev/bus/usb <image_name>
/bin/bash -c "apt update && apt install sudo && deployment_tools/demo/demo_security_barrier_camera.sh -d MYRIAD -sample-options -no_show"
```

**HDDL**:
```sh
docker run -itu root:root --rm --device=/dev/ion:/dev/ion -v /var/tmp:/var/tmp --device /dev/dri:/dev/dri --device-cgroup-rule='c 189:* rmw' -v /dev/bus/usb:/dev/bus/usb <image_name>
/bin/bash -c "apt update && apt install sudo && deployment_tools/demo/demo_security_barrier_camera.sh -d HDDL -sample-options -no_show"
```

## Use a Docker* Image for FPGA

Intel will be transitioning to the next-generation programmable deep-learning solution based on FPGAs in order to increase the level of customization possible in FPGA deep-learning. As part of this transition, future standard releases (i.e., non-LTS releases) of Intel® Distribution of OpenVINO™ toolkit will no longer include the Intel® Vision Accelerator Design with an Intel® Arria® 10 FPGA and the Intel® Programmable Acceleration Card with Intel® Arria® 10 GX FPGA.

Intel® Distribution of OpenVINO™ toolkit 2020.3.X LTS release will continue to support Intel® Vision Accelerator Design with an Intel® Arria® 10 FPGA and the Intel® Programmable Acceleration Card with Intel® Arria® 10 GX FPGA. For questions about next-generation programmable deep-learning solutions based on FPGAs, please talk to your sales representative or contact us to get the latest FPGA updates.

For instructions for previous releases with FPGA Support, see documentation for the [2020.4 version](https://docs.openvinotoolkit.org/2020.4/openvino_docs_install_guides_installing_openvino_docker_linux.html#use_a_docker_image_for_fpga) or lower.

## Troubleshooting

If you got proxy issues, please setup proxy settings for Docker. See the Proxy section in the [Install the DL Workbench from Docker Hub* ](@ref workbench_docs_Workbench_DG_Install_from_Docker_Hub) topic.

## Additional Resources

* [DockerHub CI Framework](https://github.com/openvinotoolkit/docker_ci) for Intel® Distribution of OpenVINO™ toolkit. The Framework can generate a Dockerfile, build, test, and deploy an image with the Intel® Distribution of OpenVINO™ toolkit. You can reuse available Dockerfiles, add your layer and customize the image of OpenVINO™ for your needs.

* Intel® Distribution of OpenVINO™ toolkit home page: [https://software.intel.com/en-us/openvino-toolkit](https://software.intel.com/en-us/openvino-toolkit)  

* OpenVINO™ toolkit documentation: [https://docs.openvinotoolkit.org](https://docs.openvinotoolkit.org)

* Intel® Neural Compute Stick 2 Get Started: [https://software.intel.com/en-us/neural-compute-stick/get-started](https://software.intel.com/en-us/neural-compute-stick/get-started)

* Intel® Distribution of OpenVINO™ toolkit Docker Hub* home page: [https://hub.docker.com/u/openvino](https://hub.docker.com/u/openvino)
