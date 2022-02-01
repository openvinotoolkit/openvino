# Configuration Guide for the Intel® Distribution of OpenVINO™ toolkit and the Intel® Vision Accelerator Design with Intel® Movidius™ VPUs on Windows* {#openvino_docs_install_guides_installing_openvino_windows_ivad_vpu}

@sphinxdirective

.. _vpu guide windows:

@endsphinxdirective


To enable inference on Intel® Vision Accelerator Design with Intel® Movidius™ VPUs, the following additional installation steps are required:

  1. Download and install <a href="https://www.microsoft.com/en-us/download/details.aspx?id=48145">Visual C++ Redistributable for Visual Studio 2017</a>
  2. Check with a support engineer if your Intel® Vision Accelerator Design with Intel® Movidius™ VPUs card requires SMBUS connection to PCIe slot (most unlikely). Install the SMBUS driver only if confirmed (by default, it's not required):
      1. Go to the `<INSTALL_DIR>\deployment_tools\inference-engine\external\hddl\drivers\SMBusDriver` directory, where `<INSTALL_DIR>` is the directory in which the Intel Distribution of OpenVINO™ toolkit is installed.
      2. Right click on the `hddlsmbus.inf` file and choose **Install** from the pop up menu.

You are done installing your device driver and are ready to use your Intel® Vision Accelerator Design with Intel® Movidius™ VPUs.

See also: 

* For advanced configuration steps for your IEI Mustang-V100-MX8 accelerator, see [Intel® Movidius™ VPUs Setup Guide for Use with Intel® Distribution of OpenVINO™ toolkit](movidius-setup-guide.md).

* After you've configured your Intel® Vision Accelerator Design with Intel® Movidius™ VPUs, see [Intel® Movidius™ VPUs Programming Guide for Use with Intel® Distribution of OpenVINO™ toolkit](movidius-programming-guide.md) to learn how to distribute a model across all 8 VPUs to maximize performance.

After configuration is done, you are ready to go to <a href="#get-started">Start Using the Toolkit</a> section to learn the basic OpenVINO™ toolkit workflow and run code samples and demo applications.
