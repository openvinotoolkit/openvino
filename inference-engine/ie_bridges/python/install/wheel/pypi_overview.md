## OpenVINO™ Toolkit

OpenVINO™ toolkit quickly deploys applications and solutions that emulate human vision. Based on Convolutional Neural Networks (CNNs), the toolkit extends computer vision (CV) workloads across Intel® hardware, maximizing performance. The OpenVINO™ toolkit includes the Deep Learning Deployment Toolkit (DLDT).

OpenVINO™ toolkit:

- Enables CNN-based deep learning inference on the edge
- Supports heterogeneous execution across an Intel® CPU, Intel® Integrated Graphics, Intel® FPGA,  Intel® Neural Compute Stick 2, and Intel® Vision Accelerator Design with Intel® Movidius™ VPUs
- Speeds time-to-market via an easy-to-use library of computer vision functions and pre-optimized kernels
- Includes optimized calls for computer vision standards, including OpenCV\* and OpenCL™

Operating Systems:
- Ubuntu* 18.04 long-term support (LTS), 64-bit
- Windows* 10, 64-bit
- macOS* 10.15, 64-bit

## Install the Runtime Package Using the PyPI Repository
1. Set up and update pip to the highest version:
   ```sh
   python3 -m pip install --upgrade pip
   ```
2. Install the Intel® distribution of OpenVINO™ toolkit:
   ```sh
   pip install openvino-python
   ```
3. Add PATH to environment variables.
 - Ubuntu* 18.04 and macOS*:
   ```sh
   export LD_LIBRARY_PATH=<library_dir>:${LD_LIBRARY_PATH}
   ```
 - Windows* 10:
    ```sh
   set PATH=<library_dir>;%PATH%
   ```
  How to find `library_dir`:
 - Ubuntu\*, macOS\*:
   - standard user:
     ```sh
     echo $(python3 -m site --user-base)/lib
     ```
   - root or sudo user:
     ```sh
     /usr/local/lib
     ```
   - virtual environments or custom Python installations (from sources or tarball):
     ```sh
     echo $(which python3)/../../lib
     ```
 - Windows\*:
   - standard Python:
     ```sh
      python -c "import os, sys; print((os.path.dirname(sys.executable))+'\Library\\bin')"
     ```
   - virtual environments or custom Python installations (from sources or tarball):
     ```sh
      python -c "import os, sys; print((os.path.dirname(sys.executable))+'\..\Library\\bin')"
     ```
4. Verify that the package is installed:
   ```sh
   python3 -c "import openvino"
   ```
   
Now you are ready to develop and run your application.