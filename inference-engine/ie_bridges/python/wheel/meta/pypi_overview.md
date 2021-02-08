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
   pip install openvino
   ```

3. Verify that the package is installed:
   ```sh
   python3 -c "from openvino.inference_engine import IECore"
   ```
   
Now you are ready to develop and run your application.