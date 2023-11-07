# Post-training Optimization Tool

Starting with the 2020.1 version, OpenVINO&trade; toolkit delivers the Post-Training Optimization Tool designed to accelerate the inference of DL models by converting them into a more hardware-friendly representation by applying specific methods that do not require re-training, for example, post-training quantization.
For more details about the low-precision flow in OpenVINO&trade;, refer to the [Low Precision Optimization Guide](docs/LowPrecisionOptimizationGuide.md).

Post-Training Optimization Tool includes standalone command-line tool and Python* API that provide the following key features:

## Key features:

* Two supported post-training quantization algorithms: fast [DefaultQuantization](openvino/tools/pot/algorithms/quantization/default/README.md) and precise [AccuracyAwareQuantization](openvino/tools/pot/algorithms/quantization/accuracy_aware/README.md), as well as multiple experimental methods.

* Symmetric and asymmetric quantization schemes. For more details, see the [Quantization](openvino/tools/pot/algorithms/quantization/README.md) section.
* Per-channel quantization for Convolutional and Fully-Connected layers.
* Multiple domains: Computer Vision, Recommendation Systems.
* Ability to implement custom calibration pipeline via supported [API](openvino/tools/pot/api/README.md).
* Compression for different HW targets such as CPU, GPU, NPU.
* Post-training sparsity.

## Usage

### System requirements
- Ubuntu 18.04 or later (64-bit)
- Python 3.8 or later
- OpenVINO

### Installation (Temporary)
1) Clone the openvino repo: `git clone https://github.com/openvinotoolkit/openvino`
2) Download submodules:
   ```
   git submodule init
   git submodule update
   ```
3) Setup model conversion API. 
    You can setup model conversion API that needs for POT purposed with the two ways:
    1. Install model conversion API with pip using "python setup.py install" at the mo folder (`<openvino_path>/tools/mo/setup.py`)
    2. Setup model conversion API for Python using PYTHONPATH environment variable. Add the following `<openvino_path>/tools/mo` into PYTHONPATH.
4) Install requirements for accuracy checker:
    - From POT root: `cd ./thirdparty/open_model_zoo/tools/accuracy_checker`
    - Call setup script: `python3 setup.py install`
    - Get back to root POT dir: `cd <PATH_TO_POT_DIR>`
5) Install requirements for the tool:
    - Call setup script: `python3 setup.py develop`

### Run
1) Prepare configuration file for the tool based on the examples in the `configs` folder
2) Navigate to compression tool directory
3) Launch the tool running the following command:
    `python3 main.py -c <path to config file> -e`

To test the tool you can use PyTorch Mobilenet_v2 model from `tests/data/models/mobilenetv2_example/mobilenetv2_example.onnx`

- If there're some errors with imports in ModelOptimizer, first of all make the following steps:
    - If you've installed ModelOptimizer with setting _PYTHONPATH_ variable, checkout the path. It should be as following `<openvino_path>/tools/mo.` The whole command can be found in step 3 Installation (Temporary) guide above.
