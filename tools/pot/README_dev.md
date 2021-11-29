# Post-training Optimization Tool {#pot_README_dev}

Starting with the 2020.1 version, OpenVINO&trade; toolkit delivers the Post-Training Optimization Tool designed to accelerate the inference of DL models by converting them into a more hardware-friendly representation by applying specific methods that do not require re-training, for example, post-training quantization.
For more details about the low-precision flow in OpenVINO&trade;, refer to the [Low Precision Optimization Guide](docs/LowPrecisionOptimizationGuide.md).

Post-Training Optimization Tool includes standalone command-line tool and Python* API that provide the following key features:

## Key features:

* Two supported post-training quantization algorithms: fast [DefaultQuantization](openvino/tools/pot/algorithms/quantization/default/README.md) and precise [AccuracyAwareQuantization](openvino/tools/pot/algorithms/quantization/accuracy_aware/README.md), as well as multiple experimental methods.
* Global optimization of post-training quantization parameters using [Tree-structured Parzen Estimator](openvino/tools/pot/optimization/tpe/README.md).
* Symmetric and asymmetric quantization schemes. For more details, see the [Quantization](openvino/tools/pot/algorithms/quantization/README.md) section.
* Per-channel quantization for Convolutional and Fully-Connected layers.
* Multiple domains: Computer Vision, Recommendation Systems.
* Ability to implement custom calibration pipeline via supported [API](openvino/tools/pot/api/README.md).
* Compression for different HW targets such as CPU, GPU, VPU.
* Post-training sparsity.

## Usage

### System requirements
- Ubuntu 18.04 or later (64-bit)
- Python 3.6 or later
- OpenVINO

### Installation (Temporary)
1) Clone compression tool repo: `git clone git@gitlab-icv.inn.intel.com:algo/post-training-compression-tool.git`
2) Download submodules:
   ```
   git submodule init
   git submodule update
   ```
3) Clone DLDT repo: `git clone https://gitlab-icv.inn.intel.com/inference-engine/dldt` (Not into the post-training-compression-tool)
4) Switch dldt to required branch: `feature/low_precision/develop_fp_v10`
5) Build inference engine (Instruction can be found in dldt repo)
6) Switch dldt to _mkaglins/poc_ branch (Inference engine is built from _feature/low_precision/develop_fp_v10_ branch to support `FakeQuantize` layers. ModelOptimizer is used from _mkaglins/poc_ branch. So stay on _mkaglins/poc_ branch as you've built IE and don't build it from there again)
7) Set _PYTHONPATH_ variable: `export PYTHONPATH=<path to DLDT bins>/bin/intel64/Release/lib/python_api/python3.6:<path to DLDT>/dldt/model-optimizer`
8) Install requirements for accuracy checker:
    - From POT root: `cd ./thirdparty/open_model_zoo/tools/accuracy_checker`
    - Call setup script: `python3 setup.py install`
    - Get back to root POT dir: `cd <PATH_TO_POT_DIR>`
9) Install requirements for the tool:
    - Call setup script: `python3 setup.py install`

### Run
1) Prepare configuration file for the tool based on the examples in the `configs` folder
2) Navigate to compression tool directory
3) Launch the tool running the following command:
    `python3 main.py -c <path to config file> -e`

To test the tool you can use PyTorch Mobilenet_v2 model from `tests/data/models/mobilenetv2/mobilenetv2.onnx`

 - If there're some errors with imports in ModelOptimizer first of all make the following steps:
    - Checkout _mkaglins/poc_ branch in DLDT (It's important!)
