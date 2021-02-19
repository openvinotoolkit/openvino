# Install Intel® Distribution of OpenVINO™ Toolkit from PyPI Repository {#openvino_docs_install_guides_installing_openvino_pip}

> **LEGAL NOTICE**: Your use of this software and any required dependent software (the
“Software Package”) is subject to the terms and conditions of the [software license agreements](https://software.intel.com/en-us/license/eula-for-intel-software-development-products) for the Software Package, which may also include notices, disclaimers, or
license terms for third party or open source software included in or with the Software Package, and your use indicates your acceptance of all such terms. Please refer to the “third-party-programs.txt” or other similarly-named text file included with the Software Package for additional details.


This guide provides installation steps for the Intel® distribution of OpenVINO™ toolkit distributed through the PyPI repository.

## System Requirements

* [Python* distribution](https://www.python.org/) 3.6, 3.7, 3.8
* Operating Systems:
  - Ubuntu* 18.04 long-term support (LTS), 64-bit (python 3.6 or 3.7)
  - Ubuntu* 20.04 long-term support (LTS), 64-bit (python 3.8)
  - macOS* 10.15.x versions
  - Windows 10*, 64-bit Pro, Enterprise or Education (1607 Anniversary Update, Build 14393 or higher) editions
  - Windows Server* 2016 or higher

**Runtime Packages**
- Ubuntu 18.04:  `openvino`
- Ubuntu 20.04:  `openvino-ubuntu20`

**Developer Packages**
`openvino-dev`

## Install the runtime or developer packages using the PyPI repository

### Step 1. Set up and update pip to the highest version

Run the command below:
```sh
python3 -m pip install --upgrade pip
```

### Step 2. Install the Intel® distribution of OpenVINO™ toolkit

Run the command below: <br>
**Runtime Package**:
   ```sh
   pip install openvino
   ```
**Developer Package**
   ```sh
   pip install openvino-dev
   ```

### Step 4. Verify that the Runtime package is installed

Run the command below:
```sh
python3 -c "from openvino.inference_engine import IECore"
```
   
Now you are ready to develop and run your application.

## Additional Resources

- [Intel® Distribution of OpenVINO™ toolkit](https://software.intel.com/en-us/openvino-toolkit).
- [OpenVINO™ toolkit online documentation](https://docs.openvinotoolkit.org).
- [Model Optimizer Developer Guide](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
- [Inference Engine Developer Guide](../IE_DG/Deep_Learning_Inference_Engine_DevGuide.md).
- For more information on Sample Applications, see the [Inference Engine Samples Overview](../IE_DG/Samples_Overview.md).
- [Intel® Distribution of OpenVINO™ toolkit PIP home page](https://pypi.org/project/openvino/)

