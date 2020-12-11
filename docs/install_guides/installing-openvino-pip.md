# Install Intel® Distribution of OpenVINO™ Toolkit from PyPI Repository {#openvino_docs_install_guides_installing_openvino_pip}

This guide provides installation steps for the Intel® distribution of OpenVINO™ toolkit distributed through the PyPI repository.

## System Requirements

* [Python* distribution](https://www.python.org/) 3.6 or 3.7
* Operating Systems:
  - Ubuntu* 18.04 long-term support (LTS), 64-bit
  - macOS* 10.15.x versions
  - Windows 10*, 64-bit Pro, Enterprise or Education (1607 Anniversary Update, Build 14393 or higher) editions
  - Windows Server* 2016 or higher

## Install the Runtime Package Using the PyPI Repository

### Step 1. Set up and update pip to the highest version

Run the command below:
```sh
python3 -m pip install --upgrade pip
```

### Step 2. Install the Intel® distribution of OpenVINO™ toolkit

Run the command below:
   ```sh
   pip install openvino
   ```

### Step 3. Add PATH to environment variables

Run a command for your operating system:
- Ubuntu 18.04 and macOS:
```sh
export LD_LIBRARY_PATH=<library_dir>:${LD_LIBRARY_PATH}
```
- Windows* 10:
```sh
set PATH=<library_dir>;%PATH%
```
To find `library_dir`:   
**Ubuntu, macOS**:
- Standard user:
```sh
echo $(python3 -m site --user-base)/lib
```
- Root or sudo user:
```sh
/usr/local/lib
```
- Virtual environments or custom Python installations (from sources or tarball):
```sh
echo $(which python3)/../../lib
```
**Windows**:
- Standard Python:
```sh
python -c "import os, sys; print((os.path.dirname(sys.executable))+'\Library\\bin')"
```
- Virtual environments or custom Python installations (from sources or tarball):
```sh
python -c "import os, sys; print((os.path.dirname(sys.executable))+'\..\Library\\bin')"
```

### Step 4. Verify that the package is installed

Run the command below:
```sh
python3 -c "import openvino"
```
   
Now you are ready to develop and run your application.

## Additional Resources

- [Intel® Distribution of OpenVINO™ toolkit](https://software.intel.com/en-us/openvino-toolkit).
- [OpenVINO™ toolkit online documentation](https://docs.openvinotoolkit.org).
- [Model Optimizer Developer Guide](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
- [Inference Engine Developer Guide](../IE_DG/Deep_Learning_Inference_Engine_DevGuide.md).
- For more information on Sample Applications, see the [Inference Engine Samples Overview](../IE_DG/Samples_Overview.md).
- [Intel® Distribution of OpenVINO™ toolkit PIP home page](https://pypi.org/project/openvino/)

