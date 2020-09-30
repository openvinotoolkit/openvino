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
   export LD_LIBRARY_PATH=<python_dir>/lib:${LD_LIBRARY_PATH}
   ```
 - Windows* 10:
    ```sh
   set PATH=<python_dir>/Library/bin;%PATH%
   ```
4. Verify that the package is installed:
   ```sh
   python3 -c "import openvino"
   ```
   
Now you are ready to develop and run your application.


## Additional Resources

- [Model Optimizer Developer Guide](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
- [Inference Engine Developer Guide](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Deep_Learning_Inference_Engine_DevGuide.html)
- [Inference Engine Samples Overview](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Samples_Overview.html)
- [Inference Engine Tutorials](https://github.com/intel-iot-devkit/inference-tutorials-generic)
- [Intel® distribution of OpenVINO™ toolkit PIP home page](https://pypi.org/project/openvino-python/)

