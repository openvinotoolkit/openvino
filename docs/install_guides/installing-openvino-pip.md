# Install Intel® Distribution of OpenVINO™ Toolkit from PyPI Repository {#openvino_docs_install_guides_installing_openvino_pip}

You can install Intel® Distribution of OpenVINO™ Toolkit through the PyPI repository, including:

* OpenVINO Runtime with the Inference Engine inside. This page provides basic steps for installing OpenVINO Runtime. For more details, see <https://pypi.org/project/openvino/>.
* OpenVINO Model Development Tools that includes Model Optimizer, Benchmark Tool, Accuracy Checker, Post-Training Optimization Tool, Model Downloader and other Open Model Zoo tools. From 2022.1 release, the OpenVINO Model Development Tools can only be installed via PyPI. See [Install OpenVINO Model Development Tools](../installing-model-dev-tools.md) or <https://pypi.org/project/openvino-dev> for detailed steps.

## Install OpenVINO Runtime

For system requirements and troubleshooting, see <https://pypi.org/project/openvino> for more details.

**Step 1. Set up Python virtual environment**

To avoid dependency conflicts, use a virtual environment. Skip this step only if you do want to install all dependencies globally.

To create virtual environment, run the following commands:
```python
python -m pip install --user virtualenv
python -m venv openvino_env
```

>**NOTE**: On Linux and macOS, you may need to type `python3` instead of `python`. You may also need to [install pip](https://pip.pypa.io/en/stable/installing/).

**Step 2. Activate virtual environment**

* On Linux and macOS, run the following command:
  ```
  source openvino_env/bin/activate
  ```
* On Windows, run the following command:
  ```
  openvino_env\Scripts\activate
  ```

**Step 3. Set up and update PIP to the highest version**

Run the following command:
```
python -m pip install --upgrade pip
```

**Step 4. Install the package**

Run the following command:
```
pip install openvino
```

**Step 5. Verify that the package is installed**

Run the following command:
```
python -c "from openvino.inference_engine import IECore"
```

If installation was successful, you will not see any error messages (no console output).



## Additional Resources

- [Intel® Distribution of OpenVINO™ toolkit](https://software.intel.com/en-us/openvino-toolkit)
- [Model Optimizer Developer Guide](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
- [Inference Engine Developer Guide](../IE_DG/Deep_Learning_Inference_Engine_DevGuide.md)
- [Inference Engine Samples Overview](../IE_DG/Samples_Overview.md)
