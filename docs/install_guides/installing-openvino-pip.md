# Install Intel® Distribution of OpenVINO™ Toolkit from PyPI Repository {#openvino_docs_install_guides_installing_openvino_pip}

You can install both OpenVINO™ Runtime and OpenVINO Development Tools through the PyPI repository. This page provides the main steps for installing OpenVINO Runtime.

> **NOTE**: From the 2022.1 release, the OpenVINO™ Development Tools can only be installed via PyPI. See [Install OpenVINO Development Tools](installing-model-dev-tools.md) for detailed steps.

## Installing OpenVINO Runtime

For system requirements and troubleshooting, see <https://pypi.org/project/openvino/>.

### Step 1. Set Up Python Virtual Environment

Use a virtual environment to avoid dependency conflicts. 

To create a virtual environment, use the following command:

@sphinxdirective

.. tab:: Linux and macOS

   .. code-block:: sh
   
      python3 -m venv openvino_env
   
.. tab:: Windows

   .. code-block:: sh
   
      python -m venv openvino_env
     
     
@endsphinxdirective

### Step 2. Activate Virtual Environment

@sphinxdirective

.. tab:: Linux and macOS

   .. code-block:: sh
   
      source openvino_env/bin/activate
   
.. tab:: Windows

   .. code-block:: sh
   
      openvino_env\Scripts\activate
     
.. important::

   The above command must be re-run every time a new command terminal window is opened.
   
@endsphinxdirective

### Step 3. Set Up and Update PIP to the Highest Version

Use the following command:
```sh
python -m pip install --upgrade pip
```

### Step 4. Install the Package

Use the following command:
```
pip install openvino
```

### Step 5. Verify that the Package Is Installed

Run the command below:
```sh
python -c "from openvino.runtime import Core"
```

If installation was successful, you will not see any error messages (no console output).

Congratulations! You finished installing OpenVINO Runtime. Now you can start exploring OpenVINO's functionality through Jupyter Notebooks and sample applications. See the <a href="#whats-next">What's Next</a> section to learn more!

## Installing OpenVINO Development Tools
OpenVINO Development Tools adds even more functionality to OpenVINO. It provides tools like Model Optimizer, Benchmark Tool, Post-Training Optimization Tool, and Open Model Zoo Downloader. If you install OpenVINO Development Tools, OpenVINO Runtime will also be installed as a dependency, so you don't need to install OpenVINO Runtime separately. 

See the [Install OpenVINO Development Tools](installing-model-dev-tools.md) page for step-by-step installation instructions.

<a name="whats-next"></a>
## What's Next?
Now that you've installed OpenVINO Runtime, you're ready to run your own machine learning applications! Learn more about how to integrate a model in OpenVINO applications by trying out the following tutorials.

<img src="https://user-images.githubusercontent.com/15709723/127752390-f6aa371f-31b5-4846-84b9-18dd4f662406.gif" width=400>

Try the [Python Quick Start Example](https://docs.openvino.ai/2022.3/notebooks/201-vision-monodepth-with-output.html) to estimate depth in a scene using an OpenVINO monodepth model in a Jupyter Notebook inside your web browser.

### Get started with Python
Visit the [Tutorials](../tutorials.md) page for more Jupyter Notebooks to get you started with OpenVINO, such as:
* [OpenVINO Python API Tutorial](https://docs.openvino.ai/2022.3/notebooks/002-openvino-api-with-output.html)
* [Basic image classification program with Hello Image Classification](https://docs.openvino.ai/2022.3/notebooks/001-hello-world-with-output.html)
* [Convert a PyTorch model and use it for image background removal](https://docs.openvino.ai/2022.3/notebooks/205-vision-background-removal-with-output.html)

### Run OpenVINO on accelerated devices
OpenVINO Runtime has a plugin architecture that enables you to run inference on multiple devices without rewriting your code. Supported devices include integrated GPUs, discrete GPUs, NCS2, VPUs, and GNAs. Visit the [Additional Configurations](configurations-header.md) page for instructions on how to configure your hardware devices to work with OpenVINO.

## Additional Resources

- Intel® Distribution of OpenVINO™ toolkit home page: <https://software.intel.com/en-us/openvino-toolkit>
- For IoT Libraries & Code Samples, see [Intel® IoT Developer Kit](https://github.com/intel-iot-devkit).
- [OpenVINO Installation Selector Tool](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html)