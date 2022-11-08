# Install OpenVINO™ Development Tools {#openvino_docs_install_guides_install_dev_tools}

If you want to download, convert, optimize and tune pre-trained deep learning models, install OpenVINO™ Development Tools, which provides the following tools:

* Model Optimizer
* Benchmark Tool
* Accuracy Checker and Annotation Converter
* Post-Training Optimization Tool
* Model Downloader and other Open Model Zoo tools

> **NOTE**: From the 2022.1 release, the OpenVINO™ Development Tools can only be installed via PyPI.

In both cases, Python 3.6 - 3.9 need be installed on your machine before starting.

> **NOTE**: From the 2022.1 release, the OpenVINO™ Development Tools can only be installed via PyPI.




While installing OpenVINO Development Tools, OpenVINO Runtime will also be installed as a dependency, so you don't need to install OpenVINO Runtime separately.

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


@endsphinxdirective


### Step 3. Set Up and Update PIP to the Highest Version

Use the following command:
```sh
python -m pip install --upgrade pip
```

### Step 4. Install the Package

To install and configure the components of the development package for working with specific frameworks, use the following command:
```
pip install openvino-dev[extras]
```
where the `extras` parameter specifies one or more deep learning frameworks via these values: `caffe`, `kaldi`, `mxnet`, `onnx`, `pytorch`, `tensorflow`, `tensorflow2`. Make sure that you install the corresponding frameworks for your models.

For example, to install and configure the components for working with TensorFlow 2.x and ONNX, use the following command:
```
pip install openvino-dev[tensorflow2,onnx]
```

> **NOTE**: Model Optimizer support for TensorFlow 1.x environment has been deprecated. Use TensorFlow 2.x environment to convert both TensorFlow 1.x and 2.x models. Use the `tensorflow2` value as much as possible. The `tensorflow` value is provided only for compatibility reasons.


### Step 5. Verify the Installation

To verify if the package is properly installed, run the command below (this may take a few seconds):
```sh
mo -h
```
You will see the help message for Model Optimizer if installation finished successfully.



## <a name="get-started"></a>What's Next?
Now that you've installed OpenVINO Runtime, you're ready to run your own machine learning applications! Learn more about how to integrate a model in OpenVINO applications by trying out the following tutorials.

@sphinxdirective
.. tab:: Get started with Python

   Try the `Python Quick Start Example <https://docs.openvino.ai/2022.2/notebooks/201-vision-monodepth-with-output.html>`_ to estimate depth in a scene using an OpenVINO monodepth model in a Jupyter Notebook inside your web browser.
   
   .. image:: https://user-images.githubusercontent.com/15709723/127752390-f6aa371f-31b5-4846-84b9-18dd4f662406.gif
      :width: 400

   Visit the :ref:`Tutorials <notebook tutorials>` page for more Jupyter Notebooks to get you started with OpenVINO, such as:
   
   * `OpenVINO Python API Tutorial <https://docs.openvino.ai/2022.2/notebooks/002-openvino-api-with-output.html>`_
   * `Basic image classification program with Hello Image Classification <https://docs.openvino.ai/2022.2/notebooks/001-hello-world-with-output.html>`_
   * `Convert a PyTorch model and use it for image background removal <https://docs.openvino.ai/2022.2/notebooks/205-vision-background-removal-with-output.html>`_

.. tab:: Get started with C++

   Try the `C++ Quick Start Example <openvino_docs_get_started_get_started_demos.html>`_ for step-by-step instructions on building and running a basic image classification C++ application.
   
   .. image:: https://user-images.githubusercontent.com/36741649/127170593-86976dc3-e5e4-40be-b0a6-206379cd7df5.jpg
      :width: 400

   Visit the :ref:`Samples <code samples>` page for other C++ example applications to get you started with OpenVINO, such as:
   
   * `Basic object detection with the Hello Reshape SSD C++ sample <openvino_inference_engine_samples_hello_reshape_ssd_README.html>`_
   * `Automatic speech recognition C++ sample <openvino_inference_engine_samples_speech_sample_README.html>`_

@endsphinxdirective

> **NOTE**: Model Optimizer support for TensorFlow 1.x environment has been deprecated. Use TensorFlow 2.x environment to convert both TensorFlow 1.x and 2.x models. The `tensorflow` value is provided only for compatibility reasons, use the `tensorflow2` value instead.

For more details, see <https://pypi.org/project/openvino-dev/>.

## What's Next?

Now you may continue with the following tasks:

* To convert models for use with OpenVINO, see [Model Optimizer Developer Guide](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
* See pre-trained deep learning models in our [Open Model Zoo](../model_zoo.md).
* Try out OpenVINO via [OpenVINO Notebooks](https://docs.openvino.ai/latest/notebooks/notebooks.html).
* To write your own OpenVINO™ applications, see [OpenVINO Runtime User Guide](../OV_Runtime_UG/openvino_intro.md).
* See sample applications in [OpenVINO™ Toolkit Samples Overview](../OV_Runtime_UG/Samples_Overview.md).

## Additional Resources

- Intel® Distribution of OpenVINO™ toolkit home page: <https://software.intel.com/en-us/openvino-toolkit>
- For IoT Libraries & Code Samples, see [Intel® IoT Developer Kit](https://github.com/intel-iot-devkit).
