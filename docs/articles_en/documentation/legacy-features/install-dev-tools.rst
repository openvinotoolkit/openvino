Install OpenVINO™ Development Tools
=====================================


.. meta::
   :description: Learn how to install OpenVINO™ Development Tools on Windows,
                 Linux, and macOS operating systems, using a PyPi package.

OpenVINO Development Tools is a set of utilities that make it easy to develop and
optimize models and applications for OpenVINO. It provides the following tools:

* Model conversion API
* Benchmark Tool
* Accuracy Checker and Annotation Converter
* Model Downloader and other Open Model Zoo tools

The instructions on this page show how to install OpenVINO Development Tools. If you are a
Python developer, it only takes a few simple steps to install the tools with PyPI. If you
are developing in C/C++, OpenVINO Runtime must be installed separately before installing
OpenVINO Development Tools.

In both cases, Python 3.9 - 3.12 needs to be installed on your system before starting.

.. note::

   From the 2022.1 release, the OpenVINO™ Development Tools can only be installed via PyPI.

.. _python_developers:

For Python Developers
#####################

If you are a Python developer, follow the steps in the
:ref:`Installing OpenVINO Development Tools <install_dev_tools>` section on this page to
install it. Installing OpenVINO Development Tools will also install OpenVINO Runtime as
a dependency, so you don’t need to install OpenVINO Runtime separately. This option is
recommended for new users.

.. _cpp_developers:

For C/C++ Developers
#######################

If you are a C/C++ developer, you must first install OpenVINO Runtime separately to set
up the C/C++ libraries, sample code, and dependencies for building applications with
OpenVINO. These files are not included with the PyPI distribution. See the
:doc:`Selector Tool <../../get-started/install-openvino>` page to install OpenVINO Runtime
from an archive file for your operating system.

Once OpenVINO Runtime is installed, you may install OpenVINO Development Tools for access
to tools like ``mo``, Model Downloader, Benchmark Tool, and other utilities that will help
you optimize your model and develop your application. Follow the steps in the
:ref:`Installing OpenVINO Development Tools <install_dev_tools>` section on this page
to install it.

.. _install_dev_tools:

Installing OpenVINO™ Development Tools
######################################

Follow these step-by-step instructions to install OpenVINO Development Tools on your computer.
There are two options to install OpenVINO Development Tools: installation into an existing
environment with a deep learning framework that was used for model training or creation;
or installation into a new environment.

Installation into an Existing Environment with the Source Deep Learning Framework
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

To install OpenVINO Development Tools (see the :ref:`Install the Package <install_the_package>`
section of this article) into an existing environment with the deep learning framework used
for the model training or creation, run the following command:

.. code-block:: sh

   pip install openvino-dev


Installation in a New Environment
+++++++++++++++++++++++++++++++++

If you do not have an environment with a deep learning framework for the input model or you
encounter any compatibility issues between OpenVINO and your version of deep learning
framework, you may install OpenVINO Development Tools with validated versions of
frameworks into a new environment.

Step 1. Set Up Python Virtual Environment
-----------------------------------------

Create a virtual Python environment to avoid dependency conflicts. To create a virtual
environment, use the following command:

.. tab-set::

   .. tab-item:: Windows
      :sync: windows

      .. code-block:: sh

         python -m venv openvino_env

   .. tab-item:: Linux and macOS
      :sync: linux-and-macos

      .. code-block:: sh

         python3 -m venv openvino_env



Step 2. Activate Virtual Environment
------------------------------------

Activate the newly created Python virtual environment by issuing this command:

.. tab-set::

   .. tab-item:: Windows
      :sync: windows

      .. code-block:: sh

         openvino_env\Scripts\activate

   .. tab-item:: Linux and macOS
      :sync: linux-and-macos

      .. code-block:: sh

         source openvino_env/bin/activate

.. important::

   The above command must be re-run every time a new command terminal window is opened.


Step 3. Set Up and Update PIP to the Highest Version
----------------------------------------------------

Make sure `pip` is installed in your environment and upgrade it to the latest version by
issuing the following command:

.. code-block:: sh

   python -m pip install --upgrade pip


.. _install_the_package:

Step 4. Install the Package
---------------------------

To install and configure the components of the development package together with validated
versions of specific frameworks, use the commands below.

.. code-block:: sh

   pip install openvino-dev[extras]


where the ``extras`` parameter specifies the source deep learning framework for the input model
and is one or more of the following values separated with "," :  ``onnx``, ``pytorch``,
``tensorflow``, ``tensorflow2``.

For example, to install and configure dependencies required for working with TensorFlow 2.x
and ONNX models, use the following command:

.. code-block:: sh

   pip install openvino-dev[tensorflow2,onnx]


.. note::

   Model conversion API support for TensorFlow 1.x environment has been deprecated. Use the
   ``tensorflow2`` parameter to install a TensorFlow 2.x environment that can convert both
   TensorFlow 1.x and 2.x models. If your model isn't compatible with the TensorFlow 2.x
   environment, use the `tensorflow` parameter to install the TensorFlow 1.x environment.
   The TF 1.x environment is provided only for legacy compatibility reasons.

For more details on the openvino-dev PyPI package, see
`pypi.org <https://pypi.org/project/openvino-dev/2023.2.0>`__ .

Step 5. Test the Installation
------------------------------

To verify the package is properly installed, run the command below (this may take a few seconds):

.. code-block:: sh

   mo -h

You will see the help message for ``mo`` if installation finished successfully. If you get an
error, refer to the :doc:`Troubleshooting Guide <../../get-started/troubleshooting-install-config>`
for possible solutions.

Congratulations! You finished installing OpenVINO Development Tools with C/C++ capability.
Now you can start exploring OpenVINO's functionality through example C/C++ applications.
See the "What's Next?" section to learn more!

What's Next?
############

Learn more about OpenVINO and use it in your own application by trying out some of these examples!

Get started with Python
+++++++++++++++++++++++

.. image:: ../../assets/images/get_started_with_python.gif
  :width: 400

Try the `Python Quick Start Example <../../notebooks/vision-monodepth-with-output.html>`__
to estimate depth in a scene using an OpenVINO monodepth model in a Jupyter Notebook
inside your web browser.

Visit the :doc:`Tutorials <../../learn-openvino/interactive-tutorials-python>` page for more
Jupyter Notebooks to get you started with OpenVINO, such as:

* `OpenVINO Python API Tutorial <../../notebooks/openvino-api-with-output.html>`__
* `Basic image classification program with Hello Image Classification <../../notebooks/hello-world-with-output.html>`__
* `Convert a PyTorch model and use it for image background removal <../../notebooks/vision-background-removal-with-output.html>`__

Get started with C++
++++++++++++++++++++

.. image:: ../../assets/images/get_started_with_cpp.jpg
  :width: 400


Try the :doc:`C++ Quick Start Example <../../learn-openvino/openvino-samples/get-started-demos>`
for step-by-step instructions on building and running a basic image classification C++ application.

Visit the :doc:`Samples <../../learn-openvino/openvino-samples>` page for other C++
example applications to get you started with OpenVINO, such as:

* :doc:`Basic object detection with the Hello Reshape SSD C++ sample <../../learn-openvino/openvino-samples/hello-reshape-ssd>`
* :doc:`Object classification sample <../../learn-openvino/openvino-samples/hello-classification>`

Learn OpenVINO Development Tools
++++++++++++++++++++++++++++++++

* Explore a variety of pre-trained deep learning models in the
  :doc:`Open Model Zoo <model-zoo>` and deploy them in demo applications to see how they work.

  .. important::

     Due to the deprecation of Open Model Zoo, models in the OpenVINO IR format are now
     published on `Hugging Face <https://huggingface.co/OpenVINO>`__.

* Want to import a model from another framework and optimize its performance with OpenVINO?
  Visit the :doc:`Convert a Model <transition-legacy-conversion-api/legacy-conversion-api>` page.
* Accelerate your model's speed even further with quantization and other compression techniques
  using :doc:`Neural Network Compression Framework (NNCF) <../../openvino-workflow/model-optimization-guide/quantizing-models-post-training>`.
* Benchmark your model's inference speed with one simple command using the
  :doc:`Benchmark Tool <../../learn-openvino/openvino-samples/benchmark-tool>`.

Additional Resources
####################

- `Intel® Distribution of OpenVINO™ toolkit home page <https://software.intel.com/en-us/openvino-toolkit>`__
