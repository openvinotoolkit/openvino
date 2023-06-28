# Install Intel® Distribution of OpenVINO™ Toolkit from PyPI Repository {#openvino_docs_install_guides_installing_openvino_pip}

@sphinxdirective

.. meta::
   :description: Learn how to install OpenVINO™ Runtime on Windows, Linux, and 
                 macOS operating systems, using a PyPi package.


Using the PyPI repository, you can install either OpenVINO™ Runtime or OpenVINO Development Tools on Windows, Linux, and macOS systems.
This article focuses on OpenVINO™ Runtime.

.. note

   If you install OpenVINO Development Tools, OpenVINO Runtime will also be installed as a dependency, so you don't need to install it separately.


Installing OpenVINO Runtime
###########################

For system requirements and troubleshooting, see https://pypi.org/project/openvino/

Step 1. Set Up Python Virtual Environment
+++++++++++++++++++++++++++++++++++++++++

Use a virtual environment to avoid dependency conflicts.

To create a virtual environment, use the following command:


.. tab-set::

    .. tab-item:: Linux and macOS
       :sync: linux-and-macos

       .. code-block:: sh

          python3 -m venv openvino_env

    .. tab-item:: Windows
       :sync: windows

       .. code-block:: sh

          python -m venv openvino_env


Step 2. Activate Virtual Environment
++++++++++++++++++++++++++++++++++++


.. tab-set::

    .. tab-item:: Linux and macOS
       :sync: linux-and-macos

       .. code-block:: sh

          source openvino_env/bin/activate

    .. tab-item:: Windows
       :sync: windows

       .. code-block:: sh

          openvino_env\Scripts\activate


.. important::

   The above command must be re-run every time a new command terminal window is opened.


Step 3. Set Up and Update PIP to the Highest Version
++++++++++++++++++++++++++++++++++++++++++++++++++++

Use the following command:

.. code-block:: sh

   python -m pip install --upgrade pip


Step 4. Install the Package
+++++++++++++++++++++++++++

Use the following command:

.. code-block:: sh

   python -m pip install openvino


Step 5. Verify that the Package Is Installed
++++++++++++++++++++++++++++++++++++++++++++

Run the command below:

.. code-block:: sh

   python -c "from openvino.runtime import Core; print(Core().available_devices)"

If installation was successful, you will see the list of available devices. Congratulations! You have finished installing OpenVINO Runtime.


What's Next?
####################

Now that you've installed OpenVINO Runtime, you're ready to run your own machine learning applications! Learn more about how to integrate a model in OpenVINO applications by trying out the following tutorials.

.. image:: https://user-images.githubusercontent.com/15709723/127752390-f6aa371f-31b5-4846-84b9-18dd4f662406.gif
   :width: 400

Try the `Python Quick Start Example <notebooks/201-vision-monodepth-with-output.html>`__ to estimate depth in a scene using an OpenVINO monodepth model in a Jupyter Notebook inside your web browser.

Get started with Python
+++++++++++++++++++++++

Visit the :doc:`Tutorials <tutorials>` page for more Jupyter Notebooks to get you started with OpenVINO, such as:

* `OpenVINO Python API Tutorial <notebooks/002-openvino-api-with-output.html>`__
* `Basic image classification program with Hello Image Classification <notebooks/001-hello-world-with-output.html>`__
* `Convert a PyTorch model and use it for image background removal <notebooks/205-vision-background-removal-with-output.html>`__

Run OpenVINO on accelerated devices
+++++++++++++++++++++++++++++++++++

OpenVINO Runtime has a plugin architecture that enables you to run inference on multiple devices without rewriting your code. Supported devices include integrated GPUs, discrete GPUs and GNAs. Visit the :doc:`Additional Configurations <openvino_docs_install_guides_configurations_header>` page for instructions on how to configure your hardware devices to work with OpenVINO.

Additional Resources
####################

- Intel® Distribution of OpenVINO™ toolkit home page: https://software.intel.com/en-us/openvino-toolkit
- For IoT Libraries & Code Samples, see `Intel® IoT Developer Kit <https://github.com/intel-iot-devkit>`__.

@endsphinxdirective
