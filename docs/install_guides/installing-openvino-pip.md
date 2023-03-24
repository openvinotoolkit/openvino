# Install Intel® Distribution of OpenVINO™ Toolkit from PyPI Repository {#openvino_docs_install_guides_installing_openvino_pip}

@sphinxdirective

You can install both OpenVINO™ Runtime and OpenVINO Development Tools through the PyPI repository. This page provides the main steps for installing OpenVINO Runtime.

.. note:

   From the 2022.1 release, the OpenVINO™ Development Tools can only be installed via PyPI. See :doc:`Install OpenVINO Development Tools <openvino_docs_install_guides_install_dev_tools>` for detailed steps.


Installing OpenVINO Runtime
###########################

For system requirements and troubleshooting, see https://pypi.org/project/openvino/

Step 1. Set Up Python Virtual Environment
+++++++++++++++++++++++++++++++++++++++++

Use a virtual environment to avoid dependency conflicts.

To create a virtual environment, use the following command:


.. tab-set::

    .. tab-item:: Linux and macOS
       :sync: linmac

       .. code-block:: sh

          python3 -m venv openvino_env

    .. tab-item:: Windows
       :sync: win

       .. code-block:: sh

          python -m venv openvino_env


Step 2. Activate Virtual Environment
++++++++++++++++++++++++++++++++++++


.. tab-set::

    .. tab-item:: Linux and macOS
       :sync: linmac

       .. code-block:: sh

          source openvino_env/bin/activate

    .. tab-item:: Windows
       :sync: win

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

   pip install openvino


Step 5. Verify that the Package Is Installed
++++++++++++++++++++++++++++++++++++++++++++

Run the command below:

.. code-block:: sh

   python -c "from openvino.runtime import Core"


If installation was successful, you will not see any error messages (no console output).

Congratulations! You finished installing OpenVINO Runtime. Now you can start exploring OpenVINO's functionality through Jupyter Notebooks and sample applications. See the :ref:`What's Next <whats-next>` section to learn more!

Installing OpenVINO Development Tools
#####################################

OpenVINO Development Tools adds even more functionality to OpenVINO. It provides tools like Model Optimizer, Benchmark Tool, Post-Training Optimization Tool, and Open Model Zoo Downloader. If you install OpenVINO Development Tools, OpenVINO Runtime will also be installed as a dependency, so you don't need to install OpenVINO Runtime separately.

See the :doc:`Install OpenVINO Development Tools <openvino_docs_install_guides_install_dev_tools>` page for step-by-step installation instructions.

.. _whats-next:

What's Next?
####################

Now that you've installed OpenVINO Runtime, you're ready to run your own machine learning applications! Learn more about how to integrate a model in OpenVINO applications by trying out the following tutorials.

.. image:: https://user-images.githubusercontent.com/15709723/127752390-f6aa371f-31b5-4846-84b9-18dd4f662406.gif
   :width: 400

Try the `Python Quick Start Example <https://docs.openvino.ai/nightly/notebooks/201-vision-monodepth-with-output.html>`__ to estimate depth in a scene using an OpenVINO monodepth model in a Jupyter Notebook inside your web browser.

Get started with Python
+++++++++++++++++++++++

Visit the :doc:`Tutorials <tutorials>` page for more Jupyter Notebooks to get you started with OpenVINO, such as:

* `OpenVINO Python API Tutorial <https://docs.openvino.ai/nightly/notebooks/002-openvino-api-with-output.html>`__
* `Basic image classification program with Hello Image Classification <https://docs.openvino.ai/nightly/notebooks/001-hello-world-with-output.html>`__
* `Convert a PyTorch model and use it for image background removal <https://docs.openvino.ai/nightly/notebooks/205-vision-background-removal-with-output.html>`__

Run OpenVINO on accelerated devices
+++++++++++++++++++++++++++++++++++

OpenVINO Runtime has a plugin architecture that enables you to run inference on multiple devices without rewriting your code. Supported devices include integrated GPUs, discrete GPUs and GNAs. Visit the :doc:`Additional Configurations <openvino_docs_install_guides_configurations_header>` page for instructions on how to configure your hardware devices to work with OpenVINO.

Additional Resources
####################

- Intel® Distribution of OpenVINO™ toolkit home page: https://software.intel.com/en-us/openvino-toolkit
- For IoT Libraries & Code Samples, see `Intel® IoT Developer Kit <https://github.com/intel-iot-devkit>`__.
- `OpenVINO Installation Selector Tool <https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html>`__

@endsphinxdirective
