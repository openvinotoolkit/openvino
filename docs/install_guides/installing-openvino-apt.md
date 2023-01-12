# Install Intel® Distribution of OpenVINO™ Toolkit for Linux Using APT Repository {#openvino_docs_install_guides_installing_openvino_apt}

@sphinxdirective

This guide provides detailed steps for installing OpenVINO™ Runtime through the APT repository and guidelines for installing OpenVINO Development Tools.

.. note:: From the 2022.1 release, OpenVINO™ Development Tools can be installed via PyPI only. See :ref:`Install OpenVINO Development Tools <installing-openvino-development-tools>` for more information.

.. warning:: By downloading and using this container and the included software, you agree to the terms and conditions of the `software license agreements <https://software.intel.com/content/dam/develop/external/us/en/documents/intel-openvino-license-agreements.pdf>`_.


.. tab:: System Requirements

   | Full requirement listing is available in:
   | `System Requirements Page <https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/system-requirements.html>`_

.. tab:: Processor Notes

  Processor graphics are not included in all processors.
  See `Product Specifications`_ for information about your processor.

  .. _Product Specifications: https://ark.intel.com/

.. tab:: Software Requirements

  * `CMake 3.13 or higher, 64-bit <https://cmake.org/download/>`_
  * GCC 7.5.0 (for Ubuntu 18.04) or GCC 9.3.0 (for Ubuntu 20.04)
  * `Python 3.7 - 3.10, 64-bit <https://www.python.org/downloads/>`_

| 

.. _installing-openvino-runtime:

Installing OpenVINO Runtime
===========================

Step 1: Set Up the OpenVINO Toolkit APT Repository
--------------------------------------------------

#. Install the GPG key for the repository

   a. Download the `GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB <https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB>`_

      You can also use the following command:

      .. code-block:: sh

         wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

   b. Add this key to the system keyring:

      .. code-block:: sh

         sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

      .. note::

         You might need to install GnuPG:

         .. code-block::

            sudo apt-get install gnupg

#. Add the repository via the following command:

   .. tab:: Ubuntu 18

      .. code-block:: sh

         echo "deb https://apt.repos.intel.com/openvino/2022 bionic main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2022.list

   .. tab:: Ubuntu 20

      .. code-block:: sh

         echo "deb https://apt.repos.intel.com/openvino/2022 focal main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2022.list


#. Update the list of packages via the update command:

   .. code-block:: sh

      sudo apt update


#. Verify that the APT repository is properly set up. Run the apt-cache command to see a list of all available OpenVINO packages and components:

   .. code-block:: sh

      apt-cache search openvino

Step 2: Install OpenVINO Runtime Using the APT Package Manager
--------------------------------------------------------------

Install OpenVINO Runtime
^^^^^^^^^^^^^^^^^^^^^^^^

.. tab:: The Latest Version

   Run the following command:

   .. code-block:: sh

      sudo apt install openvino


.. tab::  A Specific Version

   #. Get a list of OpenVINO packages available for installation:

      .. code-block:: sh

         sudo apt-cache search openvino

   #. Install a specific version of an OpenVINO package:

      .. code-block:: sh

         sudo apt install openvino-<VERSION>.<UPDATE>.<PATCH>

      For example:

      .. code-block:: sh

         sudo apt install openvino-2022.3.0

.. note::

   You can use ``--no-install-recommends`` option to install only required packages. Keep in mind that the build tools must be installed **separately** if you want to compile the samples.


Check for Installed Packages and Versions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the following command:

.. code-block:: sh

   apt list --installed | grep openvino


Uninstall OpenVINO Runtime
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tab:: The Latest Version

   Run the following command:

   .. code-block:: sh

      sudo apt autoremove openvino


.. tab::  A Specific Version

   Run the following command:

   .. code-block:: sh

      sudo apt autoremove openvino-<VERSION>.<UPDATE>.<PATCH>

   For example:

   .. code-block:: sh

      sudo apt autoremove openvino-2022.3.0


Step 3 (Optional): Install Additional Components
------------------------------------------------

OpenVINO Development Tools is a set of utilities for working with OpenVINO and OpenVINO models. It provides tools like Model Optimizer, Benchmark Tool, Post-Training Optimization Tool, and Open Model Zoo Downloader. If you install OpenVINO Runtime using APT, OpenVINO Development Tools must be installed separately.

See the **For C++ Developers** section on the :doc:`Install OpenVINO Model Development Tools <openvino_docs_install_guides_install_dev_tools>` page for instructions.

Step 4 (Optional): Configure Inference on Non-CPU Devices
---------------------------------------------------------

To enable the toolkit components to use processor graphics (GPU) on your system, follow the steps in :doc:`GPU Setup Guide <openvino_docs_install_guides_configurations_for_intel_gpu>`.

Step 5: Build Samples
---------------------

To build the C++ or C sample applications for Linux, run the ``build_samples.sh`` script:

.. tab:: C++

   .. code-block:: sh

      /usr/share/openvino/samples/cpp/build_samples.sh

.. tab:: C

   .. code-block:: sh

      /usr/share/openvino/samples/c/build_samples.sh


For more information, refer to :ref:`Build the Sample Applications on Linux <build-samples-linux>`.


What's Next?
============

Now you may continue with the following tasks:

* To convert models for use with OpenVINO, see :doc:`Model Optimizer Developer Guide <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`.
* See pre-trained deep learning models in our :doc:`Open Model Zoo <model_zoo>`.
* Try out OpenVINO via `OpenVINO Notebooks <https://docs.openvino.ai/2022.3/notebooks/notebooks.html>`_.
* To write your own OpenVINO™ applications, see :doc:`OpenVINO Runtime User Guide <openvino_docs_OV_UG_OV_Runtime_User_Guide>`.
* See sample applications in :doc:`OpenVINO™ Toolkit Samples Overview <openvino_docs_OV_UG_Samples_Overview>`.

Additional Resources
====================

- Intel® Distribution of OpenVINO™ toolkit home page: https://software.intel.com/en-us/openvino-toolkit.
- For IoT Libraries & Code Samples see the `Intel® IoT Developer Kit <https://github.com/intel-iot-devkit>`_.

@endsphinxdirective
