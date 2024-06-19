.. {#openvino_docs_install_guides_installing_openvino_apt}

Install Intel® Distribution of OpenVINO™ Toolkit for Linux Using APT Repository
==================================================================================


.. meta::
   :description: Learn how to install OpenVINO™ Runtime on the Linux operating
                 system, using the APT repository.

.. note::

   Note that the APT distribution:

   * offers both C/C++ and Python APIs
   * does not offer support for GNA and NPU inference
   * is dedicated to Linux users only
   * additionally includes code samples


.. tab-set::

   .. tab-item:: System Requirements
      :sync: system-requirements

      | Full requirement listing is available in:
      | :doc:`System Requirements Page <system_requirements>`

   .. tab-item:: Processor Notes
      :sync: processor-notes

      | To see if your processor includes the integrated graphics technology and supports iGPU inference, refer to:
      | `Product Specifications <https://ark.intel.com/>`__

   .. tab-item:: Software Requirements
      :sync: software-requirements

      * `CMake 3.13 or higher, 64-bit <https://cmake.org/download/>`__
      * GCC 7.5.0 (for Ubuntu 18.04), GCC 9.3.0 (for Ubuntu 20.04) or GCC 11.3.0 (for Ubuntu 22.04)
      * `Python 3.8 - 3.11, 64-bit <https://www.python.org/downloads/>`__


Installing OpenVINO Runtime
#######################################

Step 1: Set Up the OpenVINO Toolkit APT Repository
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

1. Install the GPG key for the repository

   a. Download the `GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB <https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB>`__

      You can also use the following command:

      .. code-block:: sh

         wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

   b. Add this key to the system keyring:

      .. code-block:: sh

         sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

      .. note::

         You might need to install GnuPG:

         .. code-block:: sh

            sudo apt-get install gnupg

2. Add the repository via the following command:

   .. tab-set::

      .. tab-item:: Ubuntu 22
         :sync: ubuntu-22

         .. code-block:: sh

            echo "deb https://apt.repos.intel.com/openvino/2023 ubuntu22 main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2023.list

      .. tab-item:: Ubuntu 20
         :sync: ubuntu-20

         .. code-block:: sh

            echo "deb https://apt.repos.intel.com/openvino/2023 ubuntu20 main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2023.list

      .. tab-item:: Ubuntu 18
         :sync: ubuntu-18

         .. code-block:: sh

            echo "deb https://apt.repos.intel.com/openvino/2023 ubuntu18 main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2023.list


3. Update the list of packages via the update command:

   .. code-block:: sh

      sudo apt update


4. Verify that the APT repository is properly set up. Run the apt-cache command to see a list of all available OpenVINO packages and components:

   .. code-block:: sh

      apt-cache search openvino



Step 2: Install OpenVINO Runtime Using the APT Package Manager
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

1. Install OpenVINO Runtime


.. tab-set::

   .. tab-item:: The Latest Version
      :sync: latest-version

      Run the following command:

      .. code-block:: sh

         sudo apt install openvino


   .. tab-item:: A Specific Version
      :sync: specific-version

      #. Get a list of OpenVINO packages available for installation:

         .. code-block:: sh

            sudo apt-cache search openvino

      #. Install a specific version of an OpenVINO package:

         .. code-block:: sh

            sudo apt install openvino-<VERSION>.<UPDATE>.<PATCH>

         For example:

         .. code-block:: sh


            sudo apt install openvino-2023.3.0

.. note::

   You can use ``--no-install-recommends`` option to install only required packages.
   Keep in mind that the build tools must be installed **separately** if you want to compile the samples.


2. Check for Installed Packages and Versions

Run the following command:

.. code-block:: sh

   apt list --installed | grep openvino


Congratulations! You've just Installed OpenVINO! For some use cases you may still
need to install additional components. Check the
:doc:`list of additional configurations <openvino_docs_install_guides_configurations_header>`
to see if your case needs any of them.

With the APT distribution, you can build OpenVINO sample files, as explained in the
:doc:`guide for OpenVINO sample applications <openvino_docs_OV_UG_Samples_Overview>`.
For C++ and C, just run the ``build_samples.sh`` script:

.. tab-set::

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: sh

         /usr/share/openvino/samples/cpp/build_samples.sh

   .. tab-item:: C
      :sync: c

      .. code-block:: sh

         /usr/share/openvino/samples/c/build_samples.sh

Python samples can run as following:

.. code-block:: sh

   python3 /usr/share/openvino/samples/python/hello_query_device/hello_query_device.py

Uninstalling OpenVINO Runtime
#######################################

To uninstall OpenVINO Runtime via APT, run the following command based on your needs:

.. tab-set::

   .. tab-item:: The Latest Version
      :sync: latest-version

      .. code-block:: sh

         sudo apt autoremove openvino

   .. tab-item:: A Specific Version
      :sync: specific-version

      .. code-block:: sh

         sudo apt autoremove openvino-<VERSION>.<UPDATE>.<PATCH>

      For example:

      .. code-block:: sh

         sudo apt autoremove openvino-2023.3.0


What's Next?
#######################################

Now that you've installed OpenVINO Runtime, you're ready to run your own machine learning applications!
Learn more about how to integrate a model in OpenVINO applications by trying out the following tutorials:

* Try the `C++ Quick Start Example <openvino_docs_get_started_get_started_demos.html>`_ for step-by-step
  instructions on building and running a basic image classification C++ application.

  .. image:: https://user-images.githubusercontent.com/36741649/127170593-86976dc3-e5e4-40be-b0a6-206379cd7df5.jpg
     :width: 400

* Visit the :ref:`Samples <code samples>` page for other C++ example applications to get you started with OpenVINO, such as:

  * `Basic object detection with the Hello Reshape SSD C++ sample <openvino_sample_hello_reshape_ssd.html>`_
  * `Object classification sample <openvino_sample_hello_classification.html>`_

You can also try the following:

* Learn more about :doc:`OpenVINO Workflow <openvino_workflow>`.
* To prepare your models for working with OpenVINO, see :doc:`Model Preparation <openvino_docs_model_processing_introduction>`.
* See pre-trained deep learning models in our :doc:`Open Model Zoo <model_zoo>`.
* Learn more about :doc:`Inference with OpenVINO Runtime <openvino_docs_OV_UG_OV_Runtime_User_Guide>`.
* See sample applications in :doc:`OpenVINO toolkit Samples Overview <openvino_docs_OV_UG_Samples_Overview>`.
* Take a glance at the OpenVINO product home page: https://software.intel.com/en-us/openvino-toolkit.






