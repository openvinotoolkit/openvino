Install OpenVINO™ Runtime on Linux From YUM Repository
========================================================


.. meta::
   :description: Learn how to install OpenVINO™ Runtime on Linux operating
                 system, using the YUM repository.

.. note::

   Note that the YUM distribution:

   * offers both C/C++ and Python APIs
   * does not offer support for NPU inference
   * is dedicated to Linux users only
   * additionally includes code samples

.. tab-set::

   .. tab-item:: System Requirements
      :sync: system-requirements

      | Full requirement listing is available in:
      | :doc:`System Requirements Page <../../../about-openvino/release-notes-openvino/system-requirements>`

      .. note::

         OpenVINO RPM packages are compatible with and can be run on the following operating systems:

         - RHEL 8.2 and higher
         - Amazon Linux 2022 and 2023
         - Rocky Linux 8.7, 8.8 and 9.2-9.3
         - Alma Linux 8.7, 8.8 and 9.2-9.4
         - Oracle Linux 8.7, 8.8 and 9.2-9.4
         - Fedora 29 and higher up to 41
         - OpenEuler 20.03, 22.03, 23.03 and 24.03
         - Anolis OS 8.6 and 8.8
         - CentOS Stream 8 and 9

   .. tab-item:: Processor Notes
      :sync: processor-notes

      | To see if your processor includes the integrated graphics technology and supports iGPU inference, refer to:
      | `Product Specifications <https://ark.intel.com/>`__

   .. tab-item:: Software
      :sync: software

      * `CMake 3.13 or higher, 64-bit <https://cmake.org/download/>`_
      * GCC 8.2.0
      * `Python 3.9 - 3.12, 64-bit <https://www.python.org/downloads/>`_


Install OpenVINO Runtime
########################

Step 1: Set Up the Repository
+++++++++++++++++++++++++++++


1. Create a YUM repository file (``openvino-2024.repo``) in the ``/tmp`` directory as a normal user:

   .. code-block:: sh

      tee > /tmp/openvino-2024.repo << EOF
      [OpenVINO]
      name=Intel(R) Distribution of OpenVINO 2024
      baseurl=https://yum.repos.intel.com/openvino/2024
      enabled=1
      gpgcheck=1
      repo_gpgcheck=1
      gpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
      EOF

2. Move the new ``openvino-2024.repo`` file to the YUM configuration directory, i.e. ``/etc/yum.repos.d``:

   .. code-block:: sh

      sudo mv /tmp/openvino-2024.repo /etc/yum.repos.d

3. Verify that the new repository is set up properly.

   .. code-block:: sh

      yum repolist | grep -i openvino

   You will see the available list of packages.


To list available OpenVINO packages, use the following command:

.. code-block:: sh

   yum list 'openvino*'



Step 2: Install OpenVINO Runtime Using the YUM Package Manager
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Install OpenVINO Runtime
-------------------------

.. tab-set::

   .. tab-item:: The Latest Version
      :sync: latest-version

      Run the following command:

      .. code-block:: sh

         sudo yum install openvino

   .. tab-item:: A Specific Version
      :sync: specific-version

      Run the following command:

      .. code-block:: sh

         sudo yum install openvino-<VERSION>.<UPDATE>.<PATCH>

      For example:

      .. code-block:: sh


         sudo yum install openvino-2024.4.0



Check for Installed Packages and Version
-----------------------------------------


Run the following command:

.. code-block:: sh

   yum list installed 'openvino*'

.. note::

   You can additionally install Python API using one of the alternative methods (:doc:`conda <install-openvino-conda>` or :doc:`pip <install-openvino-pip>`).

Congratulations! You've just Installed OpenVINO! For some use cases you may still
need to install additional components. Check the
:doc:`list of additional configurations <../configurations>`
to see if your case needs any of them.

With the YUM distribution, you can build OpenVINO sample files, as explained in the
:doc:`guide for OpenVINO sample applications <../../../learn-openvino/openvino-samples>`.
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



Uninstalling OpenVINO Runtime
##############################

To uninstall OpenVINO Runtime via YUM, run the following command based on your needs:

.. tab-set::

   .. tab-item:: The Latest Version
      :sync: latest-version

      .. code-block:: sh

         sudo yum autoremove openvino


   .. tab-item:: A Specific Version
      :sync: specific-version

      .. code-block:: sh

         sudo yum autoremove openvino-<VERSION>.<UPDATE>.<PATCH>

      For example:

      .. code-block:: sh

         sudo yum autoremove openvino-2024.4.0




What's Next?
#############

Now that you've installed OpenVINO Runtime, you're ready to run your own machine learning applications!
Learn more about how to integrate a model in OpenVINO applications by trying out the following tutorials:

* Try the :doc:`C++ Quick Start Example <../../../learn-openvino/openvino-samples/get-started-demos>`
  for step-by-step instructions on building and running a basic image classification C++ application.

  .. image:: https://user-images.githubusercontent.com/36741649/127170593-86976dc3-e5e4-40be-b0a6-206379cd7df5.jpg
     :width: 400

* Visit the :ref:`Samples <code samples>` page for other C++ example applications to get you started with OpenVINO, such as:

  * :doc:`Basic object detection with the Hello Reshape SSD C++ sample <../../../learn-openvino/openvino-samples/hello-reshape-ssd>`
  * :doc:`Object classification sample <../../../learn-openvino/openvino-samples/hello-classification>`

You can also try the following things:

* Learn more about :doc:`OpenVINO Workflow <../../../openvino-workflow>`.
* To prepare your models for working with OpenVINO, see :doc:`Model Preparation <../../../openvino-workflow/model-preparation>`.
* See pre-trained deep learning models in our :doc:`Open Model Zoo <../../../documentation/legacy-features/model-zoo>`.

  .. important::

     Due to the deprecation of Open Model Zoo, models in the OpenVINO IR format are now
     published on `Hugging Face <https://huggingface.co/OpenVINO>`__.

* Learn more about :doc:`Inference with OpenVINO Runtime <../../../openvino-workflow/running-inference>`.
* See sample applications in :doc:`OpenVINO toolkit Samples Overview <../../../learn-openvino/openvino-samples>`.
* Take a glance at the OpenVINO `product home page <https://software.intel.com/en-us/openvino-toolkit>`__ .




