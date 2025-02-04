Install OpenVINO™ Runtime on Linux From ZYPPER Repository
=========================================================


.. meta::
   :description: Learn how to install OpenVINO™ Runtime on Linux operating
                 system, using the ZYPPER repository.

.. note::

   Note that the ZYPPER distribution:

   * offers both C/C++ APIs
   * does not offer support for NPU inference
   * is dedicated to Linux users only
   * additionally includes code samples

   Before installing OpenVINO, see the
   :doc:`System Requirements page <../../../about-openvino/release-notes-openvino/system-requirements>`.

Install OpenVINO Runtime
########################

Step 1: Update the repository
+++++++++++++++++++++++++++++


1. Update the official factory repository to obtain the latest release:

   .. code-block:: sh

      sudo zypper refresh


2. To list available OpenVINO packages, use the following command:

   .. code-block:: sh

      zypper se openvino

Step 2: Install OpenVINO Runtime Using the ZYPPER Package Manager
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Install OpenVINO Runtime
-------------------------

Run the following command:

.. code-block:: sh

   sudo zypper install openvino-devel openvino-sample

Check for Installed Packages and Version
-----------------------------------------


Run the following command:

.. code-block:: sh

   zypper se -i openvino

.. note::
   You can additionally install Python API using one of the alternative methods (:doc:`conda <install-openvino-conda>` or :doc:`pip <install-openvino-pip>`).

Congratulations! You've just Installed OpenVINO! For some use cases you may still
need to install additional components. Check the
:doc:`list of additional configurations <./configurations>`
to see if your case needs any of them.

With the ZYPPER distribution, you can build OpenVINO sample files, as explained in the
:doc:`guide for OpenVINO sample applications <../../../get-started/learn-openvino/openvino-samples>`.
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

To uninstall OpenVINO Runtime via ZYPPER, run the following command based on your needs:

.. tab-set::

   .. tab-item:: The Latest Version
      :sync: latest-version

      .. code-block:: sh

         sudo zypper remove *openvino*


   .. tab-item:: A Specific Version
      :sync: specific-version

      .. code-block:: sh

         sudo zypper remove *openvino-<VERSION>.<UPDATE>.<PATCH>*

      For example:

      .. code-block:: sh

         sudo zypper remove *openvino-2025.0.0*




What's Next?
#############

Now that you've installed OpenVINO Runtime, you're ready to run your own machine learning applications!
Learn more about how to integrate a model in OpenVINO applications by trying out the following tutorials:

* Try the :doc:`C++ Quick Start Example <../../../get-started/learn-openvino/openvino-samples/get-started-demos>`
  for step-by-step instructions on building and running a basic image classification C++ application.

  .. image:: https://user-images.githubusercontent.com/36741649/127170593-86976dc3-e5e4-40be-b0a6-206379cd7df5.jpg
     :width: 400

* Visit the :ref:`Samples <code samples>` page for other C++ example applications to get you started with OpenVINO, such as:

  * :doc:`Basic object detection with the Hello Reshape SSD C++ sample <../../../get-started/learn-openvino/openvino-samples/hello-reshape-ssd>`
  * :doc:`Object classification sample <../../../get-started/learn-openvino/openvino-samples/hello-classification>`

You can also try the following things:

* Learn more about :doc:`OpenVINO Workflow <../../../openvino-workflow>`.
* To prepare your models for working with OpenVINO, see :doc:`Model Preparation <../../../openvino-workflow/model-preparation>`.
* See pre-trained deep learning models on `Hugging Face <https://huggingface.co/OpenVINO>`__.
* Learn more about :doc:`Inference with OpenVINO Runtime <../../../openvino-workflow/running-inference>`.
* See sample applications in :doc:`OpenVINO toolkit Samples Overview <../../../get-started/learn-openvino/openvino-samples>`.
* Take a glance at the OpenVINO `product home page <https://software.intel.com/en-us/openvino-toolkit>`__ .




