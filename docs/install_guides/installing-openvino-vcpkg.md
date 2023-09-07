# Install OpenVINO™ Runtime via VCPKG {#openvino_docs_install_guides_installing_openvino_vcpkg}

@sphinxdirective

.. meta::
   :description: Learn how to install OpenVINO™ Runtime on Windows, Linux, and macOS 
                 operating systems, using VCPKG.

.. note::
   
   Note that the VCPKG distribution:

   * offers C++ API only
   * does not offer support for GNA and NPU inference
   * is dedicated to users of all major OSs: Windows, Linux, macOS.
   * may offer different hardware support under different operating systems.

.. tab-set::

   .. tab-item:: System Requirements
      :sync: system-requirements

      | Full requirement listing is available in:
      | `System Requirements Page <https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/system-requirements.html>`__
   
   .. tab-item:: Processor Notes
      :sync: processor-notes
   
      | To see if your processor includes the integrated graphics technology and supports iGPU inference, refer to:
      | `Product Specifications <https://ark.intel.com/>`__

   .. tab-item:: Software Requirements
      :sync: software-requirements

      * `vcpkg <https://vcpkg.io/en/getting-started>`__



Installing OpenVINO Runtime
###########################

1. Make sure that you have installed VCPKG on your system. If not, follow the 
   `VCPKG installation instructions <https://vcpkg.io/en/getting-started>`__.


2. Install OpenVINO using the following terminal command:

   .. code-block:: sh

      vcpkg install openvino

   VCPKG also enables you to install only selected components, by specifying them in the command.
   See the list of `available features <https://vcpkg.link/ports/openvino>`__, for example: 

   .. code-block:: sh

      vcpkg install openvino[cpu,ir]

Note that the VCPKG installation means building all packages and dependencies from source, 
which means the compiler stage will require additional time to complete the process. 

 
Congratulations! You've just Installed OpenVINO! For some use cases you may still 
need to install additional components. Check the 
:doc:`list of additional configurations <openvino_docs_install_guides_configurations_header>`
to see if your case needs any of them.




Uninstalling OpenVINO
#####################

To uninstall OpenVINO via VCPKG, use the following command:

.. code-block:: sh

   vcpkg uninstall openvino


What's Next?
####################

Now that you've installed OpenVINO Runtime, you can try the following things:

* Learn more about :doc:`OpenVINO Workflow <openvino_workflow>`.
* To prepare your models for working with OpenVINO, see :doc:`Model Preparation <openvino_docs_model_processing_introduction>`.
* See pre-trained deep learning models in our :doc:`Open Model Zoo <model_zoo>`.
* Learn more about :doc:`Inference with OpenVINO Runtime <openvino_docs_OV_UG_OV_Runtime_User_Guide>`.
* See sample applications in :doc:`OpenVINO toolkit Samples Overview <openvino_docs_OV_UG_Samples_Overview>`.
* Check out the OpenVINO product home page: https://software.intel.com/en-us/openvino-toolkit.



@endsphinxdirective
